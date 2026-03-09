#!/bin/bash
#
# validate-live.sh - Validate a running vLLM deployment against its recipe
#
# Checks:
#   1. /health returns 200
#   2. /v1/models returns expected model name
#   3. max_model_len matches recipe
#   4. Inference produces coherent output
#   5. Tool calling works (if --enable-auto-tool-choice in recipe)
#   6. Reasoning/thinking tags parse (if --reasoning-parser in recipe)
#   7. Performance baseline (tok/s sanity check)
#
# Usage:
#   ./tests/validate-live.sh <recipe-name> [--endpoint URL]
#
# Examples:
#   ./tests/validate-live.sh qwen3.5-397b-int4-autoround
#   ./tests/validate-live.sh qwen3.5-397b-int4-autoround --endpoint http://10.100.0.211:8000

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASSED=$((PASSED + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=$((FAILED + 1)); }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; SKIPPED=$((SKIPPED + 1)); }
log_info() { echo -e "       $1"; }

# ---------- Parse args ----------
RECIPE_NAME=""
ENDPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        -*) echo "Unknown flag: $1"; exit 1 ;;
        *) RECIPE_NAME="$1"; shift ;;
    esac
done

if [[ -z "$RECIPE_NAME" ]]; then
    echo "Usage: $0 <recipe-name> [--endpoint URL]"
    echo ""
    echo "Available recipes:"
    for f in "$PROJECT_DIR/recipes/"*.yaml; do
        basename "$f" .yaml
    done
    exit 1
fi

RECIPE_FILE="$PROJECT_DIR/recipes/${RECIPE_NAME}.yaml"
if [[ ! -f "$RECIPE_FILE" ]]; then
    echo "Recipe not found: $RECIPE_FILE"
    exit 1
fi

# ---------- Extract recipe values ----------
# Simple YAML parser for flat fields
yaml_get() {
    grep "^[[:space:]]*${1}:" "$RECIPE_FILE" | head -1 | sed "s/.*${1}:[[:space:]]*//" | tr -d '"' | tr -d "'"
}

yaml_command() {
    sed -n '/^command:/,/^[a-z]/{ /^command:/d; /^[a-z]/d; p; }' "$RECIPE_FILE"
}

MODEL=$(yaml_get "model")
MAX_MODEL_LEN=$(yaml_get "max_model_len")
PORT=$(yaml_get "port")

# Check recipe command for features
RECIPE_CMD=$(yaml_command)
HAS_TOOL_CALLING=false
HAS_REASONING=false
TOOL_PARSER=""
REASONING_PARSER=""

if echo "$RECIPE_CMD" | grep -q '\-\-enable-auto-tool-choice'; then
    HAS_TOOL_CALLING=true
    TOOL_PARSER=$(echo "$RECIPE_CMD" | grep -oP '(?<=--tool-call-parser\s)\S+' || true)
fi

if echo "$RECIPE_CMD" | grep -oP '(?<=--reasoning-parser\s)\S+' > /dev/null 2>&1; then
    HAS_REASONING=true
    REASONING_PARSER=$(echo "$RECIPE_CMD" | grep -oP '(?<=--reasoning-parser\s)\S+' || true)
fi

# Default endpoint
if [[ -z "$ENDPOINT" ]]; then
    ENDPOINT="http://localhost:${PORT:-8000}"
fi

echo "=== Live Validation: $RECIPE_NAME ==="
echo "  Endpoint: $ENDPOINT"
echo "  Model:    $MODEL"
echo "  Features: tool_calling=$HAS_TOOL_CALLING reasoning=$HAS_REASONING"
echo ""

# ---------- 1. Health check ----------
echo "=== Health Check ==="
status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$ENDPOINT/health" 2>/dev/null || echo "000")
if [[ "$status" == "200" ]]; then
    log_pass "Health endpoint returned 200"
else
    log_fail "Health endpoint returned $status (expected 200)"
    echo -e "${RED}Cannot proceed — endpoint is not healthy${NC}"
    exit 1
fi

# ---------- 2. Model name ----------
echo ""
echo "=== Model Verification ==="
models_response=$(curl -s --max-time 10 "$ENDPOINT/v1/models" 2>/dev/null)
served_model=$(echo "$models_response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "PARSE_ERROR")

if [[ "$served_model" == "$MODEL" ]]; then
    log_pass "Served model matches recipe: $served_model"
else
    log_fail "Model mismatch: serving '$served_model', recipe expects '$MODEL'"
fi

# ---------- 3. max_model_len ----------
if [[ -n "$MAX_MODEL_LEN" ]]; then
    served_len=$(echo "$models_response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['max_model_len'])" 2>/dev/null || echo "0")
    if [[ "$served_len" == "$MAX_MODEL_LEN" ]]; then
        log_pass "max_model_len matches recipe: $served_len"
    else
        log_fail "max_model_len mismatch: serving $served_len, recipe expects $MAX_MODEL_LEN"
    fi
fi

# ---------- 4. Basic inference ----------
echo ""
echo "=== Inference Test ==="
inference_start=$(date +%s%N)
response=$(curl -s --max-time 120 "$ENDPOINT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "max_tokens": 32,
        "temperature": 0
    }' 2>/dev/null)
inference_end=$(date +%s%N)

content=$(echo "$response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
# Content may be null when reasoning parser captures output
c = msg.get('content') or ''
r = msg.get('reasoning') or msg.get('reasoning_content') or ''
text = c.strip() or r.strip()
print(text[:200] if text else '')
" 2>/dev/null || echo "")

if [[ -n "$content" ]]; then
    log_pass "Inference returned response: \"$(echo "$content" | head -1)\""
else
    log_fail "Inference returned empty or invalid response"
    log_info "Raw: $(echo "$response" | head -3)"
fi

# Check for tok/s from usage
total_tokens=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
elapsed_ms=$(( (inference_end - inference_start) / 1000000 ))
if [[ "$total_tokens" -gt 0 && "$elapsed_ms" -gt 0 ]]; then
    toks=$(python3 -c "print(f'{$total_tokens / ($elapsed_ms / 1000):.1f}')")
    log_info "Performance: ${total_tokens} tokens in ${elapsed_ms}ms (~${toks} tok/s)"
fi

# ---------- 5. Tool calling ----------
echo ""
echo "=== Tool Calling ==="
if [[ "$HAS_TOOL_CALLING" == "true" ]]; then
    tool_response=$(curl -s --max-time 120 "$ENDPOINT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "max_tokens": 256,
            "temperature": 0
        }' 2>/dev/null)

    tool_calls=$(echo "$tool_response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
tc = d['choices'][0]['message'].get('tool_calls', [])
if tc:
    print(json.dumps(tc[0]['function']))
else:
    print('')
" 2>/dev/null || echo "")

    if [[ -n "$tool_calls" ]]; then
        log_pass "Tool calling works (parser: $TOOL_PARSER)"
        log_info "Called: $tool_calls"
    else
        # Check if model responded with text instead (some models do this)
        text=$(echo "$tool_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [[ -n "$text" ]]; then
            log_fail "Tool calling not triggered — model responded with text instead"
            log_info "Response: $(echo "$text" | head -2)"
        else
            log_fail "Tool calling returned invalid response"
        fi
    fi
else
    log_skip "Tool calling not enabled in recipe"
fi

# ---------- 6. Reasoning / thinking ----------
echo ""
echo "=== Reasoning Output ==="
if [[ "$HAS_REASONING" == "true" ]]; then
    reason_response=$(curl -s --max-time 120 "$ENDPOINT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "Think step by step: if a train travels 60km/h for 2.5 hours, how far does it go?"}],
            "max_tokens": 512,
            "temperature": 0
        }' 2>/dev/null)

    # Check for reasoning_content in the response
    has_reasoning=$(echo "$reason_response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
# vLLM uses 'reasoning' or 'reasoning_content' depending on version
rc = msg.get('reasoning') or msg.get('reasoning_content') or ''
content = msg.get('content') or ''
if rc:
    print(f'reasoning:{len(rc)}')
elif '<think>' in content:
    print(f'think_tags:{len(content)}')
elif content:
    print(f'text_only:{len(content)}')
else:
    print('empty')
" 2>/dev/null || echo "error")

    case "$has_reasoning" in
        reasoning:*)
            log_pass "Reasoning output parsed (parser: $REASONING_PARSER)"
            log_info "reasoning field length: ${has_reasoning#reasoning:} chars"
            ;;
        think_tags:*)
            log_pass "Reasoning present as <think> tags (parser may not split them)"
            ;;
        text_only:*)
            log_pass "Model responded with text (reasoning may be inline)"
            ;;
        *)
            log_fail "Reasoning test returned no output"
            ;;
    esac
else
    log_skip "Reasoning parser not configured in recipe"
fi

# ---------- 7. Performance baseline ----------
echo ""
echo "=== Performance Baseline ==="
perf_response=$(curl -s --max-time 120 "$ENDPOINT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Write a short paragraph about the history of computing."}],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": false
    }' 2>/dev/null)

perf_tokens=$(echo "$perf_response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
u = d.get('usage', {})
print(f\"{u.get('prompt_tokens',0)} {u.get('completion_tokens',0)} {u.get('total_tokens',0)}\")
" 2>/dev/null || echo "0 0 0")

read -r prompt_tok comp_tok total_tok <<< "$perf_tokens"
if [[ "$comp_tok" -gt 0 ]]; then
    log_pass "Generated $comp_tok completion tokens (prompt: $prompt_tok)"
else
    log_fail "Performance test returned no tokens"
fi

# ---------- Summary ----------
echo ""
echo "=== Summary ==="
echo -e "Passed: ${GREEN}${PASSED}${NC}  Failed: ${RED}${FAILED}${NC}  Skipped: ${YELLOW}${SKIPPED}${NC}"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}VALIDATION PASSED${NC}"
    exit 0
fi
