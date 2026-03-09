#!/bin/bash
#
# gpu-smoke-test.sh - GPU smoke test for DGX Spark
#
# Runs on a self-hosted runner (Spark 1). Validates:
#   1. Docker image exists
#   2. Container launches and model loads
#   3. /health endpoint returns 200
#   4. Inference produces valid response
#
# Safety: Skips if a vLLM container is already running (production guard).
#
# Usage:
#   ./tests/gpu-smoke-test.sh [--image IMAGE] [--model MODEL]
#
# Defaults:
#   --image vllm-node-tf5
#   --model Qwen/Qwen3-0.6B

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="vllm-node-tf5"
MODEL="Qwen/Qwen3-0.6B"
CONTAINER_NAME="vllm-smoke-test"
PORT=8199
TIMEOUT=600  # 10 minutes max
HEALTH_TIMEOUT=300  # 5 minutes for model load

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    echo "Cleaning up..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT

# ---------- Guard: Check for running vLLM ----------
echo "=== Pre-flight Checks ==="

if docker ps --format '{{.Names}}' | grep -q 'vllm'; then
    echo -e "${YELLOW}[SKIP]${NC} vLLM container is running (production). Skipping GPU tests."
    echo "::notice::GPU smoke test skipped — production vLLM is running"
    exit 0
fi

# ---------- Check Docker image ----------
if docker image inspect "$IMAGE" &>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} Docker image exists: $IMAGE"
else
    echo -e "${RED}[FAIL]${NC} Docker image not found: $IMAGE"
    echo "::error::Docker image $IMAGE not found on runner"
    exit 1
fi

# ---------- Check model exists ----------
MODEL_DIR="$HOME/.cache/huggingface/hub/models--$(echo "$MODEL" | tr '/' '--')"
if [[ -d "$MODEL_DIR" ]]; then
    echo -e "${GREEN}[PASS]${NC} Model cached: $MODEL"
else
    echo -e "${YELLOW}[INFO]${NC} Model not cached, will download: $MODEL"
fi

# ---------- Launch container ----------
echo ""
echo "=== Launching Smoke Test Container ==="

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --privileged \
    --network host \
    --ipc=host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" \
    bash -c -i "vllm serve $MODEL \
        --port $PORT \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.3 \
        --max-model-len 1024 \
        --max-num-batched-tokens 1024 \
        --enforce-eager \
        --trust-remote-code"

echo "Container started. Waiting for health..."

# ---------- Wait for health ----------
start_time=$(date +%s)
while true; do
    elapsed=$(( $(date +%s) - start_time ))

    if [[ $elapsed -gt $HEALTH_TIMEOUT ]]; then
        echo -e "${RED}[FAIL]${NC} Health check timeout after ${HEALTH_TIMEOUT}s"
        echo "Last logs:"
        docker logs --tail 30 "$CONTAINER_NAME" 2>&1
        exit 1
    fi

    status=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
    if [[ "$status" == "200" ]]; then
        echo -e "${GREEN}[PASS]${NC} Health endpoint returned 200 (${elapsed}s)"
        break
    fi

    # Check container still running
    if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo -e "${RED}[FAIL]${NC} Container exited unexpectedly"
        docker logs --tail 50 "$CONTAINER_NAME" 2>&1
        exit 1
    fi

    sleep 5
done

# ---------- Inference test ----------
echo ""
echo "=== Inference Test ==="

response=$(curl -s --max-time 60 "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0
    }')

# Validate response
if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content'].strip()" 2>/dev/null; then
    content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())")
    echo -e "${GREEN}[PASS]${NC} Inference returned valid response: \"$content\""
else
    echo -e "${RED}[FAIL]${NC} Invalid inference response:"
    echo "$response" | head -5
    exit 1
fi

# ---------- Summary ----------
echo ""
echo -e "${GREEN}=== GPU Smoke Test PASSED ===${NC}"
exit 0
