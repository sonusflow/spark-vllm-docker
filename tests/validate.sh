#!/bin/bash
#
# validate.sh - Static validation for spark-vllm-docker
#
# Checks:
#   1. ShellCheck on all .sh files
#   2. Recipe YAML schema (required fields)
#   3. Mod structure (run.sh exists)
#   4. No hardcoded private IPs
#   5. No credential patterns
#   6. Recipe mod references resolve
#
# Usage:
#   ./tests/validate.sh          # Run all checks
#   ./tests/validate.sh -v       # Verbose output

set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERBOSE="${1:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNED=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASSED=$((PASSED + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=$((FAILED + 1)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARNED=$((WARNED + 1)); }
log_info() { [[ "$VERBOSE" == "-v" ]] && echo "       $1"; }

# ---------- 1. ShellCheck ----------
echo ""
echo "=== ShellCheck ==="

if command -v shellcheck &>/dev/null; then
    shell_errors=0
    while IFS= read -r script; do
        if shellcheck -S warning -e SC1091,SC2034,SC2086,SC2155 "$script" &>/dev/null; then
            log_info "OK: $script"
        else
            log_fail "ShellCheck errors in $(basename "$script")"
            [[ "$VERBOSE" == "-v" ]] && shellcheck -S warning -e SC1091,SC2034,SC2086,SC2155 "$script" 2>&1 | head -20
            shell_errors=$((shell_errors + 1))
        fi
    done < <(find "$PROJECT_DIR" -maxdepth 1 -name "*.sh" -type f)

    if [[ $shell_errors -eq 0 ]]; then
        log_pass "All shell scripts pass ShellCheck"
    fi
else
    log_warn "ShellCheck not installed, skipping"
fi

# ---------- 2. Recipe YAML Schema ----------
echo ""
echo "=== Recipe Schema ==="

for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
    [[ -f "$recipe" ]] || continue
    name="$(basename "$recipe")"

    # Required fields
    has_version=$(grep -c '^recipe_version:' "$recipe")
    has_name=$(grep -c '^name:' "$recipe")
    has_container=$(grep -c '^container:' "$recipe")
    has_command=$(grep -c '^command:' "$recipe")

    if [[ $has_version -gt 0 && $has_name -gt 0 && $has_container -gt 0 && $has_command -gt 0 ]]; then
        log_pass "Recipe schema valid: $name"
    else
        missing=""
        [[ $has_version -eq 0 ]] && missing="$missing recipe_version"
        [[ $has_name -eq 0 ]] && missing="$missing name"
        [[ $has_container -eq 0 ]] && missing="$missing container"
        [[ $has_command -eq 0 ]] && missing="$missing command"
        log_fail "Recipe missing required fields ($missing): $name"
    fi
done

# ---------- 3. Mod Structure ----------
echo ""
echo "=== Mod Structure ==="

for mod_dir in "$PROJECT_DIR/mods/"*/; do
    [[ -d "$mod_dir" ]] || continue
    mod_name="$(basename "$mod_dir")"

    if [[ -f "$mod_dir/run.sh" ]]; then
        log_pass "Mod has run.sh: $mod_name"
    else
        log_fail "Mod missing run.sh: $mod_name"
    fi
done

# ---------- 4. Recipe Mod References ----------
echo ""
echo "=== Mod References ==="

for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
    [[ -f "$recipe" ]] || continue
    name="$(basename "$recipe")"

    # Extract mod paths from recipe
    while IFS= read -r mod_path; do
        mod_path=$(echo "$mod_path" | sed 's/^[[:space:]]*- //' | tr -d '[:space:]')
        [[ -z "$mod_path" ]] && continue

        if [[ -d "$PROJECT_DIR/$mod_path" ]]; then
            log_info "Mod exists: $mod_path (in $name)"
        else
            log_fail "Referenced mod not found: $mod_path (in $name)"
        fi
    done < <(sed -n '/^mods:/,/^[^ ]/{ /^  - /p }' "$recipe")
done

# ---------- 5. No Private IPs ----------
echo ""
echo "=== Secrets Scan ==="

ip_hits=$(grep -rn --include='*.yaml' --include='*.sh' --include='*.py' \
    -E '(10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+)' \
    "$PROJECT_DIR" \
    --exclude-dir=.git --exclude-dir=tests \
    2>/dev/null | grep -v '#.*example' | grep -v 'CLUSTER_NODES' || true)

if [[ -z "$ip_hits" ]]; then
    log_pass "No hardcoded private IPs found"
else
    log_fail "Hardcoded private IPs detected:"
    echo "$ip_hits" | head -10
fi

# ---------- 6. No Credentials ----------
cred_hits=$(grep -rn --include='*.yaml' --include='*.sh' --include='*.py' \
    -iE '(password|token|secret|api_key)\s*[=:]' \
    "$PROJECT_DIR" \
    --exclude-dir=.git --exclude-dir=tests --exclude='validate.sh' \
    2>/dev/null | grep -v '^#' | grep -v 'HF_TOKEN' | grep -v 'example' || true)

if [[ -z "$cred_hits" ]]; then
    log_pass "No credential patterns found"
else
    log_fail "Potential credentials detected:"
    echo "$cred_hits" | head -10
fi

# ---------- Summary ----------
echo ""
echo "=== Summary ==="
echo -e "Passed: ${GREEN}${PASSED}${NC}  Failed: ${RED}${FAILED}${NC}  Warnings: ${YELLOW}${WARNED}${NC}"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}VALIDATION PASSED${NC}"
    exit 0
fi
