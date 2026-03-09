#!/bin/bash
#
# rollback-37toks.sh - Restore the proven 37 tok/s Qwen3.5-397B setup
#
# This script restores the exact configuration that achieved 37 tok/s
# single-user decode on the 4-node DGX Spark cluster (2026-03-09).
#
# What it does:
#   1. Stops any running vLLM containers on all 4 nodes
#   2. Verifies the tagged image exists (vllm-node-tf5:37toks-2026-03-09)
#   3. Restores :latest tag to the known-good image
#   4. Launches the cluster using the proven recipe
#
# Prerequisites:
#   - Driver 580.126.09 on all nodes (590.x will NOT work)
#   - Model cached at ~/.cache/huggingface/hub/models--Intel--Qwen3.5-397B-A17B-int4-AutoRound
#   - Docker image tagged vllm-node-tf5:37toks-2026-03-09 on all nodes
#
# Usage:
#   ./rollback-37toks.sh              # Full rollback and launch
#   ./rollback-37toks.sh --check      # Verify everything is in place without launching
#
# Cluster: Spark 1-4 (10.100.0.211-214)
# Image:   vllm-node-tf5:37toks-2026-03-09 (sha256:e8262d819075)
# Recipe:  qwen3.5-397b-int4-autoround
# Result:  37 tok/s single-user, 103 tok/s aggregate (4 concurrent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES="10.100.0.211 10.100.0.212 10.100.0.213 10.100.0.214"
IMAGE_TAG="vllm-node-tf5:37toks-2026-03-09"
IMAGE_SHA="e8262d819075"
CONTAINER_NAME="vllm_node"
RECIPE="qwen3.5-397b-int4-autoround"
CHECK_ONLY=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

echo "=== Rollback Check: 37 tok/s Configuration ==="
echo ""

errors=0

# ---------- Verify all nodes ----------
for ip in $NODES; do
    node_num=$(( ${ip##*.} - 210 ))
    echo "--- Spark $node_num ($ip) ---"

    # Check SSH
    if ! ssh -o ConnectTimeout=5 "sf-ai-cc@$ip" "true" 2>/dev/null; then
        echo -e "${RED}[FAIL]${NC} Cannot SSH to $ip"
        errors=$((errors + 1))
        continue
    fi

    # Check tagged image exists
    img_id=$(ssh "sf-ai-cc@$ip" "docker images '$IMAGE_TAG' --format '{{.ID}}'" 2>/dev/null)
    if [[ "$img_id" == "$IMAGE_SHA"* ]]; then
        echo -e "${GREEN}[OK]${NC} Tagged image exists: $IMAGE_TAG ($img_id)"
    else
        echo -e "${RED}[FAIL]${NC} Tagged image missing or wrong ID: got '$img_id', expected '$IMAGE_SHA'"
        errors=$((errors + 1))
    fi

    # Check driver
    driver=$(ssh "sf-ai-cc@$ip" "cat /proc/driver/nvidia/version 2>/dev/null | grep -oP '5[89]0\.[0-9.]+' | head -1" 2>/dev/null)
    if [[ "$driver" == 580.* ]]; then
        echo -e "${GREEN}[OK]${NC} Driver: $driver"
    else
        echo -e "${RED}[FAIL]${NC} Driver: $driver (need 580.x)"
        errors=$((errors + 1))
    fi

    # Check model cache
    model_exists=$(ssh "sf-ai-cc@$ip" "test -d ~/.cache/huggingface/hub/models--Intel--Qwen3.5-397B-A17B-int4-AutoRound && echo yes || echo no" 2>/dev/null)
    if [[ "$model_exists" == "yes" ]]; then
        echo -e "${GREEN}[OK]${NC} Model cached"
    else
        echo -e "${RED}[FAIL]${NC} Model not cached"
        errors=$((errors + 1))
    fi

    # Check current state
    running=$(ssh "sf-ai-cc@$ip" "docker ps --format '{{.Names}}' | grep -c vllm || echo 0" 2>/dev/null)
    if [[ "$running" -gt 0 ]]; then
        echo -e "${YELLOW}[INFO]${NC} vLLM container currently running"
    else
        echo -e "       No vLLM container running"
    fi
    echo ""
done

if [[ $errors -gt 0 ]]; then
    echo -e "${RED}$errors errors found. Fix before rollback.${NC}"
    exit 1
fi

echo -e "${GREEN}All nodes verified.${NC}"

if [[ "$CHECK_ONLY" == "true" ]]; then
    echo ""
    echo "Check-only mode. Run without --check to execute rollback."
    exit 0
fi

echo ""
echo "=== Executing Rollback ==="
echo ""

# ---------- Stop existing containers ----------
echo "Stopping existing containers..."
for ip in $NODES; do
    ssh "sf-ai-cc@$ip" "docker stop $CONTAINER_NAME 2>/dev/null; docker rm $CONTAINER_NAME 2>/dev/null" &
done
wait
echo "All containers stopped."

# ---------- Restore :latest tag ----------
echo "Restoring :latest tag from $IMAGE_TAG..."
for ip in $NODES; do
    ssh "sf-ai-cc@$ip" "docker tag $IMAGE_TAG vllm-node-tf5:latest"
done
echo "Tags restored."

# ---------- Clear stale caches ----------
echo "Clearing compile caches (prevents stale cache hangs)..."
for ip in $NODES; do
    ssh "sf-ai-cc@$ip" "rm -rf ~/.cache/vllm/torch_compile_cache/ ~/.cache/flashinfer/ ~/.triton/" &
done
wait
echo "Caches cleared."

# ---------- Launch via recipe ----------
echo ""
echo "Launching cluster with recipe: $RECIPE"
echo "Using: $SCRIPT_DIR/run-recipe.sh"
echo ""

cd "$SCRIPT_DIR"
if [[ -x "./run-recipe.py" ]]; then
    ./run-recipe.py "$RECIPE" -n "10.100.0.211,10.100.0.212,10.100.0.213,10.100.0.214"
else
    echo -e "${RED}run-recipe.py not found in $SCRIPT_DIR${NC}"
    echo "Manually launch with:"
    echo "  cd $SCRIPT_DIR"
    echo "  ./run-recipe.py $RECIPE -n 10.100.0.211,10.100.0.212,10.100.0.213,10.100.0.214"
    exit 1
fi
