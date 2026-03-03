#!/bin/bash
# Fix Marlin TP=4 constraint for Qwen3.5-397B: in_proj_ba output_size=128 / TP=4 = 32 < min_thread_n=64
# Solution: Replace MergedColumnParallelLinear with two ReplicatedLinear for B/A projections

set -e
MOD_DIR="$(dirname "$0")"
MODELS_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

echo "[fix-qwen35-tp4-marlin] Backing up originals..."
cp "$MODELS_DIR/qwen3_next.py" "$MODELS_DIR/qwen3_next.py.bak" 2>/dev/null || true
cp "$MODELS_DIR/qwen3_5.py" "$MODELS_DIR/qwen3_5.py.bak" 2>/dev/null || true

echo "[fix-qwen35-tp4-marlin] Installing patched files..."
cp "$MOD_DIR/qwen3_next.py" "$MODELS_DIR/qwen3_next.py"
cp "$MOD_DIR/qwen3_5.py" "$MODELS_DIR/qwen3_5.py"

echo "[fix-qwen35-tp4-marlin] Verifying no in_proj_ba references remain..."
if grep -q "in_proj_ba" "$MODELS_DIR/qwen3_next.py" "$MODELS_DIR/qwen3_5.py" 2>/dev/null; then
    echo "[fix-qwen35-tp4-marlin] WARNING: in_proj_ba still found in patched files!"
else
    echo "[fix-qwen35-tp4-marlin] OK — in_proj_ba fully replaced with in_proj_b + in_proj_a"
fi

echo "[fix-qwen35-tp4-marlin] Fixing ignore_keys_at_rope_validation list→set..."
python3 "$MOD_DIR/fix_rope.py"
