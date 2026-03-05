#!/bin/bash
# PROFILE: Intel Qwen3.5-397B-A17B INT4 (TP=4, all 4 Sparks)
# DESCRIPTION: vLLM serving Qwen3.5-397B with TP=4 across 4 DGX Spark nodes
# REQUIRES: fix-qwen35-tp4-marlin mod applied

export VLLM_MARLIN_USE_ATOMIC_ADD=1

vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.78 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
