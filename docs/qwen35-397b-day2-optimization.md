---
layout: default
title: "Day 2: Qwen3.5-397B Optimization & dtype Investigation"
---

# Day 2: Qwen3.5-397B Optimization & dtype Investigation

**Date:** 2026-03-04
**Cluster:** 4× NVIDIA DGX Spark (GB10, SM121, ARM64, 128 GiB unified memory each)
**Interconnect:** 100 Gbps RoCEv2 via MikroTik CRS812
**vLLM:** v0.16.1rc1 (NGC 26.01 base)
**FlashInfer:** v0.6.4
**Baseline from Day 1:** 36.15 tok/s (bfloat16, TP=4, `--enforce-eager`)

---

## Summary

Day 2 focused on two tracks: **performance optimization** (removing `--enforce-eager`, testing torch.compile, MTP speculative decoding, expert parallelism) and **investigating `--dtype float16`** as a potential fix for the Marlin TP constraint reported in [vllm#35924](https://github.com/vllm-project/vllm/issues/35924). Key findings:

1. **torch.compile works** — removing `--enforce-eager` gives a massive speedup (~77% on prior tests, now the default)
2. **MTP speculative decoding** — not yet supported for Qwen3.5 hybrid architecture in current vLLM
3. **Expert parallelism** — slightly slower than TP=4 at single-user, deferred for concurrent testing
4. **`--dtype float16` produces garbage on the 397B model** — confirmed through controlled A/B/C/D testing (see below)
5. **Container restart policy fixed** — `--restart unless-stopped` for production resilience

---

## Performance Optimization

### torch.compile (Enabled)

Removed `--enforce-eager` from the recipe and launch script. torch.compile was previously disabled due to hangs during early bring-up (Day 1), but now completes successfully with cached compilation.

| Setting | tok/s | Notes |
|---------|-------|-------|
| `--enforce-eager` (Day 1) | 36.15 | No compilation, eager execution |
| torch.compile (Day 2) | ~37 | Cached graph, ~48s compile time |

The compiled cache persists across restarts (`/root/.cache/vllm/torch_compile_cache/`), so subsequent launches skip the compilation step.

**Files changed:**
- `recipes/qwen3.5-397b-int4-autoround.yaml` — removed `--enforce-eager`
- `examples/vllm-qwen35-397b-tp4.sh` — removed `--enforce-eager`

### MTP Speculative Decoding (Not Available)

Qwen3.5 has built-in MTP (Multi-Token Prediction) heads. Attempted to enable with:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

**Result:** Not supported for the `Qwen3_5MoeForConditionalGeneration` architecture in vLLM v0.16.1rc1. The hybrid linear attention + MoE architecture doesn't have the MTP runner integration yet.

### Expert Parallelism (Deferred)

Tested `--enable-expert-parallel` to distribute 512 MoE experts across nodes instead of replicating them:

**Result:** Slightly slower than TP=4 at single-user concurrency. EP is expected to shine at higher concurrency (c2+) where different users can route to different expert partitions. Needs proper benchmarking with `llama-benchy` at various concurrency levels.

### Container Restart Policy (Fixed)

Changed from `--rm` to `--restart unless-stopped` in `launch-cluster.sh` to survive node reboots. Added proper cleanup in the stop logic to remove persistent containers.

**File changed:** `launch-cluster.sh`

---

## The `--dtype float16` Investigation

### Background

[Issue #35924](https://github.com/vllm-project/vllm/issues/35924) reported the Marlin `MIN_THREAD_N=64` constraint breaking Qwen3.5 at TP>=4. vLLM maintainer [@Isotr0py](https://github.com/Isotr0py) suggested using `--dtype float16` to enable ExllamaLinearKernel fallback for the small `in_proj_ba` layers that don't meet Marlin's requirements.

Isotr0py demonstrated this working on `Intel/Qwen3.5-35B-A3B-int4-AutoRound` with TP=2. We tested it on our GB10/SM121 hardware with both the 35B and 397B models.

### Controlled Test Matrix

All tests on 4× DGX Spark (GB10, SM121, ARM64), vLLM v0.16.1rc1, FlashInfer v0.6.4.

| Test | Model | dtype | TP | Kernel(s) Used | Output |
|------|-------|-------|----|----------------|--------|
| **A** | 397B INT4-AutoRound | bfloat16 | 4 | MarlinLinearKernel (ReplicatedLinear fix) | **Correct** |
| **B** | 397B INT4-AutoRound | float16 | 4 | MarlinLinearKernel + ExllamaLinearKernel | **Garbage** |
| **C** | 397B INT4-AutoRound | float16 | 4 | MarlinLinearKernel (ReplicatedLinear fix) | **Garbage** |
| **D** | 35B INT4-AutoRound | float16 | 2 | MarlinLinearKernel + ExllamaLinearKernel | **Correct** |

### Test Details

**Test A (baseline):** Our working production config — bfloat16 with ReplicatedLinear fix for `in_proj_ba`. Correct output on all prompts.

**Test B (Isotr0py's suggestion on 397B):** `--dtype float16` without our Marlin fix. vLLM selects ExllamaLinearKernel for the small `in_proj_ba` layers, MarlinLinearKernel for everything else.

```
Prompt: "What is the capital of France? Answer with just the city name."
Output: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..." (repeating garbage)
```

All 4 test prompts produced garbage — repeating `!` characters with no coherent text.

**Test C (isolating dtype vs kernel):** `--dtype float16` WITH our ReplicatedLinear fix applied. This means Exllama is NOT used — all layers go through MarlinLinearKernel. Still garbage output.

This proves the problem is `float16` dtype itself, not the Exllama kernel.

**Test D (replicating Isotr0py's exact 35B test):** Matching the exact command from the issue:

```bash
vllm serve Intel/Qwen3.5-35B-A3B-int4-AutoRound \
    --language-model-only \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --max-model-len 2048
```

Results — all correct:

| Prompt | Output |
|--------|--------|
| Capital of France | "Paris" (with reasoning) |
| "Hello, my name is" | "[Your Name], and I am a student at [Your School]..." |
| What is 2+2? | Correct reasoning, 2+2=4 |
| Explain gravity | Coherent physics explanation |

Kernel logs confirmed identical kernel selection: `MarlinLinearKernel + ExllamaLinearKernel`.

### Conclusion

`--dtype float16` works correctly on the **35B** model but produces garbage on the **397B** model, regardless of which kernel handles `in_proj_ba` (Marlin or Exllama). The root cause is **model-specific numerical overflow**:

- **float16** max value: ~65,504
- **bfloat16** max value: ~3.4 × 10³⁸

The 397B model has 512 MoE experts (vs 36 in the 35B), which likely produces larger intermediate activation values that overflow float16's limited range. The INT4 AutoRound quantization was calibrated assuming bfloat16 activations.

**`--dtype float16` is not a viable workaround for the 397B model.** Our ReplicatedLinear patch remains the only working solution.

---

## Alternative Quantizations Explored

### Qwen/Qwen3.5-397B-A17B-GPTQ-Int4

Downloaded the official Qwen GPTQ-Int4 quantization (236 GB, 108 files) to Spark 2. This uses `--quantization moe_wna16` which routes MoE expert layers through Triton/CUDA WNA16 kernels instead of Marlin, but attention layers still go through GPTQMarlin.

**Status:** Downloaded, not yet tested. Recipe prepared at `recipes/qwen3.5-397b-gptq-int4.yaml`.

Expected behavior: The `moe_wna16` quantization may still need our ReplicatedLinear fix for `in_proj_ba` since the non-MoE attention layers still use Marlin kernels.

### QuantTrio/Qwen3.5-397B-A17B-AWQ

Identified as another alternative (228 GB, AWQ quantization). Not downloaded yet.

---

## Recipes Created

### Test recipes (for reproducibility)

| Recipe | Purpose | Status |
|--------|---------|--------|
| `qwen3.5-397b-int4-autoround.yaml` | Production (bfloat16, MarlinFix, TP=4) | **Active** |
| `qwen3.5-397b-int4-autoround-fp16.yaml` | Test B: float16 + Exllama fallback | Garbage output |
| `qwen3.5-397b-int4-autoround-fp16-marlin.yaml` | Test C: float16 + MarlinFix | Garbage output |
| `qwen3.5-35b-fp16-test.yaml` | Test D: 35B float16 (match Isotr0py) | Correct output |
| `qwen3.5-397b-gptq-int4.yaml` | GPTQ-Int4 with moe_wna16 | Not yet tested |

---

## Current Production Configuration

```yaml
# recipes/qwen3.5-397b-int4-autoround.yaml
model: Intel/Qwen3.5-397B-A17B-int4-AutoRound
container: vllm-node-tf5
mods:
  - mods/fix-qwen3-coder-next    # Qwen3-Coder tool/reasoning parser
  - mods/fix-qwen35-tp4-marlin   # ReplicatedLinear fix for in_proj_ba

defaults:
  tensor_parallel: 4
  gpu_memory_utilization: 0.78
  max_model_len: 32768
  max_num_batched_tokens: 8192

env:
  VLLM_MARLIN_USE_ATOMIC_ADD: 1
  VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: 1

command: |
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
    --host 0.0.0.0 --port 8000
```

Key changes from Day 1:
- Removed `--enforce-eager` (torch.compile now works)
- Added `--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice`
- Added `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`
- Added `--enable-prefix-caching`

---

## Next Steps

1. **FlashInfer v0.6.5 rebuild** — Dropped today with fused MoE+GEMM AOT modules for SM121. Community reports up to 75% speedup. Requires `build-and-copy.sh --flashinfer-ref v0.6.5 --rebuild-flashinfer`.
2. **GPTQ-Int4 benchmark** — Test `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` with `--quantization moe_wna16` and compare tok/s.
3. **MTP speculative decoding** — Monitor vLLM main for Qwen3.5 MTP support.
4. **Expert parallelism at concurrency** — Benchmark EP vs TP at c2/c4/c8 with `llama-benchy`.
5. **Upstream PR** — Submit the ReplicatedLinear fix as a proper PR to vllm-project/vllm.

---

## Related Links

- [Day 1: TP=4 Marlin Fix](qwen35-397b-tp4) — Initial bring-up, Marlin constraint root cause, ReplicatedLinear patch
- [vllm#35924](https://github.com/vllm-project/vllm/issues/35924) — Our upstream issue with the dtype investigation findings
- [NETWORKING](NETWORKING) — RDMA fabric setup and CRS812 QoS configuration
