# Qwen3.5-397B-A17B INT4 AutoRound — Benchmark Results

Benchmark results from a 4-node DGX Spark cluster running Qwen3.5-397B-A17B with INT4 AutoRound quantization.

## Hardware

- **Nodes:** 4× DGX Spark (GB10 / SM121)
- **Memory:** 128 GB unified per node (512 GB total)
- **Interconnect:** 2× ConnectX-7 per node, 200G RoCEv2, dual-rail
- **Switches:** MikroTik CRS812 + CRS804 (400G inter-switch link)

## Software

- **Driver:** NVIDIA 580.126.09 (see [Driver Compatibility](#driver-compatibility))
- **CUDA:** 13.0 (host) / 13.1 (container, forward compat layer)
- **vLLM:** Built from main branch (2026-03-05)
- **Container:** Based on `nvcr.io/nvidia/pytorch:26.01-py3`
- **Recipe:** `qwen3.5-397b-int4-autoround.yaml`

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Intel/Qwen3.5-397B-A17B-int4-AutoRound |
| Tensor Parallelism | 4 (1 GPU per node) |
| Quantization | INT4 AutoRound (Marlin kernels) |
| KV Cache | FP8 |
| Max Model Length | 32,768 tokens |
| Max Batched Tokens | 8,192 |
| Prefix Caching | Enabled |
| CUDAGraphs | Enabled (default) |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | 1 |

## Results

### Single User (Sequential)

| Output Tokens | tok/s (avg) | Variance |
|--------------|-------------|----------|
| 128 | 37.4 | ±0.1 |
| 256 | 37.2 | ±0.2 |
| 512 | 36.8 | ±0.2 |
| 1024 | 36.2 | ±0.1 |

### Concurrent Users

| Concurrency | Per-User tok/s | Aggregate tok/s |
|-------------|---------------|-----------------|
| 1 | 37.4 | 37.4 |
| 2 | 33.1 | 66.2 |
| 4 | 25.8 | 103.2 |

### llama-benchy v0.3.4 (Full Suite)

| Prompt (pp) | Generate (tg) | Concurrency | tok/s |
|------------|--------------|-------------|-------|
| 512 | 128 | 1 | 37.3 |
| 2048 | 128 | 1 | 36.9 |
| 4096 | 128 | 1 | 36.5 |
| 16384 | 128 | 1 | 33.8 |
| 512 | 512 | 1 | 36.8 |
| 512 | 128 | 4 | 121.4 (aggregate) |

## Driver Compatibility

> **CRITICAL: Stay on driver 580.126.09 for DGX Spark.**

Driver 590.48.01 introduces two bugs on GB10 unified memory architecture:

1. **UMA Memory Leak** — After CUDA processes exit, ~80-96 GiB is not released. Requires `echo 3 > /proc/sys/vm/drop_caches` before each launch.
2. **CUDAGraph Capture Deadlock** — After model loads, CUDAGraph capture hangs indefinitely at 0% GPU. Using `--enforce-eager` bypasses this but costs ~17 tok/s (37→20).

Driver 580.126.09 is the officially supported driver for DGX Spark (confirmed by NVIDIA forum staff). The container's CUDA 13.1 forward compatibility layer works correctly with host driver 580 — no performance penalty.

## Pre-Launch Checklist

Before launching vLLM on a Spark cluster:

1. **Drop caches** on all nodes: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
2. **Verify driver**: `nvidia-smi` should show 580.126.09
3. **Clear compile caches** (if image/config changed):
   - `~/.cache/vllm/torch_compile_cache/`
   - `~/.cache/flashinfer/`
   - `~/.triton/`
4. **Check RDMA interfaces**: `ibstatus` should show ACTIVE/LinkUp on all ports
5. **Verify GPU persistence**: `nvidia-smi -pm 1` (already set on boot)

## Startup Timeline (Typical)

| Phase | Duration | Notes |
|-------|----------|-------|
| Container startup + Ray init | ~30s | Head + 3 workers join |
| Model loading | ~3 min | 49.3 GiB per node from HF cache |
| torch.compile | ~2 min | First run only; cached after |
| FlashInfer autotuning | ~30s | Tunes MoE kernels |
| CUDAGraph capture | ~1.5 min | 80+ graphs captured |
| **Total cold start** | **~8 min** | Subsequent starts ~5 min (cached) |
