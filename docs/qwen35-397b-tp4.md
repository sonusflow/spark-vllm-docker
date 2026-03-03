---
layout: default
title: "Qwen3.5-397B on 4x DGX Spark — TP=4 Marlin Fix"
---

# Qwen3.5-397B-A17B INT4 on 4× DGX Spark with TP=4

**Date:** 2026-03-03  
**Model:** [Intel/Qwen3.5-397B-A17B-int4-AutoRound](https://huggingface.co/Intel/Qwen3.5-397B-A17B-int4-AutoRound)  
**Cluster:** 4× NVIDIA DGX Spark (GB10, 128 GiB unified memory each)  
**Interconnect:** 100 Gbps RoCEv2 via MikroTik CRS812 switch  
**vLLM version:** 0.9.x (NGC 26.01 container base)

---

## The Problem

Running Qwen3.5-397B with **tensor parallelism = 4** across 4 DGX Spark nodes fails immediately with:

```
ValueError: output_size_per_partition = 32 is not divisible by
min_thread_n = 64 (GPTQ_MARLIN_MIN_THREAD_N)
```

The Marlin GPTQ kernel requires output dimensions to be at least 64 per TP partition. With TP=4, certain weight dimensions become too small.

## Root Cause Analysis

### Weight Shape Investigation

We extracted actual weight shapes from the INT4 AutoRound safetensors (40 shards, 199 GiB total):

| Weight | Real Shape | TP=4 Output | Status |
|--------|-----------|-------------|--------|
| `linear_attn.in_proj_a.qweight` | [4096, 64] | 16 | **FAILS** |
| `linear_attn.in_proj_b.qweight` | [4096, 64] | 16 | **FAILS** |
| `linear_attn.in_proj_qkv.qweight` | [4096, 12288] | 3072 | OK |
| `linear_attn.in_proj_z.qweight` | [4096, 8192] | 2048 | OK |
| `experts.X.gate_proj.qweight` | [4096, 1024] | 256 | OK |
| `self_attn.q_proj.qweight` | [4096, 16384] | 4096 | OK |
| `self_attn.k_proj.qweight` | [4096, 512] | 128 | OK |
| `self_attn.v_proj.qweight` | [4096, 512] | 128 | OK |

### The Culprit: `in_proj_ba`

Qwen3.5 uses a **hybrid architecture** mixing standard Transformer attention with linear attention (DeltaNet/Mamba-style state-space layers). The linear attention layers have B and A state projections with `linear_num_value_heads = 64`.

In vLLM, these two projections are fused into a single `MergedColumnParallelLinear`:

```python
# qwen3_next.py — ORIGINAL
self.in_proj_ba = MergedColumnParallelLinear(
    input_size=self.hidden_size,       # 4096
    output_sizes=[self.num_v_heads] * 2,  # [64, 64] → total 128
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.in_proj_ba",
)
```

With TP=4, `MergedColumnParallelLinear` splits the output: **128 / 4 = 32**, which is below Marlin's `min_thread_n = 64`.

The weight files store them separately as `in_proj_b` (shape 64) and `in_proj_a` (shape 64). vLLM's `stacked_params_mapping` merges them at load time:

```python
# qwen3_5.py — stacking config
("in_proj_ba", "in_proj_b", 0),
("in_proj_ba", "in_proj_a", 1),
```

## The Fix: ReplicatedLinear with Manual TP Slicing

Instead of merging B and A into a single TP-sharded layer (which breaks Marlin), we use **`ReplicatedLinear`** — every rank loads the full weight and slices its partition in the forward pass.

### Patch 1: Layer Definition (`qwen3_next.py`)

```diff
-        self.in_proj_ba = MergedColumnParallelLinear(
-            input_size=self.hidden_size,
-            output_sizes=[self.num_v_heads] * 2,
-            bias=False,
-            quant_config=quant_config,
-            prefix=f"{prefix}.in_proj_ba",
-        )
+        self.in_proj_b = ReplicatedLinear(
+            input_size=self.hidden_size,
+            output_size=self.num_v_heads,
+            bias=False,
+            quant_config=quant_config,
+            prefix=f"{prefix}.in_proj_b",
+        )
+        self.in_proj_a = ReplicatedLinear(
+            input_size=self.hidden_size,
+            output_size=self.num_v_heads,
+            bias=False,
+            quant_config=quant_config,
+            prefix=f"{prefix}.in_proj_a",
+        )
```

### Patch 2: Forward Pass (`qwen3_next.py`)

```diff
-        projected_states_ba, _ = self.in_proj_ba(hidden_states)
+        b_full, _ = self.in_proj_b(hidden_states)
+        a_full, _ = self.in_proj_a(hidden_states)
+        _ba_chunk = self.num_v_heads // self.tp_size
+        _ba_start = self.tp_rank * _ba_chunk
+        projected_states_ba = torch.cat([
+            b_full[:, _ba_start:_ba_start+_ba_chunk],
+            a_full[:, _ba_start:_ba_start+_ba_chunk],
+        ], dim=-1)
```

### Patch 3: Forward Pass (`qwen3_5.py`)

```diff
-        ba, _ = self.in_proj_ba(hidden_states)
-        b, a = ba.chunk(2, dim=-1)
-        b = b.contiguous()
-        a = a.contiguous()
+        b_full, _ = self.in_proj_b(hidden_states)
+        a_full, _ = self.in_proj_a(hidden_states)
+        _ba_chunk = self.num_v_heads // self.tp_size
+        _ba_start = self.tp_rank * _ba_chunk
+        b = b_full[:, _ba_start:_ba_start+_ba_chunk].contiguous()
+        a = a_full[:, _ba_start:_ba_start+_ba_chunk].contiguous()
```

### Patch 4: Weight Mapping Cleanup

Remove stacking and packing entries since weights now load directly:

```diff
 # qwen3_5.py — stacked_params_mapping
-            ("in_proj_ba", "in_proj_b", 0),
-            ("in_proj_ba", "in_proj_a", 1),

 # qwen3_5.py — packed_modules_mapping
-        "in_proj_ba": ["in_proj_b", "in_proj_a"],

 # qwen3_next.py — packed_modules_mapping
-        "in_proj_ba": ["in_proj_ba"],
```

### Patch 5: Transformers Rope Validation Bug

A separate pre-existing bug in `vllm/transformers_utils/configs/qwen3_5_moe.py`:

```diff
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
             "mrope_section",
             "mrope_interleaved",
-        ]
+        }
```

The transformers library uses `|` (set union) on this field, but vLLM's Qwen3.5 config returns a `list` instead of a `set`.

## Additional Fixes Required

### NCCL IB Device Selection

DGX Spark (sf-ai-spark) has **4 RDMA devices**:

| Device | Interface | IP | Purpose |
|--------|-----------|-----|---------|
| `rocep1s0f0` | `enp1s0f0np0` | 192.168.200.1 | CRS812 fabric |
| `rocep1s0f1` | `enp1s0f1np1` | 169.254.60.178 | Unused (link-local) |
| `roceP2p1s0f0` | `enP2p1s0f0np0` | 169.254.227.230 | Unused (link-local) |
| `roceP2p1s0f1` | `enP2p1s0f1np1` | 192.168.201.1 | P2P to 5090 |

The auto-detection sets `NCCL_IB_HCA` to all 4 devices. NCCL then tries to reach remote nodes via the link-local addresses, causing:

```
ibv_modify_qp failed with 110 Connection timed out,
remote GID ::ffff:169.254.60.178
```

**Fix:** Explicitly specify `--ib-if rocep1s0f0` and set `NCCL_IB_GID_INDEX=3` (RoCEv2).

### GPU Memory Utilization

The head node (SparkA) runs Ray head services (GCS, dashboard, monitors) alongside the GPU worker. These consume ~23 GiB of the 120 GiB unified memory, leaving insufficient room for `--gpu-memory-utilization 0.85`.

**Fix:** Lower to `0.78` (93.3 GiB per node).

## Final Launch Command

```bash
./launch-cluster.sh \
    --ib-if rocep1s0f0 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_ROCE_VERSION_NUM=2 \
    --apply-mod mods/fix-qwen3-coder-next \
    --apply-mod mods/fix-qwen35-tp4-marlin \
    --launch-script examples/vllm-qwen35-397b-tp4.sh \
    -d
```

vLLM serve arguments:

```bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1

vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.78 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

## Results

### Cluster Status

| Node | Hostname | RDMA IP | GPU Memory | Role |
|------|----------|---------|------------|------|
| Spark 1 | sf-ai-spark | 192.168.200.1 | 97.9 GiB / 120 GiB | Head (rank 0) |
| Spark 2 | spark-e36a | 192.168.200.3 | 97.9 GiB / 120 GiB | Worker (rank 1) |
| Spark 3 | spark-20cb | 192.168.200.4 | 97.9 GiB / 120 GiB | Worker (rank 2) |
| Spark 4 | spark-e37c | 192.168.200.5 | 97.9 GiB / 120 GiB | Worker (rank 3) |

### Memory Breakdown (per node)

| Component | Size |
|-----------|------|
| Model weights (INT4) | 49.33 GiB |
| KV cache (FP8) | 54.42 GiB |
| CUDA context + NCCL | ~3.5 GiB |
| **Total GPU** | **~97.9 GiB** |

### Loading Times

| Phase | Head Node | Workers |
|-------|-----------|---------|
| Weight download (safetensors read) | 1104s | ~145s |
| Weight loading (GPU transfer) | 272s | ~165s |
| **Total model init** | **~23 min** | **~3 min** |

> Head node is slower due to Ray head services (GCS, dashboard, monitors, metric agents) competing for CPU and memory bandwidth on the same GB10 SoC.

### Inference

- **API endpoint:** `http://192.168.200.1:8000/v1/chat/completions`
- **Max context:** 32,768 tokens
- **Quantization:** INT4 (AutoRound GPTQ) with Marlin kernels
- **Attention:** FlashInfer backend
- **KV cache:** FP8
- **Reasoning:** Built-in chain-of-thought (Qwen3.5 reasoning model)

### Sample Output

```
User: Hello, who are you?

Qwen3.5: Hello\! I'm Qwen3.5, the latest large language model developed
by Tongyi Lab. I'm designed to assist with a wide range of tasks, from
answering questions and creating content to coding, logical reasoning,
and even analyzing complex documents or visuals. I support over 100
languages, handle ultra-long contexts (up to 256K tokens), and have
been optimized for precision, efficiency, and natural interaction.
```

## Architecture Notes

### Why ReplicatedLinear?

Three approaches were considered:

1. **Pipeline Parallelism (PP=4)**: Would avoid the Marlin constraint entirely but underutilizes the 100G RDMA fabric — pipeline bubbles reduce throughput.

2. **Hybrid TP=2/PP=2**: Splits 4 nodes into 2 TP groups of 2, pipelined. Better than PP=4 but still has bubble overhead and complex scheduling.

3. **Patch vLLM (chosen)**: Replace the problematic fused layer with replicated weights. Each rank loads the full 64-element B and A projections (tiny — 4096×64 INT4 = 32 KiB each) and slices its TP partition in the forward pass. Zero performance impact since these are negligible compared to the MoE expert weights (512 experts × 3 projections × 4096×1024 each).

### Qwen3.5 Hybrid Architecture

```
60 layers total:
├── Linear Attention (DeltaNet-style) — 45 layers
│   ├── in_proj_qkv: [4096, 12288]  — Q/K/V for linear attention
│   ├── in_proj_z:   [4096, 8192]   — gate projection
│   ├── in_proj_b:   [4096, 64]     — B state (← THE PROBLEM)
│   ├── in_proj_a:   [4096, 64]     — A state (← THE PROBLEM)
│   ├── out_proj:    [8192, 4096]   — output
│   └── MoE: 512 experts × {gate, up, down}_proj
│
└── Full Attention (standard Transformer) — 15 layers
    ├── q_proj: [4096, 16384]
    ├── k_proj: [4096, 512]
    ├── v_proj: [4096, 512]
    ├── o_proj: [8192, 4096]
    └── MoE: 512 experts × {gate, up, down}_proj
```

## Backward Compatibility

The ReplicatedLinear fix is inherently backward compatible across all TP values:

| TP Size | B/A Slice per Rank | Original Behavior | Fix Behavior |
|---------|-------------------|-------------------|--------------|
| **TP=1** | 64 (full tensor) | `MergedColumnParallelLinear` output=128, no split | `ReplicatedLinear` full output, slice=full tensor — functionally identical |
| **TP=2** | 32 elements | Output=128/2=64 — passes Marlin min_thread_n=64 barely | `ReplicatedLinear` full output, slice to 32 — works, fix still necessary for safety margin |
| **TP=4** | 16 elements | Output=128/4=32 — **FAILS** (below Marlin min_thread_n=64) | `ReplicatedLinear` full output, slice to 16 — works (original failure case) |

- **Other models**: Patches only modify `qwen3_next.py` and `qwen3_5.py` — no effect on GLM, GPT-OSS, MiniMax, or any other model files.
- **Recipe override**: `./run-recipe.py qwen3.5-397b-int4-autoround --tensor_parallel 2` works — defaults to TP=4 but accepts any value.

## Files Modified

All patches are applied at container startup via the mod system — no permanent changes to the Docker image. Patches are delivered as unified diffs (portable across vLLM versions).

```
mods/fix-qwen35-tp4-marlin/
├── run.sh              # Mod installer (applies patches + runs fix_rope.py)
├── qwen3_next.patch    # Unified diff: init, forward, packed_modules_mapping
├── qwen3_5.patch       # Unified diff: forward, stacked_params, packed_modules ×2
└── fix_rope.py         # Fixes ignore_keys_at_rope_validation list→set

recipes/
└── qwen3.5-397b-int4-autoround.yaml  # Recipe for 4-node TP=4 deployment

examples/
└── vllm-qwen35-397b-tp4.sh  # Launch script with tuned parameters
```
