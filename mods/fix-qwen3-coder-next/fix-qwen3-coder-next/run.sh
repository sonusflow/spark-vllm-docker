#!/bin/bash
set -e

echo "Patching Qwen3-Coder-Next crashing on start"
patch -p1 -d /usr/local/lib/python3.12/dist-packages < fix_crash.diff || echo "Patch is not applicable, skipping"

# Restoring this one because the PR has been reverted in main
echo "Reverting PR #34279 that causes slowness"
patch -p1 -R -d /usr/local/lib/python3.12/dist-packages < fix_slowness.diff || echo "Can't revert PR #34279, skipping as it was reverted in recent commits"

# if grep -q "Cast to int64 to prevent overflow in stride" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py; then
#     echo "PR #34507 already applied, skipping."
# else
#     echo "Applying PR #34507 for slowness fix..."
#     curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34507.diff | patch -p1 -d /usr/local/lib/python3.12/dist-packages
# fi

echo "Fixing Triton allocator bug"
cp _triton* /usr/local/lib/python3.12/dist-packages/

echo "Fixing ignore_keys_at_rope_validation list-vs-set bug"
python3 << 'PYFIX'
p = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"
with open(p) as f:
    s = f.read()
old = 'kwargs["ignore_keys_at_rope_validation"] = [\n            "mrope_section",\n            "mrope_interleaved",\n        ]'
new = 'kwargs["ignore_keys_at_rope_validation"] = {\n            "mrope_section",\n            "mrope_interleaved",\n        }'
if old in s:
    s = s.replace(old, new)
    with open(p, "w") as f:
        f.write(s)
    print("Fixed: list -> set")
else:
    print("Already fixed or pattern changed")
PYFIX
