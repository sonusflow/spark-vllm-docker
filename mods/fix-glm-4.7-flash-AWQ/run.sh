#!/bin/bash
set -e
echo "--- Applying GLM 4.7 AWQ speed patch..."
patch -p1 -d / < glm47_flash.patch
echo "=== OK"
echo "--- Applying vLLM crash patch..."
patch -p1 -d /usr/local/lib/python3.12/dist-packages < glm47_vllm_bug.patch || echo "=== Patch is not applicable, skipping"
echo "=== OK"
