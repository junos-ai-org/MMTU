#!/bin/bash
set -e

# Deploy latest code from image to persistent volume
echo "Deploying MMTU code to /workspace/MMTU..."
mkdir -p /workspace/MMTU
cp -r /opt/MMTU/. /workspace/MMTU/
echo "  Done."

# Use persistent cache on /workspace (survives pod restarts with network volume)
export HF_HOME="/workspace/.cache/huggingface"

# Download T5Gemma weights
T5GEMMA_MODEL="${T5GEMMA_MODEL_PATH:-google/t5gemma-9b-9b-ul2-it}"
echo "Checking model weights for ${T5GEMMA_MODEL}..."
huggingface-cli download "$T5GEMMA_MODEL"
echo "  T5Gemma model weights ready."

echo ""
echo "============================================================"
echo "  T5Gemma image ready"
echo "============================================================"
echo ""
echo "  cd /workspace/MMTU"
echo ""
echo "  # Smoke test"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-smoke.yaml"
echo ""
echo "  # Full baseline"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full.yaml"
echo ""
echo "  # Column permutation"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full-colperm.yaml"
echo ""

exec sleep infinity
