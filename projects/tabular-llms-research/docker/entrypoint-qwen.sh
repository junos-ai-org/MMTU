#!/bin/bash
set -e

# Deploy latest code from image to persistent volume
echo "Deploying MMTU code to /workspace/MMTU..."
mkdir -p /workspace/MMTU
cp -r /opt/MMTU/. /workspace/MMTU/
echo "  Done."

# Use persistent cache on /workspace (survives pod restarts with network volume)
export HF_HOME="/workspace/.cache/huggingface"

# Download Qwen2.5 weights
QWEN_MODEL="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
echo "Checking model weights for ${QWEN_MODEL}..."
huggingface-cli download "$QWEN_MODEL"
echo "  Qwen2.5 model weights ready."

echo ""
echo "============================================================"
echo "  Qwen2.5 image ready"
echo "============================================================"
echo ""
echo "  cd /workspace/MMTU"
echo ""
echo "  # Start vLLM server"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model ${QWEN_MODEL} &"
echo ""
echo "  # Smoke test (wait for vLLM to start)"
echo "  sleep 60 && python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-smoke.yaml"
echo ""
echo "  # Full baseline"
echo "  python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-full.yaml"
echo ""
echo "  # Column permutation"
echo "  python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-full-colperm.yaml"
echo ""

exec sleep infinity
