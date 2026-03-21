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
T5GEMMA_MODEL="${T5GEMMA_MODEL_PATH:-google/t5gemma-2b-2b-ul2-it}"
echo "Checking model weights for ${T5GEMMA_MODEL}..."
huggingface-cli download "$T5GEMMA_MODEL"
echo "  T5Gemma model weights ready."

# Optionally download Qwen3 weights
QWEN_MODEL="${QWEN_MODEL_PATH:-Qwen/Qwen3-4B-Instruct}"
if [ "${DOWNLOAD_QWEN:-true}" = "true" ]; then
    echo "Checking model weights for ${QWEN_MODEL}..."
    huggingface-cli download "$QWEN_MODEL"
    echo "  Qwen3 model weights ready."
fi

echo ""
echo "============================================================"
echo "  tabular-llms-research ready"
echo "============================================================"
echo ""
echo "  cd /workspace/MMTU"
echo ""
echo "  # Smoke tests"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-2b-smoke.yaml"
echo ""
echo "  # Qwen3 (start vLLM first, then run)"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model ${QWEN_MODEL} &"
echo "  sleep 60 && python projects/tabular-llms-research/run.py run configs/qwen3-4b-smoke.yaml"
echo ""
echo "  # Full baseline"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-2b-full.yaml"
echo "  python projects/tabular-llms-research/run.py run configs/qwen3-4b-full.yaml"
echo ""
echo "  # Column permutation"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-2b-full-colperm.yaml"
echo "  python projects/tabular-llms-research/run.py run configs/qwen3-4b-full-colperm.yaml"
echo ""
echo "  # Natural language table format (Qwen2.5-7B)"
echo "  python projects/tabular-llms-research/run.py run configs/qwen3-4b-smoke-nl.yaml"
echo "  python projects/tabular-llms-research/run.py run configs/qwen3-4b-full-nl.yaml"
echo ""

exec sleep infinity
