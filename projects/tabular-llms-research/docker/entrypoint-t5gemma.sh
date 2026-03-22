#!/bin/bash
set -e

# Set up SSH deploy key from RunPod secret (if provided)
if [ -n "$DEPLOY_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$DEPLOY_KEY" > ~/.ssh/id_ed25519
    chmod 600 ~/.ssh/id_ed25519
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
    echo "SSH deploy key configured."
fi

# Clone or update MMTU code
MMTU_REPO="${MMTU_GIT_URL:-git@github.com:junos-ai-org/MMTU.git}"
MMTU_REF="${MMTU_GIT_REF:-main}"

if [ -d /workspace/MMTU/.git ]; then
    echo "Updating MMTU code (ref: ${MMTU_REF})..."
    cd /workspace/MMTU && git fetch origin && git checkout "$MMTU_REF" && git pull origin "$MMTU_REF" || true
    cd /
else
    echo "Cloning MMTU code (ref: ${MMTU_REF})..."
    git clone --depth 1 --branch "$MMTU_REF" "$MMTU_REPO" /workspace/MMTU
fi

pip install -q -r /workspace/MMTU/requirements.txt

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
echo "  # Update code mid-session (no restart needed)"
echo "  mmtu-update"
echo ""

exec sleep infinity
