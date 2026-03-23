#!/bin/bash
set -euo pipefail

# Run a full experiment: build dataset (if needed) + inference + evaluate + analyze.
#
# Usage:
#   ./projects/tabular-llms-research/run_experiment.sh <dataset_config> <run_config>
#
# Examples:
#   ./projects/tabular-llms-research/run_experiment.sh \
#       experiments/encoder_vs_decoder_baseline/configs/dataset.yaml \
#       experiments/encoder_vs_decoder_baseline/configs/run_qwen.yaml
#
#   # All context_growth buckets
#   for bucket in 0_24k 24k_48k 48k_72k 72k_96k 96k_120k; do
#       ./projects/tabular-llms-research/run_experiment.sh \
#           experiments/context_growth/configs/dataset_${bucket}.yaml \
#           experiments/context_growth/configs/run_qwen_${bucket}.yaml
#   done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $(basename "$0") <dataset_config> <run_config>"
    echo ""
    echo "  dataset_config  Path to dataset YAML (relative to project dir or absolute)"
    echo "  run_config      Path to run YAML (relative to project dir or absolute)"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

DATASET_CONFIG="$1"
RUN_CONFIG="$2"

# Resolve dataset config path
if [ ! -f "$DATASET_CONFIG" ]; then
    DATASET_CONFIG="${SCRIPT_DIR}/${DATASET_CONFIG}"
fi
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "Error: dataset config not found: $1"
    exit 1
fi

# Resolve run config path
if [ ! -f "$RUN_CONFIG" ]; then
    RUN_CONFIG="${SCRIPT_DIR}/${RUN_CONFIG}"
fi
if [ ! -f "$RUN_CONFIG" ]; then
    echo "Error: run config not found: $2"
    exit 1
fi

# Extract artifact_path from dataset config and resolve relative to config dir
ARTIFACT_REL=$(python3 -c "
import yaml, sys
with open('${DATASET_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['output']['artifact_path'])
")

CONFIG_DIR="$(dirname "$(realpath "$DATASET_CONFIG")")"
ARTIFACT_PATH="${CONFIG_DIR}/${ARTIFACT_REL}"
ARTIFACT_PATH="$(realpath -m "$ARTIFACT_PATH")"

echo "============================================================"
echo "  run_experiment.sh"
echo "============================================================"
echo "  Dataset config: ${DATASET_CONFIG}"
echo "  Run config:     ${RUN_CONFIG}"
echo "  Artifact:       ${ARTIFACT_PATH}"
echo "============================================================"
echo ""

# Step 1: Build dataset (skip if artifact exists)
if [ -f "$ARTIFACT_PATH" ]; then
    echo "[1/2] Dataset artifact already exists, skipping build."
    echo "      ${ARTIFACT_PATH}"
else
    echo "[1/2] Building dataset artifact..."
    python3 "${SCRIPT_DIR}/build_dataset.py" "$DATASET_CONFIG"
fi

echo ""

# Step 2: Run inference (includes evaluate + analyze)
echo "[2/2] Running inference..."
python3 "${SCRIPT_DIR}/run.py" run "$RUN_CONFIG"

echo ""
echo "Done."
