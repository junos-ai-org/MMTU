#!/bin/bash
set -euo pipefail

# Run all context_growth buckets sequentially.
#
# Usage:
#   ./projects/tabular-llms-research/run_context_growth.sh [--force-dataset-rebuild]
#
# Runs Qwen2.5-14B on 5 token-length buckets: 0-24K, 24K-48K, 48K-72K, 72K-96K, 96K-120K.
# The vLLM server must already be running with the Qwen2.5-14B model.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FORCE_FLAG=""

if [[ "${1:-}" == "--force-dataset-rebuild" ]]; then
    FORCE_FLAG="--force-dataset-rebuild"
fi

BUCKETS=(0_24k 24k_48k 48k_72k 72k_96k 96k_120k)
CONFIGS_DIR="${SCRIPT_DIR}/experiments/context_growth/configs"

echo "============================================================"
echo "  run_context_growth.sh"
echo "============================================================"
echo "  Buckets: ${BUCKETS[*]}"
echo "============================================================"
echo ""

for i in "${!BUCKETS[@]}"; do
    BUCKET="${BUCKETS[$i]}"
    DATASET_CFG="${CONFIGS_DIR}/dataset_${BUCKET}.yaml"
    RUN_CFG="${CONFIGS_DIR}/run_qwen_${BUCKET}.yaml"

    echo "------------------------------------------------------------"
    echo "  [$(( i + 1 ))/${#BUCKETS[@]}] Bucket: ${BUCKET}"
    echo "------------------------------------------------------------"

    # Only force-rebuild on the first bucket (all share the same flag intent)
    if [ "$i" -eq 0 ] && [ -n "$FORCE_FLAG" ]; then
        "${SCRIPT_DIR}/run_experiment.sh" $FORCE_FLAG "$DATASET_CFG" "$RUN_CFG"
    else
        "${SCRIPT_DIR}/run_experiment.sh" ${FORCE_FLAG:+"$FORCE_FLAG"} "$DATASET_CFG" "$RUN_CFG"
    fi

    echo ""
done

echo "============================================================"
echo "  All buckets complete."
echo "============================================================"
echo ""
echo "Compare results with:"
echo "  python ${SCRIPT_DIR}/compare.py \\"
for BUCKET in "${BUCKETS[@]}"; do
    echo "    experiments/context_growth/output/Qwen2.5-14B-Instruct/latest \\"
done
echo ""
