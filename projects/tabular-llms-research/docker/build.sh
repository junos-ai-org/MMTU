#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TAG_PREFIX="${DOCKER_TAG_PREFIX:-achithanar/mmtu-tabular-llms}"

echo "Building T5Gemma image..."
docker build -f "$REPO_ROOT/projects/tabular-llms-research/docker/Dockerfile.t5gemma" \
    -t "${TAG_PREFIX}:t5gemma" "$REPO_ROOT"

echo ""
echo "Building Qwen image..."
docker build -f "$REPO_ROOT/projects/tabular-llms-research/docker/Dockerfile.qwen" \
    -t "${TAG_PREFIX}:qwen" "$REPO_ROOT"

echo ""
echo "Done. Images:"
echo "  ${TAG_PREFIX}:t5gemma"
echo "  ${TAG_PREFIX}:qwen"
