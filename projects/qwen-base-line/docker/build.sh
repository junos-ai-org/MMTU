#!/bin/bash
set -e

# Build layered Docker images for qwen-base-line project.
#
# Usage:
#   bash projects/qwen-base-line/docker/build.sh --all              # rebuild everything
#   bash projects/qwen-base-line/docker/build.sh --model            # rebuild from model layer
#   bash projects/qwen-base-line/docker/build.sh --code             # rebuild code layer only (default)
#   bash projects/qwen-base-line/docker/build.sh --code --push      # rebuild + push to registry
#
# Options:
#   --registry REG   Docker registry prefix (e.g. docker.io/username)
#   --push           Push images after building

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MMTU_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

# Defaults
BUILD_BASE=false
BUILD_MODEL=false
BUILD_CODE=true
REGISTRY=""
PUSH=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            BUILD_BASE=true
            BUILD_MODEL=true
            BUILD_CODE=true
            shift ;;
        --model)
            BUILD_MODEL=true
            BUILD_CODE=true
            shift ;;
        --code)
            BUILD_CODE=true
            shift ;;
        --registry)
            REGISTRY="$2"
            shift 2 ;;
        --push)
            PUSH=true
            shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all|--model|--code] [--registry REG] [--push]"
            exit 1 ;;
    esac
done

tag_and_push() {
    local local_tag="$1"
    if [ -n "$REGISTRY" ]; then
        local remote_tag="${REGISTRY}/${local_tag}"
        docker tag "$local_tag" "$remote_tag"
        if [ "$PUSH" = true ]; then
            echo "Pushing ${remote_tag}..."
            docker push "$remote_tag"
        fi
    fi
}

cd "$MMTU_ROOT"

if [ "$BUILD_BASE" = true ]; then
    echo "=== Building Layer 1: mmtu-base ==="
    docker build -t mmtu-base:latest \
        -f projects/qwen-base-line/docker/Dockerfile.base .
    tag_and_push "mmtu-base:latest"
fi

if [ "$BUILD_MODEL" = true ]; then
    echo "=== Building Layer 2: mmtu-qwen7b ==="
    docker build -t mmtu-qwen7b:latest \
        -f projects/qwen-base-line/docker/Dockerfile.model .
    tag_and_push "mmtu-qwen7b:latest"
fi

if [ "$BUILD_CODE" = true ]; then
    echo "=== Building Layer 3: mmtu-qwen-baseline ==="
    docker build -t mmtu-qwen-baseline:latest \
        -f projects/qwen-base-line/docker/Dockerfile .
    tag_and_push "mmtu-qwen-baseline:latest"
fi

echo ""
echo "Done! Images built:"
[ "$BUILD_BASE" = true ] && echo "  mmtu-base:latest"
[ "$BUILD_MODEL" = true ] && echo "  mmtu-qwen7b:latest"
[ "$BUILD_CODE" = true ] && echo "  mmtu-qwen-baseline:latest"
