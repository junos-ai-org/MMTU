#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

DEFAULT_REPO="achithanar"
DEFAULT_TAG="latest"

usage() {
    cat <<EOF
Build and push Docker images for ${PROJECT_NAME}.

Usage: $(basename "$0") [OPTIONS]

Options:
  --repo REPO    Docker Hub repo prefix (default: ${DEFAULT_REPO})
  --tag TAG      Image tag (default: ${DEFAULT_TAG})
  --no-push      Build only, don't push
  -h, --help     Show this help

Images built:
  REPO/${PROJECT_NAME}:TAG-t5gemma
  REPO/${PROJECT_NAME}:TAG-qwen

Examples:
  $(basename "$0")                          # ${DEFAULT_REPO}/${PROJECT_NAME}:latest-t5gemma
  $(basename "$0") --tag v2                 # ${DEFAULT_REPO}/${PROJECT_NAME}:v2-t5gemma
  $(basename "$0") --repo myorg --tag v2    # myorg/${PROJECT_NAME}:v2-t5gemma
  $(basename "$0") --no-push               # build only
EOF
}

REPO="$DEFAULT_REPO"
TAG="$DEFAULT_TAG"
PUSH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)  REPO="$2"; shift 2 ;;
        --tag)   TAG="$2"; shift 2 ;;
        --no-push) PUSH=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

IMAGE="${REPO}/${PROJECT_NAME}"
IMAGES=()

echo "Building T5Gemma image..."
docker build -f "$SCRIPT_DIR/Dockerfile.t5gemma" \
    -t "${IMAGE}:${TAG}-t5gemma" "$REPO_ROOT"
IMAGES+=("${IMAGE}:${TAG}-t5gemma")

echo ""
echo "Building Qwen image..."
docker build -f "$SCRIPT_DIR/Dockerfile.qwen" \
    -t "${IMAGE}:${TAG}-qwen" "$REPO_ROOT"
IMAGES+=("${IMAGE}:${TAG}-qwen")

if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing images..."
    for img in "${IMAGES[@]}"; do
        docker push "$img"
    done
fi

echo ""
echo "Done:"
for img in "${IMAGES[@]}"; do
    echo "  $img"
done
