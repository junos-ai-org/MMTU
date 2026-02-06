#!/bin/bash
set -e

REGISTRY="${REGISTRY:?Set REGISTRY env var (e.g. export REGISTRY=docker.io/achithanar)}"
LLADA_DIR="${LLADA_DIR:?Set LLADA_DIR env var (e.g. export LLADA_DIR=~/projects/llada)}"
MMTU_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_IMAGE="$REGISTRY/mmtu-base:latest"
LLADA_IMAGE="$REGISTRY/mmtu-llada:latest"
RUNPOD_IMAGE="$REGISTRY/mmtu-runpod:latest"

case "${1:-all}" in
  base)
    echo "==> Building base image..."
    docker build -f "$MMTU_DIR/Dockerfile.base" -t "$BASE_IMAGE" "$MMTU_DIR"
    docker push "$BASE_IMAGE"
    ;;
  llada)
    echo "==> Building LLaDA image..."
    docker build -f "$MMTU_DIR/Dockerfile.llada" --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$LLADA_IMAGE" "$LLADA_DIR"
    docker push "$LLADA_IMAGE"
    ;;
  mmtu)
    echo "==> Building MMTU image..."
    docker build -f "$MMTU_DIR/Dockerfile.mmtu" --build-arg BASE_IMAGE="$LLADA_IMAGE" -t "$RUNPOD_IMAGE" "$MMTU_DIR"
    docker push "$RUNPOD_IMAGE"
    ;;
  all)
    echo "==> Building all images..."
    "$0" base
    "$0" llada
    "$0" mmtu
    echo "==> Done. RunPod image: $RUNPOD_IMAGE"
    ;;
  *)
    echo "Usage: $0 {base|llada|mmtu|all}"
    exit 1
    ;;
esac
