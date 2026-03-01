# qwen-base-line

## Goal
Establish a baseline for Qwen2.5-7B-Instruct on a subset of MMTU tasks,
served via vLLM on RunPod.

## Status
- [ ] Phase 1: Smoke test (5 Entity-Matching questions)
- [ ] Phase 2: Baseline (99 questions, 11 tasks x 9)

## Key Decisions
- **Uses inference.py**: Fixed `create_query_funtion_openai()` to use modern
  `base_url=` (was broken with openai>=1.0). Imports `query_chat_endpoint`
  directly to reuse multi-threading, resumption, and error handling.
- **Subprocess evaluation**: Calls `evaluate.py` via subprocess to avoid
  `args` global dependency.
- **Model alias**: `Qwen2_5-7B-Instruct` (dots/slashes replaced) for
  evaluate.py filename parsing compatibility.
- **Temperature 0**: Deterministic baseline.
- **TNIAH excluded**: `Table-needle-in-a-haystack` evaluator reads local
  filesystem paths for ground truth; fails on RunPod.
- **Runtime model download**: Weights not baked into image (~11GB vs ~25GB).
  Cached on RunPod network volume (`/workspace/.cache/huggingface`).

## Workflow

```
1. BUILD (any x86_64 Linux, e.g. AWS EC2 t3.medium)
   git clone ... && cd MMTU
   bash projects/qwen-base-line/docker/build.sh \
     --registry docker.io/<username> --push

2. RUNPOD
   Create GPU pod (A40/A100, >=24GB VRAM)
   Template image: docker.io/<username>/mmtu-qwen-baseline:latest
   Optional: network volume on /workspace (caches model weights)

3. RUN (SSH into pod, entrypoint auto-downloads model + starts vLLM)
   cd /workspace/MMTU
   python projects/qwen-base-line/run.py smoke
   python projects/qwen-base-line/run.py baseline
```
