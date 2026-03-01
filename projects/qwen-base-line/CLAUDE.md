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
- **3-layer Docker**: base (deps) -> model (weights) -> code.
  Code changes rebuild only the top layer (seconds).
