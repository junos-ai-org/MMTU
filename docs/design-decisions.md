# Design Decisions

This document tracks key architectural and tooling decisions made for this project.

## 001: LLaDA integration via `pip install -e` (not git submodule)

**Context**: MMTU and LLaDA are separate repositories. We need them to interact for inference.

**Decision**: Install LLaDA as an editable Python package (`pip install -e`) into MMTU's venv rather than using git submodules.

**Alternatives considered**:
- **Git submodule**: Rejected because the user has a private fork of LLaDA and is actively developing it. Submodules create friction — every LLaDA change requires a submodule pointer update commit in MMTU.
- **vLLM server**: Rejected because vLLM does not support diffusion-based LLMs like LLaDA (open feature request: vllm-project/vllm#18532).

**Integration point**: `self_deploy_query_function()` in `inference.py` imports LLaDA's `generate()` function.

---

## 002: Three-layer Docker image architecture

**Context**: Running on RunPod wastes GPU budget if dependencies and model weights are installed at pod startup.

**Decision**: Split the Docker setup into three layers:
1. `Dockerfile.base` — CUDA + PyTorch + transformers + model weights (~20GB, rarely changes)
2. `Dockerfile.llada` — LLaDA fork installed on top of base (changes when fork changes)
3. `Dockerfile.mmtu` — MMTU benchmark code + pip deps (changes most often)

**Rationale**: Each layer caches independently. Iterating on MMTU code (the most common case) only rebuilds a thin layer in seconds, while the expensive base layer (model weight download) is built once.

**Alternatives considered**:
- **Single Dockerfile**: Simpler but any code change triggers a full rebuild including re-downloading model weights.
- **Git submodule for LLaDA in Docker**: Same friction as decision 001.

---

## 003: Multi-stage build for base image

**Context**: The base image needs compiler toolchain (gcc, python-dev) to build native extensions, but these aren't needed at runtime.

**Decision**: Use Docker multi-stage build — compile in a `devel` stage, copy only installed packages to a slim `runtime` stage.

**Tradeoff**: Saves ~2-3GB on the base image. Proportionally modest when model weights are ~16GB, but matters more if weights are loaded from a network volume instead of baked in.

---

## 004: W&B for experiment tracking (not DVC, not git)

**Context**: Need a reproducible way to track evaluation results across runs with different models and parameters.

**Decision**: Integrate Weights & Biases into `evaluate.py` with an optional `--wandb` flag.

**What gets logged**:
- Per-task accuracy/F1 scores
- Overall MMTU score
- Run config (model name, result file)
- Result JSONL file as a W&B artifact

**Alternatives considered**:
- **Git**: Good for code, not for comparing metrics across runs.
- **DVC**: Overkill — result JSONL files are small (a few MB). DVC shines for large datasets/models.
- **Manual spreadsheets**: Not reproducible.

**Integration**: Optional — `evaluate.py` works without wandb installed.

---

## 005: GitHub Actions for Docker builds (not local, not AWS)

**Context**: Need to build Docker images without paying for GPU time.

**Decision**: Added `.github/workflows/build-docker.yml` that builds and pushes images on push to main (MMTU layer) or via manual trigger (base/llada layers).

**Rationale**: Free, no local Docker installation needed, native amd64 (no cross-compilation). Can also build locally with `./build_docker.sh` if preferred.

---

## 006: Dataset filtering flags for development

**Context**: The full MMTU benchmark has ~28K questions. During development, running the full benchmark is slow and expensive.

**Decision**: Added `--tasks`, `--max_samples`, and `--max_samples_per_task` flags to `inference.py`.

**Usage**: `python3 inference.py self_deploy --tasks NL2SQL --max_samples_per_task 10`

**Rationale**: Allows quick iteration during development without modifying the dataset or configs.
