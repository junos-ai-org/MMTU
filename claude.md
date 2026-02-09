# MMTU - Massive Multi-Task Table Understanding and Reasoning Benchmark

## Overview

MMTU is a large-scale NeurIPS'25 benchmark containing ~28,136 questions across 25 real-world table tasks. It evaluates LLMs' ability to understand, reason about, and manipulate tabular data at expert level. The benchmark was curated from 52 datasets across research communities in data management (SIGMOD/VLDB), programming languages (PLDI/POPL), and web data (WWW/WSDM).

Paper: https://arxiv.org/abs/2506.05587

## Project Structure

```
MMTU/
├── build_data.py          # Generates prompts from raw data and configurations
├── inference.py           # Runs model inference (OpenAI, Azure, self-deploy/LLaDA)
├── evaluate.py            # Evaluates model outputs, optional W&B logging
├── requirements.txt       # Python 3.11 dependencies (includes wandb)
├── configurations/        # 25+ task directories with prompt/data configs (~410 config files)
├── evaluators/            # 27 task-specific evaluator modules (base + 26 tasks)
├── utils/                 # Table serialization, processing, and helper utilities
├── docs/                  # Design decisions and architecture documentation
│   └── design-decisions.md
├── Dockerfile.base        # CUDA + PyTorch + model weights (rarely changes)
├── Dockerfile.llada       # LLaDA fork layer (changes when fork changes)
├── Dockerfile.mmtu        # MMTU code layer (changes most often)
├── Dockerfile.evaluate    # Lightweight image for sandboxed evaluation
├── build_docker.sh        # Builds and pushes all three Docker layers
├── docker-compose.evaluate.yml
├── run_evaluate_docker.sh
└── .github/workflows/build-docker.yml  # CI: auto-build Docker images
```

## Key Scripts

- **`build_data.py`**: Reads raw data, applies prompt templates, counts tokens, filters by token limit. Run with `python3 build_data.py one --config <config_path>`.
- **`inference.py`**: Sends prompts to LLM APIs, handles retries/threading. Supports `--tasks`, `--max_samples`, `--max_samples_per_task` for filtering. The `self_deploy` provider integrates LLaDA via `self_deploy_query_function()`.
- **`evaluate.py`**: Loads task evaluators, parses model JSON outputs, computes accuracy/F1 metrics. Supports `--wandb` for logging to Weights & Biases.

## LLaDA Integration

LLaDA (a diffusion-based LLM) is integrated via `self_deploy_query_function()` in `inference.py`. The two repos stay separate — LLaDA is installed as an editable package (`pip install -e`) into MMTU's venv. See `docs/design-decisions.md` (decision 001) for rationale.

```bash
pip install -e /path/to/llada-fork
python3 inference.py self_deploy --tasks NL2SQL --max_samples_per_task 10
```

LLaDA-specific CLI args: `--model`, `--steps`, `--gen_length`, `--block_length`.

## Experiment Tracking (W&B)

Evaluation results can be logged to Weights & Biases:

```bash
python3 evaluate.py result.jsonl --wandb --wandb_project mmtu-benchmark
```

This logs per-task scores, overall MMTU score, and uploads the result file as an artifact. W&B is optional — evaluate.py works without it.

## Docker / RunPod Setup

Three-layer Docker architecture to avoid wasting GPU budget on setup:
- `Dockerfile.base` — CUDA runtime + PyTorch + model weights (~20GB)
- `Dockerfile.llada` — Private LLaDA fork
- `Dockerfile.mmtu` — Benchmark code

Build with `./build_docker.sh all` (requires `REGISTRY` and `LLADA_DIR` env vars). CI builds via GitHub Actions on push to main. Docker Hub registry: `docker.io/achithanar`.

## The 25 Tasks

| Category | Tasks |
|---|---|
| NL-to-SQL & QA | NL2SQL, Table-QA |
| Data Transformation | Data-transform-pbe, Data-transform-reshape, Transform-by-input-output-table, Transform-by-output-target-schema, semantic-transform |
| Entity/Schema Matching | Entity-Matching, Schema-Matching, semantic-join, equi-join-detect, header-value-matching |
| Data Quality | Error-Detect, Data-Imputation |
| Annotation | Column-type-annotation, Cell-entity-annotation, Columns-property-anotation |
| Constraint Discovery | Functional-Dependency, Arithmetic-Relationship, String-Relationship |
| Fact Verification & Search | Table-Fact-Verification, Table-needle-in-a-haystack, Table-Locate-by-Row-Col |
| Other | Formula-prediction-context, List-to-table |

## Configuration System

Each task has multiple configuration variants under `configurations/<TASK>/v1.0/`. Configs define:
- `prompt_template`: Jinja-style template with placeholders like `{{{table}}}`, `{{{question}}}`
- `dataset_config`: Data paths, field specs (`text`, `table.csv`, `table.csv.path`, `list`, `fewshot`)
- Table serialization format (CSV, JSON, HTML, Markdown)
- Dataset size variant (sample200, sample1000, full)
- Few-shot variant (0-shot, 3-shot)

## Evaluators

Located in `evaluators/`. Each inherits from `BaseEvaluator` in `base_evaluator.py`. Tasks use either accuracy (`acc`) or F1 score (`f1`) metrics. Some evaluators execute model-generated code (NL2SQL, data transforms) - use Docker for sandboxed evaluation.

## Table Serialization

`utils/table_serializer.py` provides 13+ serializer classes: Text, Markdown, HTML, JSON, CSV - each with variants for index/header inclusion. `utils/table_processor.py` has row-shuffling and row-limiting processors.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Design Decisions

Documented in `docs/design-decisions.md`. Key decisions:
- LLaDA via editable install, not git submodule (001)
- Three-layer Docker images for RunPod (002)
- Multi-stage build for base image (003)
- W&B for experiment tracking (004)
- GitHub Actions for Docker CI (005)
- Dataset filtering flags for development (006)

## Development Notes

- Data is not stored in the repo; it's downloaded from HuggingFace during inference or from OneDrive for raw access.
- Code execution happens during evaluation for some tasks (NL2SQL, transforms) - always use sandboxed environments.
- To add a new task: create a config directory, a prompt template, and an evaluator class inheriting `BaseEvaluator`.
- To customize prompts: edit the `prompt_template` in the relevant config file, then regenerate with `build_data.py`.
