# Plan: Restructure + New Experiments

## Overview

Three changes: (1) restructure configs/artifacts/output by experiment, (2) add context_growth experiment, (3) add column_shuffle experiment. All exclude 6 hard tasks.

### Hard tasks to exclude everywhere
- Data-transform-reshape
- Error-Detect
- Formula-prediction-context
- String-Relationship
- Columns-property-anotation
- Arithmetic-Relationship

### Non-hard tasks (15 tasks)
- Table-QA
- Table-Fact-Verification
- Entity-Matching
- Table-needle-in-a-haystack
- Table-Locate-by-Row-Col
- Schema-Matching
- Data-Imputation
- List-to-table
- semantic-transform
- semantic-join
- header-value-matching
- Functional-Dependency
- Cell-entity-annotation
- Column-type-annotation
- equi-join-detect

---

## Step 1: Restructure directory layout

**Current:**
```
projects/tabular-llms-research/
├── configs/           # flat: all dataset + run configs mixed
├── artifacts/         # flat
└── output/            # grouped by model name, not experiment
```

**New:**
```
projects/tabular-llms-research/
├── experiments/
│   ├── encoder_vs_decoder_baseline/
│   │   ├── configs/
│   │   │   ├── dataset.yaml
│   │   │   ├── dataset_smoke.yaml
│   │   │   ├── run_qwen.yaml
│   │   │   ├── run_qwen_smoke.yaml
│   │   │   ├── run_t5gemma.yaml
│   │   │   └── run_t5gemma_smoke.yaml
│   │   ├── artifacts/
│   │   └── output/
│   ├── context_growth/
│   │   ├── configs/
│   │   │   ├── dataset_bucket_0_24k.yaml
│   │   │   ├── dataset_bucket_24k_48k.yaml
│   │   │   ├── dataset_bucket_48k_72k.yaml
│   │   │   ├── dataset_bucket_72k_96k.yaml
│   │   │   ├── dataset_bucket_96k_120k.yaml
│   │   │   ├── dataset_smoke.yaml        # small sample from bucket 0-24k
│   │   │   ├── run_qwen_bucket_0_24k.yaml
│   │   │   ├── ... (one per bucket)
│   │   │   ├── run_qwen_smoke.yaml
│   │   │   └── ...
│   │   ├── artifacts/
│   │   └── output/
│   └── column_shuffle/
│       ├── configs/
│       │   ├── dataset_perm1.yaml
│       │   ├── dataset_perm2.yaml
│       │   ├── dataset_perm3.yaml
│       │   ├── dataset_perm4.yaml
│       │   ├── dataset_perm5.yaml
│       │   ├── dataset_smoke.yaml        # 2 samples/task, single perm
│       │   ├── run_qwen_perm1.yaml
│       │   ├── ... (one per perm per model)
│       │   ├── run_t5gemma_perm1.yaml
│       │   ├── ...
│       │   ├── run_qwen_smoke.yaml
│       │   └── run_t5gemma_smoke.yaml
│       ├── artifacts/
│       └── output/
├── backends/          # unchanged
├── docker/            # unchanged
├── build_dataset.py   # updated to resolve paths relative to experiment dir
├── run.py             # updated to resolve paths relative to experiment dir
├── analyze.py         # unchanged
├── compare.py         # unchanged
└── table_permuter.py  # unchanged
```

### Code changes for restructure

**build_dataset.py**: The `output.artifact_path` in YAML is already a relative path. Config files will use paths like `artifacts/bucket_0_24k.jsonl`. We need `build_dataset.py` to resolve this relative to the config file's parent directory (the experiment's configs/ dir, then up one to the experiment dir). Simplest: resolve relative to the config file's directory's parent.

Actually, looking at the current code, `artifact_path` is resolved relative to `_get_project_dir()`. We should change it to resolve relative to the config file's parent directory. Same for `run.py`.

**run.py changes:**
- `artifact_path` currently resolved relative to `_get_project_dir()`. Change to resolve relative to config file's parent dir.
- `output` dir currently `_get_project_dir() / "output" / exp_name / model_alias / run_key`. Change to resolve relative to config file's parent dir: `config_dir / "output" / model_alias / run_key` (experiment name is implicit in the directory).
- Keep experiment name in provenance/logging but don't use it for dir structure.

Wait, actually let me think about this more carefully. The simplest approach:
- Add an optional `base_dir` field to configs, or just use the config file's location to infer the experiment directory.
- For `build_dataset.py`: artifact_path relative to config's parent's parent (experiment dir). Config lives in `experiments/<exp>/configs/dataset.yaml`, so `config.parent.parent` = experiment dir.
- For `run.py`: same logic. Output goes to `experiment_dir / output / model_alias / run_key`.

**Changes to build_dataset.py:**
- `load_config()` returns both config dict and resolved config path
- `build_dataset()` takes a `base_dir` parameter for resolving relative paths
- `base_dir` = config_path.parent.parent (go up from configs/ to experiment dir)

**Changes to run.py:**
- Same `base_dir` logic
- Output dir: `base_dir / "output" / model_alias / run_key`
- Artifact path resolution: relative to `base_dir`

## Step 2: Migrate existing configs to new structure

Move existing configs into `experiments/encoder_vs_decoder_baseline/configs/` and update paths.

The existing `output/` directory has results from prior runs. Move those into the experiment's output dir.

Old configs that were flat in `configs/` get moved. Old `artifacts/` directory gets moved.

## Step 3: context_growth experiment

**Goal:** See how Qwen performs as input length grows. 5 token-length buckets.

**Buckets (input tokens):**
- 0-24K
- 24K-48K
- 48K-72K
- 72K-96K
- 96K-120K

**Dataset configs:** Each bucket has its own dataset config with `min_input_tokens` and `max_input_tokens`. We need to add `min_input_tokens` support to `build_dataset.py`.

Currently `build_dataset.py` only has `max_input_tokens`. Add `min_input_tokens` (default 0).

`samples_per_task`: 8 per task per bucket (or total_samples if some buckets are sparse).

**Run configs:** One per bucket for qwen only (user said "how qwen performs").
**Smoke:** One dataset with bucket 0-24k, 2 samples/task. One qwen run config.

## Step 4: column_shuffle experiment

**Goal:** See how Qwen and T5Gemma perform when columns are shuffled. 5 datasets with different seeds, same questions but different column permutations.

**Dataset configs:** 5 configs, each with different `permutation.seed` (e.g., 1, 2, 3, 4, 5), all with `shuffle_columns: true`. Same task list, same `samples_per_task`, same `seed` for question sampling (so same questions chosen).

**Run configs:** 5 * 2 = 10 run configs (one per permutation per model).
**Smoke:** 1 dataset (perm seed 1, 2 samples/task). 2 run configs (qwen + t5gemma).

## Step 5: Docker images

Already have per-model images (Dockerfile.qwen, Dockerfile.t5gemma). No changes needed — the user confirmed "image per model, as usual." The existing setup already does this.

## Step 6: Update CLAUDE.md

Update the project CLAUDE.md to reflect the new directory structure and experiments.

---

## Execution Order

1. Modify `build_dataset.py` to support `min_input_tokens` and resolve paths relative to config file location
2. Modify `run.py` to resolve paths relative to config file location
3. Create `experiments/` directory structure
4. Move existing configs, artifacts, output into `experiments/encoder_vs_decoder_baseline/`
5. Create context_growth configs (5 dataset + 5 run + 1 smoke dataset + 1 smoke run)
6. Create column_shuffle configs (5 dataset + 10 run + 1 smoke dataset + 2 smoke run)
7. Update CLAUDE.md
8. Remove old empty `configs/` directory
9. Commit and push
