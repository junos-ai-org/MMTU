# tabular-llms-research

## Goal
Compare encoder-decoder (T5Gemma 9B-9B UL2 IT) vs decoder-only (Qwen2.5-14B-Instruct) architectures
on MMTU table understanding tasks. Investigate whether encoder-decoder's bidirectional
attention provides structural advantages for table comprehension, and measure
robustness to column/row permutations.

## Architecture

```
projects/tabular-llms-research/
├── build_dataset.py            # Deterministic artifact generator
├── run.py                      # Entry point (run, evaluate, list)
├── analyze.py                  # Post-run analysis
├── compare.py                  # Side-by-side model comparison
├── table_permuter.py           # Used by build_dataset.py
├── backends/
│   ├── base.py
│   ├── qwen_backend.py         # Qwen2.5 via vLLM OpenAI API
│   └── t5gemma_backend.py      # T5Gemma via HuggingFace transformers
├── docker/
│   ├── build.sh                # Build all Docker images
│   ├── Dockerfile.base         # Shared base image
│   ├── Dockerfile.qwen         # Qwen backend (vLLM)
│   └── Dockerfile.t5gemma      # T5Gemma backend (transformers)
└── experiments/                # Each experiment is self-contained
    ├── encoder_vs_decoder_baseline/
    │   ├── configs/            # dataset.yaml, run_qwen.yaml, etc.
    │   ├── artifacts/          # Generated JSONL datasets
    │   └── output/             # Model outputs per run
    ├── context_growth/         # Qwen accuracy vs input token length
    │   ├── configs/            # 5 bucket datasets + qwen runs + smoke
    │   ├── artifacts/
    │   └── output/
    └── column_shuffle/         # Robustness to column permutation
        ├── configs/            # 5 perm datasets + qwen/t5gemma runs + smoke
        ├── artifacts/
        └── output/
```

## Workflow (Deterministic & Tracked)

To ensure fair comparison, models DO NOT sample datasets independently. We first generate a static deterministic dataset artifact, then run both models against it.
Outputs are grouped by *experiment*, not by model, so you can easily compare runs. A registry row is also automatically appended to `experiments.md` upon completion.

### Step 1: Generate Dataset Artifact
```bash
# Build dataset for an experiment (paths resolve relative to config file)
python projects/tabular-llms-research/build_dataset.py \
    experiments/encoder_vs_decoder_baseline/configs/dataset.yaml
```

### Step 2: Run Inference
```bash
# Both models evaluate the exact same artifact
# Note: Add `wandb: true` to the `experiment` block in your YAML to enable W&B.
python projects/tabular-llms-research/run.py run \
    experiments/encoder_vs_decoder_baseline/configs/run_qwen.yaml
python projects/tabular-llms-research/run.py run \
    experiments/encoder_vs_decoder_baseline/configs/run_t5gemma.yaml
```

### Step 3: Compare Results
```bash
python projects/tabular-llms-research/compare.py \
    experiments/encoder_vs_decoder_baseline/output/Qwen2.5-14B-Instruct/latest \
    experiments/encoder_vs_decoder_baseline/output/t5gemma-9b-9b-ul2-it/latest
```

## Experiments

### encoder_vs_decoder_baseline
Baseline comparison of Qwen2.5-14B (decoder-only) vs T5Gemma-9B (encoder-decoder)
on 15 non-hard MMTU tasks. 8 samples/task.

### context_growth (Qwen only)
How does Qwen2.5-14B accuracy change as input context grows? 5 token-length buckets:
0-24K, 24K-48K, 48K-72K, 72K-96K, 96K-120K. 8 samples/task per bucket.

### column_shuffle (Qwen + T5Gemma)
Robustness to column permutation. 5 datasets with the same questions but different
random column orderings (permutation seeds 1-5). 8 samples/task. Tests whether
encoder-decoder architecture is more robust to structural changes.

## Hard Tasks (Near-Total Failure)

These tasks have 0-12.5% accuracy for both models. They share a common theme:
exact-match evaluation over massive/combinatorial output spaces with no partial credit.

### Data-transform-reshape (0% both models)
Models must predict a sequence of pandas reshape operations (stack, pivot, transpose,
wide_to_long, explode, ffill, subtitle) with exact index parameters. Models misidentify
the transformation type entirely -- e.g. predicting "transpose" when the answer is
"stack" or "pivot". Even getting the right operator isn't enough; off-by-one index
errors (stack_end_idx 10 vs 11) are scored as total failure. The task requires
structural pattern recognition (repeating column groups, nested headers) that current
LLMs cannot reliably perform from serialized table text.

### Error-Detect (0-12.5%)
Ground truth is usually [] (no errors). Models hallucinate errors that aren't there,
flagging unusual-but-valid cell values. F1 metric penalizes false positives hard.

### Formula-prediction-context (0-12.5%)
Must guess the exact Excel formula. Off-by-one cell reference or wrong function = zero.
Infinite output space (any Excel formula) with exact string match evaluation.

### String-Relationship (0-12.5%)
Models over-identify relationships. Expected: 1 relationship. Models return 8+ spurious
ones with wrong column names. Output space is exponential (2^N column subsets), evaluated
via F1 on set of tuples.

### Columns-property-annotation (12.5%)
Models pick semantically plausible but wrong DBPedia ontology properties -- e.g.
"birthDate" instead of "foalDate", "sibling" instead of "relation". Thousands of
candidate properties, evaluated by exact string match. No credit for "close" answers.

### Arithmetic-Relationship (12.5%)
Models find plausible but wrong formulas, or miss multi-column sums. e.g. predicts
G=C+D when truth is G=C+E+F. Combinatorial search space (all +,-,*,/ combos across
N columns), evaluated via F1 on formula sets. Numerical precision and zero-handling
make detection fragile.

### Key takeaway
These tasks require either (a) exact recall from a huge ontology/formula space, or
(b) structural reasoning over table geometry. Both are weaknesses of current LLMs
operating on serialized text. They represent the hardest frontier for tabular LLM
research.
