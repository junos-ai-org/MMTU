# tabular-llms-research

## Goal
Compare encoder-decoder (T5Gemma 9B-9B UL2 IT) vs decoder-only (Qwen2.5-14B-Instruct) architectures
on MMTU table understanding tasks. Investigate whether encoder-decoder's bidirectional
attention provides structural advantages for table comprehension, and measure
robustness to column/row permutations.

## Architecture

```
projects/tabular-llms-research/
├── build_dataset.py            # Step 1: Deterministic artifact generator
├── run.py                      # Step 2: Entry point (run, evaluate, list)
├── analyze.py                  # Step 3: Post-run analysis
├── compare.py                  # Step 4: Side-by-side model comparison
├── table_permuter.py           # Used by build_dataset.py
├── backends/
│   ├── base.py
│   ├── qwen_backend.py         # Qwen2.5 via vLLM OpenAI API
│   └── t5gemma_backend.py      # T5Gemma via HuggingFace transformers
├── configs/
│   ├── dataset_baseline.yaml   # Config for dataset generation
│   ├── dataset_colperm.yaml
│   ├── run_qwen_baseline.yaml  # Config for model inference
│   └── run_t5gemma_baseline.yaml
└── output/
    └── encoder_vs_decoder_baseline/     # Experiment Name
        ├── Qwen2.5-14B-Instruct/        # Model output directory
        │   ├── config.yaml
        │   ├── result.jsonl
        │   └── analysis.md
        └── t5gemma-9b-9b-ul2-it/
```

## Workflow (Deterministic & Tracked)

To ensure fair comparison, models DO NOT sample datasets independently. We first generate a static deterministic dataset artifact, then run both models against it.
Outputs are grouped by *experiment*, not by model, so you can easily compare runs. A registry row is also automatically appended to `experiments.md` upon completion.

### Step 1: Generate Dataset Artifact
```bash
# Builds artifacts/datasets/baseline_150_samples.jsonl
python projects/tabular-llms-research/build_dataset.py configs/dataset_baseline.yaml
```

### Step 2: Run Inference
```bash
# Both models evaluate the exact same artifact
# Note: Add `wandb: true` to the `experiment` block in your YAML to automatically
# sync your metrics, inputs, and outputs to the Weights & Biases dashboard!
python projects/tabular-llms-research/run.py run configs/run_qwen_baseline.yaml
python projects/tabular-llms-research/run.py run configs/run_t5gemma_baseline.yaml
```

### Step 3: Compare Results
```bash
# View side-by-side delta of the two runs
python projects/tabular-llms-research/compare.py \
    projects/tabular-llms-research/output/encoder_vs_decoder_baseline/Qwen2.5-14B-Instruct \
    projects/tabular-llms-research/output/encoder_vs_decoder_baseline/t5gemma-9b-9b-ul2-it
```

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
