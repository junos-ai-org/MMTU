# tabular-llms-research

## Goal
Compare encoder-decoder (T5Gemma 2B-2B) vs decoder-only (Qwen3 4B) architectures
on MMTU table understanding tasks. Investigate whether encoder-decoder's bidirectional
attention provides structural advantages for table comprehension, and measure
robustness to column/row permutations.

## Status
- [ ] Project structure and backend interface
- [ ] T5Gemma seq2seq backend (HuggingFace transformers)
- [ ] Qwen3 backend (vLLM OpenAI-compatible API)
- [ ] Table permutation utility (parse + permute markdown tables in prompts)
- [ ] Smoke test runs (both models)
- [ ] Full baseline runs (150 samples, balanced across 21 tasks)
- [ ] Column permutation experiment
- [ ] Analysis and comparison

## Architecture

```
projects/tabular-llms-research/
├── run.py                      # Entry point (run, evaluate, list)
├── analyze.py                  # Post-run analysis (per-task scores, comparison)
├── table_permuter.py           # Parse/permute markdown tables in prompts
├── backends/
│   ├── base.py                 # InferenceBackend ABC (symlink to dllm-alpha)
│   ├── qwen3_backend.py        # Qwen3 4B via vLLM OpenAI API
│   └── t5gemma_backend.py      # T5Gemma 2B-2B via HuggingFace transformers
├── configs/
│   ├── qwen3-4b-smoke.yaml
│   ├── qwen3-4b-full.yaml
│   ├── qwen3-4b-full-colperm.yaml
│   ├── t5gemma-2b-smoke.yaml
│   ├── t5gemma-2b-full.yaml
│   └── t5gemma-2b-full-colperm.yaml
├── tests/
│   └── test_table_permuter.py
└── output/
```

## Tasks (21 non-execution MMTU tasks)

Excluded (require code execution): NL2SQL, Data-transform-pbe,
Transform-by-output-target-schema, Transform-by-input-output-table.

| Task | Metric |
|---|---|
| Table-QA | acc |
| Table-Fact-Verification | acc |
| Error-Detect | f1 |
| Entity-Matching | acc |
| Table-needle-in-a-haystack | acc |
| Table-Locate-by-Row-Col | acc |
| Schema-Matching | f1 |
| Data-transform-reshape | acc |
| Data-Imputation | acc |
| List-to-table | acc |
| Formula-prediction-context | acc |
| semantic-transform | acc |
| semantic-join | f1 |
| header-value-matching | acc |
| Arithmetic-Relationship | f1 |
| Functional-Dependency | f1 |
| String-Relationship | f1 |
| Cell-entity-annotation | acc |
| Column-type-annotation | acc |
| Columns-property-anotation | acc |
| equi-join-detect | f1 |

## Experiments

### Experiment 1: Baseline (no permutation)
- Qwen3 4B vs T5Gemma 2B-2B
- 150 samples, balanced across 21 tasks (~7 per task)
- Markdown table format
- Temperature: 0.0

### Experiment 2: Column permutation
- Same setup as Experiment 1
- Columns shuffled with seed=42
- Compare accuracy drop and stability

## Dataset
- **Smoke**: 25 flat random samples for quick validation
- **Full**: 150 samples, balanced per task (~7 per task)
- Token filter: max_input_tokens = 3566 (4096 - 530)

## Commands

```bash
# Smoke tests
python projects/tabular-llms-research/run.py run configs/qwen3-4b-smoke.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-2b-smoke.yaml

# Full baseline
python projects/tabular-llms-research/run.py run configs/qwen3-4b-full.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-2b-full.yaml

# Column permutation
python projects/tabular-llms-research/run.py run configs/qwen3-4b-full-colperm.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-2b-full-colperm.yaml

# Evaluate standalone
python projects/tabular-llms-research/run.py evaluate <result_file>

# List experiments
python projects/tabular-llms-research/run.py list
```

## Key Decisions
- **Table permutation in prompts**: Parse markdown tables from pre-built HF prompts,
  permute columns/rows, re-serialize. Avoids rebuilding from raw data via build_data.py.
- **T5Gemma backend**: Direct HuggingFace transformers (AutoModelForSeq2SeqLM), not
  vLLM (no seq2seq support in vLLM).
- **Qwen3 backend**: vLLM OpenAI-compatible API (same pattern as dllm-alpha).
- **150 samples balanced**: ~7 samples per task ensures each task contributes to the
  overall score without any single task dominating.
