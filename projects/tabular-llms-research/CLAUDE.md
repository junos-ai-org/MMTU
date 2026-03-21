# tabular-llms-research

## Goal
Compare encoder-decoder (T5Gemma 9B-9B, ~18B total / ~9B active) vs decoder-only
(Qwen2.5-14B-Instruct) architectures on MMTU table understanding tasks. Investigate
whether encoder-decoder's bidirectional attention provides structural advantages for
table comprehension, and measure robustness to column/row permutations.

**Design choice**: Models have different parameter profiles — T5Gemma has more total
parameters (~18B vs 14B) but fewer active during decoding (~9B vs 14B). This tests
whether encoder-decoder architecture can leverage its total capacity efficiently.

## Status
- [x] Project structure and backend interface
- [x] T5Gemma seq2seq backend (HuggingFace transformers)
- [x] Qwen2.5 backend (vLLM OpenAI-compatible API)
- [x] Table permutation utility (parse + permute markdown tables in prompts)
- [ ] Smoke test runs (both models)
- [ ] Full baseline runs (168 samples, balanced across 21 tasks)
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
│   ├── qwen_backend.py         # Qwen2.5-14B-Instruct via vLLM OpenAI API
│   └── t5gemma_backend.py      # T5Gemma 9B-9B UL2 IT via HuggingFace transformers
├── configs/
│   ├── qwen2.5-14b-smoke.yaml
│   ├── qwen2.5-14b-full.yaml
│   ├── qwen2.5-14b-full-colperm.yaml
│   ├── t5gemma-9b-smoke.yaml
│   ├── t5gemma-9b-full.yaml
│   └── t5gemma-9b-full-colperm.yaml
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── tests/
│   └── test_table_permuter.py
└── output/
```

## Models

| Model | Type | Total Params | Active Params (decoding) | HF ID |
|---|---|---|---|---|
| T5Gemma 9B-9B UL2 IT | encoder-decoder | ~18B | ~9B | `google/t5gemma-9b-9b-ul2-it` |
| Qwen2.5-14B-Instruct | decoder-only | 14B | 14B | `Qwen/Qwen2.5-14B-Instruct` |

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
- Qwen2.5-14B-Instruct vs T5Gemma 9B-9B UL2 IT
- 168 samples, balanced across 21 tasks (~8 per task)
- Markdown table format
- Temperature: 0.0

### Experiment 2: Column permutation
- Same setup as Experiment 1
- Columns shuffled with seed=42
- Compare accuracy drop and stability

## Dataset
- **Smoke**: 25 flat random samples for quick validation
- **Full**: 168 samples, balanced per task (~8 per task)
- Token filter: max_input_tokens = 3566 (4096 - 530)

## Commands

```bash
# Smoke tests
python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-smoke.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-9b-smoke.yaml

# Full baseline
python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-full.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full.yaml

# Column permutation
python projects/tabular-llms-research/run.py run configs/qwen2.5-14b-full-colperm.yaml
python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full-colperm.yaml

# Evaluate standalone
python projects/tabular-llms-research/run.py evaluate <result_file>

# List experiments
python projects/tabular-llms-research/run.py list
```

## HuggingFace Inference Optimization (reusable across projects)

When running HuggingFace models directly (not via vLLM), these optimizations
apply to any model and are worth enabling by default:

| Optimization | Speedup | Code | Notes |
|---|---|---|---|
| Flash Attention 2 | ~1.5x | `attn_implementation="flash_attention_2"` in `from_pretrained()` | Requires `flash-attn` package, CUDA GPUs (Ampere+). Fused attention kernels, lower memory. |
| `torch.compile` | ~1.5-2x | `model = torch.compile(model)` after loading | First call is slow (compilation), subsequent calls faster. Works with any model. |
| BetterTransformer | ~1.3x | `model.to_bettertransformer()` | PyTorch native. Largely superseded by Flash Attention 2. |
| Mixed precision | varies | `torch_dtype=torch.bfloat16` in `from_pretrained()` | Always use bf16 on Ampere+ GPUs. |

**Not available for encoder-decoder models:**
- **vLLM**: Decoder-only architectures only (no seq2seq support).
- **TensorRT-LLM**: Supports enc-dec but complex setup, overkill for small experiments.
- **SGLang**: Decoder-only only.

**Applied in this project**: T5Gemma backend uses Flash Attention 2 + `torch.compile`.

## Infrastructure
- **Docker Hub**: Images hosted under `achithanar/` on Docker Hub
  - `achithanar/mmtu-tabular-llms:tfix` — GPU image for T5Gemma/Qwen2.5 inference
- **Compute**: Running evaluations on RunPod (GPU cloud, Spot instances)

## Key Decisions
- **Model sizing**: Different parameter profiles — T5Gemma ~18B total / ~9B active
  during decoding vs Qwen2.5 14B always active. Tests architectural efficiency.
- **Table permutation in prompts**: Parse markdown tables from pre-built HF prompts,
  permute columns/rows, re-serialize. Avoids rebuilding from raw data via build_data.py.
- **T5Gemma backend**: Direct HuggingFace transformers (AutoModelForSeq2SeqLM), not
  vLLM (no seq2seq support in vLLM).
- **Qwen2.5 backend**: vLLM OpenAI-compatible API (same pattern as dllm-alpha).
- **168 samples balanced**: ~8 samples per task ensures each task contributes to the
  overall score without any single task dominating.
