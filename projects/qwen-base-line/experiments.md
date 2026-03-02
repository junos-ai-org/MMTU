# Experiments

## Experiment 1: Baseline (99 questions, 11 tasks x 9)

**Date**: 2026-03-02
**Model**: Qwen2.5-7B-Instruct via vLLM
**Settings**: temperature=0, max_model_len=32768, seed=42
**Image**: `achithanar/mmtu-qwen-baseline:logging`
**GPU**: RunPod (A40/A100)

### Results

**Overall MMTU score: 0.398**

| Task | Dataset | Metric | Score |
|---|---|---|---|
| Entity-Matching | DBLP-ACM | acc | 1.000 |
| Entity-Matching | DBLP-Scholar | acc | 1.000 |
| Entity-Matching | Fodors-Zagats | acc | 1.000 |
| Entity-Matching | Walmart-Amazon | acc | 1.000 |
| Schema-Matching | HXD | f1 | 1.000 |
| semantic-join | SEMA-join | f1 | 0.734 |
| Functional-Dependency | Auto-Relate | f1 | 0.709 |
| semantic-transform | SEMA-join | acc | 0.667 |
| Schema-Matching | assays | f1 | 0.498 |
| List-to-table | TEGRA | acc | 0.444 |
| semantic-transform | DataXFormer | acc | 0.400 |
| equi-join-detect | Auto-BI | f1 | 0.320 |
| Schema-Matching | prospect | f1 | 0.250 |
| semantic-join | DataXFormer | f1 | 0.236 |
| Schema-Matching | miller2 | f1 | 0.222 |
| Data-Imputation | WebTable | acc | 0.000 |
| Data-Imputation | tablib | acc | 0.000 |
| String-Relationship | Auto-Relate | f1 | 0.000 |
| Table-Locate-by-Row-Col | experiment5 | acc | 0.000 |
| Arithmetic-Relationship | Auto-Relate | f1 | NaN |

### Issues

- **13/99 queries failed** with HTTP 400: prompts were 32,769 tokens, exceeding
  the vLLM `--max-model-len 32768` by exactly 1 token.
- Arithmetic-Relationship returned NaN — likely all 9 samples exceeded the
  context limit.
- **Fix**: Bumped `--max-model-len` to 65536 in entrypoint.sh for the next run.
  Qwen2.5-7B natively supports 131K context.
