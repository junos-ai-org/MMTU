# Qwen2.5-7B-Instruct Baseline Results

**Date**: 2026-03-02
**Model**: Qwen2.5-7B-Instruct via vLLM
**Settings**: temperature=0, max_model_len=65536, seed=42
**Samples**: 99 (11 tasks x 9 per task)

## Overall Score: 0.398

## Results by Task

| Task | Dataset | Metric | Score |
|:--|:--|:--|--:|
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

## Results by Task (Aggregated)

| Task | Metric | Avg Score |
|:--|:--|--:|
| Entity-Matching | acc | 1.000 |
| Schema-Matching | f1 | 0.493 |
| semantic-join | f1 | 0.485 |
| Functional-Dependency | f1 | 0.709 |
| semantic-transform | acc | 0.533 |
| List-to-table | acc | 0.444 |
| equi-join-detect | f1 | 0.320 |
| Data-Imputation | acc | 0.000 |
| String-Relationship | f1 | 0.000 |
| Table-Locate-by-Row-Col | acc | 0.000 |
| Arithmetic-Relationship | f1 | NaN |

## Notes

- 99/99 queries received responses (0 context-limit failures after bumping
  `--max-model-len` to 65536).
- Arithmetic-Relationship NaN is a format mismatch: the evaluator filters
  predictions against `labeled_formulas` and the model's output format doesn't
  match.
- Zero-score tasks (Data-Imputation, String-Relationship, Table-Locate-by-Row-Col)
  reflect model capability limits at 7B scale, not infrastructure issues.
