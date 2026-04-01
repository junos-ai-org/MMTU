# Insight: Data-Imputation & List-to-table Deep Dive

**Date:** 2026-04-01
**Scope:** 133 paired questions (66 Data-Imputation, 67 List-to-table)
**Models:** Qwen2.5-14B-Instruct vs T5Gemma-9B-9B-UL2-IT
**Method:** Per-question LLM-assisted analysis (Claude Sonnet 4.6, 5 parallel agents)

## Headlines

- **T5Gemma wins on both tasks.** It's correct on 20 questions where Qwen fails; Qwen only wins 6 where T5Gemma fails.
- **Both models fail on >50% of questions** — 73/133 are both_wrong.
- The failure modes are task-specific and largely distinct between the two tasks.

## Score Summary

| Category | Data-Imputation (66) | List-to-table (67) | Total (133) |
|---|---|---|---|
| Both correct | 18 (27%) | 16 (24%) | 34 (26%) |
| Both wrong | 38 (58%) | 35 (52%) | 73 (55%) |
| T5Gemma only | 9 (14%) | 11 (16%) | 20 (15%) |
| Qwen only | 1 (2%) | 5 (7%) | 6 (5%) |

## Data-Imputation Findings

**Task:** Given a table with a missing cell, predict the correct value.

### Why T5Gemma wins (+9 vs +1)

T5Gemma is more reliable at **row-lookup in large tables**. The recurring "datatables.net"
employee table (cases 30, 41, 43, 44, 46-48, 51) requires finding the right row by name
and extracting a specific column value. Qwen frequently retrieves values from the wrong
row. T5Gemma's bidirectional attention may help it maintain row-column alignment in wide
tables.

### Dominant failure mode: hallucination (33/66 tagged)

Both models fail identically on questions requiring **world knowledge not in the table** —
stadium capacities, sports records, geologic stage names, historical dates. They produce
plausible but incorrect values. This accounts for the bulk of the 38 both_wrong cases.

### Other patterns

- **Null/sentinel labels** (`nan`, `?`, `-`): Both models refuse to output these as
  imputed values, instead hallucinating real content (3 cases).
- **Sequential/pattern values** (episode numbers, repeated categories): Trivially solved
  by both models — these are the 18 both_correct cases.
- **Shared hallucinations**: In at least 2 cases, both models produce the *exact same*
  wrong answer (e.g., "June 18 & 19" for idx 52), suggesting shared training data bias.

### Tag frequency

| Tag | Count | Notes |
|---|---|---|
| value_extraction | 33 | Core task: extract the right value |
| hallucination | 33 | Plausible but wrong values |
| correct_approach | 28 | Right reasoning, right answer |
| reasoning_error | 25 | Wrong inference from available data |
| wrong_row | 23 | Retrieved value from incorrect row |
| completely_off | 20 | Not even close to correct |
| both_trivial | 18 | Pattern completion, sequential numbers |
| close_but_wrong | 12 | Almost right (off by small details) |

## List-to-table Findings

**Task:** Convert a raw text list into a structured table with `||` column delimiters.

### Why T5Gemma wins (+11 vs +5)

T5Gemma is better at **preserving compound tokens** — keeping multi-word values like
city+state ("Smyrna TN"), name+position codes ("Dexter FowlerCF"), and range expressions
("4 - 0") intact as single cells. Qwen tends to over-split on whitespace and punctuation
boundaries.

### Why Qwen wins when it does (+5)

Qwen handles some **ambiguous delimiters** better — particularly dash-separated ranges
("X - Y") that should be kept atomic (idx 73), and cases where T5Gemma gets truncated
by output length limits.

### Dominant failure mode: column boundary errors (39/67 tagged wrong_column)

The core challenge is **tokenization ambiguity**: where does one column end and the next
begin? Both models struggle with:

- Compound names with embedded spaces (address strings, multi-word titles)
- Adjacent numeric fields without clear delimiters
- Leading rank numbers attached to names
- Multi-level headers / colspan structures

### Delimiter spacing artifact

Qwen consistently uses ` || ` (with spaces) while T5Gemma uses `||`. The evaluator
is **inconsistent** about accepting this — sometimes Qwen gets credit, sometimes not.
This is a pure formatting artifact that inflates T5Gemma's apparent advantage. Affected
cases: idx 87, 88, 91, 95, 98, 101, 104, 117.

**Estimated impact:** At least 3-4 of Qwen's "failures" are delimiter spacing alone,
which would narrow the List-to-table gap from +6pp to roughly +2-3pp.

### Other patterns

- **Truncation**: Both models hit output length limits on long tables with URLs or HTML
  entities (5-7 cases). T5Gemma is slightly more compact.
- **Unicode handling**: Qwen converts accented characters to Unicode escapes (idx 85),
  failing on Spanish/international text where T5Gemma preserves original encoding.
- **Sparse tables**: Tables with blank cells or variable column counts defeat both models.

### Tag frequency

| Tag | Count | Notes |
|---|---|---|
| structural_error | 45 | Table structure mis-identified |
| wrong_column | 39 | Column boundary errors |
| delimiter_error | 32 | Wrong delimiter format/placement |
| correct_approach | 19 | Right structure, right answer |
| both_trivial | 16 | Simple uniform tables |
| format_error | 13 | Output formatting issues |
| truncation | 7 | Output cut off |

## Cross-task Themes

1. **T5Gemma's bidirectional attention helps with alignment.** Both row-lookup in
   Data-Imputation and column-boundary detection in List-to-table require tracking
   positional relationships across the table. T5Gemma is more reliable at this.

2. **Evaluator artifacts inflate gaps.** Delimiter spacing in List-to-table and
   null-value handling in Data-Imputation are formatting issues, not comprehension
   failures. Roughly 3-5 of the 20 T5Gemma-only wins may be evaluator artifacts.

3. **The "both wrong" majority is real.** 55% of questions are genuinely hard —
   requiring world knowledge (imputation) or resolving ambiguous tokenization
   (list-to-table). Neither architecture solves these.

4. **Easy cases are trivially easy.** The 34 both_correct cases are mostly pattern
   completion (sequential numbers, repeated values) or simple uniform tables. There's
   a sharp cliff between "trivial" and "impossible" with little middle ground.

## Files

- `paired_data.jsonl` — 133 joined pairs with scores
- `analyzed_pairs.jsonl` — 133 pairs with per-question explanations and tags
- `analysis_chunk_{0-4}.jsonl` — Raw agent outputs (5 chunks)
- `build_pairs.py` — Script to reproduce the paired data
