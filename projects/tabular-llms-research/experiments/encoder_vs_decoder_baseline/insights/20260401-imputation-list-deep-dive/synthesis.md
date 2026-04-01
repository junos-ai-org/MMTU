# Insight: Data-Imputation & List-to-table Deep Dive

**Date:** 2026-04-01
**Scope:** 133 paired questions (66 Data-Imputation, 67 List-to-table)
**Models:** Qwen2.5-14B-Instruct vs T5Gemma-9B-9B-UL2-IT
**Method:** Per-question LLM-assisted analysis (Claude Sonnet 4.6, 5 parallel agents)

## Headlines

- **T5Gemma wins on both tasks.** It's correct on 20 questions where Qwen fails; Qwen only wins 6 where T5Gemma fails.
- **Both models fail on >50% of questions** — 73/133 are both_wrong.
- The failure modes are task-specific: **hallucination** dominates Data-Imputation, **column boundary errors** dominate List-to-table.

## Score Summary

| Category | Data-Imputation (66) | List-to-table (67) | Total (133) |
|---|---|---|---|
| Both correct | 18 (27%) | 16 (24%) | 34 (26%) |
| Both wrong | 38 (58%) | 35 (52%) | 73 (55%) |
| T5Gemma only | 9 (14%) | 11 (16%) | 20 (15%) |
| Qwen only | 1 (2%) | 5 (7%) | 6 (5%) |

---

## Data-Imputation

**Task:** Given a table with a missing cell, predict the correct value.

### Failure Mode Counts

| Failure Mode | Both Wrong | T5Gemma Only | Qwen Only | Both Correct | Total |
|---|---|---|---|---|---|
| hallucination | 30 | 2 | 1 | 0 | 33 |
| reasoning_error | 24 | 1 | 0 | 0 | 25 |
| wrong_row | 18 | 5 | 0 | 0 | 23 |
| completely_off | 20 | 0 | 0 | 0 | 20 |
| close_but_wrong | 11 | 1 | 0 | 0 | 12 |
| both_trivial | 0 | 0 | 0 | 18 | 18 |

### Failure Mode 1: Hallucination (33/66 cases)

Both models produce plausible but fabricated values when the answer requires **world knowledge not present in the table**. This is the single largest failure mode.

**Example — Tamil film title (idx 1, both_wrong):**
- **Question:** What film does the song "Eecham pazham" (2015) belong to?
- **Label:** `Puriyadha Anandham Pudhidhaga Arambam`
- **Qwen:** `Irandam Kathirvelan Kadhal` — hallucinated a real Tamil film name, wrong one
- **T5Gemma:** `Eecham Pazham` — returned the song name itself, confusing column identities
- **Why:** Niche cultural knowledge (Tamil film-song associations) not memorized by either model

**Example — F1 constructor (idx 17, Qwen wins):**
- **Question:** What constructor did John Watson drive at the 1975 Race of Champions?
- **Label:** `Lotus-Ford`
- **Qwen:** `Lotus-Ford` — correct, likely memorized this specific F1 dataset
- **T5Gemma:** `Brabham-Ford` — guessed another Ford-powered team from the same era
- **Why:** Both models attempted factual recall; Qwen had stronger memorization of this specific race

### Failure Mode 2: Wrong-Row Lookup (23/66 cases)

The recurring "datatables.net" employee table appears in ~12 cases with the same structure but different target employees. The model must find an employee by name and extract a column value. Qwen frequently retrieves values from adjacent rows.

**Example — employee age lookup (idx 30, T5Gemma wins):**
- **Question:** What is the age of Quinn Flynn (Support Lead, Edinburgh, $342,000)?
- **Label:** `22`
- **Qwen:** `30` — pulled the age of a different employee in the table
- **T5Gemma:** `22` — correctly located Quinn Flynn's row
- **Why:** T5Gemma's bidirectional attention may help maintain row-column alignment in wide, multi-row tables

**Example — employee age lookup (idx 9, T5Gemma wins):**
- **Question:** What is the age of Michael Bruce (Javascript Developer, Edinburgh, $4,080)?
- **Label:** `29`
- **Qwen:** `25` — off by 4, likely a different Michael or adjacent row
- **T5Gemma:** `29` — correct match on name + other attributes

T5Gemma wins 5 of the 23 wrong_row cases; Qwen wins 0. This is the clearest architectural advantage.

### Failure Mode 3: Null/Sentinel Labels (3 cases)

When the ground truth is a sentinel value (`?`, `nan`, `-`), both models refuse to output it and instead hallucinate a real value.

**Example — horse racing fence column (idx 27, both_wrong):**
- **Question:** What is the fence number for this race entry?
- **Label:** `?` (literally — the source data has no known value)
- **Qwen:** `6`
- **T5Gemma:** `8`
- **Why:** Both models treated `?` as a placeholder to fill rather than a legitimate table value that appears in many rows of the same column

### Failure Mode 4: Partial Value Extraction (1 case, illustrative)

**Example — multi-person attribution (idx 3, T5Gemma wins):**
- **Question:** Who directed the documentary "The War"?
- **Label:** `Ken Burns, Lynn Novick`
- **Qwen:** `Ken Burns` — returned only the primary director
- **T5Gemma:** `Ken Burns, Lynn Novick` — returned the complete multi-value answer
- **Why:** Qwen truncated to the most prominent name; T5Gemma preserved the full cell contents

### What "Both Correct" Looks Like (18 cases)

All 18 both_correct cases are trivially solvable — sequential episode numbers, repeated category values, `nan` propagation from all-null columns. There is no middle ground between "trivial" and "hard."

---

## List-to-table

**Task:** Convert a raw text list into a structured table with `||` column delimiters.

### Failure Mode Counts

| Failure Mode | Both Wrong | T5Gemma Only | Qwen Only | Both Correct | Total |
|---|---|---|---|---|---|
| structural_error | 32 | 9 | 4 | 0 | 45 |
| wrong_column | 26 | 9 | 4 | 0 | 39 |
| delimiter_error | 26 | 3 | 3 | 0 | 32 |
| format_error | 11 | 1 | 1 | 0 | 13 |
| truncation | 6 | 0 | 1 | 0 | 7 |
| both_trivial | 0 | 0 | 0 | 16 | 16 |

### Failure Mode 1: Column Boundary / Structural Errors (45/67 cases)

The dominant failure. Both models struggle to determine where one column ends and the next begins when the input contains compound values with embedded spaces.

**Example — compound place names (idx 72, T5Gemma wins):**
- **Label:** `7||Mariental Land||NA.HA.MR||...`
- **Qwen:** `7 || Mariental || Land || NA.HA.MR||...` — split "Mariental Land" into two columns
- **T5Gemma:** `7||Mariental Land||NA.HA.MR||...` — kept the compound name intact
- **Why:** Qwen used a naive space-splitting heuristic; T5Gemma recognized the compound geographic name

**Example — size ranges (idx 73, Qwen wins):**
- **Label:** `XS||34||78 - 81||62 - 64||86 - 89`
- **Qwen:** `XS || 34 || 78 - 81 || 62 - 64 || 86 - 89` — correctly kept ranges as atomic values (5 columns)
- **T5Gemma:** `XS||34||78||-||81||62||-||64||86||-||89` — split on dashes, producing 13 columns
- **Why:** T5Gemma naively split on ` - ` delimiters; Qwen recognized "X - Y" as a range expression

**Example — institution name with city (idx 82, both_wrong):**
- **Label:** `Bethel College Mishawaka||Mishawaka||Masters||2,097`
- **Qwen:** `Bethel College||Mishawaka||Mishawaka||Masters||2,097` — split institution name, creating 5 columns
- **T5Gemma:** `Bethel College||Mishawaka||Mishawaka||Masters||2,097` — identical error
- **Why:** "Bethel College Mishawaka" is the full institution name, but "Mishawaka" coincidentally matches the city column. Both models made the same wrong split.

**Example — locale-specific number formats (idx 123, both_wrong):**
- **Label:** `Magne Langedal||70 år||213 441||1 200 559||42 914`
- **Qwen:** `Magne Langedal || 70 år || 213 || 441 || 1 || 200 || 559 || 42 || 914`
- **T5Gemma:** `Magne Langedal||70 år||213||441||1||200||559||42||914`
- **Why:** Norwegian numbers use space as thousands separator (213 441 = 213,441). Both models split every space, fragmenting 3 monetary columns into 7+.

### Failure Mode 2: Delimiter Spacing Artifact (affects ~8 cases)

Qwen consistently uses ` || ` (with spaces) while T5Gemma uses `||`. The evaluator is inconsistent about accepting this.

**Example — athletics results (idx 117, T5Gemma wins on formatting alone):**
- **Label:** `2||5||Alli Ngaimoko (UGA)||0.208||21.21||Q`
- **Qwen:** `2 || 5 || Alli Ngaimoko (UGA) || 0.208 || 21.21 || Q` — structurally correct, spacing rejected
- **T5Gemma:** `2||5||Alli Ngaimoko (UGA)||0.208||21.21||Q` — exact match
- **Why:** Pure formatting artifact. Qwen's parsing is correct; it just adds spaces around delimiters.

**Estimated impact:** 3-4 of Qwen's failures in List-to-table are delimiter spacing alone, narrowing the real gap from +6pp to ~+2-3pp.

### Failure Mode 3: Unicode/Encoding (1-2 cases)

**Example — Spanish tournament results (idx 85, T5Gemma wins):**
- **Label:** `2008||6||3||0||3||8||5||9||+3||1ª Ronda`
- **Qwen:** `2008||6||3||0||3||8||5||9||+3||1\u00b0 Ronda` — escaped the `ª` character
- **T5Gemma:** `2008||6||3||0||3||8||5||9||+3||1ª Ronda` — preserved UTF-8
- **Why:** Qwen's tokenizer converts accented/special characters to Unicode escapes; T5Gemma preserves them verbatim

### Failure Mode 4: Truncation (7 cases)

**Example — website traffic data (idx 116, Qwen wins):**
- **Label:** Full table with 10+ rows of French hospital URLs and percentages
- **Qwen:** Complete output, all rows present
- **T5Gemma:** Truncated at the last row: `telemedecine.aphp.org||0.2%||1.` — missing final `0`
- **Why:** T5Gemma hit its output length limit just before completing. Qwen's longer generation window saved it here.

### What "Both Correct" Looks Like (16 cases)

All 16 are simple, uniform tables — consistent column counts, no compound values, no special characters. Clean tabular data that any model can parse.

---

## Cross-task Themes

### 1. T5Gemma's bidirectional attention helps with positional alignment

Both row-lookup in Data-Imputation and column-boundary detection in List-to-table require tracking positional relationships across the table. T5Gemma wins 5/23 wrong-row cases (Qwen wins 0) and 9/39 wrong-column cases (Qwen wins 4). The encoder's ability to attend in both directions appears to help with spatial reasoning over table structure.

### 2. Evaluator artifacts inflate the gap

| Artifact | Task | Estimated Cases | Effect |
|---|---|---|---|
| Delimiter spacing (` || ` vs `||`) | List-to-table | 3-4 | Inflates T5Gemma advantage |
| Null/sentinel labels (`?`, `nan`) | Data-Imputation | 2-3 | Penalizes both equally |
| Unicode escaping | List-to-table | 1-2 | Inflates T5Gemma advantage |

After adjusting for evaluator artifacts, T5Gemma's real advantage is ~15 cases vs Qwen's ~6, down from 20 vs 6.

### 3. The "both wrong" majority is real

55% of questions are genuinely hard — requiring world knowledge (imputation) or resolving ambiguous tokenization (list-to-table). Neither architecture solves these. The failures are often identical between models, suggesting shared limitations of the text-in/text-out paradigm for table understanding.

### 4. Sharp cliff between trivial and impossible

The 34 both_correct cases are all trivially solvable (sequential numbers, uniform tables). The 73 both_wrong cases are genuinely difficult. There is almost no "medium difficulty" range where one model consistently outperforms — the disagreements (26 total) are scattered across diverse edge cases rather than forming a coherent difficulty tier.

---

## Files

- `paired_data.jsonl` — 133 joined pairs with scores
- `analyzed_pairs.jsonl` — 133 pairs with per-question explanations and tags
- `analysis_chunk_{0-4}.jsonl` — Raw agent outputs (5 chunks)
- `build_pairs.py` — Script to reproduce the paired data
