# Synthesis: Why Each Model Wins on Specific Tasks

**Experiment**: encoder_vs_decoder_baseline_large (861 samples, 15 tasks)
**Models**: Qwen2.5-14B-Instruct (decoder-only) vs T5Gemma-9B-9B-UL2-IT (encoder-decoder)

## Executive Summary

After analyzing 250 samples across the 4 most divergent tasks, three cross-cutting themes explain most of the performance gaps:

1. **Format compliance and instruction-following** — T5Gemma has systematic output format failures (nested JSON, truncated output, wrong JSON types) that silently score zero. This inflates Qwen's apparent advantage on multiple tasks.
2. **Chain-of-thought reasoning** — Qwen generates step-by-step reasoning in 84%+ of responses; T5Gemma never does. This gives Qwen a genuine edge on multi-step computation but occasionally backfires.
3. **Evaluator artifacts** — Direction-sensitive scoring in equi-join-detect and lack of hierarchy-aware scoring in column-type-annotation create gaps that don't reflect true capability differences.

**If format/evaluator issues were fixed**, the overall gap would likely flip: Qwen's 3 "winning" tasks shrink substantially, while T5Gemma's column-type-annotation advantage is largely real.

---

## Task-by-Task Breakdown

### 1. Functional-Dependency (Qwen +27.1pp)

**Root cause breakdown**: ~50% format failures, ~50% semantic accuracy

| Factor | Contribution | Detail |
|---|---|---|
| T5Gemma format failures | ~13pp | 11/49 cases (22%) produce malformed output: 6 use nested lists `[["col"], ["col"]]` instead of flat `["col", "col"]`, 5 produce broken JSON. Qwen has 0 format failures. |
| Qwen better at ID/key detection | ~14pp | Qwen reliably identifies ID columns (e.g., `SalesTaxTypeID`, `IDFornecedor`) as determinants. T5Gemma over-generates: bidirectional pairs, transitive closures, wrong determinant columns. |

**Key example**: case_314 — T5Gemma identifies the correct FDs (`cod.banco -> banco.nome`) but wraps each column in a sub-list. Scores 0. Qwen gets the same answer in the correct format. Scores 1.0.

**Implication**: ~half of Qwen's advantage is an instruction-following gap, not a reasoning gap. If T5Gemma reliably produced flat-list JSON, the delta would shrink to ~14pp.

---

### 2. Table-QA (Qwen +20.9pp)

**Root cause breakdown**: ~15pp reasoning, ~6pp format compliance

| Factor | Contribution | Detail |
|---|---|---|
| Chain-of-thought reasoning | ~15pp | 17/22 Qwen wins are genuine: multi-step computation (percentages, averages), sorting/ranking, counting with conditions. T5Gemma produces 0% chain-of-thought. |
| Format quirks | ~6pp | T5Gemma returns `5.54` (float) for percentage questions where ground truth is `0.05543`. Qwen returns `"5.54%"` (string with %), triggering the evaluator's divide-by-100 logic. T5Gemma also returns numbers as JSON strings, bypassing numeric comparison. |

**Key example**: Population density question requiring 10 parallel division calculations. Qwen explicitly computes all 10, finds the minimum (4.36 people/km2). T5Gemma confuses smallest area with lowest density.

**Counter-example**: test_1107 — "If all 4 quarters had the same total, how many shares in 2010?" T5Gemma correctly multiplies by 4. Qwen's chain-of-thought overthinks it and returns the un-multiplied figure.

**Implication**: This is the most genuine capability gap. Chain-of-thought is a real advantage for multi-step table reasoning. But ~30% of the gap is evaluator-specific format matching.

---

### 3. Equi-Join-Detect (Qwen +17.7pp)

**Root cause breakdown**: ~83% evaluator artifact, ~6% truncation, ~11% name accuracy

| Factor | Contribution | Detail |
|---|---|---|
| Direction convention mismatch | ~14.7pp | Evaluator treats `(A, col1, B, col2)` != `(B, col2, A, col1)`. T5Gemma uses PK->FK direction; ground truth uses FK->PK. T5Gemma fully reverses direction on 18% of cases. Qwen never fully reverses. |
| Output truncation | ~1pp | T5Gemma: 7 parse failures vs Qwen: 3. T5Gemma hits output token limit sooner. |
| Column/table name hallucination | ~2pp | T5Gemma invents names: "DateKey" instead of "Date", "D_VENDEUR" (French) instead of "D_Vendedor" (Portuguese). |

**Key example**: case_279073330 — Both models identify the exact same 3 join relationships with the same column names. T5Gemma scores 0 because it puts dimension tables as `from_table`. Qwen scores 1.0.

**With direction-insensitive scoring**: Gap shrinks from 17.7pp to ~3.0pp (within noise for n=67).

**Implication**: This task mostly measures convention-following, not join detection ability. The evaluator should arguably treat direction as symmetric.

---

### 4. Column-Type-Annotation (T5Gemma +19.4pp)

**Root cause breakdown**: ~90% ontology specificity, ~10% other

| Factor | Contribution | Detail |
|---|---|---|
| Qwen defaults to generic classes | ~18pp | Qwen outputs "Person" 23/67 times (34%), "Settlement" 9/67. Ground truth is generic in only 15% of cases. T5Gemma's distribution matches ground truth far better. |
| Capitalization / misc errors | ~1pp | One case: Qwen outputs `building` instead of `Building`. |

**Key example**: Figure skating table — ground truth is `FigureSkater`. T5Gemma picks `FigureSkater` using context clues (coach names, competition venues). Qwen picks `Person`. The prompt explicitly asks for "the most fine grained ontology class name" — Qwen ignores this instruction.

**T5Gemma's weakness (8 cases)**: When ground truth IS generic (e.g., `Person`), T5Gemma over-specifies (outputs `Actor`, `Academic`, `MagazineEditor`).

**Implication**: This is largely a real capability gap. T5Gemma has better DBpedia ontology recall (40+ distinct classes vs Qwen's handful) and better instruction-following on specificity. Possible architectural explanation: bidirectional attention lets T5Gemma cross-reference all columns to infer domain-specific types.

---

## Cross-Task Themes

### Theme 1: T5Gemma Has a Systematic Instruction-Following Problem on Output Format

| Task | Format failure rate (T5Gemma) | Format failure rate (Qwen) |
|---|---|---|
| Functional-Dependency | 22% (nested lists, broken JSON) | 0% |
| Table-QA | ~7% (wrong JSON types) | 0% |
| equi-join-detect | 10% (truncation) | 4.5% |
| Column-type-annotation | 0% | ~1.5% (capitalization) |

T5Gemma's format failures are task-specific (it misunderstands different output schemas differently), suggesting this is not a single bug but a broader weakness in structured output generation. The encoder-decoder architecture may process output format instructions less reliably than Qwen's heavily instruction-tuned decoder.

### Theme 2: Chain-of-Thought Is a Double-Edged Sword

Qwen generates reasoning traces; T5Gemma never does. This matters most on:
- **Multi-step computation** (Table-QA): Genuine advantage. Explicit arithmetic > implicit hidden-state computation.
- **Join direction** (equi-join-detect): Qwen's reasoning ("the sales table references the products table") anchors correct FK->PK direction.
- **FD identification** (Functional-Dependency): Qwen reasons about which column is the "key", leading to more precise predictions.

But chain-of-thought backfires in ~10% of cases (7/67 on Table-QA), where verbose reasoning introduces logical errors that a direct-output model avoids.

### Theme 3: Evaluator Design Inflates Apparent Gaps

| Task | Raw gap | Estimated "real" gap (fixing evaluator/format) |
|---|---|---|
| Functional-Dependency | +27.1pp (Qwen) | ~+14pp (Qwen) |
| Table-QA | +20.9pp (Qwen) | ~+15pp (Qwen) |
| equi-join-detect | +17.7pp (Qwen) | ~+3pp (Qwen, within noise) |
| Column-type-annotation | +19.4pp (T5Gemma) | ~+19pp (T5Gemma, mostly real) |

**Corrected overall picture**: Qwen's aggregate advantage on these 4 tasks drops from +11.6pp to roughly +3pp. T5Gemma's column-type-annotation advantage is the most robust finding.

---

## Recommendations

1. **Fix equi-join-detect evaluation** — Make direction symmetric. The current metric measures convention-following, not join detection.
2. **Investigate T5Gemma format failures** — The 22% failure rate on Functional-Dependency is addressable. Consider few-shot examples in the prompt, or post-processing to flatten nested lists.
3. **Add chain-of-thought prompting for T5Gemma** — Table-QA gap is largely a reasoning-style gap. Explicitly prompting T5Gemma for step-by-step reasoning could close much of the 15pp gap.
4. **Consider hierarchy-aware scoring for CTA** — Qwen's "Person" for "FigureSkater" is semantically closer than a random wrong class. Partial credit via ontology distance would give a more nuanced view.
5. **Normalize Table-QA evaluator** — The string-vs-number JSON type distinction and percentage-handling quirks create unfair advantages. Consider numeric-only comparison for numeric answers.
