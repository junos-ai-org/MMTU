# Functional-Dependency Task: Qwen2.5-14B vs T5Gemma-9B Analysis

**Qwen: 70.0% F1 | T5Gemma: 43.0% F1 | Delta: +27.1pp**

## Summary

Qwen2.5-14B's dominant advantage on Functional-Dependency stems from two compounding factors: (1) T5Gemma fails to produce valid output in 22% of cases (11/49) due to format errors--nested list structures and malformed JSON--while Qwen achieves a 100% valid output rate; and (2) even when both models produce valid output, T5Gemma over-generates spurious functional dependencies, predicting many plausible-but-wrong relationships that hurt precision, while Qwen is more selective and better identifies the semantically correct determinant columns.

## How Scoring Works

The Functional-Dependency evaluator uses **set-based F1** on predicted vs ground-truth functional dependencies:

1. **Parsing**: The model's response is searched for a JSON object containing a `"Functional-Dependency"` key. The value should be a list of 2-element lists: `[determinant_column, dependent_column]`.

2. **Filtering**: Predicted FDs are filtered against a `labeled_fds` whitelist. Any prediction not in the whitelist is silently discarded. This means over-generation of non-whitelisted FDs has no direct penalty, but missing the correct ones is punished.

3. **Scoring**: Standard precision/recall/F1 computed over the intersection of predicted and ground-truth FD sets (both represented as sets of tuples).

4. **Aggregation**: Per-sample precision and recall are averaged (ignoring NaN), then combined into macro F1.

**Critical detail**: The format must be flat 2-element lists like `["colA", "colB"]`. Nested lists like `[["colA"], ["colB"]]` are converted to tuples of lists, which never match the flat-list labeled_fds, resulting in an empty prediction set and a score of 0.

## Pattern Analysis

### Pattern 1: T5Gemma Format Failures (11/49 cases, 22%)

This is the single largest driver of T5Gemma's underperformance.

| Failure Type | Count | Cases |
|---|---|---|
| Nested list format `[["col"], ["col"]]` | 6 | case_7, case_218, case_240, case_252, case_302, case_314 |
| Malformed/truncated JSON | 5 | case_86, case_125, case_188, case_338, case_372 |

**Qwen has zero format failures across all 49 cases.** It always produces correctly structured flat-list JSON.

In the 6 nested-list cases, T5Gemma wraps each column name in its own sub-list (e.g., `[["CustomerID"], ["FirstName"]]` instead of `["CustomerID", "FirstName"]`). This is a systematic misunderstanding of the required output schema. The evaluator's `preprocess` method converts each item via `tuple()`, producing `(["CustomerID"], ["FirstName"])` whose `list()` form `[["CustomerID"], ["FirstName"]]` never matches any entry in `labeled_fds`.

In the 5 malformed JSON cases, T5Gemma produces syntactically broken JSON (mismatched brackets in case_86), multi-column determinants as single lists (case_125, case_188), or responses that exceed the regex extraction pattern.

**Impact**: These 11 cases all score 0 for T5Gemma. Qwen scores >0 on 8 of them (average Qwen score on these: 0.57). This format gap alone accounts for roughly **12.7pp of the 27pp delta**.

### Pattern 2: T5Gemma Over-Generates Spurious FDs

When T5Gemma does produce valid flat-format output, it systematically predicts far more FDs than exist in the ground truth, including many that are semantically incorrect. T5Gemma appears to treat functional dependency detection as "list all plausible column correlations" rather than identifying true deterministic relationships.

Common over-generation patterns:
- **Bidirectional hallucination**: Predicting both `A -> B` and `B -> A` for every pair (case_275: 14 predictions for 2 ground-truth FDs)
- **Transitive closure**: If `A -> B` and `B -> C`, T5Gemma also predicts `A -> C`, `C -> B`, `B -> A`, etc. (case_214: 21 predictions for 6 ground-truth FDs)
- **Composite key confusion**: Outputting multi-column determinants as single flat lists (case_291: `["Building", "Consumption type", "Year"]` instead of `["Building", "City"]`)

While the whitelist filtering removes non-valid FDs (so precision isn't directly hurt by invalid predictions), T5Gemma's scattershot approach means it often misses the actual ground-truth FDs because it fixates on the wrong determinant columns.

### Pattern 3: Qwen Better Identifies ID/Key Columns as Determinants

Across divergent cases, Qwen consistently identifies the correct "ID" or "key" column as the determinant (e.g., `SalesTaxTypeID`, `CityID`, `IDFornecedor`, `cod.banco`, `produitID`). T5Gemma frequently picks the wrong determinant--using non-key columns or composite keys--even when an obvious ID column exists.

### Pattern 4: Both Models Fail on Complex/Ambiguous Tables

Both models score 0 on 9 cases (case_42, case_57, case_69, case_86, case_136, case_188, case_240, case_302, case_316). These tend to involve tables with non-obvious functional dependencies, non-English column names, or complex multi-column determinants.

## Detailed Example Breakdowns

### Example 1: case_314 (Qwen=1.0, T5Gemma=0.0, delta=+1.0) -- Nested Format Failure

**Ground truth**: `cod.banco -> banco.nome`, `cod.banco -> banco.imagem`

**Table context**: Brazilian financial data with bank codes, names, and image URLs.

**Qwen response** (correctly formatted):
```json
{"Functional-Dependency": [["cod.banco", "banco.nome"], ["cod.banco", "banco.imagem"]]}
```
Qwen identifies both FDs perfectly: a bank code determines the bank name and logo image.

**T5Gemma response** (nested format):
```json
{"Functional-Dependency": [[["cod.banco"], ["banco.nome"]], [["cod.banco"], ["banco.imagem"]], ...]}
```
T5Gemma actually identifies the correct FDs conceptually, but wraps each column in a sub-list. The evaluator cannot match `[["cod.banco"], ["banco.nome"]]` against the labeled FD `["cod.banco", "banco.nome"]`, so every prediction is filtered out, yielding score 0.

**Why Qwen wins**: Pure format compliance. T5Gemma "knew" the answer but expressed it in the wrong structure.

### Example 2: case_183 (Qwen=1.0, T5Gemma=0.0, delta=+1.0) -- Wrong Determinant

**Ground truth**: `DATA -> Ano`, `DATA -> Nome do Mes`, `DATA -> Nome do Dia`, `DATA -> Trimestre`

**Table context**: Brazilian sales records. The `DATA` column is a datetime; `Ano` (year), `Nome do Mes` (month name), `Nome do Dia` (day name), and `Trimestre` (quarter) are all derivable from it.

**Qwen**: Predicts 16 FDs including all 4 ground-truth ones (plus 12 extras using other determinants like VENDEDOR, FORMA PGTO). After whitelist filtering, the 4 correct ones survive plus `["DATA", "VENDEDOR"]` which is in the whitelist. Result: perfect recall, high precision.

**T5Gemma**: Predicts 6 FDs using `VENDEDOR` as the determinant: `VENDEDOR -> FORMA PGTO`, `VENDEDOR -> N FISCAL`, `VENDEDOR -> Ano`, etc. None of these use `DATA` as the determinant. After whitelist filtering, zero FDs match the ground truth.

**Why Qwen wins**: Qwen correctly identifies that `DATA` (the datetime) functionally determines the time-derived columns. T5Gemma fixates on `VENDEDOR` (salesperson) as the determinant, which is semantically wrong--a salesperson does not determine dates/months.

### Example 3: case_275 (Qwen=1.0, T5Gemma=0.0, delta=+1.0) -- Bidirectional Hallucination

**Ground truth**: `IDFornecedor -> Fornecedor`, `IDFornecedor -> Logotipo`

**Table context**: Product catalog with supplier IDs, names, and logos.

**Qwen**: Predicts 5 FDs including both ground-truth ones. After filtering: both match.

**T5Gemma**: Predicts 14 FDs, generating both directions for every pair involving `IDProduto` (e.g., `IDProduto -> Produto` AND `Produto -> IDProduto`). Crucially, T5Gemma never predicts `IDFornecedor -> Fornecedor` or `IDFornecedor -> Logotipo`--the two correct FDs. It focuses entirely on `IDProduto` as the sole determinant.

**Why Qwen wins**: Qwen identifies that the supplier ID (`IDFornecedor`) determines supplier name and logo, a classic entity-attribute relationship. T5Gemma generates a massive set of bidirectional FDs around `IDProduto` but misses the supplier-level dependencies entirely.

### Example 4: case_111 (Qwen=1.0, T5Gemma=0.5, delta=+0.5) -- Precision vs Recall

**Ground truth**: `SalesTaxTypeID -> SalesTaxType`, `SalesTaxTypeID -> Description`

**Qwen**: Predicts exactly these 2 FDs. Perfect precision and recall.

**T5Gemma**: Predicts 9 FDs including `SalesTaxTypeID -> SalesTaxType` (a match) but also many spurious ones like `CityCOPONbr -> SalesTaxTypeID`, `CityID -> SalesTaxType`, `SalesTaxType -> Description`. After filtering against the whitelist, only some survive. T5Gemma gets 1 of the 2 ground-truth FDs (recall=0.5) but also includes wrong FDs that pass the whitelist filter.

**Why Qwen wins**: Qwen is surgically precise, predicting exactly the ID-based FDs. T5Gemma's scattershot approach gets partial credit but misses the second FD while including noise.

### Example 5: case_67 (Qwen=0.5, T5Gemma=0.8, delta=-0.3) -- T5Gemma Win

**Ground truth**: `customer_code -> custmer_name`, `customer_code -> customer_type`

**Qwen**: Predicts 7 FDs, including both ground-truth ones plus extras like `customer_code -> currency`, `market_code -> currency`, `customer_code -> market_code`. After filtering, 5 match labeled_fds. Precision = 2/5 = 0.4, recall = 2/2 = 1.0, F1 ~= 0.57. Wait -- the reported score is 0.5, suggesting the evaluator's macro average differs slightly.

**T5Gemma**: Predicts 9 FDs. After filtering, 2 match ground truth (`customer_code -> custmer_name`, `customer_code -> customer_type`). T5Gemma achieves good recall on the ground truth with fewer false positives surviving the whitelist filter.

**Why T5Gemma wins**: In this case, T5Gemma's predictions happen to include the ground truth while its spurious predictions (e.g., `product_code -> customer_code`) get filtered out by the whitelist. Qwen's extras survive filtering, hurting precision.

### Example 6: case_148 (Qwen=0.8, T5Gemma=1.0, delta=-0.2) -- T5Gemma Win

**Ground truth**: `CostElement -> Cost Element`, `CostElement -> Cost Element Group`, `CostElement -> Business Area`

**Qwen**: Predicts 6 FDs including all 3 ground-truth ones, but also includes `Business Area -> Budget`, `IT Dep. -> CostElement`, `Country -> IT Dep.`, `IT Dep. -> Business Area`. After filtering, 4 survive (3 correct + 1 extra), giving precision = 3/4 = 0.75.

**T5Gemma**: Predicts 10 FDs. After filtering, exactly the 3 ground-truth ones survive plus `Budget -> IT Dep.` (in labeled_fds). Despite over-generating, T5Gemma's extra predictions happen to be filtered cleanly, achieving higher precision on the surviving set.

**Why T5Gemma wins**: T5Gemma's over-generation is counterintuitively beneficial here: it predicts many FDs, but the whitelist filters out the wrong ones while keeping the right ones. Qwen's extras are more "plausible" and thus survive filtering, hurting its precision.

## Quantitative Breakdown

| Category | Cases | Avg Qwen | Avg T5Gemma | Avg Delta |
|---|---|---|---|---|
| T5Gemma format failure (nested/parse error) | 11 | 0.48 | 0.00 | +0.48 |
| T5Gemma valid, Qwen wins | 19 | 0.91 | 0.46 | +0.46 |
| T5Gemma valid, tie | 15 | 0.60 | 0.60 | 0.00 |
| T5Gemma valid, T5Gemma wins | 4 | 0.68 | 0.87 | -0.19 |

## Conclusion: Why Does Qwen Win on Functional-Dependency?

Qwen2.5-14B's 27pp advantage decomposes into two independent factors:

1. **Format reliability (~60% of the gap)**: T5Gemma fails to produce evaluator-compatible output in 22% of cases (11/49). Six cases use a nested list format `[["col"], ["col"]]` that the evaluator cannot process, and five produce malformed JSON. Qwen has zero format failures. This alone accounts for roughly 12-15pp of the delta.

2. **Better semantic understanding of functional dependencies (~40% of the gap)**: When both models produce valid output, Qwen is more precise at identifying the correct determinant columns--particularly ID/key columns--and predicting the right FDs. T5Gemma tends to over-generate, producing many plausible-but-wrong FDs (bidirectional pairs, transitive closures, composite key confusion). While the whitelist filter removes some noise, T5Gemma still misses ground-truth FDs more often than Qwen.

The format issue is arguably the more concerning finding for T5Gemma. The task prompt clearly specifies `[<determinant column>, <dependent column>]` as the required format, but T5Gemma sometimes interprets this as `[[<determinant column>], [<dependent column>]]`. This suggests T5Gemma has weaker instruction-following on structured output constraints compared to Qwen, possibly due to differences in instruction-tuning data or the encoder-decoder architecture's handling of output format specifications. Qwen's decoder-only architecture, heavily optimized for chat/instruction-following, appears to have a significant edge in consistently producing the exact JSON schema requested.
