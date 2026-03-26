# Encoder vs Decoder Baseline — Error Analysis

Qualitative analysis of divergent predictions between **Qwen2.5-14B-Instruct** (decoder-only) and **T5Gemma-9B** (encoder-decoder) on the large baseline (861 samples, 13 tasks).

## Summary Table

| Task | N | Both right | Qwen only | T5 only | Neither |
|---|---|---|---|---|---|
| Functional-Dependency (f1) | 49 | 5 | **17** | 2 | 25 |
| Table-QA (acc) | 67 | 14 | **22** | 8 | 23 |
| equi-join-detect (f1) | 67 | 5 | 5 | 9 | **48** |
| Column-type-annotation (acc) | 67 | 5 | 8 | **21** | 33 |
| Data-Imputation (acc) | 66 | 18 | 1 | **9** | 38 |
| List-to-table (acc) | 67 | 16 | 5 | **10** | 36 |

---

## 1. Functional-Dependency (Qwen +27pp)

Qwen dominates this task (70.0% vs 43.0%). Of 49 questions, Qwen gets 17 exclusively right while T5 gets 0 exclusively right at exact match.

### Example A: `case_372` (Qwen 1.0, T5 0.0)

**Table**: Product sales data with columns `SalesOrderID`, `SalesOrderDetailID`, `OrderQty`, `ProductID`, `UnitPriceDiscount`, `Color_many`, `Name`, `ProductNumber`, `Color`.

**Ground truth**: `ProductID -> Name`, `ProductID -> ProductNumber`, `ProductID -> Color`

**Qwen** correctly identifies the 3 ground-truth dependencies plus a few extra (which don't hurt under F1):
```json
{"Functional-Dependency": [["ProductNumber", "Color"], ["ProductNumber", "Name"],
  ["ProductNumber", "Color_many"], ["SalesOrderID", "SalesOrderDetailID"],
  ["ProductID", "Color"], ["ProductID", "Name"], ["ProductID", "ProductNumber"]]}
```

**T5** correctly identifies the 3 ground-truth pairs but then **over-generates massively** — listing nearly every column pair as a dependency (e.g., `SalesOrderID -> ProductID`, `SalesOrderID -> OrderQty`, `OrderQty -> ProductID`, etc.). This floods the output with false positives, destroying F1 precision.

**Pattern**: T5 treats functional dependency detection as "list all correlations" rather than identifying semantically meaningful determinant relationships. Qwen is more selective.

### Example B: `case_183` (Qwen 1.0, T5 0.0)

**Table**: Brazilian sales records with columns `DATA` (date), `VENDEDOR` (seller), `FORMA PGTO` (payment method), `N FISCAL`, `Ano`, `Nome do Mes`, `Nome do Dia`, `Trimestre`.

**Ground truth**: `DATA -> Ano`, `DATA -> Nome do Mes`, `DATA -> Nome do Dia`, `DATA -> Trimestre`

**Qwen** correctly identifies all 4 date-derived dependencies. It over-generates some spurious ones but includes all ground truth.

**T5** focuses on `VENDEDOR` as the determinant instead of `DATA`, completely missing the core insight that date determines year/month/day/quarter. It outputs `VENDEDOR -> FORMA PGTO`, `VENDEDOR -> N FISCAL`, etc. — plausible-sounding but wrong.

**Pattern**: T5 latches onto the first "entity-like" column (seller name) as the key, while Qwen correctly identifies that temporal decomposition (date -> month, quarter, etc.) is the real functional dependency.

---

## 2. Table-QA (Qwen +21pp)

Qwen significantly outperforms on question answering (53.7% vs 32.8%), with 22 exclusive wins vs 8 for T5.

### Example A: `7185f1c0...` — Numerical reasoning (Qwen 1.0, T5 0.0)

**Question**: "In which year did the entity experience the largest difference between 'property taxes' and 'investment earnings'?"

**Table**: Financial data with property taxes and investment earnings from 2000-2005.

**Qwen** shows its work — computes the difference for each year (e.g., 2005: 24,384,900 - 255,041 = 24,129,859) and correctly identifies **2005**.

**T5** answers **2002** with no reasoning. Simply wrong.

**Pattern**: Qwen's chain-of-thought reasoning (computing differences, comparing) gives it a major edge on multi-step numerical questions. T5 tends to guess without showing intermediate steps.

### Example B: `test_8` — Comparison reasoning (Qwen 1.0, T5 0.0)

**Question**: "Did Altria outperform the S&P 500?"

**Table**: Investment comparison showing Altria at $286.61 vs S&P 500 at $198.09 by Dec 2016 (both starting at $100).

**Qwen** compares endpoint values and answers **"yes"**.

**T5** answers **"Sometimes"** — hedging instead of reading the data. The question asks about overall performance, not year-by-year.

**Pattern**: T5 sometimes gives vague/hedged answers instead of committing to what the data shows.

### Example C: `test_333` — Simple arithmetic (T5 1.0, Qwen 0.0)

**Question**: "What is the average size (in square feet) of call centers in 2017?"

**Table**: Facility types with counts and total square footage. Call centers: 17 facilities, 1,400,000 total sq ft.

**T5** correctly computes 1,400,000 / 17 = **82,352.94** (matching ground truth).

**Qwen** overthinks it — questions whether the table has temporal data, notes there's only one row for call centers, and talks itself out of answering. The verbose reasoning becomes a liability when the computation is straightforward.

### Example D: `table-nu-1447` — Counting (T5 1.0, Qwen 0.0)

**Question**: "How many episodes featured Mario Batali?"

**Qwen** answers **4**, T5 answers **3** (correct). Qwen likely miscounts rows in the table.

**Pattern**: On simple lookup/counting tasks, T5's concise output format (just the answer) avoids the over-reasoning trap that sometimes leads Qwen astray.

---

## 3. equi-join-detect (Qwen +18pp)

Both models struggle (65.6% vs 47.8%), with 48/67 questions wrong for both at exact match. However, with partial credit (f1 > 0), Qwen gets partial credit on 15 exclusive questions vs T5's 1.

### Example A: `279073330` — Star schema joins (Qwen 1.0, T5 0.0)

**Tables**: `sales`, `products`, `locations`, `people` — a classic BI star schema.

**Ground truth**: 3 joins — `sales.Product = products.Product`, `sales.Geography = locations.Geo`, `sales.Sales Person = people.Sales person`

**Qwen** identifies all 3 correctly through semantic reasoning about the business logic.

**T5** also identifies the same 3 joins but uses a **different column name** (`DateKey` instead of `Date`) in some relationships — technically correct semantically but wrong for exact match. The output format is correct but column name precision fails.

### Example B: `939121682` — Complex BI schema (Qwen 1.0, T5 0.0)

**Tables**: `DIM_Calendar`, `DIM_Customers`, `DIM_Products`, `FACT_InternetSales`, `FACT_Budget` — a full data warehouse schema.

**Ground truth**: 4 joins including `FACT_InternetSales.OrderDateKey = DIM_Calendar.Date`

**Qwen** correctly identifies the join column as `Date` in `DIM_Calendar`.

**T5** uses `DateKey` instead of `Date` for the calendar dimension — a plausible column name that happens to be wrong. T5 also hallucinates extra joins (e.g., `DueDateKey`, `ShipDateKey`) that aren't in the ground truth.

**Pattern**: T5 struggles with exact column name matching in join detection, often using semantically similar but incorrect column names. It also tends to over-generate join relationships.

---

## 4. Column-type-annotation (T5 +19pp)

T5 dramatically outperforms Qwen (38.8% vs 19.4%), with 21 exclusive wins. This is T5's strongest relative advantage.

### Example A: `case_1198` — Case sensitivity (T5 1.0, Qwen 0.0)

**Table**: Architects and their buildings (George Gilbert Scott, George Dance the Younger, etc.)

**Ground truth**: `http://dbpedia.org/ontology/Building`

**T5** outputs exactly `Building` (capital B). **Qwen** outputs `building` (lowercase b). The evaluation is case-sensitive, so Qwen gets zero for a single-character difference.

**Pattern**: Qwen frequently gets the right ontology concept but with wrong casing. T5 appears to have better recall of the exact DBpedia ontology URI conventions.

### Example B: `case_1708` — Ontology vocabulary (T5 1.0, Qwen 0.0)

**Table**: Naval vessels (USS Zuni, USS Yorktown, Voima, etc.) with dates.

**Ground truth**: `http://dbpedia.org/ontology/Ship`

**T5** correctly outputs `Ship`. **Qwen** outputs `Vessel` — a reasonable synonym that simply isn't the DBpedia class name.

**Pattern**: Qwen uses semantically correct but non-canonical class names. T5 has better memorization of the specific DBpedia ontology vocabulary — likely because the encoder's bidirectional attention helps it learn exact string patterns from training data.

### Example C: `case_405` — Correct generalization (Qwen 1.0, T5 0.0)

**Table**: Portuguese castles with district columns (Vila Real, Portalegre, Faro, Braga, etc.)

**Ground truth**: `http://dbpedia.org/ontology/Settlement`

**Qwen** correctly identifies `Settlement`. **T5** outputs `PortugueseDistrict` — too specific and not a standard DBpedia class.

**Pattern**: When T5 fails, it tends to over-specify (guessing a narrow subclass). When Qwen fails, it under-specifies or uses synonyms.

---

## 5. Data-Imputation (T5 +12pp)

T5 outperforms on missing value prediction (40.6% vs 29.0%), with 9 exclusive wins. Notably, Qwen has only 1 exclusive win — when Qwen gets it right, T5 almost always does too.

### Example A: `case_125` — Multi-value cells (T5 1.0, Qwen 0.0)

**Table**: Documentary films with directors.

**Ground truth**: `Ken Burns, Lynn Novick`

**Qwen** outputs `Ken Burns` — gets the primary director but misses the co-director. **T5** correctly outputs the full value `Ken Burns, Lynn Novick`.

**Pattern**: T5 is better at reproducing compound cell values (multiple names, full strings). Qwen tends to truncate or simplify.

### Example B: `case_5` — Numerical imputation (T5 1.0, Qwen 0.0)

**Table**: Employee records with Name, Position, Office, Age, Start date, Salary.

**Ground truth**: `22` (age)

**Qwen** guesses `30`. **T5** correctly predicts `22`.

**Pattern**: T5 appears to better leverage contextual patterns in the surrounding data to predict missing numerical values. The encoder's bidirectional view of the full table may help it pick up on distributional patterns.

### Example C: `case_303` — Domain knowledge (Qwen 1.0, T5 0.0)

**Table**: 1975 F1 qualifying results — Driver #6 John Watson's constructor is `[MISSING]`.

**Ground truth**: `Lotus-Ford`

**Qwen** correctly answers `Lotus-Ford`. **T5** guesses `Brabham-Ford` — wrong team, though the right engine manufacturer suffix.

**Pattern**: Qwen's larger parameter count (14B vs 9B active) gives it an edge on questions requiring specific factual knowledge (1975 F1 constructor assignments).

---

## 6. List-to-table (T5 +8pp)

T5 outperforms on column delimiter detection (38.8% vs 31.3%), with 10 exclusive wins.

### Example A: `case_487` — Ambiguous column boundaries (T5 1.0, Qwen 0.0)

**Table**: Vacation rental pricing with dates, weekly rates (in multiple currencies), and nightly rates.

**Ground truth**: `&euro; 504US$ 680GB£ 424` should be a **single cell** under "Weekly rate".

**Qwen** incorrectly splits the multi-currency value into separate columns: `&euro; 504||US$ 680||GB£ 424`. This creates too many columns.

**T5** correctly keeps the concatenated currency string as one cell: `&euro; 504US$ 680GB£ 424||Get instant quote`.

**Pattern**: T5 better recognizes that multi-currency values belong together as a single cell. The encoder's bidirectional context helps it see the overall column structure before deciding where to split.

### Example B: `case_247` — Column count consistency (T5 1.0, Qwen 0.0)

**Table**: BitTorrent tracker list with URLs, status, dates, and numeric columns.

**Ground truth**: 6 columns per row.

**Qwen** produces rows with 5 columns (missing one delimiter), making the column count inconsistent.

**T5** correctly produces 6 columns per row, matching the expected structure.

**Pattern**: T5 is more consistent at maintaining uniform column counts across all rows. The encoder's ability to see the full input at once helps it count columns globally rather than deciding row-by-row.

### Example C: `case_608` — First-column boundary (Qwen 1.0, T5 0.0)

**Table**: Product specifications (Power Supply, Bulbs, Watts, etc.)

**Ground truth**: First column delimiter after the attribute name (e.g., `Power Supply||AC||...`)

**T5** merges `Power Supply AC` into a single cell, missing the first delimiter. **Qwen** correctly separates them.

**Pattern**: When T5 fails at List-to-table, it tends to merge the first two columns. The first column boundary is often the hardest to detect because attribute names can look like they belong with the first value.

---

## Cross-Task Patterns

### Qwen's strengths (decoder-only, 14B)
1. **Multi-step reasoning**: Chain-of-thought on numerical QA, computing differences, comparisons
2. **Selective output**: Better precision — avoids flooding output with false positives (Functional-Dependency)
3. **Domain knowledge**: Larger model memorizes more facts (F1 constructors, financial terminology)

### T5Gemma's strengths (encoder-decoder, 9B active)
1. **Exact string recall**: Better at reproducing exact ontology URIs, compound values, canonical names (Column-type-annotation, Data-Imputation)
2. **Structural comprehension**: Better at understanding table geometry — column counts, cell boundaries (List-to-table)
3. **Concise answers**: Less prone to over-reasoning that leads to wrong conclusions (Table-QA simple lookups)
4. **Bidirectional context**: Sees the full table at once, helping with global structure decisions

### Shared weaknesses
1. **Both fail on 48/67 equi-join-detect questions** — join detection across complex schemas remains hard
2. **Neither exceeds 40% on Data-Imputation** — predicting missing values is fundamentally difficult
3. **Both under 40% on List-to-table and Column-type-annotation** — these tasks have very strict exact-match evaluation
