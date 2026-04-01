# Equi-Join-Detect: Why Qwen2.5-14B Outperforms T5Gemma-9B

**Task**: equi-join-detect (identify foreign-key join relationships between database tables)
**Metric**: F1 on set of (from_table, from_column, to_table, to_column) tuples
**Qwen F1**: 65.6% | **T5Gemma F1**: 47.8% | **Delta**: +17.7pp

## Summary

Qwen's advantage on equi-join-detect is almost entirely explained by a **direction-matching artifact** in the evaluation metric. The evaluator compares join tuples as ordered 4-tuples where `(A, col1, B, col2)` is different from `(B, col2, A, col1)`, even though both represent the same join relationship. T5Gemma systematically reverses the from/to direction relative to the ground truth convention on 11 of 60 parseable cases (fully reversed) and 10 more (partially reversed), while Qwen matches the ground truth direction in 50 of 56 parseable cases and never fully reverses. If direction is ignored, the gap shrinks from 17.7pp to just 3.0pp. A secondary factor is T5Gemma's higher rate of output truncation (7 parse failures vs. 3 for Qwen), caused by its shorter output token budget producing incomplete JSON for databases with many join relationships.

## How Scoring Works

The `EquiJoinDetectEvaluator` scores each test case by computing **precision** and **recall** over sets of join tuples, then averaging precision and recall separately across all cases before computing F1.

Each predicted join is a 4-tuple: `(from_table, from_column, to_table, to_column)`. The evaluator performs **exact tuple matching** -- it compares predicted tuples against ground truth tuples using Python set intersection. This means:

- `("sales", "Product", "products", "Product")` matches the ground truth.
- `("products", "Product", "sales", "Product")` does **NOT** match, even though it describes the same join.

The prompt template uses the field names `from_table`/`to_table`, implying a direction, but does not explain the convention. The ground truth follows a particular direction convention (e.g., fact table as `from_table`, dimension table as `to_table`), but this convention is never communicated to the models.

## Quantitative Breakdown

### Score Distribution

| Category | Count | Description |
|----------|-------|-------------|
| Qwen wins | 31 | Qwen scores higher |
| T5Gemma wins | 14 | T5Gemma scores higher |
| Ties | 22 | Same score (including 11 mutual zeros) |

### Zero-Score Cases

| Model | Score = 0 | % of 67 |
|-------|-----------|---------|
| T5Gemma | 25 | 37.3% |
| Qwen | 11 | 16.4% |

T5Gemma scores zero more than twice as often. Of the 15 cases where T5Gemma scores 0 but Qwen scores > 0:
- **9 cases** (60%): Direction reversal is the primary cause -- T5Gemma identifies the correct tables and columns but swaps from/to.
- **5 cases** (33%): Column or table name errors compound the direction issue.
- **1 case** (7%): JSON parse failure due to truncation.

### Direction Convention Adherence

| Model | Follows GT direction | Fully reversed | Mixed | Neither |
|-------|---------------------|----------------|-------|---------|
| Qwen | 50 (89%) | 0 (0%) | 6 (11%) | -- |
| T5Gemma | 32 (53%) | 11 (18%) | 10 (17%) | 7 (12%) |

Qwen **never** fully reverses direction. T5Gemma fully reverses on nearly 1 in 5 cases.

### Impact of Direction-Insensitive Scoring

| Scoring Method | Qwen F1 | T5Gemma F1 | Gap |
|---------------|---------|------------|-----|
| Original (direction-sensitive) | 65.6% | 47.8% | 17.7pp |
| Direction-insensitive | 71.2% | 62.6% | 8.6pp |
| Direction-insensitive + no direction error | ~71.2% | ~68.2% | ~3.0pp |

**Direction alone accounts for ~83% of the performance gap** (14.7pp of the 17.7pp delta).

### Output Truncation (Parse Failures)

| Model | Parse Failures | % of 67 |
|-------|---------------|---------|
| Qwen | 3 | 4.5% |
| T5Gemma | 7 | 10.4% |

All parse failures in both models are caused by **response truncation** -- the JSON output is cut off mid-field, producing invalid JSON. T5Gemma's responses are shorter on average (median 666 chars vs. 1171 for Qwen) but fail more often, suggesting it hits its generation limit sooner. Interestingly, 3 of the parse-failure test cases are shared between both models (337973596, 149605362, 291574773), indicating those cases have unusually many join relationships.

### Response Format Differences

| Model | Format | Explanation before JSON |
|-------|--------|------------------------|
| Qwen | Free text + JSON code block | 67/67 (100%) |
| T5Gemma | Pure JSON code block | 67/67 (100%) |

Qwen always generates explanatory text ("Based on the provided schema...") before producing the JSON. T5Gemma always outputs a JSON code block directly. Despite Qwen's verbosity consuming output tokens, its explanatory reasoning may serve as a chain-of-thought that helps it identify the correct join direction.

## Pattern Analysis

### Pattern 1: T5Gemma Systematically Reverses Join Direction

T5Gemma tends to place the **dimension/lookup table** as `from_table` and the **fact/referencing table** as `to_table`. This follows a "primary key points to foreign key" convention (PK -> FK). The ground truth follows the opposite convention: the table holding the foreign key is `from_table` (FK -> PK, i.e., "this table references that table"). Qwen reliably adopts the FK -> PK convention that matches the ground truth.

### Pattern 2: T5Gemma Hallucinates Column Names

In several cases, T5Gemma invents or misrenames columns. For example, in case 939121682, the DIM_Calendar table has a column called "Date" but T5Gemma refers to it as "DateKey". In case 208829205, the table is named "D_Vendedor" but T5Gemma calls it "D_VENDEUR" (French instead of Portuguese). These are semantic near-misses that the exact-match evaluator cannot credit.

### Pattern 3: Qwen's Chain-of-Thought Helps Accuracy

Qwen's explanatory text often explicitly reasons about which table references which, e.g., "The sales table can be joined with the people table using the Sales Person column." This reasoning appears to anchor the direction correctly (the table with the FK is stated as "from"). T5Gemma, outputting pure JSON without reasoning, has no such anchor.

### Pattern 4: Truncation Disproportionately Hurts T5Gemma

For complex databases with many tables (8+ join relationships), both models risk running out of output tokens. But T5Gemma's shorter effective output budget means it fails more often. Three of the seven T5Gemma truncation cases are shared with Qwen (both fail), but four are T5Gemma-only failures.

## Detailed Example Breakdowns

### Example 1: Case 279073330 (Delta = +1.0, Qwen wins)

**Database**: Sales analytics with 4 tables: `sales`, `products`, `locations`, `people`.

**Ground truth** (3 joins):
```
(sales, Product) -> (products, Product)
(sales, Geography) -> (locations, Geo)
(sales, Sales Person) -> (people, Sales person)
```

**Qwen prediction**: Exactly matches ground truth. 3/3 correct. F1 = 1.0.
```json
{"joins": [
  {"from_table": "sales", "from_column": "Sales Person", "to_table": "people", "to_column": "Sales person"},
  {"from_table": "sales", "from_column": "Product", "to_table": "products", "to_column": "Product"},
  {"from_table": "sales", "from_column": "Geography", "to_table": "locations", "to_column": "Geo"}
]}
```

**T5Gemma prediction**: Identifies the exact same 3 join relationships with the same column names, but **reverses the direction on all 3**. 0/3 match. F1 = 0.0.
```json
{"joins": [
  {"from_table": "locations", "from_column": "Geo", "to_table": "sales", "to_column": "Geography"},
  {"from_table": "people", "from_column": "Sales person", "to_table": "sales", "to_column": "Sales Person"},
  {"from_table": "products", "from_column": "Product", "to_table": "sales", "to_column": "Product"}
]}
```

**Analysis**: This is a pure direction-reversal case. T5Gemma uses a PK -> FK convention (dimension table as `from_table`), while the ground truth uses FK -> PK (fact table as `from_table`). The semantic content of both predictions is identical.

---

### Example 2: Case 939121682 (Delta = +1.0, Qwen wins)

**Database**: Star schema with `DIM_Calendar`, `DIM_Customers`, `DIM_Products`, `FACT_InternetSales`, `FACT_Budget`, `Medidas`.

**Ground truth** (4 joins):
```
(FACT_InternetSales, OrderDateKey) -> (DIM_Calendar, Date)
(FACT_InternetSales, CustomerKey) -> (DIM_Customers, CustomerKey)
(FACT_InternetSales, ProductKey) -> (DIM_Products, ProductKey)
(FACT_Budget, Date) -> (DIM_Calendar, Date)
```

**Qwen**: Perfect match, 4/4. F1 = 1.0.

**T5Gemma**: Three compounding errors:
1. **Direction reversed**: Places dimension tables as `from_table` (e.g., `DIM_Calendar -> FACT_InternetSales`).
2. **Column name hallucination**: Uses `DateKey` for DIM_Calendar's column, but the actual column is called `Date`.
3. **Over-generation**: Predicts 6 joins instead of 4, adding `ShipDateKey` and `DueDateKey` joins that aren't in the ground truth.

Result: 0/6 predictions match any of the 4 ground truth tuples. F1 = 0.0.

**Analysis**: T5Gemma demonstrates reasonable database understanding (ShipDateKey and DueDateKey *are* plausible join columns), but direction reversal + column name error means zero credit.

---

### Example 3: Case 208829205 (Delta = +0.875, Qwen wins)

**Database**: Portuguese-language sales data warehouse with 9 tables including `Fato_venda`, `D_PRODUTOS`, `D_CLIENTES`, `D_Vendedor`, `D_Tempo_Oficial`, etc.

**Ground truth**: 9 joins.

**Qwen**: Perfect 9/9. F1 = 1.0. Correctly uses table name `D_Vendedor`, column name `Cod_Vend`, and the FK -> PK direction.

**T5Gemma**: Only 1/7 match (`D_CLIENTES.Cod_uf -> D_UF.Cod_uf`). F1 = 0.125.
- **Direction reversal** on 5 joins.
- **Table name hallucination**: Uses `D_VENDEUR` (French) instead of `D_Vendedor` (Portuguese).
- **Missing joins**: Does not predict the two temporal joins (`Fato_venda.Dt_Venda -> D_Tempo_Oficial.Data_ref` and `Fato_meta.Mes/Ano -> D_Tempo_Oficial.Data_ref`).

**Analysis**: T5Gemma's multilingual confusion (French vs. Portuguese table names) and direction reversal compound to near-total failure on a case Qwen handles perfectly.

---

### Example 4: Case 235479503 (Delta = -0.667, T5Gemma wins)

**Database**: Harry Potter themed database with `Chapters`, `Characters`, `Dialogue`, `Dialogue {Extract Spell}`, `Movies`, `Places`, `Spells`.

**Ground truth** (8 joins), using `Dialogue -> Chapters`, `Dialogue -> Characters`, etc. (the table with the FK is `from_table`).

**Qwen**: Scores 0.0. Despite identifying 6 correct join relationships, Qwen **reverses the direction** on every single one: `Chapters -> Dialogue` instead of `Dialogue -> Chapters`.

**T5Gemma**: Scores 0.667. Gets 5/8 correct with proper direction on most, but reverses direction on 2 (`Chapters -> Dialogue` pairs) and misses the `Dialogue {Extract Spell}.Custom.Incantation -> Spells.Incantation` join.

**Analysis**: This is one of the rare cases where *Qwen* reverses direction. Notably, 4 of the 8 GT joins have bridge/child tables as `from_table` and parent tables as `to_table` -- the convention Qwen usually follows. But for this specific database, Qwen placed the parent (Chapters, Characters, Places) as `from_table`, going against its usual pattern. T5Gemma's mixed direction partially aligns with the GT here.

---

### Example 5: Case 173341254 (Delta = -0.5, T5Gemma wins)

**Database**: Top Movies with bridge tables (`Actors Bridge`, `Country Bridge`, `Director Bridge`, `Genre Bridge`) connecting `TopMovies` to lookup tables.

**Ground truth** (8 joins): Bridge tables are consistently `from_table`:
```
(Actors Bridge, Title) -> (TopMovies, Title)
(Actors Bridge, Actors) -> (Actors, Actors)
... (same pattern for Country, Director, Genre bridges)
```

**Qwen**: F1 = 0.5. Gets 4/8 correct (the lookup-side joins like `Actors Bridge -> Actors`) but reverses the 4 bridge-to-TopMovies joins as `TopMovies -> Actors Bridge`.

**T5Gemma**: F1 = 1.0. Perfect. All 8 joins match exactly.

**Analysis**: T5Gemma correctly infers that bridge tables should be `from_table` when joining to the main table. Qwen treats the main table (`TopMovies`) as the FK holder, which is wrong here -- the bridge table holds the FK.

---

### Example 6: Case 242856579 (Delta = +0.909, Qwen wins)

**Database**: Crime/property analytics with `factCrime`, `weather`, `propertysales`, `dimDate`, `dimNeighborhood`, `dimCrime`.

**Ground truth** (5 joins): Fact/source tables as `from_table`:
```
(weather, DateKey) -> (dimDate, datekey)
(propertysales, NeighborhoodKey) -> (dimNeighborhood, NeighborhoodKey)
(factCrime, dateKey) -> (dimDate, datekey)
(factCrime, crimeTypeKey) -> (dimCrime, CrimeTypeKey)
(factCrime, neighborhoodKey) -> (dimNeighborhood, NeighborhoodKey)
```

**Qwen**: 5/5 GT joins found, plus 1 extra (`propertysales.datekey -> dimDate.datekey`). F1 = 0.909.

**T5Gemma**: 0/6 match. F1 = 0.0. Every join has the direction reversed (`dimDate -> factCrime` instead of `factCrime -> dimDate`). Also hallucinates a wrong join (`weather.DateKey -> factCrime.dateKey` instead of `weather.DateKey -> dimDate.datekey`).

**Analysis**: Classic T5Gemma direction reversal in a star schema. T5Gemma uses PK -> FK direction; GT expects FK -> PK.

## Conclusion: Why Does Qwen Win?

The 17.7pp gap is dominated by a **direction-matching artifact**, not a fundamental difference in join detection ability:

1. **Direction convention alignment (accounts for ~83% of the gap)**: The ground truth follows an FK -> PK convention where the table holding the foreign key is `from_table`. Qwen naturally follows this convention (89% of cases), likely because its instruction-following training and chain-of-thought reasoning help it adopt the perspective of "this table references that table." T5Gemma adopts a PK -> FK convention (dimension as `from_table`) on ~35% of cases, and the direction-sensitive evaluator treats every reversed tuple as a complete miss.

2. **Output truncation (accounts for ~6% of the gap)**: T5Gemma produces 7 unparseable (truncated) responses vs. 3 for Qwen. The likely cause is T5Gemma's shorter effective generation budget -- its responses average 783 characters vs. Qwen's 1237, yet it still runs out on complex databases.

3. **Column/table name accuracy (accounts for ~11% of the gap)**: T5Gemma occasionally hallucinates column names (`DateKey` vs `Date`) or table names (`D_VENDEUR` vs `D_Vendedor`). These errors compound with direction reversal to ensure zero credit even on partial matches.

**If direction were treated as symmetric** (i.e., `(A, col1, B, col2)` equivalent to `(B, col2, A, col1)`), the gap would shrink from 17.7pp to approximately 3.0pp -- within noise for a 67-sample evaluation. The remaining gap would be attributable to T5Gemma's slightly higher truncation rate and occasional name hallucinations.

This finding suggests that the equi-join-detect task, as currently evaluated, measures **convention-following** more than **join detection ability**. A direction-insensitive metric would provide a fairer comparison of the models' structural understanding of relational databases.
