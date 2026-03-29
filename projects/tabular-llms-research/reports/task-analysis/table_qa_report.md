# Table-QA Task Analysis: Qwen2.5-14B vs T5Gemma-9B

**Qwen accuracy: 53.7% (36/67) | T5Gemma accuracy: 32.8% (22/67) | Delta: +20.9pp**

## Summary

Qwen2.5-14B's advantage on Table-QA stems from two reinforcing factors: (1) chain-of-thought reasoning that enables correct multi-step computation and comparison, and (2) better format compliance with the evaluator's strict matching rules, particularly around percentage notation. Of the 22 cases where Qwen wins, approximately 17 are genuine reasoning improvements (T5Gemma produces a wrong answer) and 5 are format-related (T5Gemma computes approximately the right value but outputs it in a format the evaluator cannot match). T5Gemma wins only 8 cases, almost all due to Qwen reasoning errors rather than any systematic T5Gemma advantage.

## How Scoring Works

The evaluator (`evaluators/tableqa_evaluator.py`) extracts a JSON `{"answer": <value>}` from the model response and compares it to a ground-truth label using these rules:

1. **Exact string match** (case-insensitive): `str(y_true).lower() == str(y_pred).lower()`
2. **Percentage handling**: If `y_pred` is a string containing `%`, the evaluator strips the `%` sign and tries both `round(y_true, 2) == round(y_pred, 2)` and `round(y_true, 2) == round(y_pred/100, 2)`. This is critical because many ground-truth labels are stored as decimals (e.g., `0.05543`) while answers are naturally expressed as percentages (e.g., `5.54%`).
3. **Dollar handling**: If `y_pred` is a string containing `$`, strips `$` and commas, then compares rounded floats.
4. **Numeric fallback**: If `y_pred` is not a string (i.e., a JSON number), compares `round(float(y_true), 2) == round(float(y_pred), 2)`.

**Key quirk**: If `y_pred` is a string that does NOT contain `%` or `$`, there is no numeric fallback. The evaluator returns `correct=0` even if the string represents the right number. This penalizes T5Gemma, which sometimes returns `"0.132"` (string) instead of `0.132` (number).

## Response Format Differences

| Property | Qwen2.5-14B | T5Gemma-9B |
|---|---|---|
| Chain-of-thought reasoning | 56/67 (84%) | 0/67 (0%) |
| Response format | Reasoning text followed by `{"answer": ...}` | Always `` ```json\n{"answer": ...}\n``` `` |
| Numeric answers | Returns JSON numbers (float/int) | Returns JSON numbers or strings |
| Percentage answers | Often includes `%` sign in string | Returns raw number without `%` |

T5Gemma always produces a terse code-block JSON response with no reasoning. Qwen almost always shows its work before producing the answer.

## Pattern Analysis

### Breakdown of Qwen's 22 Wins

| Category | Count | Description |
|---|---|---|
| T5Gemma reasoning error | 17 | T5Gemma produces a factually wrong answer |
| T5Gemma format mismatch | 5 | T5Gemma's answer is approximately correct but formatted so the evaluator rejects it |

### Breakdown of T5Gemma's 8 Wins

| Category | Count | Description |
|---|---|---|
| Qwen reasoning error | 7 | Qwen produces a factually wrong answer |
| Qwen borderline numeric error | 1 | Qwen off by 1 (26 vs 27) |

### Question Type Analysis

Among Qwen's 22 wins:
- **Multi-step computation** (11 cases): Percentage calculations, averages, differences, ratios. Qwen's chain-of-thought lets it correctly identify operands and perform arithmetic.
- **Lookup/comparison** (7 cases): Finding which row has the max/min, ordering by date, comparing values. Qwen's reasoning traces through the table systematically.
- **Counting** (4 cases): "How many X?" questions requiring filtering and counting rows.

Among T5Gemma's 8 wins, the distribution is more even (3 computation, 2 lookup, 2 counting, 1 calculation), suggesting T5Gemma has no systematic advantage on any question type.

## Detailed Example Breakdowns

### Example 1: Percentage Format Advantage (test_33) -- Qwen wins

**Question**: "In 2018 what was the percent of the CIB Markets net interest income as part of the managed interest income?"

**Table**: Financial data with rows including "net interest income - managed basis" ($55,687M in 2018) and "CIB Markets net interest income" ($3,087M in 2018).

**Ground truth**: `0.05543000000000001` (stored as decimal fraction)

**Qwen response** (with chain-of-thought):
> 1. CIB Markets Net Interest Income (2018): $3087 million
> 2. Net Interest Income Managed Basis (2018): $55687 million
> 3. Percentage = (3087 / 55687) x 100 = 5.54%
>
> `{"answer": "5.54%"}`

**T5Gemma response**: `{"answer": 5.54}`

**Why Qwen wins**: Both models compute the correct value (5.54%). But Qwen includes the `%` sign, which triggers the evaluator's percentage-handling logic: it strips the `%`, divides 5.54 by 100 to get 0.0554, and `round(0.05543, 2) == round(0.0554, 2)` yields `0.06 == 0.06` -- match. T5Gemma returns the bare number `5.54` as a float, so the evaluator compares `round(0.05543, 2) == round(5.54, 2)` which is `0.06 == 5.54` -- no match. **T5Gemma gets the math right but loses on format.**

### Example 2: Chain-of-Thought Enables Complex Reasoning (e5309753c35c2748bbaaab0ab2655983) -- Qwen wins

**Question**: "Population Density is calculated as the total population of a place divided by its area in square kilometers. Based on this definition, which place in the table has the lowest population density?"

**Table**: 10 South African places with population and area columns (e.g., Addo Elephant National Park: pop 90, area 1.08 km2; Remainder of the Municipality: pop 15,218, area 3,491.83 km2).

**Ground truth**: `remainder of the municipality`

**Qwen response** (chain-of-thought):
> Calculates density for all 10 places:
> - Addo Elephant National Park: 90/1.08 = 83.33
> - Paterson: 671/0.22 = 3050
> - Remainder of Municipality: 15218/3491.83 = 4.36
>
> The place with the lowest population density is "Remainder of the Municipality" with approximately 4.36 people per km2.
>
> `{"answer": "remainder of the municipality"}`

**T5Gemma response**: `{"answer": "paterson"}`

**Why Qwen wins**: This question requires computing population/area for every row and finding the minimum. Qwen explicitly calculates all 10 densities in its chain-of-thought and correctly identifies the outlier (the large rural area with 4.36 people/km2). T5Gemma appears to confuse the smallest area with the lowest density -- Paterson has the smallest area (0.22 km2) but actually has one of the higher densities (3050/km2). Without chain-of-thought, T5Gemma cannot track 10 parallel calculations.

### Example 3: Temporal Comparison Requires Reasoning (table-nu-3762) -- Qwen wins

**Question**: "Which episode came first, Young Forever or You Oughta Know?"

**Table**: TV episode listing with columns for season number, series number, title, Canadian airdate, US airdate, and production code.

**Ground truth**: `Young Forever`

**Qwen response**:
> "Young Forever" aired on August 22, 2013. "You Oughta Know" aired on October 17, 2013. Based on these dates, "Young Forever" came before "You Oughta Know."
>
> `{"answer": "Young Forever"}`

**T5Gemma response**: `{"answer": "You Oughta Know"}`

**Why Qwen wins**: Qwen extracts the specific dates from the table and compares them explicitly. T5Gemma gives the wrong answer (possibly confusing the ordering or misreading the dates). This is a simple two-row comparison, but T5Gemma fails without visible reasoning.

### Example 4: Sorting and Second-Highest (table-nu-2632) -- Qwen wins

**Question**: "Who is the second with the most pennant?"

**Table**: 8 ships with pennant numbers (Leander: 75, Orion: 85, Neptune: 20, Ajax: 22, Achilles: 70, Amphion: 29, Apollo: 63, Sydney: 48).

**Ground truth**: `Leander`

**Qwen response**:
> Lists all pennant numbers, sorts in descending order: 85, 75, 70, 63, 48, 29, 22, 20. The second highest is 75, which is "Leander".
>
> `{"answer": "Leander"}`

**T5Gemma response**: `{"answer": "Orion"}`

**Why Qwen wins**: The question asks for the second-highest, but T5Gemma returns the highest (Orion, pennant 85). T5Gemma may have misinterpreted "second with the most" or failed to properly sort the values without explicit reasoning steps.

### Example 5: Qwen Overthinks, T5Gemma Gets It Right (test_1107) -- T5Gemma wins

**Question**: "Had all four quarters of 2010 had the same number of total shares purchased, how many total shares were purchased in 2010?"

**Table**: Share repurchase data showing Q4 2010 monthly data and a total row (3,063,816 total shares for Q4).

**Ground truth**: `12255264` (= 3,063,816 x 4)

**Qwen response**:
> "The total number of shares purchased across all periods listed is 3063816. Since the question asks us to assume each quarter had the same number of shares purchased, we can use this total directly as the answer."
>
> `{"answer": 3063816}`

**T5Gemma response**: `{"answer": 12255264}`

**Why T5Gemma wins**: The question is a hypothetical: "if all four quarters had the same number as this quarter, what would the annual total be?" The answer is simply 3,063,816 x 4 = 12,255,264. T5Gemma correctly interprets this and multiplies by 4. Qwen's chain-of-thought actually leads it astray -- it overthinks the problem and concludes the total "already represents an equal distribution," returning the un-multiplied Q4 figure. This is a case where Qwen's verbose reasoning introduces a logical error.

### Example 6: T5Gemma Computes Correctly but Loses on Format (test_457) -- Qwen wins

**Question**: "What portion of the total full-time employees of mainline operations are pilots and flight crew instructors?"

**Table**: Employee breakdown showing pilots/flight crew instructors (13,400) and total mainline operations (101,500).

**Ground truth**: `0.13202`

**Qwen response**:
> Portion = 13400 / 101500 = 0.132
>
> `{"answer": 0.132}`

**T5Gemma response**: `{"answer": "0.132"}`

**Why Qwen wins**: Both models compute the same value (0.132). But Qwen returns it as a JSON number (`0.132`), which enters the evaluator's numeric branch: `round(0.13202, 2) == round(0.132, 2)` gives `0.13 == 0.13` -- match. T5Gemma returns it as a JSON string (`"0.132"`), which enters the string comparison branch: `"0.13202" == "0.132"` -- no match. Since the string contains neither `%` nor `$`, there is no fallback. **T5Gemma has the right answer but the wrong JSON type.**

## Conclusions: Why Qwen Wins on Table-QA

### 1. Chain-of-thought reasoning is the primary driver (~17 of 22 wins)

Qwen produces explicit step-by-step reasoning in 84% of its responses, while T5Gemma never does. This manifests as:

- **Multi-step computation**: Qwen can identify the right operands from the table, perform division/multiplication/averaging, and arrive at the correct number. T5Gemma frequently selects wrong operands (e.g., returning a single value instead of an average) or makes arithmetic errors (e.g., 86929 instead of 85929).
- **Sorting and ranking**: Questions asking for "second highest" or "which came first" require comparing multiple values. Qwen explicitly lists and sorts; T5Gemma often returns the wrong rank.
- **Counting with conditions**: "How many X?" questions require filtering rows by a criterion. Qwen's reasoning traces through rows one by one; T5Gemma miscounts (e.g., 6 instead of 8 Disney movies).

### 2. Format compliance gives Qwen an evaluator-specific edge (~5 of 22 wins)

The evaluator's strict matching rules interact differently with each model's output style:

- **Percentage notation**: When ground truth is a decimal (e.g., 0.05543), Qwen's habit of including `%` in percentage answers (e.g., `"5.54%"`) triggers the evaluator's divide-by-100 logic, producing a match. T5Gemma returns bare numbers (e.g., `5.54`) that fail the comparison against the decimal ground truth. This accounts for at least 3 wins (test_33, test_153, test_603).
- **JSON type (string vs number)**: T5Gemma sometimes returns numeric values as JSON strings (e.g., `"0.132"` instead of `0.132`), which bypass the numeric rounding comparison and fail on exact string match. This accounts for at least 1 win (test_457).
- **Determiners**: T5Gemma occasionally adds articles like "the" to answers (e.g., "the UNESCO World Heritage List" vs "UNESCO World Heritage List"), breaking exact match.

### 3. T5Gemma's lack of reasoning is its fundamental weakness

T5Gemma's 0% chain-of-thought rate is not simply a stylistic choice -- it reflects the model's inability to perform intermediate computation in text. When a question requires looking up two values and dividing them, T5Gemma must do this implicitly in its hidden states, which is unreliable for precise arithmetic. Qwen can "show its work" and self-correct.

### 4. Qwen's reasoning occasionally backfires

In 7 of T5Gemma's 8 wins, Qwen makes a reasoning error despite its chain-of-thought. In test_1107, Qwen's elaborate reasoning leads it to the wrong interpretation ("the total already represents equal distribution"). In table-nu-1447, Qwen overcounts (4 instead of 3 episodes featuring Mario Batali). Chain-of-thought is powerful but not infallible -- verbose reasoning can introduce logical errors that a more direct model avoids.

### Net Assessment

Removing the ~5 format-related wins, Qwen still holds a substantial **~12pp advantage** from pure reasoning quality (17 reasoning wins - 7 reasoning losses = +10 net cases out of 67 = +14.9pp). The format advantage adds another ~7.5pp. Both factors are real, but reasoning quality is the larger contributor to Qwen's Table-QA dominance.
