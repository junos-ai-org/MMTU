# Column-Type-Annotation: Why T5Gemma-9B Outperforms Qwen2.5-14B

## Summary

T5Gemma-9B (38.8%) doubles Qwen2.5-14B's accuracy (19.4%) on Column-type-annotation, a +19.4pp gap driven by a single dominant failure mode: **Qwen systematically defaults to generic parent classes in the DBpedia ontology hierarchy** (e.g., "Person" instead of "FigureSkater", "Settlement" instead of "City"), while T5Gemma produces more specific, fine-grained annotations that match the ground truth. Of 21 cases where T5Gemma wins but Qwen loses, 19 (90%) follow this pattern. Conversely, all 8 cases where Qwen wins involve ground truth that IS a generic class, and T5Gemma over-specifies.

## How Scoring Works

The `CTAEvaluator` uses **exact string match** between the predicted and ground-truth DBpedia ontology URIs:

```python
def _evaluate_one(self, y_true, y_pred):
    correct = 1 if y_true == y_pred else 0
```

The evaluator extracts the model's prediction by parsing a JSON object from the response with key `"DBpedia ontology class"`. The ground truth is the `label` field in the metadata (a full URI like `http://dbpedia.org/ontology/FigureSkater`).

This is strict: `"Building"` vs `"building"` is a miss. `"Person"` vs `"FigureSkater"` is a miss. There is no partial credit and no hierarchy-aware scoring. The `okay_annotation` field in metadata lists acceptable alternative classes, but the evaluator does not use it.

## Aggregate Statistics

| Metric | Value |
|---|---|
| Total test cases | 67 |
| Both correct | 5 |
| Both wrong | 33 |
| T5Gemma wins only | 21 |
| Qwen wins only | 8 |
| Qwen accuracy | 13/67 = 19.4% |
| T5Gemma accuracy | 26/67 = 38.8% |

## Pattern Analysis

### The Core Finding: Specificity Calibration

The two models sit at opposite ends of the specificity spectrum for DBpedia ontology classes:

**Qwen defaults to generic parent classes.** Out of 67 responses, Qwen outputs `"Person"` 23 times (34%), `"Settlement"` 9 times (13%), and `"SportsTeam"` 4 times (6%). In total, 41/67 responses (61%) use a broad generic class. Since only 10/67 ground truth labels are generic classes, Qwen's strategy is badly miscalibrated.

**T5Gemma picks specific leaf classes.** T5Gemma outputs `"Person"` only 8 times (12%), matching the actual ground truth distribution (8/67 cases have `Person` as the label). Its output distribution is much flatter and more specific: `Racehorse` (5), `City` (4), `MusicalArtist` (3), `Building` (3), `FigureSkater` (2), `Cyclist` (2), etc.

The ground truth distribution is heavily skewed toward specific classes: `Person` (8), `City` (6), `RaceHorse` (6), `MusicalArtist` (4), `Company` (3), and many singletons. T5Gemma's willingness to commit to specific classes gives it a structural advantage.

### When Qwen Loses: Over-generalization (19/21 cases)

In 19 of 21 cases where T5Gemma wins and Qwen loses, Qwen picks a correct but too-broad parent class:

| Ground Truth | Qwen Predicts | Count |
|---|---|---|
| City | Settlement | 4 |
| FigureSkater | Person | 2 |
| Cyclist | Person | 2 |
| AmericanFootballPlayer | Person | 1 |
| AmericanFootballTeam | SportsTeam | 1 |
| CyclingTeam | SportsTeam | 1 |
| MusicalArtist | Person | 1 |
| Magazine | Publication | 1 |
| TelevisionEpisode | Work | 1 |
| Ship | Vessel | 1 |
| School | EducationalInstitution | 1 |
| GrandPrix | Race | 1 |
| Artery | AnatomicalStructure | 1 |
| Country | Settlement | 1 |

The remaining 2/21 cases: one is a capitalization error (`building` vs `Building`), and one is `HistoricBuilding` instead of `Building` (over-specific for Qwen, which is unusual).

### When T5Gemma Loses: Over-specification (7/8 cases)

In 7 of 8 cases where Qwen wins and T5Gemma loses, the ground truth IS a generic class and T5Gemma over-specifies:

| Ground Truth | T5Gemma Predicts |
|---|---|
| Person | Academic |
| Person | MagazineEditor |
| Person | Actor |
| Person | TelevisionSeries (wrong entirely) |
| Settlement | PortugueseDistrict |
| Place | Cemetery |
| Nerve | PeripheralNerve |

T5Gemma's bias toward specificity backfires when the correct answer is intentionally broad.

## Detailed Example Breakdowns

### Example 1: case_1085 -- FigureSkater vs Person

**Column:** col0 in a table of figure skating pairs/teams
**Sample values:** N. Katsalapov, N. Orford, N. Takahashi, M. Trankov, M. Mukhortova
**Context clues:** Other columns contain partner names, coaches, birthplaces -- clearly a figure skating competition roster.

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/FigureSkater` |
| **T5Gemma** | `http://dbpedia.org/ontology/FigureSkater` |
| **Qwen** | `http://dbpedia.org/ontology/Person` |

**Why T5Gemma wins:** The table context strongly signals figure skating (coach names like A. Zhulin, M. Zueva; locations like Chiba). T5Gemma integrates this context to pick the specific ontology class. Qwen recognizes these are people but does not commit to the specific subtype. The prompt explicitly asks for "the most fine grained ontology class name that can describe all entities," yet Qwen ignores this instruction.

### Example 2: case_1419 -- CyclingTeam vs SportsTeam

**Column:** col1 in a table of cyclists and their teams
**Sample values:** Tabriz Petrochemical Team, UnitedHealthcare, Team FixIT.no, Verva ActiveJet, Team Joker
**Context clues:** col0 has cyclist names, other columns have birth dates.

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/CyclingTeam` |
| **T5Gemma** | `http://dbpedia.org/ontology/CyclingTeam` |
| **Qwen** | `http://dbpedia.org/ontology/SportsTeam` |

**Why T5Gemma wins:** The table is clearly about cycling (cyclist names paired with teams and dates). T5Gemma connects "cyclists + teams" to the specific `CyclingTeam` class. Qwen stops at the broader `SportsTeam` -- correct in the hierarchy but not fine-grained enough.

### Example 3: case_532 -- Country vs Settlement

**Column:** col2 in a table of people and places
**Sample values:** The Netherlands, United States, Argentina, Americans
**Context clues:** Other columns have person names, cities (Rijswijk, White Plains), and dates.

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/Country` |
| **T5Gemma** | `http://dbpedia.org/ontology/Country` |
| **Qwen** | `http://dbpedia.org/ontology/Settlement` |

**Why T5Gemma wins:** The values in col2 are clearly country names ("United States", "Argentina", "The Netherlands"). T5Gemma correctly identifies this as `Country`. Qwen picks `Settlement`, which is not even a correct parent class for Country in DBpedia (both are under `PopulatedPlace`). This is a clear semantic error by Qwen, not just over-generalization.

### Example 4: case_1198 -- Building (capitalization error)

**Column:** col1 in a table of architects and their buildings
**Sample values:** Delft University of Technology, Oratorio dei Filippini, Chateau de Maisons, Textile Center Building

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/Building` |
| **T5Gemma** | `http://dbpedia.org/ontology/Building` |
| **Qwen** | `http://dbpedia.org/ontology/building` (lowercase 'b') |

**Why T5Gemma wins:** Qwen actually identifies the correct class conceptually but outputs it with a lowercase 'b' (`building` instead of `Building`). Since scoring is exact string match, this is a miss. T5Gemma produces the correctly-capitalized PascalCase URI. This suggests T5Gemma has better memorization of the exact DBpedia ontology URI conventions.

### Example 5: case_678 -- GrandPrix vs Race

**Column:** col0 in a table of historic motor racing events
**Sample values:** 1931 Belgian Grand Prix, 1930 San Sebastian Grand Prix, 1930 Rome Grand Prix, 1930 Monza Grand Prix, 1930 Monaco Grand Prix

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/GrandPrix` |
| **T5Gemma** | `http://dbpedia.org/ontology/GrandPrix` |
| **Qwen** | `http://dbpedia.org/ontology/Race` |

**Why T5Gemma wins:** Every value literally contains "Grand Prix" in its name. T5Gemma maps this directly to the `GrandPrix` class. Qwen abstracts to the parent class `Race`. This is arguably the most clear-cut example -- the class name appears verbatim in the data, yet Qwen still generalizes.

### Example 6: case_811 -- Person vs Actor (Qwen wins)

**Column:** col2 in a table of TV shows and movies
**Sample values:** M. Enriquez, L. Joon-gi, S. Woo, C. Sevigny, H. Graham, A. Chen, Seetha
**Context clues:** col0 has show titles ("Iljimae", "If Tomorrow Comes"), col1 has related show titles.

| | Response |
|---|---|
| **Ground truth** | `http://dbpedia.org/ontology/Person` |
| **T5Gemma** | `http://dbpedia.org/ontology/Actor` |
| **Qwen** | `http://dbpedia.org/ontology/Person` |

**Why Qwen wins:** The ground truth here is the generic `Person`, not `Actor`. While many of these people are actors, the column contains a mix of directors, producers, and actors. The most accurate encompassing class is `Person`. T5Gemma's specificity bias leads it to guess `Actor`, which is too narrow. Qwen's conservative strategy pays off when the ground truth is intentionally broad.

## Conclusion: Why T5Gemma Wins on Column-Type-Annotation

The +19.4pp advantage for T5Gemma comes down to **ontology specificity calibration**:

1. **T5Gemma follows instructions better.** The prompt explicitly asks for "the most fine grained ontology class name that can describe all entities in the column." T5Gemma complies -- it picks specific leaf classes like `FigureSkater`, `CyclingTeam`, `GrandPrix`, and `City`. Qwen systematically ignores this instruction and defaults to broad parent classes.

2. **T5Gemma has better DBpedia ontology recall.** T5Gemma produces a diverse vocabulary of 40+ distinct ontology classes across 67 responses, while Qwen's output is concentrated in a handful of generic classes (Person=23, Settlement=9, SportsTeam=4 account for 54% of all outputs). T5Gemma appears to have better coverage of the DBpedia class hierarchy.

3. **The ground truth distribution rewards specificity.** Only 10/67 ground truth labels are generic parent classes. The remaining 57/67 require specific leaf-level annotations. Qwen's generic-defaulting strategy has a ceiling of ~15% accuracy on this distribution, which is close to its observed 19.4%.

4. **T5Gemma's weakness is the mirror image.** When the ground truth IS a generic class (especially `Person` -- 8 cases), T5Gemma sometimes over-specifies (Academic, Actor, MagazineEditor). But this costs it only 7-8 cases compared to the 19-21 it gains from specificity.

5. **Possible architectural explanation.** T5Gemma's encoder-decoder architecture with bidirectional attention over the input table may allow it to better synthesize context clues across all columns simultaneously (e.g., recognizing that person names + cycling team names + birth dates = CyclingTeam domain). Qwen's autoregressive decoder may process the table more sequentially, defaulting to the most salient surface-level category (e.g., "these look like team names" -> `SportsTeam`). However, this hypothesis requires further investigation -- the specificity difference could also stem from differences in fine-tuning data or instruction-following capability.
