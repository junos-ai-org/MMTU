# Experiments Log

## Research Question
Does encoder-decoder architecture (T5Gemma 9B-9B UL2 IT, ~18B total / ~9B active)
provide advantages over decoder-only (Qwen2.5-14B-Instruct, 14B active) for table
understanding? Is it more robust to structural permutations?

Note: Different parameter profiles — T5Gemma has ~18B total / ~9B active during
decoding, Qwen2.5 has 14B always active.

---

## encoder_vs_decoder_baseline_large (2025-03-25)

**Dataset:** `dataset_large.yaml` — 67 samples/task × 15 tasks = 1005 built, 861 evaluated (some tasks had fewer available samples)
**Tokenizer:** `google/t5gemma-9b-9b-ul2-it` (SentencePiece), max_input_tokens=4000
**Models:** Qwen2.5-14B-Instruct (decoder-only, vLLM) vs T5Gemma-9B-9B-UL2-IT (encoder-decoder, HF transformers)
**Run dirs:** `output/Qwen2.5-14B-Instruct/20260325-130423` | `output/t5gemma-9b-9b-ul2-it/20260325-120505`

### Results

| Task | Qwen2.5-14B | T5Gemma-9B | Delta (Qwen − T5) |
|---|---|---|---|
| Entity-Matching | 0.959 | 0.959 | 0.000 |
| Table-Fact-Verification | 0.866 | 0.851 | +0.015 |
| Schema-Matching | 0.792 | 0.799 | −0.007 |
| Functional-Dependency | 0.700 | 0.430 | **+0.271** |
| equi-join-detect | 0.656 | 0.478 | **+0.177** |
| semantic-transform | 0.560 | 0.532 | +0.029 |
| Table-QA | 0.537 | 0.328 | **+0.209** |
| Cell-entity-annotation | 0.522 | 0.567 | −0.045 |
| header-value-matching | 0.520 | 0.533 | −0.013 |
| semantic-join | 0.326 | 0.331 | −0.005 |
| List-to-table | 0.313 | 0.388 | −0.075 |
| Data-Imputation | 0.290 | 0.406 | **−0.116** |
| Column-type-annotation | 0.194 | 0.388 | **−0.194** |
| **OVERALL** | **0.556** | **0.543** | **+0.013** |

### Observations

1. **Overall scores are close** (55.6% vs 54.3%), but the per-task profile differs substantially.
2. **Qwen dominates reasoning-heavy tasks** requiring multi-step inference:
   - Functional-Dependency (+27.1pp): detecting column dependencies requires scanning column pairs and testing functional relationships
   - Table-QA (+20.9pp): answering natural language questions over tables
   - equi-join-detect (+17.7pp): identifying joinable column pairs across tables
3. **T5Gemma wins on annotation/classification tasks** that benefit from bidirectional encoding:
   - Column-type-annotation (+19.4pp): classifying column semantic types
   - Data-Imputation (+11.6pp): predicting missing cell values
   - List-to-table (+7.5pp), Cell-entity-annotation (+4.5pp)
4. **Both models are equally strong** on Entity-Matching (95.9%) and Schema-Matching (~79%).
5. **Parameter gap matters**: Qwen has 14B always-active params vs T5Gemma's ~9B active during decoding. The fact that T5Gemma is competitive (and wins on some tasks) despite fewer active parameters suggests encoder-decoder architecture provides genuine advantages for table structure comprehension.

---
