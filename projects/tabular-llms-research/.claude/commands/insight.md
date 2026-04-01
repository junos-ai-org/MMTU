# Per-question cross-model insight analysis

Run a deep per-question analysis comparing two model runs on specified MMTU tasks.

## Arguments
- `$ARGUMENTS` — comma-separated task names (e.g., "Data-Imputation, List-to-table")

## Workflow

1. **Setup**: Create the insight directory at `experiments/<experiment>/insights/<YYYYMMDD-description>/` with:
   - `insight_config.json` — objective, run references, task filters, tags
   - Per-task subdirectories

2. **Build paired data**: Load both model result JSONLs, filter to the specified tasks, join on `test_case`, and run MMTU evaluators (pure Python, no model calls) for per-row scores. Write `paired_data.jsonl`.

3. **Analyze with Sonnet subagents**: Split the paired data into ~27-row chunks. Launch one Sonnet subagent per chunk (use `model: sonnet`). Each agent reads the paired data and for each row:
   - Understands the question from the prompt
   - Compares both model responses against the ground truth label
   - Categorizes: `both_correct`, `both_wrong`, `qwen_only_correct`, `t5gemma_only_correct`
   - Writes 2-4 sentence explanation of why models succeeded or failed
   - Adds comma-separated tags from: `format_error`, `value_extraction`, `reasoning_error`, `partial_match`, `hallucination`, `truncation`, `wrong_column`, `wrong_row`, `case_mismatch`, `delimiter_error`, `structural_error`, `both_trivial`, `close_but_wrong`, `completely_off`, `correct_approach`
   - Writes a chunk JSONL file (`analysis_chunk_N.jsonl`)

4. **Merge**: Combine all chunk files into `analyzed_pairs.jsonl`, sorted by idx.

5. **Synthesize report**: Write `synthesis.md` containing:
   - Score summary table (both_correct / both_wrong / model_only counts per task)
   - **Per-task failure mode count tables** broken down by category (both_wrong, t5gemma_only, qwen_only, both_correct)
   - **Concrete examples for each failure mode** showing the question, label, both model responses, and explanation of why each model succeeded or failed
   - Cross-task themes and architectural insights
   - Evaluator artifact analysis (formatting issues that inflate/deflate gaps)

6. **Export to docx**: Convert synthesis.md to synthesis.docx using `md_to_docx.py`.

7. **Commit and push**.

## Key details
- The MMTU evaluators do NOT run the model — they just compare output vs label (string matching, F1, etc.)
- Join on `test_case` field in metadata, not on row order
- One Sonnet call per question with the shared prompt (table + question) + both responses
- Use ~5 parallel subagents to process all pairs concurrently
- The report must include failure mode count tables AND concrete examples — a report with only summary stats is too shallow
