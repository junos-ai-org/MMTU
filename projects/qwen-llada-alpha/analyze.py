#!/usr/bin/env python3
"""
Post-run analysis script for MMTU experiment results.

Produces per-question metadata (input/output tokens, correctness) and
summary distribution tables (markdown histograms).

Usage:
    python projects/qwen-llada-alpha/analyze.py <result_file>.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import tiktoken
    _encoder = tiktoken.encoding_for_model("gpt-4o")
    def count_tokens(text: str) -> int:
        return len(_encoder.encode(text))
except Exception:
    # Fallback: approximate tokens as words * 1.3
    def count_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


def get_mmtu_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_evaluators():
    """Import MMTU evaluators (needs MMTU root on sys.path)."""
    mmtu_root = get_mmtu_root()
    if str(mmtu_root) not in sys.path:
        sys.path.insert(0, str(mmtu_root))

    from evaluators.em_evaluator import EMEvaluator
    from evaluators.sm_evaluator import SMEvaluator
    from evaluators.data_imputation_evaluator import DataImputationEvaluator
    from evaluators.semantic_transform_evaluator import SemanticTransformEvaluator
    from evaluators.list_to_table_evaluator import ListToTableEvaluator
    from evaluators.tablelocate_evaluator import TableLocateEvaluator
    from evaluators.ar_evaluator import AREvaluator
    from evaluators.fd_evaluator import FDEvaluator
    from evaluators.sr_evaluator import SREvaluator
    from evaluators.ejd_evaluator import EquiJoinDetectEvaluator
    from evaluators.semantic_join_evaluator import SemanticJoinEvaluator

    return {
        "Entity-Matching": EMEvaluator(),
        "Schema-Matching": SMEvaluator(),
        "Data-Imputation": DataImputationEvaluator(),
        "semantic-transform": SemanticTransformEvaluator(),
        "List-to-table": ListToTableEvaluator(),
        "Table-Locate-by-Row-Col": TableLocateEvaluator(),
        "Arithmetic-Relationship": AREvaluator(),
        "Functional-Dependency": FDEvaluator(),
        "String-Relationship": SREvaluator(),
        "equi-join-detect": EquiJoinDetectEvaluator(),
        "semantic-join": SemanticJoinEvaluator(),
    }


def find_eom(response: str) -> str:
    """Return response up to end-of-message marker (or full response)."""
    # LLaDA uses <|im_end|> or <|endoftext|> — strip anything after
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eom|>"):
        idx = response.find(marker)
        if idx != -1:
            return response[:idx]
    return response


def evaluate_row(evaluator, metadata: dict, response: str) -> int:
    """Return 1 if correct, 0 otherwise."""
    try:
        y_pred = evaluator._get_pred(response)
        y_gt = evaluator._get_gt(metadata)
        result = evaluator._evaluate_one(y_gt, y_pred)
        return int(result.get("correct", 0))
    except Exception:
        return 0


def histogram_table(values: list[float], label: str, n_bins: int = 10) -> str:
    """Build an ASCII-bar markdown histogram."""
    if not values:
        return f"_No data for {label}_\n"

    lo, hi = min(values), max(values)
    if lo == hi:
        return f"| {label} | Count |\n|---|---|\n| {lo} | {len(values)} |\n"

    bin_width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_max = 30

    lines = [f"### {label}", "", "| Range | Count | Distribution |", "|---|---|---|"]
    for i, count in enumerate(bins):
        start = lo + i * bin_width
        end = start + bin_width
        bar_len = round(count / max_count * bar_max) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"| {start:.0f}\u2013{end:.0f} | {count} | {bar} |")

    lines.append(f"\n**Total:** {len(values)} | **Mean:** {sum(values)/len(values):.1f} | "
                 f"**Median:** {sorted(values)[len(values)//2]:.1f} | "
                 f"**Min:** {lo:.0f} | **Max:** {hi:.0f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post-run analysis of MMTU results")
    parser.add_argument("result_file", type=str, help="Path to .result.jsonl")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output markdown file (default: alongside result file)")
    args = parser.parse_args()

    result_path = Path(args.result_file).resolve()
    if not result_path.exists():
        print(f"Error: {result_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else result_path.with_suffix(".analysis.md")

    # Load results
    rows = []
    with open(result_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print("No results found.")
        sys.exit(1)

    print(f"Loaded {len(rows)} results from {result_path.name}")

    # Evaluators
    evaluators = load_evaluators()

    # Per-question analysis
    records = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        task = metadata["task"]
        response_raw = row.get("response", "")
        response = find_eom(response_raw)

        input_tokens = count_tokens(row["prompt"])
        output_tokens = count_tokens(response)

        evaluator = evaluators.get(task)
        correct = evaluate_row(evaluator, metadata, response) if evaluator else -1

        records.append({
            "task": task,
            "dataset": metadata.get("dataset", ""),
            "test_case": metadata.get("test_case", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "correct": correct,
        })

    df = pd.DataFrame(records)

    # Build markdown report
    md = []
    exp_name = rows[0].get("experiment", result_path.stem)
    md.append(f"# Analysis: {exp_name}\n")
    md.append(f"**Result file:** `{result_path.name}`  ")
    md.append(f"**Samples:** {len(df)}\n")

    # --- Per-question table ---
    md.append("## Per-Question Metadata\n")
    md.append("| # | Task | Test Case | Input Tokens | Output Tokens | Correct |")
    md.append("|---|---|---|---|---|---|")
    for i, r in df.iterrows():
        correct_str = {1: "Yes", 0: "No", -1: "N/A"}[r["correct"]]
        md.append(f"| {i+1} | {r['task']} | {r['test_case']} | {r['input_tokens']} | "
                  f"{r['output_tokens']} | {correct_str} |")
    md.append("")

    # --- Correctness distribution ---
    evaluated = df[df["correct"] >= 0]
    md.append("## Correctness Summary\n")
    if len(evaluated) > 0:
        total = len(evaluated)
        n_correct = int(evaluated["correct"].sum())
        md.append(f"**Overall:** {n_correct}/{total} correct ({100*n_correct/total:.1f}%)\n")

        md.append("| Task | Correct | Total | Accuracy |")
        md.append("|---|---|---|---|")
        for task, grp in evaluated.groupby("task"):
            tc = int(grp["correct"].sum())
            tt = len(grp)
            md.append(f"| {task} | {tc} | {tt} | {100*tc/tt:.1f}% |")
        md.append("")
    else:
        md.append("_No evaluatable tasks found._\n")

    # --- Distribution histograms ---
    md.append("## Input Token Length Distribution\n")
    md.append(histogram_table(df["input_tokens"].tolist(), "Input Tokens"))

    md.append("## Output Token Length Distribution\n")
    md.append(histogram_table(df["output_tokens"].tolist(), "Output Tokens"))

    # --- Per-task breakdown ---
    md.append("## Per-Task Token Statistics\n")
    md.append("| Task | N | Input Mean | Input Med | Output Mean | Output Med |")
    md.append("|---|---|---|---|---|---|")
    for task, grp in df.groupby("task"):
        n = len(grp)
        im = grp["input_tokens"].mean()
        imed = grp["input_tokens"].median()
        om = grp["output_tokens"].mean()
        omed = grp["output_tokens"].median()
        md.append(f"| {task} | {n} | {im:.0f} | {imed:.0f} | {om:.0f} | {omed:.0f} |")
    md.append("")

    report = "\n".join(md)

    # Write output
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nAnalysis written to: {output_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
