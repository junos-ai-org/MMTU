#!/usr/bin/env python3
"""
Build paired data for cross-model analysis of Data-Imputation and List-to-table.

Loads both model result JSONLs, filters to target tasks, joins on test_case,
runs MMTU evaluators for per-row scores, and writes paired_data.jsonl.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Setup paths
INSIGHT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = INSIGHT_DIR.parent.parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
MMTU_ROOT = PROJECT_DIR.parent.parent

# Add MMTU root and project to path for imports
sys.path.insert(0, str(MMTU_ROOT))
sys.path.insert(0, str(PROJECT_DIR))

from analyze import evaluate_per_row, load_evaluators, find_eom

TARGET_TASKS = {"Data-Imputation", "List-to-table"}

QWEN_RESULT = EXPERIMENT_DIR / "output/Qwen2.5-14B-Instruct/20260325-124613/encoder_vs_decoder_baseline_large.Qwen2.5-14B-Instruct.result.jsonl"
T5GEMMA_RESULT = EXPERIMENT_DIR / "output/t5gemma-9b-9b-ul2-it/20260325-120505/encoder_vs_decoder_baseline_large.t5gemma-9b-9b-ul2-it.result.jsonl"


def load_and_filter(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            meta = json.loads(row["metadata"])
            if meta["task"] in TARGET_TASKS:
                rows.append({
                    "task": meta["task"],
                    "test_case": meta["test_case"],
                    "dataset": meta.get("dataset", ""),
                    "label": meta["label"],
                    "prompt": row["prompt"],
                    "response": find_eom(row["response"]),
                    "metadata": row["metadata"],
                })
    return pd.DataFrame(rows)


def get_scores(df: pd.DataFrame) -> dict:
    """Run MMTU evaluators and return {(task, test_case): score}."""
    # Rebuild the format evaluate_per_row expects (needs prompt + response + metadata)
    eval_df = pd.DataFrame({
        "task": df["task"],
        "test_case": df["test_case"],
        "metadata": df["metadata"],
        "response": df["response"],
        "prompt": df["prompt"],
    })
    evaluators = load_evaluators()
    scores_df = evaluate_per_row(eval_df, evaluators)
    return {(r["task"], r["test_case"]): r["score"] for _, r in scores_df.iterrows()}


def main():
    print("Loading Qwen results...")
    qwen_df = load_and_filter(QWEN_RESULT)
    print(f"  {len(qwen_df)} rows: {qwen_df['task'].value_counts().to_dict()}")

    print("Loading T5Gemma results...")
    t5gemma_df = load_and_filter(T5GEMMA_RESULT)
    print(f"  {len(t5gemma_df)} rows: {t5gemma_df['task'].value_counts().to_dict()}")

    print("Computing Qwen scores...")
    qwen_scores = get_scores(qwen_df)

    print("Computing T5Gemma scores...")
    t5gemma_scores = get_scores(t5gemma_df)

    # Build lookup by (task, test_case)
    qwen_lookup = {(r["task"], r["test_case"]): r for _, r in qwen_df.iterrows()}
    t5gemma_lookup = {(r["task"], r["test_case"]): r for _, r in t5gemma_df.iterrows()}

    all_keys = sorted(set(qwen_lookup.keys()) & set(t5gemma_lookup.keys()))
    print(f"\nJoined pairs: {len(all_keys)}")

    output_path = INSIGHT_DIR / "paired_data.jsonl"
    with open(output_path, "w") as f:
        for i, (task, test_case) in enumerate(all_keys):
            qr = qwen_lookup[(task, test_case)]
            tr = t5gemma_lookup[(task, test_case)]
            pair = {
                "idx": i,
                "task": task,
                "test_case": test_case,
                "dataset": qr["dataset"],
                "label": qr["label"],
                "prompt": qr["prompt"],
                "qwen_response": qr["response"],
                "t5gemma_response": tr["response"],
                "qwen_score": qwen_scores.get((task, test_case), -1),
                "t5gemma_score": t5gemma_scores.get((task, test_case), -1),
            }
            f.write(json.dumps(pair) + "\n")

    # Summary
    pairs = []
    with open(output_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    for task in TARGET_TASKS:
        task_pairs = [p for p in pairs if p["task"] == task]
        both_correct = sum(1 for p in task_pairs if p["qwen_score"] == 1 and p["t5gemma_score"] == 1)
        both_wrong = sum(1 for p in task_pairs if p["qwen_score"] == 0 and p["t5gemma_score"] == 0)
        qwen_only = sum(1 for p in task_pairs if p["qwen_score"] == 1 and p["t5gemma_score"] == 0)
        t5gemma_only = sum(1 for p in task_pairs if p["qwen_score"] == 0 and p["t5gemma_score"] == 1)
        partial = len(task_pairs) - both_correct - both_wrong - qwen_only - t5gemma_only
        print(f"\n{task} ({len(task_pairs)} pairs):")
        print(f"  Both correct: {both_correct}")
        print(f"  Both wrong:   {both_wrong}")
        print(f"  Qwen only:    {qwen_only}")
        print(f"  T5Gemma only: {t5gemma_only}")
        if partial:
            print(f"  Partial:      {partial}")

    print(f"\nWrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()
