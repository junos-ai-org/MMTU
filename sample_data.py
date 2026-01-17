"""
Data sampling utilities for MMTU evaluation.
Samples verification (n=5) and evaluation (n=300) sets from selected tasks.
"""

import json
import random
import argparse
from datasets import load_dataset
import pandas as pd

SELECTED_TASKS = [
    "Entity-Matching",
    "Schema-Matching",
    "header-value-matching",
    "Data-Imputation",
    "Error-Detect",
    "List-to-table",
    "semantic-join",
    "equi-join-detect",
    "Arithmetic-Relationship",
    "Functional-Dependency",
    "String-Relationship",
    "Formula-prediction-context",
    "semantic-transform",
    "Table-needle-in-a-haystack",
    "Table-Locate-by-Row-Col",
    "Column-type-annotation",
    "Columns-property-anotation",
    "Cell-entity-annotation",
]


def load_mmtu_dataset():
    """Load MMTU dataset from HuggingFace."""
    print("Loading MMTU dataset from HuggingFace...")
    ds = load_dataset("MMTU-benchmark/MMTU", split="train")
    df = ds.to_pandas()

    # Extract task from metadata
    def extract_task(metadata):
        if isinstance(metadata, dict):
            return metadata.get('task', 'unknown')
        elif isinstance(metadata, str):
            return json.loads(metadata).get('task', 'unknown')
        return 'unknown'

    df['task'] = df['metadata'].apply(extract_task)
    return df


def sample_tasks(df, tasks, n_per_task, seed=42):
    """
    Sample n examples per task.

    Args:
        df: DataFrame with MMTU data
        tasks: List of task names to include
        n_per_task: Number of samples per task
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    random.seed(seed)

    # Filter to selected tasks
    df_filtered = df[df['task'].isin(tasks)].copy()

    # Sample per task
    sampled_dfs = []
    for task in tasks:
        task_df = df_filtered[df_filtered['task'] == task]
        n_samples = min(n_per_task, len(task_df))
        if n_samples > 0:
            sampled = task_df.sample(n=n_samples, random_state=seed)
            sampled_dfs.append(sampled)
            print(f"  {task}: {n_samples} samples")
        else:
            print(f"  {task}: No data available!")

    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True)
    return pd.DataFrame()


def create_verification_sample(df, n_per_task=5, seed=42, output_file="verification_sample.jsonl"):
    """Create verification sample (5 per task)."""
    print(f"\nCreating verification sample (n={n_per_task} per task)...")
    sampled = sample_tasks(df, SELECTED_TASKS, n_per_task, seed)

    # Save to JSONL
    sampled[['prompt', 'metadata']].to_json(output_file, orient='records', lines=True)

    print(f"\nVerification set saved to {output_file}")
    print(f"Total: {len(sampled)} examples across {sampled['task'].nunique()} tasks")
    return sampled


def create_evaluation_sample(df, n_total=300, seed=42, output_file="evaluation_sample_300.jsonl"):
    """Create evaluation sample (300 total, distributed across tasks)."""
    n_tasks = len(SELECTED_TASKS)
    n_per_task = n_total // n_tasks

    print(f"\nCreating evaluation sample (n={n_total} total, ~{n_per_task} per task)...")
    sampled = sample_tasks(df, SELECTED_TASKS, n_per_task, seed)

    # Save to JSONL
    sampled[['prompt', 'metadata']].to_json(output_file, orient='records', lines=True)

    print(f"\nEvaluation set saved to {output_file}")
    print(f"Total: {len(sampled)} examples across {sampled['task'].nunique()} tasks")
    return sampled


def print_task_statistics(df):
    """Print statistics about available tasks."""
    print("\n=== Task Statistics ===")
    task_counts = df['task'].value_counts()

    print("\nSelected tasks availability:")
    for task in SELECTED_TASKS:
        count = task_counts.get(task, 0)
        status = "OK" if count > 0 else "MISSING"
        print(f"  [{status}] {task}: {count} examples")

    print(f"\nTotal examples in selected tasks: {df[df['task'].isin(SELECTED_TASKS)].shape[0]}")


def main():
    parser = argparse.ArgumentParser(description="Sample MMTU data for evaluation")
    parser.add_argument("--verification", "-v", action="store_true",
                        help="Create verification sample (5 per task)")
    parser.add_argument("--evaluation", "-e", action="store_true",
                        help="Create evaluation sample (300 total)")
    parser.add_argument("--both", "-b", action="store_true",
                        help="Create both samples")
    parser.add_argument("--stats", "-s", action="store_true",
                        help="Print task statistics only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n-verify", type=int, default=5,
                        help="Samples per task for verification (default: 5)")
    parser.add_argument("--n-eval", type=int, default=300,
                        help="Total samples for evaluation (default: 300)")
    args = parser.parse_args()

    # Load dataset
    df = load_mmtu_dataset()

    if args.stats:
        print_task_statistics(df)
        return

    if args.both or args.verification:
        create_verification_sample(df, n_per_task=args.n_verify, seed=args.seed)

    if args.both or args.evaluation:
        create_evaluation_sample(df, n_total=args.n_eval, seed=args.seed)

    if not (args.verification or args.evaluation or args.both or args.stats):
        print("No action specified. Use --help for options.")
        print_task_statistics(df)


if __name__ == "__main__":
    main()
