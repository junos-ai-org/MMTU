"""
Compare evaluation results between LLaDA and QWEN on MMTU.

Usage:
    python compare_results.py results/llada_300.jsonl results/qwen_300.jsonl
"""

import argparse
import json
import os
import pandas as pd
from collections import defaultdict

# Import MMTU evaluators
from evaluate import evaluator_dict, summary_metric


def load_results(result_file):
    """Load results from JSONL file."""
    df = pd.read_json(result_file, lines=True)

    # Extract task from metadata
    def extract_task(metadata):
        if isinstance(metadata, dict):
            return metadata.get('task', 'unknown')
        elif isinstance(metadata, str):
            return json.loads(metadata).get('task', 'unknown')
        return 'unknown'

    df['task'] = df['metadata'].apply(extract_task)
    return df


def evaluate_results(df, model_name):
    """Run MMTU evaluators on results."""
    results_by_task = {}

    for task, group in df.groupby('task'):
        if task not in evaluator_dict:
            print(f"  Warning: No evaluator for task '{task}', skipping")
            continue

        evaluator = evaluator_dict[task]
        try:
            avg_results, detail_results, error_cnt = evaluator.evaluate(group)

            # Get primary metric for this task
            metric = summary_metric.get(task, 'acc')
            if isinstance(metric, list):
                metric = metric[0]

            # Calculate average score
            if metric in avg_results.columns:
                score = avg_results[metric].mean()
            else:
                score = None

            results_by_task[task] = {
                'metric': metric,
                'score': score,
                'n_examples': len(group),
                'avg_results': avg_results,
                'detail_results': detail_results
            }

        except Exception as e:
            print(f"  Error evaluating {task}: {e}")
            results_by_task[task] = {
                'metric': 'error',
                'score': None,
                'n_examples': len(group),
                'error': str(e)
            }

    return results_by_task


def compare_models(llada_file, qwen_file, output_dir="comparison_results"):
    """Compare LLaDA and QWEN results."""
    print("=" * 70)
    print("MMTU Model Comparison: LLaDA vs QWEN")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    llada_df = load_results(llada_file)
    qwen_df = load_results(qwen_file)

    print(f"  LLaDA: {len(llada_df)} examples")
    print(f"  QWEN:  {len(qwen_df)} examples")

    # Evaluate
    print("\nEvaluating LLaDA...")
    llada_results = evaluate_results(llada_df, "LLaDA")

    print("\nEvaluating QWEN...")
    qwen_results = evaluate_results(qwen_df, "QWEN")

    # Create comparison table
    comparison_data = []
    all_tasks = set(llada_results.keys()) | set(qwen_results.keys())

    for task in sorted(all_tasks):
        llada_info = llada_results.get(task, {})
        qwen_info = qwen_results.get(task, {})

        llada_score = llada_info.get('score')
        qwen_score = qwen_info.get('score')
        metric = llada_info.get('metric', qwen_info.get('metric', 'N/A'))

        # Calculate delta
        if llada_score is not None and qwen_score is not None:
            delta = llada_score - qwen_score
            winner = "LLaDA" if delta > 0.01 else ("QWEN" if delta < -0.01 else "Tie")
        else:
            delta = None
            winner = "N/A"

        comparison_data.append({
            'Task': task,
            'Metric': metric,
            'LLaDA': f"{llada_score:.3f}" if llada_score is not None else "N/A",
            'QWEN': f"{qwen_score:.3f}" if qwen_score is not None else "N/A",
            'Delta': f"{delta:+.3f}" if delta is not None else "N/A",
            'Winner': winner,
            'N (LLaDA)': llada_info.get('n_examples', 0),
            'N (QWEN)': qwen_info.get('n_examples', 0),
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_markdown(index=False))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    llada_wins = sum(1 for r in comparison_data if r['Winner'] == 'LLaDA')
    qwen_wins = sum(1 for r in comparison_data if r['Winner'] == 'QWEN')
    ties = sum(1 for r in comparison_data if r['Winner'] == 'Tie')

    print(f"\nTask wins:")
    print(f"  LLaDA: {llada_wins}")
    print(f"  QWEN:  {qwen_wins}")
    print(f"  Tie:   {ties}")

    # Average scores
    llada_scores = [llada_results[t]['score'] for t in llada_results if llada_results[t].get('score') is not None]
    qwen_scores = [qwen_results[t]['score'] for t in qwen_results if qwen_results[t].get('score') is not None]

    if llada_scores:
        print(f"\nAverage score (LLaDA): {sum(llada_scores)/len(llada_scores):.3f}")
    if qwen_scores:
        print(f"Average score (QWEN):  {sum(qwen_scores)/len(qwen_scores):.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
    comparison_df.to_excel(os.path.join(output_dir, "comparison_summary.xlsx"), index=False)

    print(f"\nResults saved to {output_dir}/")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Compare LLaDA and QWEN results on MMTU")
    parser.add_argument("llada_results", help="Path to LLaDA results JSONL file")
    parser.add_argument("qwen_results", help="Path to QWEN results JSONL file")
    parser.add_argument("--output", "-o", default="comparison_results",
                        help="Output directory for comparison results")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.llada_results):
        print(f"Error: LLaDA results file not found: {args.llada_results}")
        return 1

    if not os.path.exists(args.qwen_results):
        print(f"Error: QWEN results file not found: {args.qwen_results}")
        return 1

    compare_models(args.llada_results, args.qwen_results, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
