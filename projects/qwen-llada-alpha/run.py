#!/usr/bin/env python3
"""
Shared baseline runner for MMTU benchmark experiments.

Usage:
    python projects/baselines/run.py run configs/llada-8b-smoke.yaml
    python projects/baselines/run.py run configs/llada-8b-baseline.yaml
    python projects/baselines/run.py evaluate <result_file> [--n_jobs N]
    python projects/baselines/run.py list
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets import load_dataset


def _get_mmtu_root() -> Path:
    """Return the MMTU repo root (three levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _get_project_dir() -> Path:
    """Return the baselines project directory."""
    return Path(__file__).resolve().parent


def _get_output_dir(experiment_name: str) -> Path:
    """Return and create the experiment output directory."""
    out = _get_project_dir() / "output" / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_sha() -> str:
    """Return the current short git SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_get_mmtu_root()),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_config(config_path: str) -> dict:
    """Load and return a YAML experiment config."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _get_project_dir() / path
    if not path.exists():
        print(f"Error: config file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def load_mmtu_dataset(hf_dataset: str, hf_split: str) -> list[dict]:
    """Load the full MMTU dataset from HuggingFace."""
    print(f"Loading MMTU dataset from {hf_dataset} (split={hf_split})...")
    ds = load_dataset(hf_dataset, split=hf_split)
    records = [{"prompt": row["prompt"], "metadata": row["metadata"]} for row in ds]
    print(f"  Loaded {len(records)} total records.")
    return records


def sample_records(
    records: list[dict],
    tasks: list[str],
    n_per_task: int,
    seed: int,
    max_input_tokens: int,
) -> list[dict]:
    """Sample n_per_task records per task, excluding prompts that exceed max_input_tokens."""
    mmtu_root = _get_mmtu_root()
    if str(mmtu_root) not in sys.path:
        sys.path.insert(0, str(mmtu_root))
    from utils.count_token import count_tokens_mp

    task_set = set(tasks)

    # Collect task-relevant records
    task_records: list[tuple[str, dict]] = []
    for rec in records:
        meta = json.loads(rec["metadata"])
        task = meta["task"]
        if task in task_set:
            task_records.append((task, rec))

    # Batch token count
    prompts = [rec["prompt"] for _, rec in task_records]
    token_counts = count_tokens_mp(prompts)

    # Filter by token limit and bucket by task
    by_task: dict[str, list[dict]] = {t: [] for t in tasks}
    skipped_counts: dict[str, int] = {t: 0 for t in tasks}

    for (task, rec), tc in zip(task_records, token_counts):
        if tc > max_input_tokens:
            skipped_counts[task] += 1
        else:
            by_task[task].append(rec)

    # Sample
    rng = random.Random(seed)
    sampled = []
    for task in sorted(tasks):
        pool = by_task[task]
        total = len(pool) + skipped_counts[task]
        if len(pool) == 0:
            print(f"  {task}: 0/{total} fit <={max_input_tokens} tokens — SKIPPED")
            continue
        n = min(n_per_task, len(pool))
        chosen = rng.sample(pool, n)
        sampled.extend(chosen)
        print(f"  {task}: sampled {n}/{len(pool)} qualifying"
              f" ({skipped_counts[task]}/{total} exceeded token limit)")

    print(f"  Total sampled: {len(sampled)}")
    return sampled


def run_experiment(config: dict, config_path: str) -> None:
    """Execute a full experiment: load model, sample data, run inference, evaluate."""
    exp = config["experiment"]
    model_cfg = config["model"]
    inference_cfg = config["inference"]
    dataset_cfg = config["dataset"]

    experiment_name = exp["name"]
    seed = exp["seed"]
    model_alias = model_cfg["alias"]

    output_dir = _get_output_dir(experiment_name)

    print("=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Backend: {model_cfg['backend']} | Model: {model_cfg['model_path']}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Freeze config into output dir for traceability
    frozen_config = output_dir / "config.yaml"
    shutil.copy2(config_path if Path(config_path).is_absolute()
                 else str(_get_project_dir() / config_path), str(frozen_config))

    # Record provenance
    provenance = {
        "experiment_name": experiment_name,
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # 1. Load backend
    print("\n[1/4] Loading model...")
    project_dir = _get_project_dir()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from backends import get_backend

    backend_cls = get_backend(model_cfg["backend"])
    backend = backend_cls()
    backend.load(config)

    if not backend.health_check():
        print("FATAL: Backend health check failed.")
        sys.exit(1)

    # 2. Load and sample dataset
    print("\n[2/4] Loading and sampling dataset...")
    records = load_mmtu_dataset(dataset_cfg["name"], dataset_cfg["split"])
    sampled = sample_records(
        records,
        dataset_cfg["tasks"],
        dataset_cfg["samples_per_task"],
        seed,
        inference_cfg["max_input_tokens"],
    )

    # 3. Run inference
    print(f"\n[3/4] Running inference on {len(sampled)} samples...")
    input_file = output_dir / "input.jsonl"
    result_file = output_dir / f"{experiment_name}.{model_alias}.result.jsonl"

    with open(input_file, "w") as f:
        for rec in sampled:
            f.write(json.dumps(rec) + "\n")

    # Process each sample, with resumption support
    existing_results = set()
    if result_file.exists():
        with open(result_file) as f:
            for line in f:
                row = json.loads(line)
                existing_results.add(row.get("prompt", "")[:200])
        print(f"  Resuming: {len(existing_results)} existing results found.")

    completed = len(existing_results)
    total = len(sampled)

    with open(result_file, "a") as out_f:
        for i, rec in enumerate(sampled):
            # Skip already-processed prompts (resumption)
            if rec["prompt"][:200] in existing_results:
                continue

            completed += 1
            print(f"  [{completed}/{total}] Generating response...", end=" ", flush=True)
            t0 = time.time()

            try:
                response = backend.generate(rec["prompt"])
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR ({elapsed:.1f}s): {e}")
                response = f"[ERROR] {e}"

            result_row = {
                "prompt": rec["prompt"],
                "metadata": rec["metadata"],
                "response": response,
                "model": model_alias,
                "experiment": experiment_name,
            }
            out_f.write(json.dumps(result_row) + "\n")
            out_f.flush()

    # 4. Evaluate
    print("\n[4/4] Evaluating results...")
    rc = run_evaluation(result_file, _get_mmtu_root())
    if rc != 0:
        print(f"Evaluation exited with code {rc}")
    else:
        print(f"\nExperiment '{experiment_name}' complete!")
        print(f"Results: {output_dir}")


def run_evaluation(result_file: Path, mmtu_root: Path, n_jobs: int = -1) -> int:
    """Run evaluate.py via subprocess."""
    cmd = [
        sys.executable, str(mmtu_root / "evaluate.py"),
        str(result_file),
        "--n_jobs", str(n_jobs),
    ]
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(mmtu_root))
    return proc.returncode


def cmd_run(args):
    """Run an experiment from a YAML config."""
    config = load_config(args.config)
    run_experiment(config, args.config)


def cmd_evaluate(args):
    """Standalone evaluation of a result file."""
    result_file = Path(args.result_file).resolve()
    if not result_file.exists():
        print(f"Error: {result_file} does not exist")
        sys.exit(1)
    rc = run_evaluation(result_file, _get_mmtu_root(), n_jobs=args.n_jobs)
    sys.exit(rc)


def cmd_list(args):
    """List all experiments and their status."""
    output_root = _get_project_dir() / "output"
    if not output_root.exists():
        print("No experiments found.")
        return

    experiments = sorted(output_root.iterdir())
    if not experiments:
        print("No experiments found.")
        return

    print(f"{'Experiment':<35} {'Status':<12} {'Timestamp':<22}")
    print("-" * 70)

    for exp_dir in experiments:
        if not exp_dir.is_dir():
            continue
        prov_file = exp_dir / "provenance.json"
        if prov_file.exists():
            with open(prov_file) as f:
                prov = json.load(f)
            timestamp = prov.get("timestamp", "unknown")[:19]
        else:
            timestamp = "unknown"

        # Check for result files
        result_files = list(exp_dir.glob("*.result.jsonl"))
        if result_files:
            with open(result_files[0]) as f:
                n_results = sum(1 for _ in f)
            status = f"{n_results} results"
        else:
            status = "no results"

        print(f"{exp_dir.name:<35} {status:<12} {timestamp:<22}")


def main():
    parser = argparse.ArgumentParser(
        description="MMTU baseline experiment runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    sub_run = subparsers.add_parser("run", help="Run an experiment from a YAML config")
    sub_run.add_argument("config", type=str, help="Path to experiment YAML config")

    # evaluate
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate a result file")
    sub_eval.add_argument("result_file", type=str)
    sub_eval.add_argument("--n_jobs", type=int, default=-1)

    # list
    subparsers.add_parser("list", help="List all experiments")

    args = parser.parse_args()
    commands = {"run": cmd_run, "evaluate": cmd_evaluate, "list": cmd_list}
    commands[args.command](args)


if __name__ == "__main__":
    main()
