# CLAUDE.md

MMTU (Massive Multi-Task Table Understanding) — NeurIPS 2025 benchmark evaluating LLMs on 28,136 questions across 25 table-related tasks.

## Setup

```bash
export MMTU_HOME=$(pwd)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

```bash
# Inference (Azure OpenAI example)
python3 inference.py azure_openai \
    --endpoint https://<DEPLOYMENT>.openai.azure.com \
    --api_key $AZURE_API_KEY \
    --model <MODEL_NAME> \
    --api_version 2024-08-01-preview

# Evaluation
python3 evaluate.py <result_file>.jsonl

# Build data for a single config
python3 build_data.py one --config configurations/<task>/<version>/<config>.py

# Docker evaluation
./run_evaluate_docker.sh <result_file>.jsonl [--n_jobs N] [--debug] [--viz]
```

## Project Structure

```
configurations/   # 410 Python config files across 25 task directories
evaluators/       # 27 evaluator modules (1 base + 25 task-specific + 1 shared)
utils/            # table_serializer.py, table_processor.py, count_token.py, utils.py
inference.py      # LLM querying with retry logic (OpenAI, Azure OpenAI, Azure AI Foundry)
evaluate.py       # Evaluation orchestration — maps tasks to evaluators, computes metrics
build_data.py     # Data preparation — processes raw datasets into benchmark format
```

## Architecture

- **Evaluators** inherit from `BaseEvaluator` (`evaluators/base_evaluator.py`). Override `_evaluate_one` and `_compute_metric`.
- **Configs** are Python modules exporting `prompt_template` (with `{{{placeholder}}}` syntax) and a `dataset_config` dict.
- **Serializers** in `utils/table_serializer.py` convert DataFrames to string formats (Markdown, CSV, JSON, HTML, KeyValue, etc.).

## Code Conventions

- Python 3.11+
- `snake_case` for functions/variables, `PascalCase` for classes, `_` prefix for private methods
- Imports grouped: stdlib, then third-party, then local (relative imports in packages)
- Triple-brace template placeholders: `{{{field_name}}}`
- All new code must include proper type annotations and pass linter checks

## Projects & Experiments

Active projects live in `projects/`, each with its own `CLAUDE.md` (goal, status, decisions) and `experiments.md` (chronological experiment log). Check there for context on ongoing work.
