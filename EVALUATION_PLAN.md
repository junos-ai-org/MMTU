# MMTU Evaluation Plan: LLaDA vs QWEN 7B

## Task Mapping

Your selected tasks mapped to MMTU configuration names:

| Your Task Name | MMTU Task Name | Metric |
|----------------|----------------|--------|
| Entity Matching | `Entity-Matching` | acc |
| Schema Matching | `Schema-Matching` | f1 |
| Header Value Matching | `header-value-matching` | acc |
| Data Imputation | `Data-Imputation` | acc |
| Error Detection | `Error-Detect` | f1 |
| List-to-Table | `List-to-table` | acc |
| Semantic Join | `semantic-join` | f1 |
| Equi-Join Detect | `equi-join-detect` | f1 |
| Arithmetic Relationship | `Arithmetic-Relationship` | f1 |
| Functional Relationship | `Functional-Dependency` | f1 |
| String Relationship | `String-Relationship` | f1 |
| Formula-by-Context | `Formula-prediction-context` | acc |
| Semantic-Transform | `semantic-transform` | acc |
| Needle-in-a-Haystack | `Table-needle-in-a-haystack` | acc |
| Needle-Index | `Table-Locate-by-Row-Col` | acc |
| Column Type Annotation | `Column-type-annotation` | acc |
| Column Property Annotation | `Columns-property-anotation` | acc |
| Cell Entity Annotation | `Cell-entity-annotation` | acc |

**Total: 18 tasks**

---

## Infrastructure Strategy: Minimize GPU Idle Time

### The Problem
```
Traditional workflow (wasteful):
┌──────────────────────────────────────────────────────────────┐
│ GPU Instance ($$$)                                           │
│ ├── [30 min] pip install dependencies                        │
│ ├── [15 min] huggingface-cli login                          │
│ ├── [45 min] Download LLaDA weights (14GB+)                 │
│ ├── [30 min] Download QWEN weights (14GB+)                  │
│ ├── [10 min] Clone repos, setup code                        │
│ └── [????? ] Actually run experiments  <-- what you pay for │
└──────────────────────────────────────────────────────────────┘
```

### The Solution: Pre-baked Docker + S3 Weights

```
Phase 1: Build (CPU machine - FREE or cheap)
┌──────────────────────────────────────────────────────────────┐
│ Local Machine / GitHub Actions / EC2 CPU                     │
│ ├── Build Docker image with all dependencies                 │
│ ├── Push to AWS ECR                                          │
│ └── Upload model weights to S3                               │
└──────────────────────────────────────────────────────────────┘

Phase 2: Run (GPU instance - FAST)
┌──────────────────────────────────────────────────────────────┐
│ RunPod GPU Instance ($$$)                                    │
│ ├── [2 min]  Pull Docker image (cached layers)              │
│ ├── [1 min]  Mount S3 weights via s3fs or download          │
│ └── [START]  Run experiments immediately                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Docker Image Design

### Dockerfile

```dockerfile
# Base image with CUDA
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl \
    s3fs awscli \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer - won't rebuild)
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# MMTU code
COPY . /workspace/MMTU
WORKDIR /workspace/MMTU

# LLaDA code (from your private fork)
RUN git clone https://github.com/junos-ai-org/LLaDA.git /workspace/LLaDA

# Pre-download QWEN tokenizer (small, can bake in)
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')"

# Entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

### Entrypoint Script (docker/entrypoint.sh)

```bash
#!/bin/bash
set -e

# Mount S3 bucket with model weights (if AWS credentials provided)
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Mounting S3 bucket..."
    mkdir -p /models
    s3fs $S3_BUCKET /models \
        -o passwd_file=/dev/stdin \
        -o url=https://s3.amazonaws.com \
        -o use_path_request_style \
        <<< "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY"

    export LLADA_MODEL_PATH=/models/llada-8b-instruct
    export QWEN_MODEL_PATH=/models/qwen2-7b-instruct
else
    echo "No AWS credentials - expecting models at /models"
fi

exec "$@"
```

### requirements.txt additions

```
# Core MMTU dependencies
pandas
datasets
openai
tenacity
tiktoken
tqdm
openpyxl

# LLaDA dependencies
torch>=2.1.0
transformers>=4.36.0
accelerate
safetensors
sentencepiece

# S3 mounting
boto3
```

---

## LLaDA Integration

LLaDA is a **diffusion model** - it generates text by iteratively denoising from masked tokens, not via autoregressive next-token prediction. This means:

1. **Cannot use vLLM, TGI, or other standard inference servers**
2. **Must use LLaDA's native `generate()` method**
3. **Different sampling parameters** (e.g., number of diffusion steps)

### Custom Query Function for MMTU

Create `llada_inference.py`:

```python
import torch
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '/workspace/LLaDA')
from llada.modeling_llada import LLaDAForCausalLM  # Adjust import based on actual LLaDA structure

class LLaDAInference:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLaDAForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, steps=64, temperature=1.0):
        """
        LLaDA diffusion generation.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            steps: Number of diffusion steps (more = better quality, slower)
            temperature: Sampling temperature
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # LLaDA-specific generation (adjust based on actual API)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_diffusion_steps=steps,
            temperature=temperature,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response


def create_llada_query_function(model_path):
    """Factory function compatible with MMTU inference.py"""
    inference = LLaDAInference(model_path)

    def query(prompt, temperature):
        import time
        t0 = time.time()
        response = inference.generate(prompt, temperature=temperature)
        t1 = time.time()

        # Return format expected by MMTU
        return {
            "response": response,
            "prompt_tokens": None,  # LLaDA doesn't expose this easily
            "completion_tokens": None,
            "time_taken": t1 - t0
        }

    query.__name__ = "llada_8b_instruct"
    return query
```

### QWEN Integration (simpler - standard HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QWENInference:
    def __init__(self, model_path="Qwen/Qwen2-7B-Instruct", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, temperature=1.0):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
        )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response


def create_qwen_query_function(model_path="Qwen/Qwen2-7B-Instruct"):
    inference = QWENInference(model_path)

    def query(prompt, temperature):
        import time
        t0 = time.time()
        response = inference.generate(prompt, temperature=temperature)
        t1 = time.time()

        return {
            "response": response,
            "prompt_tokens": None,
            "completion_tokens": None,
            "time_taken": t1 - t0
        }

    query.__name__ = "qwen2_7b_instruct"
    return query
```

---

## Sampling Strategy: 5 → 300

### Phase 1: Verification Sample (n=5)

Quick sanity check that everything works:

```python
# run_verification.py
from datasets import load_dataset
import random

SELECTED_TASKS = [
    "Entity-Matching", "Schema-Matching", "header-value-matching",
    "Data-Imputation", "Error-Detect", "List-to-table",
    "semantic-join", "equi-join-detect", "Arithmetic-Relationship",
    "Functional-Dependency", "String-Relationship", "Formula-prediction-context",
    "semantic-transform", "Table-needle-in-a-haystack", "Table-Locate-by-Row-Col",
    "Column-type-annotation", "Columns-property-anotation", "Cell-entity-annotation"
]

def sample_verification_set(n_per_task=5, seed=42):
    """Sample 5 examples per task for verification."""
    random.seed(seed)

    ds = load_dataset("MMTU-benchmark/MMTU", split="train")
    df = ds.to_pandas()

    # Filter to selected tasks
    df['task'] = df['metadata'].apply(lambda x: x['task'] if isinstance(x, dict) else json.loads(x)['task'])
    df = df[df['task'].isin(SELECTED_TASKS)]

    # Sample n per task
    sampled = df.groupby('task').apply(lambda x: x.sample(min(n_per_task, len(x)), random_state=seed))
    sampled = sampled.reset_index(drop=True)

    return sampled

# Run
verification_set = sample_verification_set(n_per_task=5)
verification_set.to_json("verification_sample.jsonl", orient="records", lines=True)
print(f"Verification set: {len(verification_set)} examples across {verification_set['task'].nunique()} tasks")
```

### Phase 2: Full Evaluation Sample (n=300 total, stratified)

```python
def sample_evaluation_set(n_total=300, seed=42):
    """Sample 300 examples total, stratified by task."""
    random.seed(seed)

    ds = load_dataset("MMTU-benchmark/MMTU", split="train")
    df = ds.to_pandas()

    df['task'] = df['metadata'].apply(lambda x: x['task'] if isinstance(x, dict) else json.loads(x)['task'])
    df = df[df['task'].isin(SELECTED_TASKS)]

    # Calculate samples per task (proportional or equal)
    n_tasks = len(SELECTED_TASKS)
    n_per_task = n_total // n_tasks  # ~16-17 per task

    sampled = df.groupby('task').apply(
        lambda x: x.sample(min(n_per_task, len(x)), random_state=seed)
    )
    sampled = sampled.reset_index(drop=True)

    return sampled

# Run
evaluation_set = sample_evaluation_set(n_total=300)
evaluation_set.to_json("evaluation_sample_300.jsonl", orient="records", lines=True)
print(f"Evaluation set: {len(evaluation_set)} examples across {evaluation_set['task'].nunique()} tasks")
```

---

## Complete Workflow

### Step 1: One-time Setup (on CPU, no GPU cost)

```bash
# 1. Build and push Docker image
docker build -t mmtu-llada-eval .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag mmtu-llada-eval:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mmtu-llada-eval:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mmtu-llada-eval:latest

# 2. Download and upload model weights to S3 (can do on any machine)
# LLaDA weights (from your private repo or HuggingFace)
huggingface-cli download BAAI/LLaDA-8B-Instruct --local-dir ./llada-8b-instruct
aws s3 sync ./llada-8b-instruct s3://your-bucket/models/llada-8b-instruct/

# QWEN weights
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./qwen2-7b-instruct
aws s3 sync ./qwen2-7b-instruct s3://your-bucket/models/qwen2-7b-instruct/
```

### Step 2: Run on GPU (RunPod)

On RunPod, create a pod with:
- **Template**: Custom Docker image from ECR
- **GPU**: A100 40GB or similar (for running both models)
- **Environment variables**:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `S3_BUCKET=your-bucket/models`

```bash
# Inside the container, models are at /models/llada-8b-instruct and /models/qwen2-7b-instruct

# Phase 1: Quick verification (5 samples)
python run_evaluation.py --model llada --input verification_sample.jsonl --output results/llada_verify.jsonl
python run_evaluation.py --model qwen --input verification_sample.jsonl --output results/qwen_verify.jsonl

# Check results look reasonable
python evaluate.py results/llada_verify.jsonl
python evaluate.py results/qwen_verify.jsonl

# Phase 2: Full evaluation (300 samples)
python run_evaluation.py --model llada --input evaluation_sample_300.jsonl --output results/llada_300.jsonl
python run_evaluation.py --model qwen --input evaluation_sample_300.jsonl --output results/qwen_300.jsonl

# Evaluate
python evaluate.py results/llada_300.jsonl
python evaluate.py results/qwen_300.jsonl
```

### Step 3: Comparison Analysis

```python
# compare_results.py
import pandas as pd

llada_results = pd.read_excel("results/summary/overall_average.xlsx")  # After running evaluate.py
qwen_results = pd.read_excel("results_qwen/summary/overall_average.xlsx")

comparison = pd.merge(llada_results, qwen_results, on=['task', 'dataset', 'tag', 'note', 'metric'], suffixes=('_llada', '_qwen'))
comparison['delta'] = comparison['llada'] - comparison['qwen']
comparison.to_csv("llada_vs_qwen_comparison.csv")

print(comparison.groupby('task')['delta'].mean().sort_values(ascending=False))
```

---

## Cost Estimate

| Phase | Resource | Time | Cost |
|-------|----------|------|------|
| Docker build | Local/GitHub Actions | 30 min | $0 |
| Model upload to S3 | Local | 1 hour | ~$0.50 storage/month |
| Verification (5 samples) | RunPod A100 | ~10 min | ~$0.50 |
| Full eval (300 samples) | RunPod A100 | ~2-4 hours | ~$6-12 |
| **Total** | | | **~$7-13** |

Compare to traditional approach: ~$20-30+ (paying for GPU during all setup time)

---

## Files to Create

1. `Dockerfile` - Container definition
2. `docker/entrypoint.sh` - Startup script
3. `llada_inference.py` - LLaDA query function
4. `qwen_inference.py` - QWEN query function
5. `run_evaluation.py` - Main evaluation runner with sampling
6. `sample_data.py` - Data sampling utilities
7. `compare_results.py` - Comparison analysis

---

## Next Steps

1. [ ] Verify access to `junos-ai-org/LLaDA` repository
2. [ ] Check LLaDA's actual generate() API (may differ from above)
3. [ ] Set up AWS ECR repository
4. [ ] Set up S3 bucket for model weights
5. [ ] Build and test Docker image locally
6. [ ] Run verification sample on RunPod
7. [ ] Run full evaluation
8. [ ] Analyze and compare results
