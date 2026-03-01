# qwen-base-line

## Goal
Establish a baseline for Qwen2.5-7B-Instruct on a subset of MMTU tasks,
served via vLLM on RunPod.

## Status
- [ ] Phase 1: Smoke test (5 Entity-Matching questions)
- [ ] Phase 2: Baseline (99 questions, 11 tasks x 9)

## Key Decisions
- **Uses inference.py**: Fixed `create_query_funtion_openai()` to use modern
  `base_url=` (was broken with openai>=1.0). Imports `query_chat_endpoint`
  directly to reuse multi-threading, resumption, and error handling.
- **Subprocess evaluation**: Calls `evaluate.py` via subprocess to avoid
  `args` global dependency.
- **Model alias**: `Qwen2_5-7B-Instruct` (dots/slashes replaced) for
  evaluate.py filename parsing compatibility.
- **Temperature 0**: Deterministic baseline.
- **TNIAH excluded**: `Table-needle-in-a-haystack` evaluator reads local
  filesystem paths for ground truth; fails on RunPod.
- **Runtime model download**: Weights not baked into image (~11GB vs ~25GB).
  Cached on RunPod network volume (`/workspace/.cache/huggingface`).

## Workflow

### Step 1: Build on AWS EC2

1. **Launch EC2** in AWS Console:
   - AMI: Amazon Linux 2023 (or Ubuntu 22.04)
   - Instance type: `t3.medium` (~$0.04/hr, or ~$0.01/hr spot)
   - Storage: 30 GB gp3
   - Security group: allow SSH (port 22)
   - Key pair: create or select one

2. **SSH in**:
   ```bash
   ssh -i key.pem ec2-user@<public-ip>    # Amazon Linux
   ssh -i key.pem ubuntu@<public-ip>      # Ubuntu
   ```

3. **Run the build script** (edit the config vars first):
   ```bash
   curl -O https://raw.githubusercontent.com/junos-ai-org/MMTU/main/projects/qwen-base-line/docker/ec2-build-guide.sh
   # Edit DOCKERHUB_USER and GITHUB_REPO_URL in the script
   bash ec2-build-guide.sh
   ```
   Or manually:
   ```bash
   sudo yum install -y docker git && sudo systemctl start docker
   sudo usermod -aG docker $USER && newgrp docker
   git clone https://github.com/junos-ai-org/MMTU.git && cd MMTU
   docker login
   bash projects/qwen-base-line/docker/build.sh \
       --registry docker.io/<username> --push
   ```

4. **Terminate the EC2 instance** — don't leave it running.

### Step 2: Deploy on RunPod

1. Create a GPU pod (A40/A100, >=24GB VRAM)
2. Template image: `docker.io/<username>/mmtu-qwen-baseline:latest`
3. Optional: attach a network volume on `/workspace` (caches model weights)

### Step 3: Run experiments

SSH into the RunPod pod. The entrypoint auto-downloads the model + starts vLLM.

```bash
cd /workspace/MMTU
python projects/qwen-base-line/run.py smoke      # quick sanity check
python projects/qwen-base-line/run.py baseline   # full baseline
```
