#!/bin/bash
set -euo pipefail

# --- Logging & error handling ---------------------------------------------------
log()  { echo "[entrypoint] $(date '+%H:%M:%S') $*"; }
warn() { echo "[entrypoint] $(date '+%H:%M:%S') WARNING: $*" >&2; }
die()  { echo "[entrypoint] $(date '+%H:%M:%S') FATAL: $*" >&2; exit 1; }

trap 'rc=$?; echo ""; die "entrypoint failed at line $LINENO (exit code $rc). Check logs above."' ERR

# --- Deploy key -----------------------------------------------------------------
_setup_deploy_key() {
    log "Setting up GitHub deploy key..."
    mkdir -p ~/.ssh
    python3 -c "
import base64, os, sys

raw = os.environ['DEPLOY_KEY'].strip()
# Strip whitespace/newlines that sneak in from env var quoting
raw = raw.replace(' ', '').replace('\n', '').replace('\r', '')
# Pad to a multiple of 4
raw += '=' * (-len(raw) % 4)

n_data = len(raw.rstrip('='))
if n_data % 4 == 1:
    print(f'ERROR: DEPLOY_KEY has {n_data} base64 data chars (remainder 1 mod 4).', file=sys.stderr)
    print('This is not valid base64 — the key is truncated or corrupted.', file=sys.stderr)
    print('Re-generate with:  base64 -w0 < deploy_key > deploy_key.b64', file=sys.stderr)
    sys.exit(1)

try:
    decoded = base64.b64decode(raw)
except Exception as e:
    print(f'ERROR: Failed to base64-decode DEPLOY_KEY: {e}', file=sys.stderr)
    print(f'Key length (after cleanup): {len(raw)} chars', file=sys.stderr)
    sys.exit(1)

open('/root/.ssh/github_deploy_key', 'wb').write(decoded)
print(f'  Decoded deploy key ({len(decoded)} bytes).')
"
    chmod 600 ~/.ssh/github_deploy_key
    cat > ~/.ssh/config << 'SSHEOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_deploy_key
    IdentitiesOnly yes
SSHEOF
    chmod 600 ~/.ssh/config
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
    log "  Deploy key ready."
}

if [ -n "${DEPLOY_KEY:-}" ]; then
    if ! _setup_deploy_key; then
        warn "Deploy key setup failed (see errors above). Continuing without it — SSH in to debug."
    fi
else
    warn "DEPLOY_KEY not set — git clone will only work if SSH keys are mounted or repo is public."
fi

# --- Clone or update MMTU code --------------------------------------------------
MMTU_REPO="${MMTU_GIT_URL:-git@github.com:junos-ai-org/MMTU.git}"
MMTU_REF="${MMTU_GIT_REF:-main}"

if [ -d /workspace/MMTU/.git ]; then
    log "Updating MMTU code (ref: ${MMTU_REF})..."
    (cd /workspace/MMTU && git fetch origin && git checkout "$MMTU_REF" && git pull origin "$MMTU_REF") || warn "Git update failed — continuing with existing code."
else
    log "Cloning MMTU code (ref: ${MMTU_REF}) from ${MMTU_REPO}..."
    git clone --depth 1 --branch "$MMTU_REF" "$MMTU_REPO" /workspace/MMTU || warn "Git clone failed — SSH in to debug. Container will stay alive."
fi

# --- Model weights --------------------------------------------------------------
export HF_HOME="/workspace/.cache/huggingface"

QWEN_MODEL="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
log "Downloading model weights for ${QWEN_MODEL}..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('${QWEN_MODEL}')"
log "  Qwen2.5 model weights ready."

# --- Ready banner ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Qwen2.5 image ready"
echo "============================================================"
echo ""
echo "  cd /workspace/MMTU"
echo ""
echo "  # Start vLLM server"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model ${QWEN_MODEL} &"
echo ""
echo "  # --- Smoke tests ---"
echo "  sleep 60 && python projects/tabular-llms-research/run.py run \\"
echo "      experiments/encoder_vs_decoder_baseline/configs/run_qwen_smoke.yaml"
echo ""
echo "  # --- Baseline ---"
echo "  python projects/tabular-llms-research/run.py run \\"
echo "      experiments/encoder_vs_decoder_baseline/configs/run_qwen.yaml"
echo ""
echo "  # --- Context growth (all buckets) ---"
echo "  for cfg in experiments/context_growth/configs/run_qwen_[0-9]*.yaml; do"
echo "      python projects/tabular-llms-research/run.py run \"\$cfg\""
echo "  done"
echo ""
echo "  # --- Column shuffle (all permutations) ---"
echo "  for cfg in experiments/column_shuffle/configs/run_qwen_perm*.yaml; do"
echo "      python projects/tabular-llms-research/run.py run \"\$cfg\""
echo "  done"
echo ""
echo "  # Update code mid-session (no restart needed)"
echo "  mmtu-update"
echo ""

exec sleep infinity
