#!/bin/bash
set -euo pipefail

# --- Logging & error handling ---------------------------------------------------
log()  { echo "[entrypoint] $(date '+%H:%M:%S') $*"; }
warn() { echo "[entrypoint] $(date '+%H:%M:%S') WARNING: $*" >&2; }
die()  { echo "[entrypoint] $(date '+%H:%M:%S') FATAL: $*" >&2; exit 1; }

trap 'echo ""; die "entrypoint failed at line $LINENO (exit code $?). Check logs above."' ERR

# --- Deploy key -----------------------------------------------------------------
if [ -n "${DEPLOY_KEY:-}" ]; then
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
else
    warn "DEPLOY_KEY not set — git clone will only work if SSH keys are mounted or repo is public."
fi

# --- Clone or update MMTU code --------------------------------------------------
MMTU_REPO="${MMTU_GIT_URL:-git@github.com:junos-ai-org/MMTU.git}"
MMTU_REF="${MMTU_GIT_REF:-main}"

if [ -d /workspace/MMTU/.git ]; then
    log "Updating MMTU code (ref: ${MMTU_REF})..."
    cd /workspace/MMTU && git fetch origin && git checkout "$MMTU_REF" && git pull origin "$MMTU_REF" || true
    cd /
else
    log "Cloning MMTU code (ref: ${MMTU_REF}) from ${MMTU_REPO}..."
    git clone --depth 1 --branch "$MMTU_REF" "$MMTU_REPO" /workspace/MMTU
fi

log "Installing Python dependencies..."
pip install -q -r /workspace/MMTU/requirements.txt

# --- Model weights --------------------------------------------------------------
export HF_HOME="/workspace/.cache/huggingface"

T5GEMMA_MODEL="${T5GEMMA_MODEL_PATH:-google/t5gemma-9b-9b-ul2-it}"
log "Downloading model weights for ${T5GEMMA_MODEL}..."
huggingface-cli download "$T5GEMMA_MODEL"
log "  T5Gemma model weights ready."

# --- Ready banner ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  T5Gemma image ready"
echo "============================================================"
echo ""
echo "  cd /workspace/MMTU"
echo ""
echo "  # Smoke test"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-smoke.yaml"
echo ""
echo "  # Full baseline"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full.yaml"
echo ""
echo "  # Column permutation"
echo "  python projects/tabular-llms-research/run.py run configs/t5gemma-9b-full-colperm.yaml"
echo ""
echo "  # Update code mid-session (no restart needed)"
echo "  mmtu-update"
echo ""

exec sleep infinity
