#!/usr/bin/env bash
# setup_venvs.sh — Create and populate per-engine venvs for tts_server.
#
# Venv layout:
#   /workspaces/.venvs/tts_server          — host (FastAPI server, no engine packages)
#   /workspaces/.venvs/tts_server-chatterbox — Chatterbox Turbo + Full engines
#   /workspaces/.venvs/tts_server-higgs    — Higgs Audio engine (transformers<4.47.0)
#   /workspaces/.venvs/tts_server-qwen3    — Qwen3-TTS engine (transformers>=4.57.3)
#
# This script is idempotent: it skips venv creation if the directory already
# exists, but always re-runs pip install so requirements stay current.
#
# Usage:
#   bash scripts/setup_venvs.sh
#
# Requirements files are expected in the repo root (next to this script's parent dir).

set -e

PYTHON=python3.11
VENVS_ROOT=/workspaces/.venvs
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Detect Blackwell (sm_120) GPU via nvidia-smi (no torch required).
IS_BLACKWELL=0
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q "^12\."; then
    IS_BLACKWELL=1
fi

# Helper: upgrade torch to cu128 wheel if Blackwell GPU is present.
# All engine venvs need this because default pip torch pulls cu13 libs that
# require CUDA 13.0+ drivers, but RTX 50-series runs CUDA 12.x drivers.
install_blackwell_torch() {
    local name=$1
    if [ "$IS_BLACKWELL" = "1" ]; then
        echo "[override] Blackwell GPU — installing torch==2.10.0 cu128 into '$name'..."
        "$VENVS_ROOT/$name/bin/pip" install \
            torch==2.10.0 torchaudio==2.10.0 \
            --index-url https://download.pytorch.org/whl/cu128 --quiet
        echo "[ok]   torch cu128 installed for $name"
    fi
}

echo "==> tts_server venv setup"
echo "    Repo root : $REPO_ROOT"
echo "    Venvs root: $VENVS_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Helper: create venv only if it doesn't already exist
# ---------------------------------------------------------------------------
create_venv() {
    local name=$1
    local path="$VENVS_ROOT/$name"
    if [ -d "$path" ]; then
        echo "[skip] Venv '$name' already exists at $path"
    else
        echo "[create] Creating venv '$name' at $path"
        $PYTHON -m venv "$path"
        echo "[ok]   Created $path"
    fi
}

# ---------------------------------------------------------------------------
# Helper: install requirements into a venv
# ---------------------------------------------------------------------------
install_reqs() {
    local name=$1
    local reqs=$2
    local pip="$VENVS_ROOT/$name/bin/pip"
    echo "[install] Installing $reqs into '$name'..."
    "$pip" install --upgrade pip --quiet
    "$pip" install -r "$reqs"
    echo "[ok]   $name deps installed"
}

# ---------------------------------------------------------------------------
# 1. Host venv — reuse existing tts_server venv (already present in container)
#    Update it to host-only deps; engine packages are not installed here.
# ---------------------------------------------------------------------------
echo "--- Host venv (tts_server) ---"
create_venv tts_server
install_reqs tts_server "$REPO_ROOT/requirements-host.txt"
echo ""

# ---------------------------------------------------------------------------
# 2. Chatterbox venv
# ---------------------------------------------------------------------------
echo "--- Chatterbox venv (tts_server-chatterbox) ---"
create_venv tts_server-chatterbox
install_reqs tts_server-chatterbox "$REPO_ROOT/requirements-chatterbox.txt"
# chatterbox-tts pins torch==2.6.0; Blackwell needs cu128 override.
install_blackwell_torch tts_server-chatterbox
echo ""

# ---------------------------------------------------------------------------
# 3. Higgs venv
#    faster-higgs-audio is not on PyPI; it is accessed at runtime via sys.path
#    from /tmp/faster-higgs-audio (see HIGGS_REPO_PATH env var).
#    Clone it if not already present:
# ---------------------------------------------------------------------------
echo "--- Higgs venv (tts_server-higgs) ---"
create_venv tts_server-higgs
install_reqs tts_server-higgs "$REPO_ROOT/requirements-higgs.txt"
install_blackwell_torch tts_server-higgs
# flash-attn must be installed after torch (links against its CUDA extensions).
# --no-build-isolation ensures it picks up the already-installed torch headers.
echo "[install] Installing flash-attn into tts_server-higgs (compiles from source, may take ~5–10 min)..."
"$VENVS_ROOT/tts_server-higgs/bin/pip" install flash-attn --no-build-isolation --quiet \
    && echo "[ok]   flash-attn installed" \
    || echo "[warn] flash-attn build failed — Higgs will fall back to SDPA attention"

HIGGS_REPO_PATH="${HIGGS_REPO_PATH:-/tmp/faster-higgs-audio}"
if [ -d "$HIGGS_REPO_PATH" ]; then
    echo "[skip] faster-higgs-audio repo already present at $HIGGS_REPO_PATH"
else
    echo "[clone] Cloning faster-higgs-audio to $HIGGS_REPO_PATH"
    git clone https://github.com/sorbetstudio/faster-higgs-audio "$HIGGS_REPO_PATH"
    echo "[ok]   Cloned to $HIGGS_REPO_PATH"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Qwen3 venv
# ---------------------------------------------------------------------------
echo "--- Qwen3 venv (tts_server-qwen3) ---"
create_venv tts_server-qwen3
install_reqs tts_server-qwen3 "$REPO_ROOT/requirements-qwen3.txt"
install_blackwell_torch tts_server-qwen3
echo ""

echo "==> All venvs ready."
echo ""
echo "Activate a venv with:"
echo "  source $VENVS_ROOT/tts_server/bin/activate          # host"
echo "  source $VENVS_ROOT/tts_server-chatterbox/bin/activate"
echo "  source $VENVS_ROOT/tts_server-higgs/bin/activate"
echo "  source $VENVS_ROOT/tts_server-qwen3/bin/activate"
