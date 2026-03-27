# TTS Server Setup

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU | RTX 5070 Ti (12 GB VRAM, Blackwell / sm_120) or equivalent with 8+ GB VRAM |
| CUDA | 12.8+ (system-level, must match the PyTorch wheel) |
| Python | 3.11 (pinned; other versions untested) |
| faster-higgs-audio | Must be cloned locally before install (see below) |

The server targets **one model in VRAM at a time**. Recommended VRAM budget is 10 000 MB to leave headroom for the OS and driver.

---

## 2. Installation

> **Note — legacy single-venv sections below (2.1–2.9):** The current setup uses
> per-engine venvs managed by `scripts/setup_venvs.sh`. Skip to [section 2.10](#210-per-engine-venv-setup)
> for the recommended path. Only sections 2.2 (host venv) and 2.4 (server package install)
> are still required; the rest are kept for reference.

### 2.1 Clone faster-higgs-audio

The Higgs engine depends on this repo and it is not available on PyPI.

**Windows:**
```powershell
git clone https://github.com/sorbetstudio/faster-higgs-audio %USERPROFILE%\tmp\faster-higgs-audio
```

**Linux / WSL / dev container:**
```bash
git clone https://github.com/sorbetstudio/faster-higgs-audio /tmp/faster-higgs-audio
```

> **Important (Linux):** The default `HIGGS_REPO_PATH` in the worker falls back to a Windows
> path (`%USERPROFILE%\tmp\...`) when `USERPROFILE` is unset. On Linux you **must** set
> `HIGGS_REPO_PATH` explicitly in `.env` — see Section 4. Without it every Higgs request
> returns HTTP 500 with `ModuleNotFoundError: No module named 'boson_multimodal'`.

### 2.2 Create and activate the venv

```bash
python3.11 -m venv /workspaces/.venvs/tts_server
source /workspaces/.venvs/tts_server/bin/activate
```

### 2.3 Install PyTorch with CUDA 12.8 support

Install this **before** anything else. It will be overridden after chatterbox-tts is installed (see section 3).

```bash
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2.4 Install the server package

```bash
cd /workspaces/hub_1/tts_server
pip install -e .
```

This installs FastAPI, uvicorn, pydantic, soundfile, numpy, and the loose `torch>=2.0` pin declared in `requirements.txt`.

### 2.5 Install chatterbox-tts

```bash
pip install chatterbox-tts
```

**Important:** chatterbox-tts pins `torch==2.6.0`, which it will pull in here, downgrading the torch you installed in step 2.3. Fix this immediately in the next step.

### 2.6 Override torch for Blackwell (RTX 50-series)

RTX 5070 Ti (sm_120) requires torch 2.10.0+ built against CUDA 12.8. The torch==2.6.0 pulled in by chatterbox-tts does not include sm_120 kernels and will fail at runtime.

```bash
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

This overrides the chatterbox pin. chatterbox-tts itself runs correctly on torch 2.10.0 despite the declared pin.

### 2.7 Install faster-higgs-audio

```bash
pip install -e %USERPROFILE%\tmp\faster-higgs-audio
```

### 2.8 Install Higgs runtime dependencies

These are required by faster-higgs-audio but are not declared in its package metadata:

```bash
pip install "transformers>=4.45.1,<4.47.0" accelerate bitsandbytes \
    librosa vector-quantize-pytorch descript-audio-codec descript-audiotools \
    omegaconf langid jieba click loguru pyyaml dacite pandas pydub
```

### 2.9 Install validation dependencies (optional)

Required only if you plan to run STT-based quality validation:

```bash
pip install faster-whisper anthropic
```

---

## 2.10 Per-Engine Venv Setup

### Why separate venvs are needed

Each TTS engine requires a different version of `transformers` that conflicts with the others:

- **Higgs** requires `transformers<4.47.0`
- **Qwen3** requires `transformers>=4.57.3`
- **Chatterbox** bundles its own torch pin

These ranges cannot coexist in a single venv. The server therefore runs the FastAPI host process in a lightweight host venv and spawns each engine as a subprocess worker in its own isolated venv.

### Venv layout

All paths are relative to the repo root:

| Venv | Path | Engines | Key constraint |
|------|------|---------|----------------|
| Host | `.venv\` | FastAPI server | fastapi, uvicorn, pydantic — no torch, no engine packages |
| Chatterbox | `.venvs\chatterbox\` | chatterbox, chatterbox_full | chatterbox-tts |
| Higgs | `.venvs\higgs\` | higgs | transformers<4.47.0 |
| Qwen3 | `.venvs\qwen3\` | qwen3 | transformers>=4.57.3 |

### Creating all engine venvs (Windows)

Run once from the repo root after a fresh clone:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_venvs.ps1
```

This script:
- Creates/skips each venv (idempotent)
- Installs the appropriate requirements file into each
- Auto-detects Blackwell GPU (sm_12.x) and installs `torch==2.10.0+cu128` into engine venvs
- Clones faster-higgs-audio to `D:\tmp\faster-higgs-audio` if not already present

### Starting the server

Use `start_server.ps1` (or `start_server.sh` from the bash tool). This runs uvicorn as a native
Windows process to avoid VS Code port-forwarding interference:

```powershell
.\start_server.ps1                     # port 8765, 10000 MB VRAM (defaults)
.\start_server.ps1 -Port 8765 -VramMb 10000
```

The model manager automatically selects the correct engine venv when launching a subprocess worker.

---

## 3. Known Dependency Conflicts

### 3.1 chatterbox-tts pins torch==2.6.0 (no sm_120 support)

**Conflict:** chatterbox-tts declares `torch==2.6.0` as a hard requirement. torch 2.6.0 does not include CUDA kernels for Blackwell GPUs (compute capability sm_120, RTX 50-series).

**Resolution:** After installing chatterbox-tts, override with the cu128 wheel:

```bash
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

pip will warn about the broken chatterbox-tts requirement — this is expected and safe to ignore. chatterbox-tts functions correctly on torch 2.10.0.

### 3.2 protobuf version conflict (descript-audiotools vs onnx)

**Conflict:** `descript-audiotools` requires `protobuf<3.20` and `onnx` requires `protobuf>=4.25.1`. These ranges do not overlap — pip cannot satisfy both simultaneously.

**Resolution:** There is no clean resolution. Install whichever version pip lands on. Both packages work at runtime despite the declared incompatibility because neither exercises the protobuf features that differ across these major versions. Ignore pip's `ERROR: pip's dependency resolver does not currently take into account all packages that are installed` warning for this conflict.

---

## 4. Configuration

Copy the example below into a `.env` file at the repo root and edit the values to match your environment:

```bash
# .env

# VRAM budget in MB. 10000 is recommended for a 12 GB GPU to leave OS headroom.
AVAILABLE_VRAM_MB=10000

# Higgs quantization: 4 (4-bit), 8 (8-bit), or 0 (bf16, no quantization).
# 8-bit uses ~6.7 GB VRAM; 4-bit uses ~4.2 GB but takes longer to load.
HIGGS_QUANT_BITS=8

# Path to the cloned faster-higgs-audio repo (step 2.1).
# Windows default — works when USERPROFILE is set:
HIGGS_REPO_PATH=%USERPROFILE%\tmp\faster-higgs-audio
# Linux / WSL / dev container — MUST be set explicitly (no USERPROFILE fallback):
# HIGGS_REPO_PATH=/tmp/faster-higgs-audio

# HuggingFace model and tokenizer IDs for Higgs Audio v2.
HIGGS_MODEL_ID=bosonai/higgs-audio-v2-generation-3B-base
HIGGS_TOKENIZER_ID=bosonai/higgs-audio-v2-tokenizer

# Optional: enables LLM-adjudicated STT validation (Claude Haiku).
# Only needed if running tests/stt_validate.py with --llm-adjudicate.
ANTHROPIC_API_KEY=sk-ant-...
```

Additional variables (optional, defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOICES_DIR` | `./voices` | Directory for persisted voice embeddings |
| `IDLE_TIMEOUT_S` | `60` | Seconds of inactivity before the active engine is unloaded. Increase to `600` when running long multi-block renders (default 60s can fire mid-generation if model load + synthesis exceeds the timeout). |

---

## 5. Running the Server

The server is started using `start_server.ps1`, which uses the host venv (`.venv\`) — fastapi/uvicorn
only. Engine subprocess workers are launched in their own venvs automatically by the model manager.

```powershell
# From PowerShell (recommended):
.\start_server.ps1

# From Claude's bash tool (invokes PowerShell on Windows):
./start_server.sh
```

Default port is **8765**. VS Code port-forwarding can intercept processes started from bash/WSL;
the PowerShell script avoids this by spawning a native Windows process.

### Linux / WSL / dev container (no Windows PowerShell)

When running fully inside a Linux container without access to `powershell.exe`, start uvicorn
directly from the host venv:

```bash
cd /workspaces/hub_3/tts_server
AVAILABLE_VRAM_MB=10000 /workspaces/.venvs/tts_server/bin/python \
    -m uvicorn app.main:app --host 127.0.0.1 --port 8765 \
    >> tts_server.log 2>&1 &
```

**Required before starting:** ensure `HIGGS_REPO_PATH=/tmp/faster-higgs-audio` is in `.env`
(see Section 2.1). Without it the Higgs worker cannot find `boson_multimodal` and every Higgs
request returns HTTP 500.

Confirm startup is complete by polling the health endpoint:

```bash
until curl -s http://127.0.0.1:8765/health | grep -q '"status":"ok"'; do sleep 3; done
echo "Server ready"
```

Models are lazy-loaded on first request. Startup is fast (~15–20s for dep probes); the first `/tts` call will take 3–21 seconds while the model loads into VRAM.

## 5.5 Install voice fixtures

Voice fixtures are reference audio clips committed to `tests/voice_fixtures/` and required
by integration tests in this repo and dependent repos (notably **video_agent_long**). They
are not present in the runtime `voices/` directory on a fresh clone.

```bash
python tests/setup_test_voices.py
```

Run once after a fresh clone or container rebuild (before running any tests that use named voices).
The script is idempotent — already-installed voices are skipped.

**Cross-repo fixture dependency:** `video_agent_long` tests reference the `ragnar-narrator`
voice (a Higgs reference clip cloned here). If you are running video_agent_long integration
tests in a fresh container, this step is required even if you are not running tts_server tests
directly. See [Cross-repo fixtures](#cross-repo-fixtures) at the bottom of this document.

---

## 6. Verifying the Setup

### 6.1 Health check

```bash
curl http://localhost:8000/health
```

Expected response: HTTP 200 with a JSON body showing `"status": "ok"` and the current engine state (no model loaded at idle).

### 6.2 List available models

```bash
curl http://localhost:8000/models
```

Both `chatterbox` and `higgs` should appear. Check that their VRAM estimates fit within your `AVAILABLE_VRAM_MB` budget.

### 6.3 Chatterbox synthesis test

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a chatterbox test."}' \
  -o chatterbox_test.wav
```

Expect a valid WAV file (~1–3 seconds of audio). First call triggers model load (~3s).

### 6.4 Higgs synthesis test

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs",
    "text": "Hello, this is a higgs audio test.",
    "speaker_description": "Male voice, warm and conversational, moderate pace."
  }' \
  -o higgs_test.wav
```

Expect a valid WAV file. First call triggers model load (~12s for 8-bit, ~21s for 4-bit).

### 6.5 Full STT validation (optional)

Generate audio artifacts and validate transcription accuracy:

```bash
# Generate 24 WAV samples across both models
python tests/generate_artifacts.py

# Run STT validation (requires faster-whisper installed)
python tests/stt_validate.py \
  --artifacts-dir tests/artifacts/ \
  --manifest tests/manifest.json \
  -v
```

A correctly set-up server should pass at least 85% of test cases (21/24 on the reference hardware). Failures on specific proper nouns or brand names are normal model quality variance, not setup errors.

---

## Cross-repo fixtures

Some fixtures in this repo are consumed by other repos:

| Voice ID | Consumed by | Purpose |
|----------|-------------|---------|
| `ragnar-narrator` | `video_agent_long` integration tests | Higgs reference audio for consistent speaker identity across TTS blocks. voice_agent_long's `test_persona_higgs_integration.py` uses this voice ID. |

**Why this coupling exists:** Higgs does not produce a consistent speaker identity from
`speaker_description` alone — varying `scene_description` per block shifts the voice even
with a constant description and seed=0. Voice cloning (reference audio anchor) is required
for multi-block narration. The reference audio must be pre-cloned on the server before
video_agent_long tests can run.

**Setup order on a fresh machine (Windows):**
1. Clone faster-higgs-audio: `git clone https://github.com/sorbetstudio/faster-higgs-audio %USERPROFILE%\tmp\faster-higgs-audio`
2. Run venv setup: `powershell -ExecutionPolicy Bypass -File scripts\setup_venvs.ps1`
3. Start tts_server: `.\start_server.ps1`
4. Install fixtures: `.venv\Scripts\python tests\setup_test_voices.py`
5. Run video_agent_long tests (or tts_server tests)

**Setup order on a fresh machine (Linux / dev container):**
1. Clone faster-higgs-audio: `git clone https://github.com/sorbetstudio/faster-higgs-audio /tmp/faster-higgs-audio`
2. Add to `.env`: `HIGGS_REPO_PATH=/tmp/faster-higgs-audio`
3. Start tts_server: `AVAILABLE_VRAM_MB=10000 /workspaces/.venvs/tts_server/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 >> tts_server.log 2>&1 &`
4. Install fixtures: `/workspaces/.venvs/tts_server/bin/python tests/setup_test_voices.py`
5. Run video_agent_long tests (or tts_server tests) with `TTS_SERVER_URL=http://127.0.0.1:8765`

**Adding a new cross-repo fixture voice:** commit the reference audio to
`tests/voice_fixtures/<id>/`, document it in the table above, and update
`video_agent_long/CLAUDE.md` with the prerequisite.
