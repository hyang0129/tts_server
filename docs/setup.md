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

```bash
git clone https://github.com/sorbetstudio/faster-higgs-audio /tmp/faster-higgs-audio
```

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
pip install -e /tmp/faster-higgs-audio
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

| Venv | Path | Engines | Key constraint |
|------|------|---------|----------------|
| Host | `/workspaces/.venvs/tts_server/` | FastAPI server | fastapi, uvicorn, pydantic — no torch, no engine packages |
| Chatterbox | `/workspaces/.venvs/tts_server-chatterbox/` | chatterbox, chatterbox_full | chatterbox-tts |
| Higgs | `/workspaces/.venvs/tts_server-higgs/` | higgs | transformers<4.47.0 |
| Qwen3 | `/workspaces/.venvs/tts_server-qwen3/` | qwen3 | transformers>=4.57.3 |

### Creating all engine venvs

Run once from the repo root after a fresh clone or container rebuild:

```bash
bash scripts/setup_venvs.sh
```

This script creates and populates each engine venv. The host venv (`/workspaces/.venvs/tts_server/`) must already exist (created in step 2.2) — the script only creates the engine venvs.

### Starting the server

Always start the server using the host venv. The model manager automatically selects the correct engine venv when launching a subprocess worker.

```bash
AVAILABLE_VRAM_MB=10000 /workspaces/.venvs/tts_server/bin/uvicorn app.main:app \
    --host 0.0.0.0 --port 8000
```

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
HIGGS_REPO_PATH=/tmp/faster-higgs-audio

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

---

## 5. Running the Server

The server is started using the **host venv** (`/workspaces/.venvs/tts_server/`), which contains only fastapi and uvicorn. Engine subprocess workers are launched in their own venvs automatically.

```bash
cd /workspaces/hub_3/tts_server
AVAILABLE_VRAM_MB=10000 /workspaces/.venvs/tts_server/bin/uvicorn app.main:app \
    --host 0.0.0.0 --port 8000
```

Models are lazy-loaded on first request. Startup is fast; the first `/tts` call will take 3–21 seconds while the model loads into VRAM.

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

**Setup order in a fresh container:**
1. Start tts_server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Install fixtures: `python tests/setup_test_voices.py`
3. Run video_agent_long tests (or tts_server tests)

**Adding a new cross-repo fixture voice:** commit the reference audio to
`tests/voice_fixtures/<id>/`, document it in the table above, and update
`video_agent_long/CLAUDE.md` with the prerequisite.
