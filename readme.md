# TTS Server

Consolidated FastAPI server serving multiple TTS model backends behind a unified API. Currently supports **Chatterbox Turbo** and **Higgs Audio v2**.

## Features

- **Multi-model**: Chatterbox (350M, fast) and Higgs Audio (3B, expressive) behind one API
- **Lazy loading**: Models load on first request, no VRAM used at idle
- **Auto-unload**: Models unloaded after 60s idle to free VRAM
- **VRAM-aware**: Only enables models that fit within configured VRAM budget
- **Model swap**: Transparent swap when switching between models (unload → load)
- **Voice management**: Clone, blend (chatterbox), and description-only generation (higgs)

## Setup

### Prerequisites

- NVIDIA GPU with CUDA 12.8+ support (tested on RTX 5070 Ti, 12GB)
- Python 3.11
- [faster-higgs-audio](https://github.com/sorbetstudio/faster-higgs-audio) cloned locally (for higgs engine)

### Install

```bash
# Clone faster-higgs-audio (needed for higgs engine)
git clone https://github.com/sorbetstudio/faster-higgs-audio /tmp/faster-higgs-audio

# Create venv
python3.11 -m venv /workspaces/.venvs/tts_server
source /workspaces/.venvs/tts_server/bin/activate

# Install torch with CUDA 12.8 support
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install server dependencies
pip install -e .

# Install model backends
pip install chatterbox-tts                          # Chatterbox Turbo
pip install -e /tmp/faster-higgs-audio              # Higgs Audio

# Install higgs runtime deps (not declared by the package)
pip install "transformers>=4.45.1,<4.47.0" accelerate bitsandbytes \
    librosa vector-quantize-pytorch descript-audio-codec descript-audiotools \
    omegaconf langid jieba click loguru pyyaml dacite pandas pydub

# Install validation deps (optional, for STT testing)
pip install faster-whisper anthropic
```

### Configure

Copy and edit `.env`:

```bash
# .env
AVAILABLE_VRAM_MB=10000          # VRAM budget (MB). 10000 recommended for 12GB GPU.
HIGGS_QUANT_BITS=8               # 4, 8, or 0 (bf16, no quantization)
HIGGS_REPO_PATH=/tmp/faster-higgs-audio
HIGGS_MODEL_ID=bosonai/higgs-audio-v2-generation-3B-base
HIGGS_TOKENIZER_ID=bosonai/higgs-audio-v2-tokenizer
ANTHROPIC_API_KEY=sk-ant-...     # Optional, for LLM-adjudicated STT validation
```

### Run

```bash
cd /workspaces/hub/repos/tts_server
source /workspaces/.venvs/tts_server/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Quick Start

```bash
# Chatterbox TTS (default model)
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}' -o output.wav

# Higgs TTS (description-only voice)
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"model": "higgs", "text": "Hello world!", "speaker_description": "Male, warm voice, moderate pace"}' \
  -o output.wav

# Check available models
curl http://localhost:8000/models

# Health check
curl http://localhost:8000/health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tts` | Synthesize speech (`model` field selects engine, default: chatterbox) |
| POST | `/voices/clone` | Clone voice from reference audio |
| POST | `/voices/blend` | Blend two voices (chatterbox only) |
| GET | `/voices` | List voices (filterable by `?model=`) |
| GET | `/voices/{id}` | Voice details |
| DELETE | `/voices/{id}` | Delete a voice |
| GET | `/health` | Server + engine status |
| GET | `/models` | List models with load status and VRAM |

See [docs/api.md](docs/api.md) for full request/response schemas.

## VRAM Usage (RTX 5070 Ti)

| Model | Loaded | Peak | Load Time |
|-------|--------|------|-----------|
| Chatterbox Turbo | 4.2 GB | 4.4 GB | ~3s |
| Higgs 8-bit | 6.7 GB | 7.1 GB | ~12s |
| Higgs 4-bit | 4.2 GB | 5.0 GB | ~21s |

See [docs/vram_management.md](docs/vram_management.md) for full profiling data.

## Testing

```bash
# Run integration tests (server must be running on port 8000)
pytest tests/test_integration.py -v

# Generate test artifacts (24 WAVs: 6 texts x 4 voices across both models)
python tests/generate_artifacts.py --dry-run          # preview
python tests/generate_artifacts.py                     # generate
python tests/generate_artifacts.py --validate          # generate + STT validation

# Run STT validation on existing artifacts
python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json -v
```

## Architecture

```
POST /tts {model: "higgs", text: "...", speaker_description: "..."}
     │
     ▼
  ModelManager
     ├─ Is higgs loaded? No → unload chatterbox → load higgs
     └─ higgs.generate(text, ...) → WAV bytes
     │
     ▼
  Response (audio/wav)
     │
     ▼
  Idle Monitor (background task)
     └─ No requests for 60s? → unload model → free VRAM
```
