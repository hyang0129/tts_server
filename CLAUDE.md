# TTS Server

## Project overview
Consolidated FastAPI TTS server that serves multiple model backends (Chatterbox, Higgs) behind a unified API. Only one model is loaded in VRAM at a time; models are lazy-loaded on first request and auto-unloaded after 60s idle.

## Architecture
- **app/main.py** — FastAPI app, lifespan, all HTTP endpoints
- **app/model_manager.py** — ModelManager: lazy load/unload/swap engines, VRAM tracking, idle timer
- **app/engine_base.py** — TTSEngine ABC
- **app/engine_chatterbox.py** — Chatterbox Turbo wrapper
- **app/engine_higgs.py** — Higgs Audio wrapper (requires faster-higgs-audio repo)
- **app/voices.py** — Unified VoiceStore with compatible_models tracking

## Key commands
```bash
# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Lint
ruff check .

# Integration tests (requires running server on port 8000)
pytest tests/test_integration.py -v
pytest tests/test_integration.py -v -k chatterbox
pytest tests/test_integration.py -v -k higgs

# Generate test artifacts (audio samples for manual review)
python tests/generate_artifacts.py
```

## Hardware target
- NVIDIA RTX 5070 Ti Laptop GPU (12 GB VRAM, Blackwell / sm_120)
- Measured VRAM: Chatterbox ~4.2 GB, Higgs 8-bit ~6.7 GB, Higgs 4-bit ~4.2 GB
- One model at a time; swap takes 3-21s depending on model
- See `docs/vram_management.md` for full profiling data

## Environment variables
- `AVAILABLE_VRAM_MB` — VRAM budget in MB (default 12000). Recommended: 10000.
- `TTS_VOICES_DIR` — voice storage directory (default ./voices)
- `HIGGS_QUANT_BITS` — quantization bits for higgs (4, 8, or 0 for bf16)
- `HIGGS_REPO_PATH` — path to faster-higgs-audio repo (default /tmp/faster-higgs-audio)
- `HIGGS_MODEL_ID` — HuggingFace model ID for higgs
- `HIGGS_TOKENIZER_ID` — HuggingFace tokenizer ID for higgs

## Conventions
- Python 3.11, type hints throughout
- Ruff for linting (line-length 100)
- One model in VRAM at a time; swap on request
- Default model is chatterbox (when model field omitted from /tts)
- Chatterbox and Higgs use separate venvs due to incompatible torch versions

## API contract
- The API is consumed by video_agent and potentially other repos. Changes to request/response shapes, status codes, or headers can break clients.
- Adding new optional fields or headers is safe; removing, renaming, or retyping existing fields is breaking.
- Always update `docs/api.md` when the contract changes.
- See `docs/api.md` for full endpoint documentation and `docs/vram_management.md` for VRAM profiling.

## API endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | /tts | Synthesize speech (model field selects engine) |
| POST | /voices/clone | Clone voice from reference audio |
| POST | /voices/blend | Blend two voices (chatterbox only) |
| GET | /voices | List voices (?model= filter) |
| GET | /voices/{id} | Voice details with compatible_models |
| DELETE | /voices/{id} | Delete a voice |
| GET | /health | Engine status + active model |
| GET | /models | List models with load status + VRAM |
