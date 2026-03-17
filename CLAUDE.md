# TTS Server

## Project overview
Consolidated FastAPI TTS server that serves multiple model backends (Chatterbox, Higgs) behind a unified API. Only one model is loaded in VRAM at a time; models are lazy-loaded on first request and auto-unloaded after 60s idle.

## Architecture
- **app/main.py** — FastAPI app, lifespan, all HTTP endpoints
- **app/model_manager.py** — ModelManager: lazy load/unload/swap engines, VRAM tracking, idle timer
- **app/engine_base.py** — TTSEngine ABC
- **app/engine_chatterbox.py** — Chatterbox Turbo wrapper
- **app/engine_higgs.py** — Higgs Audio wrapper (requires faster-higgs-audio repo)
- **app/engine_qwen3.py** — Qwen3-TTS wrapper (requires qwen-tts package)
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

# STT validation (also serves as model setup health check)
python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json -v
```

## STT validation as a setup health check
The STT validation (`tests/stt_validate.py`) serves dual purpose: it checks transcription accuracy AND confirms a model is correctly set up. A model passing ≥ 85% of its test cases is considered correctly installed. Baseline: 21/24 pass (87.5%) on RTX 5070 Ti.

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
- `QWEN3_MODEL_ID` — HuggingFace model ID for qwen3 (default Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- `QWEN3_DTYPE` — weight dtype for qwen3 (bfloat16 or float16, default bfloat16)
- `QWEN3_VRAM_MB` — override VRAM budget estimate for qwen3 (default 5500)

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

## Adding a new TTS engine

When a user asks to add or integrate a new TTS model, follow this process exactly.
The full rationale is in `docs/agentic-qwen3-tts-workflow.md` — read it before starting.

### Step 1 — Require a source repo URL

A GitHub (or other) repo URL is mandatory before any implementation work begins.

- **If the user provided a URL:** proceed to Step 2.
- **If the user provided only a name (e.g. "add Kokoro TTS"):** do NOT guess or
  assume a repo. Instead, web-search for the model name and return 2–4 candidate
  repos with their URLs, star counts, and a one-line description of each. Ask the
  user to confirm which one to use. Do not proceed until a URL is confirmed.

### Step 2 — Create tracking issue and draft PR

Before any implementation begins:

1. Create a GitHub issue: `feat: add <ModelName> TTS engine`
   - Source repo URL
   - Brief description of the model
   - Note that implementation is in progress

2. Create a feature branch (`feat/add-<name>-tts`) and open a **draft PR** referencing
   the issue. This makes the work visible and trackable from the start.
   The PR will be marked ready for review once the Validator is fully green.

### Step 3 — Spin up the agent team

Launch the following agents. The Researcher runs first; all others are unblocked
once it delivers its memo.

| Agent | Task |
|-------|------|
| **Researcher** | Fetch the confirmed repo URL and any linked HuggingFace model card. Extract: model ID, Python inference API (exact class/function signatures), dependencies, VRAM footprint, supported paralinguistic tags, voice control surface (cloning, style, description), sampling parameters. Produce a feature delta table against Chatterbox and Higgs. Cite every source URL. |
| **Implementer** | Using the Researcher memo, write `app/engine_<name>.py` subclassing `TTSEngine`. Register in `model_manager.py`. Expose all features (including any new paralinguistic tags or voice params) as optional fields in `main.py`. Update `voices.py` `compatible_models` if cloning is supported. |
| **Test writer** | Add engine test cases to `tests/test_integration.py`. Add manifest entries to `generate_artifacts.py`. Exclude the `long` text fixture from this engine's STT validation threshold (it is a known edge case with military proper nouns that cause STT noise for all models). |
| **Docs updater** | Update `README.md` VRAM table, `docs/api.md` with new model name and any new params, and this `CLAUDE.md` under Environment variables and Architecture. |
| **Validator** | Run `ruff check .`. Confirm all five `TTSEngine` abstract methods are implemented. Confirm no existing engine tests are broken. Run `python tests/generate_artifacts.py --model <name>` to produce WAV artifacts, then run STT validation (`python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json`) for the new engine (≥ 85% pass rate, excluding `long` fixtures). Report pass/fail on every checklist item. |

### Step 4 — Agent self-review (before human handoff)

The Validator must confirm every item below before surfacing the work to the user:

```
[ ] engine_<name>.py subclasses TTSEngine with all 5 methods
[ ] unload() calls gc.collect() + torch.cuda.empty_cache()
[ ] model_manager.py — engine is registered
[ ] main.py — engine-specific params are optional fields with defaults
[ ] voices.py — compatible_models updated if cloning is supported
[ ] New paralinguistic tags (if any) are documented in docs/api.md
[ ] Env vars documented in README and CLAUDE.md
[ ] docs/api.md updated
[ ] test_integration.py — test cases present
[ ] generate_artifacts.py — manifest entries present, long fixture excluded
[ ] ruff check . — PASSED
[ ] generate_artifacts.py — `python tests/generate_artifacts.py --model <name>` run successfully
[ ] STT validation — ≥ 85% pass (excluding long fixtures)
[ ] Chatterbox and Higgs existing tests unaffected
```

If any item fails, the relevant agent fixes it before escalating.

### Step 5 — Mark PR ready for review

Once the Validator is fully green, update the draft PR (created in Step 2):
- Mark it ready for review (remove draft status)
- Update the PR body with the Validator checklist (all green), the feature delta
  summary from the Researcher memo, and instructions for the reviewer to listen
  to the WAV artifacts before approving

### Step 6 — Human review

Present the user with:
- The PR URL
- The WAV files generated by the Validator step (in `tests/artifacts/`). If not already generated, run `python tests/generate_artifacts.py --model <name>` now before presenting.
- One-line summary of any new features exposed

The human listens to the audio and approves or rejects the PR.
**Do not merge without explicit human approval.**
Approved → human merges (or explicitly asks Claude to merge).
Rejected → describe the problem, close the PR, fix, open a new one.
