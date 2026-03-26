# TTS Server

## Project overview
Consolidated FastAPI TTS server that serves multiple model backends (Chatterbox, Higgs) behind a unified API. Only one model is loaded in VRAM at a time; models are lazy-loaded on first request and auto-unloaded after 60s idle.

## Architecture
- **app/main.py** — FastAPI app, lifespan, all HTTP endpoints
- **app/model_manager.py** — ModelManager: lazy load/unload/swap engines, VRAM tracking, idle timer
- **app/engine_base.py** — TTSEngine ABC
- **app/engine_chatterbox.py** — Chatterbox Turbo wrapper
- **app/engine_chatterbox_full.py** — Chatterbox Full (original) wrapper (same chatterbox package, ResembleAI/chatterbox model)
- **app/engine_higgs.py** — Higgs Audio wrapper (requires faster-higgs-audio repo)
- **app/engine_qwen3.py** — Qwen3-TTS wrapper (requires qwen-tts package)
- **app/voices.py** — Unified VoiceStore with compatible_models tracking
- **workers/worker_protocol.py** — JSON-RPC protocol (stdlib-only)
- **workers/chatterbox_worker.py** — Chatterbox Turbo subprocess worker
- **workers/chatterbox_full_worker.py** — Chatterbox Full subprocess worker
- **workers/higgs_worker.py** — Higgs subprocess worker
- **workers/qwen3_worker.py** — Qwen3 subprocess worker

## Venv layout

The host process and each engine run in separate isolated venvs (paths relative to repo root):

| Venv | Path | Engines | Key constraint |
|------|------|---------|----------------|
| Host | `.venv\` | FastAPI server | fastapi, uvicorn, pydantic — no torch, no engine packages |
| Chatterbox | `.venvs\chatterbox\` | chatterbox, chatterbox_full | chatterbox-tts |
| Higgs | `.venvs\higgs\` | higgs | transformers<4.47.0 |
| Qwen3 | `.venvs\qwen3\` | qwen3 | transformers>=4.57.3 |

One-time setup to create all engine venvs (Windows):

```powershell
# Run from repo root in PowerShell
powershell -ExecutionPolicy Bypass -File scripts\setup_venvs.ps1
```

## Starting the server (agent instructions)

The agent can and should start the server autonomously when needed (e.g. before running
generate_artifacts.py or integration tests). Use the host venv (`.venv\`) to start the server
via `start_server.sh` — it contains fastapi and uvicorn. Each engine subprocess is launched in
its own engine venv automatically by the model manager.

```bash
# Start via the wrapper script (recommended — runs as a native Windows process on port 8765)
./start_server.sh

# Install voice fixtures (required for named-voice tests — run once after setup)
.venv/Scripts/python tests/setup_test_voices.py
```

The `.env` file in the repo root is auto-loaded by the server; it contains `HF_TOKEN` and
`ANTHROPIC_API_KEY`. For qwen3 Base voice cloning, the default `QWEN3_MODEL_ID` is already
set to the Base model — no override needed.

## Key commands
```bash
# Run the server (Windows — native process, port 8765)
./start_server.sh

# Install tracked voice fixtures (run once after fresh clone)
.venv/Scripts/python tests/setup_test_voices.py

# Lint
ruff check .

# Integration tests (requires running server on port 8765)
pytest tests/test_integration.py -v
pytest tests/test_integration.py -v -k chatterbox
pytest tests/test_integration.py -v -k higgs

# Generate test artifacts (audio samples for manual review)
python tests/generate_artifacts.py

# STT validation (also serves as model setup health check)
python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json -v
```

## Voice fixtures

Named voices required by integration tests are tracked in `tests/voice_fixtures/<voice_id>/`.
The runtime `voices/` directory is gitignored. After a fresh clone or container rebuild, install
the fixtures before running tests:

```bash
python tests/setup_test_voices.py
```

This copies fixture voices into the runtime `voices/` directory. The script is idempotent —
already-installed voices are skipped.

### Tracked fixture voices

| Voice ID | Model | Description |
|----------|-------|-------------|
| `higgs-sable` | higgs | Sable, Keeper of the Akashic Archives persona voice. Measured, dry female narrator with quiet authority. WAV sourced from `config/personas/akashic_archives/voice_ref.wav` in video_agent_long (sha256: 2e0563b2…). Used by `test_higgs_drift_integration.py` to reproduce voice identity drift from video_agent_long#158. |
| `ragnar-narrator` | higgs, chatterbox, chatterbox_full, qwen3 | Alias for the same WAV as `higgs-sable` (identical bytes). Kept for backward compatibility with video_agent_long integration tests. See video_agent_long#183 for migration to `higgs-sable`. |

### Adding a new fixture voice

1. Generate the reference audio (keep under 30s): `POST /tts` or `POST /voices/clone`
2. Copy files from `voices/<id>/` → `tests/voice_fixtures/<id>/`
3. Commit the fixture directory (reference.wav, reference.txt, metadata.json)
4. Document it in the table above

## Qwen3 Base voice cloning: reference text requirements

When cloning a voice for the Base model, `reference_text` must match **only the first
`_MAX_REF_SECONDS` (8s) of audio**, because the engine trims the reference audio to 8s
before encoding. Providing the full transcript of a longer clip causes a mismatch that
makes the model prepend the reference transcript to generated audio.

To get an accurate 8s transcript, transcribe the trimmed audio with Whisper:
```python
import scipy.io.wavfile, tempfile
from tests.stt_validate import transcribe_wav
sr, data = scipy.io.wavfile.read("voices/<id>/reference.wav")
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    scipy.io.wavfile.write(f.name, sr, data[:int(8.0 * sr)])
    text = transcribe_wav(f.name)  # use this as reference_text
```

If you update `reference.txt`, delete `qwen3_prompt.pkl` (and any downstream blend pkls)
so the engine recomputes the prompt on the next request.

## STT validation as a setup health check
The STT validation (`tests/stt_validate.py`) serves dual purpose: it checks transcription accuracy AND confirms a model is correctly set up. A model passing ≥ 85% of its test cases is considered correctly installed. Baseline: 21/24 pass (87.5%) on RTX 5070 Ti.

## Hardware target
- NVIDIA RTX 5070 Ti Laptop GPU (12 GB VRAM, Blackwell / sm_120)
- Measured VRAM: Chatterbox ~4.2 GB, Chatterbox Full ~4.7 GB (estimated), Higgs 8-bit ~6.7 GB, Higgs 4-bit ~4.2 GB
- One model at a time; swap takes 3-21s depending on model
- See `docs/vram_management.md` for full profiling data

## Environment variables
- `AVAILABLE_VRAM_MB` — VRAM budget in MB (default 12000). Recommended: 10000.
- `TTS_VOICES_DIR` — voice storage directory (default ./voices)
- `HIGGS_QUANT_BITS` — quantization bits for higgs (4, 8, or 0 for bf16)
- `HIGGS_ATTN_IMPL` — attention implementation for higgs (`flash_attention_2` default, `sdpa`, or `eager`). `flash_attention_2` requires `flash_attn` which is installed automatically by `scripts/setup_venvs.sh`. `sdpa` uses torch's built-in SDPA kernel (no extra packages — fallback if flash_attn build fails).
- `HIGGS_REPO_PATH` — path to faster-higgs-audio repo (default %USERPROFILE%\tmp\faster-higgs-audio)
- `HIGGS_MODEL_ID` — HuggingFace model ID for higgs
- `HIGGS_TOKENIZER_ID` — HuggingFace tokenizer ID for higgs
- `CB_FULL_VRAM_MB` — override VRAM budget estimate for chatterbox_full (default 4700)
- `QWEN3_MODEL_ID` — HuggingFace model ID for qwen3 (default Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- `QWEN3_DTYPE` — weight dtype for qwen3 (bfloat16 or float16, default bfloat16)
- `QWEN3_VRAM_MB` — override VRAM budget estimate for qwen3 (default 5500)

## Conventions
- Python 3.11, type hints throughout
- Ruff for linting (line-length 100)
- One model in VRAM at a time; swap on request
- Default model is chatterbox (when model field omitted from /tts)
- Each engine (Chatterbox, Higgs, Qwen3) runs in its own venv as a subprocess worker to isolate incompatible transformers versions.

## Generated audio files
- **`tests/artifacts/`** — manifest-tracked WAV samples produced by `generate_artifacts.py`. Git-ignored; regenerate with `python tests/generate_artifacts.py`.
- **`tests/samples/`** — ad-hoc / exploratory audio (one-off voice tests, blending experiments, etc.). Also git-ignored. When generating temp audio during development, save it here rather than `/tmp` so it persists across sessions and stays co-located with the repo. Never save generated audio to system `/tmp`.

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
