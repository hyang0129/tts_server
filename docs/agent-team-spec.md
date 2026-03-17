# Agent Team Spec: New TTS Engine Integration

Canonical team structure for adding a TTS engine to this server.
Referenced by `CLAUDE.md` and `agentic-qwen3-tts-workflow.md`.

---

## Execution Order

```
[Issue + Draft PR] ──► [Researcher] ──► [Implementer] ──┐
                                                          ├──► [Validator] ──► [PR ready for review] ──► human audio review
                                    ──► [Test Writer]   ──┤
                                    ──► [Docs Updater]  ──┘
```

Issue and draft PR are created first, before any implementation — work is visible
and trackable from day one. Researcher runs next and is blocking. Implementer,
Test Writer, and Docs Updater run in parallel once the Researcher memo is
available. Validator runs last and marks the PR ready for review once all checks
are green.

---

## Agent Definitions

### Researcher
**Runs:** first, alone
**Input:** confirmed GitHub repo URL (+ linked HuggingFace model card if present)
**Fetch targets:**
- Model ID (HuggingFace path or local checkpoint format)
- Python inference API — exact class names, constructor args, generate call signature
- Required pip packages and version constraints
- Estimated VRAM (bf16 baseline + any quantization options)
- Supported paralinguistic tags (syntax, full tag vocabulary)
- Voice control surface: cloning (ref audio? transcript?), speaker description, style params
- Sampling parameters: which of temperature / top_p / top_k / seed / others are supported

**Output:** a written memo containing:
1. All of the above with cited source URLs
2. Feature delta table — Qwen3 vs Chatterbox vs Higgs (or current engine set)
3. Any integration gotchas (e.g. custom sys.path injection, non-standard output format)

**Must not:** write any code or modify any files.

---

### Implementer
**Runs:** after Researcher, in parallel with Test Writer and Docs Updater
**Input:** Researcher memo
**Writes:**
- `app/engine_<name>.py` — `TTSEngine` subclass (use `engine_higgs.py` as structural template)
- `app/model_manager.py` — register new engine in engines list
- `app/main.py` — expose new params as `Optional` fields with defaults in the `/tts` request body
- `app/voices.py` — add engine name to `compatible_models` if cloning is supported

**Constraints:**
- All 5 abstract methods implemented: `load`, `unload`, `generate`, `is_loaded`, `deps_available`
- `unload()` must call `gc.collect()` and `torch.cuda.empty_cache()`
- Blocking inference wrapped in `asyncio.run_in_executor`
- New params must be optional with defaults — never break existing callers
- VRAM estimate exposed via `estimated_vram_mb` class attribute; env var override required if quantization is supported (follow `HIGGS_QUANT_BITS` pattern)
- All new paralinguistic tags documented inline and flagged to Docs Updater

---

### Test Writer
**Runs:** after Researcher, in parallel with Implementer and Docs Updater
**Input:** Researcher memo
**Writes:**
- `tests/test_integration.py` — add engine test cases (at minimum: basic synthesis, model listing, health check)
- `tests/generate_artifacts.py` — add manifest entries for new engine across standard text fixtures

**Constraints:**
- Exclude the `long` text fixture from STT validation for the new engine (WWII armor passage — known STT noise edge case, not a model quality signal)
- If the engine supports voice cloning, include at least one cloning test case
- If the engine exposes new paralinguistic tags, include at least one expressive test case using those tags
- Do not modify existing Chatterbox or Higgs test cases

---

### Docs Updater
**Runs:** after Researcher, in parallel with Implementer and Test Writer
**Input:** Researcher memo
**Writes:**
- `README.md` — VRAM table row, setup instructions for new model, env var table entries
- `docs/api.md` — new model name under supported models, new optional request params with types and defaults
- `CLAUDE.md` — env var entries, Architecture file list entry for new engine file

**Constraints:**
- Only add to existing tables/sections — do not restructure documents
- Flag any new paralinguistic tags in `docs/api.md` with a note on which engines support them

---

### Validator
**Runs:** last, after Implementer + Test Writer + Docs Updater all complete
**Input:** all modified files
**Executes:**
```bash
ruff check .
```
**Checks (static):**
```
[ ] engine_<name>.py subclasses TTSEngine with all 5 methods
[ ] unload() calls gc.collect() + torch.cuda.empty_cache()
[ ] model_manager.py — engine instantiated and registered
[ ] main.py — new params are Optional with defaults
[ ] voices.py — compatible_models updated if cloning supported
[ ] New paralinguistic tags documented in docs/api.md
[ ] Env vars in README and CLAUDE.md
[ ] test_integration.py — new test cases present
[ ] generate_artifacts.py — manifest entries present, long fixture excluded
[ ] ruff check . — PASSED
[ ] Chatterbox and Higgs test case counts unchanged
```
**Executes (runtime, if GPU available):**
```bash
python tests/generate_artifacts.py   # generate qwen3 WAVs
python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json -v
# pass threshold: ≥ 85%, excluding long_ prefixed files
```

**Output:** checklist with PASS/FAIL per item. If any item fails, the Validator
routes back to the responsible agent (Implementer, Test Writer, or Docs Updater)
with the specific failure. Does not escalate to human until all items are green.

---

## Human Handoff

Once Validator reports all green:

1. **Mark the draft PR ready for review.** Update the PR description to include:
   - Link to source repo / HuggingFace model card
   - Feature delta table (from Researcher memo)
   - Validator checklist with all items marked green
   - Instructions for the reviewer: "Listen to the WAV files in `tests/artifacts/` for the new engine before approving"

2. **Present to the user:**
   - PR URL
   - The generated WAV files from `tests/artifacts/` for the new engine
   - One-line summary of any new features (e.g. "Qwen3 adds `[whisper]` and `[gasp]` tags")

The human listens to the audio and reviews the PR.
**Approved** → merge (human merges, or explicitly asks Claude to merge).
**Rejected** → user describes the specific problem; close the PR, route back to Implementer, open a new PR once fixed.

**Claude must not merge the PR without explicit human approval.**
