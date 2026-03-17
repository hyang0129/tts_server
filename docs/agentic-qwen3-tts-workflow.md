# Agentic Workflow: Adding Qwen3-TTS to tts_server

**Scope:** Can Claude autonomously implement a new TTS engine (Qwen3) in tts_server,
given only CLAUDE.md context and a single human pass/fail review at the end?

---

## 1. The Workflow

### What the user says
> "Add Qwen3-TTS as a model to the server."

(No further specification. No API discussion. No install instructions.)

### What Claude must infer from CLAUDE.md + codebase
1. Read `repos/tts_server/CLAUDE.md` — understand the architecture, engine ABC, naming conventions, VRAM budget, venv, linting rules.
2. Read existing engines (`engine_chatterbox.py`, `engine_higgs.py`) as implementation templates.
3. Research Qwen3-TTS: model ID, install path, inference API, VRAM footprint.
4. Identify all touch points: new engine file, `model_manager.py` registration, `main.py` parameter exposure, `voices.py` `compatible_models`, `.env` docs, integration tests, STT artifact config.
5. Execute: write code, verify it lints, update docs.

### Agent team that makes sense
Claude Code supports spawning specialized subagents via the Agent tool. A reasonable decomposition:

| Agent | Role | Output |
|-------|------|--------|
| **Researcher** | Web-fetch Qwen3-TTS HuggingFace page + any official inference examples. Determine: model ID, Python API, VRAM, dependencies, whether it uses transformers/vllm/custom code. Produce a feature delta: what does Qwen3 support that Chatterbox/Higgs don't, and vice versa. | Research memo with API signatures + feature comparison table |
| **Implementer** | Write `engine_qwen3.py` conforming to `TTSEngine` ABC. Wire into `model_manager.py` and `main.py`, exposing all features identified by Researcher. | New/modified source files |
| **Test writer** | Add Qwen3 cases to `generate_artifacts.py` manifest and `test_integration.py`. Exclude `long` (tank) text from Qwen3 STT validation. | Test file changes |
| **Docs updater** | Update `README.md`, `docs/api.md`, `CLAUDE.md` with new env vars, VRAM table entry, and usage example. | Markdown file changes |
| **Linter/validator** | Run `ruff check .` and verify no import errors on the engine module in isolation. | Pass/fail signal |

The Researcher runs first; the rest run in parallel once it completes. The Linter/validator runs last.

---

## 2. Feature Cataloguing

A new engine is only as valuable as the features it exposes. The Researcher's output
must include a **feature delta** comparing Qwen3 against Chatterbox and Higgs. The
Implementer then uses this to decide what to expose, not just how to wire the model up.

### What the Researcher must catalogue

**Paralinguistic tags** — Chatterbox and Higgs both support inline tags like
`[laugh]`, `[chuckle]`, `[sigh]`. The Researcher must determine: does Qwen3
support a tag syntax? Are the supported tags a subset, superset, or different
vocabulary? If Qwen3 supports tags that neither existing engine does (e.g.
`[whisper]`, `[gasp]`, `[breath]`), these must be exposed and documented.
If it uses a different syntax, a normalization layer should be considered so
callers can use consistent tags across engines.

**Voice control surface** — Chatterbox uses cloned voice embeddings from reference
audio. Higgs uses `speaker_description` (prose) and `scene_description` (prose)
plus optional reference audio + transcript. Qwen3 may offer something different
— a style parameter, emotion labels, a different cloning mechanism. The Researcher
catalogs exactly what is available so the Implementer exposes all of it.

**Sampling parameters** — Both existing engines expose `temperature`, `top_p`,
`top_k`, `seed`. If Qwen3 supports additional generation parameters (repetition
penalty, length penalty, cfg scale), they should be added as optional fields
with sensible defaults. If it supports fewer, existing param names should be
accepted but silently ignored with a log warning (not an error).

**Feature comparison table the Researcher must produce:**

| Feature | Chatterbox | Higgs | Qwen3 |
|---------|-----------|-------|-------|
| Paralinguistic tags | `[laugh]`, `[chuckle]`, etc. | `[laugh]`, etc. | _to be determined_ |
| Voice cloning | Yes (embedding from ref audio) | Yes (ref audio + transcript) | _TBD_ |
| Speaker description (prose) | No | Yes | _TBD_ |
| Scene description | No | Yes | _TBD_ |
| Temperature / top_p / top_k | Yes | Yes | _TBD_ |
| Seed control | Yes | Yes | _TBD_ |
| Additional params | exl2 / cfg | ras_win_len | _TBD_ |

Any "Yes" in the Qwen3 column must be implemented. Any capability Qwen3 has that
neither other engine has must also be implemented and documented — not skipped because
there's no existing parallel.

### Handling feature overlap

If Qwen3 supports `speaker_description` analogously to Higgs, it should use the
**same field name** in the `/tts` request body. The goal is that a caller who
switches `"model": "higgs"` to `"model": "qwen3"` has their existing parameters
work without modification, wherever the semantics are equivalent.

If a parameter name conflicts (same name, different semantics), document the
difference in `docs/api.md` rather than silently reusing the field.

---

## 4. Feasibility Assessment

### Strong factors in favor

**Clean plugin architecture.** `TTSEngine` is a strict ABC with five methods:
`load`, `unload`, `generate`, `is_loaded`, `deps_available`. The engine is registered
in `model_manager.py` by instantiation; no factory magic, no decorators. An agent
can read `engine_higgs.py` (175 lines) and produce a structurally correct new file
with high confidence.

**CLAUDE.md is high-signal.** The project CLAUDE.md lists every file's role,
conventions (Python 3.11, ruff, line-length 100), env var naming patterns, and
the venv path. This is the exact information an agent needs to avoid mistakes.

**Existing pattern is close.** Both Chatterbox and Higgs are transformer-based
models loaded via HuggingFace. Qwen3-TTS follows the same pattern. The
`run_in_executor` pattern for blocking inference, the `gc.collect() +
torch.cuda.empty_cache()` unload sequence, and the `**params` forwarding are
all directly reusable.

**STT validation is the oracle.** The project already has `stt_validate.py`
with a defined pass threshold (≥ 85%). An agent can generate Qwen3 artifacts and
run STT validation autonomously. This gives a machine-readable correctness signal
before any human looks at the work.

**Linting is deterministic.** `ruff check .` either passes or it does not.
An agent can iterate on lint failures without human involvement.

### Factors against / uncertainty

**Qwen3-TTS Python API is not yet standardized** (as of early 2026).
The model was released in 2025 and inference examples vary — some use
`transformers.pipeline`, some use a custom `Qwen3TTS` class, some route through
`vllm`. The Researcher agent must get this right; if it picks the wrong
inference path, the Implementer will produce code that is structurally correct
but behaviorally wrong.

**VRAM is unknown at planning time.** The RTX 5070 Ti has 12 GB. Qwen3-TTS
likely lands between 4–8 GB depending on quantization. The agent must either
web-fetch benchmark data or use a conservative default, and must expose
`QWEN3_QUANT_BITS` env var analogous to `HIGGS_QUANT_BITS` so the user can tune
it without code changes.

**Voice cloning semantics may differ.** Chatterbox uses a voice embedding;
Higgs uses reference audio + transcript. Qwen3 may support neither, one, or
both. The agent must correctly set `compatible_models` in `voices.py` and must
not silently swallow a voice_ref argument that the model ignores.

**No GPU available at agent execution time.** The agent cannot actually load
the model weights and run inference. Structural correctness (linting, import
checks, ABC conformance) is verifiable; runtime correctness is not.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Researcher fetches stale or wrong API docs | Medium | High — wrong inference code | Agent must cite the exact source URL and paste the relevant code snippet into its memo; human can spot-check before implementation proceeds |
| VRAM estimate too low, model fails to load | Medium | Medium — hard error on first request, not silent | `deps_available` check and `AVAILABLE_VRAM_MB` guard will surface this at load time, not silently corrupt output |
| Voice cloning wired up when model doesn't support it | Low | Medium — silent no-op or crash | Agent should explicitly document whether cloning is supported; integration test should exercise `voice_ref_path=None` path and verify output is non-empty WAV |
| Agent introduces a breaking API change | Low | High — breaks video_agent | CLAUDE.md's API contract section is explicit; agent must be instructed to add fields as optional with defaults, never remove or rename |
| Ruff lint errors not caught before commit | Very low | Low — blocked by CI if configured | Agent runs `ruff check .` as final step; any failure causes it to fix before producing output |
| Engine registered but not exposed in `/models` response | Low | Low — functional but confusing | Integration test that calls `GET /models` and asserts `"qwen3"` in model names catches this |
| Conflicting torch/transformers version with existing engines | Medium | High — breaks other engines | Agent must install into the shared tts_server venv and verify `import chatterbox` and `import boson_multimodal` still resolve after adding deps |

---

## 6. Division of Review Responsibilities

The key insight is that code integrity review and audio quality review are
separable — and the agent is better at one, the human is irreplaceable for
the other.

### Agent validates: code integrity

Before presenting anything to the human, the agent runs its own internal
checklist and must reach a clean pass on every item. Only then does it
generate audio and hand off.

```
[ ] engine_qwen3.py — subclasses TTSEngine with all 5 methods
[ ] engine_qwen3.py — unload() calls gc.collect() + cuda.empty_cache()
[ ] model_manager.py — QwenEngine() is in the engines list
[ ] main.py — Qwen3-specific params are optional fields with defaults
[ ] voices.py — "qwen3" in compatible_models if cloning is supported
[ ] .env example / README — QWEN3_* env vars documented
[ ] docs/api.md — updated with new model name and any new params
[ ] test_integration.py — qwen3 test cases present
[ ] generate_artifacts.py — manifest includes qwen3 entries
[ ] ruff check . — PASSED (zero errors)
[ ] No existing engine tests broken (chatterbox, higgs test counts unchanged)
[ ] STT validation run — qwen3 passes ≥ 85% threshold (excluding `long` text fixtures)
```

The agent does not ask for human review until all items above are green.
If STT validation fails (< 85%), the agent diagnoses and fixes before escalating.

**STT validation note — exclude `long` text fixtures:** The `long` text label
(the WWII armor passage: Tiger I, M4 Sherman, T-34, Panzer 8 Maus) is an acknowledged
edge case. It contains military proper nouns and Roman numerals that cause
STT transcription noise across *all* models, independent of whether the model
synthesized the audio correctly. This fixture is excluded from the Qwen3 pass
threshold calculation. The test artifacts for `long` are still generated so the
human reviewer can listen to them, but they do not count toward the ≥ 85% gate.

### Human validates: the audio

The human's single review is purely perceptual. They receive a set of generated
WAV files — the same artifact set that `generate_artifacts.py` produces — and
listen to them. The question is simply: **does this sound right?**

Specifically:
- Is the speech intelligible?
- Does the voice character match the requested speaker description or voice ref?
- Are there artifacts — clipping, stuttering, silence, garbled phonemes?
- Does it feel comparable in quality to Chatterbox or Higgs output?

This takes 2–3 minutes. The human does not read code. They do not check
imports or lint output. That is the agent's job.

**Pass:** audio sounds correct → merge.
**Fail:** audio sounds wrong → agent is given the specific complaint (e.g.
"voice cloning is being ignored", "output is silent") and re-runs.

---

## 7. What Makes This Workflow Succeed or Fail

### It succeeds when

- CLAUDE.md is maintained as the canonical source of truth (conventions, VRAM table, env var patterns)
- The Researcher produces a concrete feature delta before any implementation starts
- The Researcher cites a specific, authoritative source for the Qwen3 inference API
- All features Qwen3 supports are exposed — especially paralinguistic tags and voice controls
- The agent writes the checklist *before* asking for human review — not after
- The implementation follows the Higgs engine as a template (parallel structure = fewer invented decisions)
- Test generation is included in the same PR, not as a follow-up

### It fails when

- The Researcher hallucinates an API that doesn't exist or is version-mismatched
- The Researcher omits the feature delta — Qwen3 ships with unique capabilities that never get exposed
- A feature is available but mapped to the wrong existing param name (silent semantic mismatch)
- The agent skips the test and artifact update (treats it as "extra work")
- VRAM estimate is wrong and the agent hardcodes a number without env var override
- Voice cloning is silently wired up for a model that doesn't support it
- The human review checkpoint is skipped because "the linter passed"

---

## 8. Recommended CLAUDE.md Additions to Enable This Workflow

To make the agentic workflow reliable, add the following to `repos/tts_server/CLAUDE.md`:

```markdown
## Adding a new engine

To add a new TTS engine:
1. Create `app/engine_<name>.py` subclassing `TTSEngine` (see engine_higgs.py as template)
2. Add `<Name>Engine()` to the engines list in `model_manager.py`
3. Expose engine-specific params as Optional fields in the `/tts` request body in `main.py`
4. Update `compatible_models` in `voices.py` if the engine supports voice cloning
5. Add env vars to `.env.example` and document them in this file under "Environment variables"
6. Update `docs/api.md` and VRAM table in README
7. Add test cases to `tests/test_integration.py` and entries to `generate_artifacts.py`
8. Run `ruff check .` — must pass with zero errors

VRAM estimates: measure at load time with `torch.cuda.memory_allocated()` and add
to the VRAM table in README.md. Use an env var for quantization (follow HIGGS_QUANT_BITS
pattern) so users can tune without code changes.
```

This section turns the implicit tribal knowledge into explicit machine-readable instructions
that an agent can follow step-by-step without inference gaps.

---

## 9. Verdict

**Feasible with bounded risk.** The tts_server architecture is well-suited for
agentic feature addition: clean ABC, good CLAUDE.md, existing patterns to follow,
a machine-verifiable test oracle (STT ≥ 85%), and deterministic lint gating.

The main risk is the Researcher step — getting the Qwen3-TTS inference API wrong
cascades into a structurally correct but runtime-broken implementation. Mitigated
by requiring the Researcher to cite its source and paste the actual inference
example into its memo before the Implementer runs.

The review division is what makes single-pass human review viable:
- The agent owns code integrity end-to-end — linting, test coverage, STT threshold
- The human owns audio quality — a perceptual judgment no automated tool can make

The human is not reading diffs. They are listening to WAV files. That is the
correct use of human attention in this loop.
