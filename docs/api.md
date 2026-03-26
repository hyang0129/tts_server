# TTS Server API Reference

Base URL: `http://localhost:8000`

## Endpoints

### POST /tts

Synthesize speech from text. The `model` field selects which TTS engine; the server lazy-loads it into VRAM (unloading any previously active engine).

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string (required) | -- | Text to synthesize (1–5000 chars, non-whitespace). Supports paralinguistic tags — see model docs. |
| `model` | string | `"chatterbox"` | Engine: `"chatterbox"`, `"chatterbox_full"`, `"higgs"`, or `"qwen3"` |
| `voice` | string \| null | `null` | Voice ID. Must be compatible with the selected model. |
| `voice_checksum` | string \| null | `null` | SHA-256 (64 hex chars) of the stored reference WAV. **Required when `voice` is specified.** Omitting it returns 422; wrong format returns 422; mismatch returns 409. Not validated for legacy voices registered before this feature (no-op passthrough). |
| `temperature` | float | model default | Sampling temperature (0.0–2.0). Higher = more expressive/variable. |
| `top_p` | float | 0.95 | Nucleus sampling threshold |

For model-specific parameters see [chatterbox.md](chatterbox.md), [higgs.md](higgs.md), and [qwen3.md](qwen3.md).

**Chatterbox Full-specific parameters** (ignored by other models):

| Field | Type | Default | Description |
|---|---|---|---|
| `exaggeration` | float | model default | Emotion intensity (0.0–1.0). Higher values increase expressiveness. |
| `cfg_weight` | float | 0.5 | Classifier-free guidance weight. |
| `min_p` | float | 0.05 | Minimum probability threshold for sampling. |

Note: `top_k` is **not supported** by `chatterbox_full` and will be ignored if provided.

**Chatterbox Full — paralinguistic tags:**

The full model has a different (richer) tag vocabulary than Chatterbox Turbo. Tags are placed inline in the `text` field.

Supported: `[laughter]`, `[giggle]`, `[guffaw]`, `[gasp]`, `[sigh]`, `[whisper]`, `[cough]`, `[sneeze]`, `[sniff]`, `[clear_throat]`, `[exhale]`, `[inhale]`, `[groan]`, `[cry]`, `[mumble]`, `[humming]`, `[singing]`, `[UH]`, `[UM]`

For Turbo compatibility, `[laugh]` → `[laughter]`, `[chuckle]` → `[giggle]`, `[breath]` → `[inhale]` are automatically translated.

**Qwen3-specific parameters** (ignored by other models):

| Field | Type | Default | Description |
|---|---|---|---|
| `qwen3_language` | string \| null | `"Auto"` | Language hint: `"English"`, `"Chinese"`, `"Japanese"`, etc., or `"Auto"` |
| `qwen3_speaker` | string \| null | null | Preset speaker name (CustomVoice models only). E.g. `"Ryan"`, `"Aiden"`. |
| `qwen3_instruct` | string \| null | null | Tone/emotion instruction. E.g. `"Speak with warm enthusiasm"`. |

**Higgs-specific parameters** (ignored by other models):

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | int \| null | `null` | RNG seed for reproducible generation |
| `max_new_tokens` | int \| null | 2048 | Maximum tokens to generate |
| `scene_description` | string \| null | `null` | Scene description for audio context |
| `speaker_description` | string \| null | `null` | Speaker identity description |
| `ras_win_len` | int \| null | 7 | RAS repetition window length |
| `ras_win_max_num_repeat` | int \| null | 2 | Max repetitions in RAS window |
| `force_audio_gen` | bool \| null | `null` | Force audio generation mode |
| `continuation_audio_base64` | string \| null | `null` | Base64-encoded WAV (24 kHz mono) of the preceding generated segment. Anchors speaker identity across blocks. Requires `continuation_audio_text`. |
| `continuation_audio_text` | string \| null | `null` | Transcript of `continuation_audio_base64`. Required when continuation audio is provided. |

`speaker_description` is also accepted for Qwen3 and maps to the `instruct` parameter (voice-design mode).

**Response**: `audio/wav` (PCM 16-bit, mono)

**Response headers**:

| Header | Description |
|---|---|
| `X-Audio-Duration-S` | Duration in seconds |
| `X-Sample-Rate` | Sample rate (Hz) |
| `X-Audio-Frames` | Total audio frames |
| `X-Voice-WPM` | Voice words-per-minute (if calibrated) |

**Status codes**: 200 OK, 400 model unavailable or voice incompatible, 404 voice not found, 409 voice checksum mismatch, 422 validation error.

**404 voice-not-found error body** — machine-readable so clients can distinguish "voice missing, register it" from other 404s:
```json
{
  "detail": {
    "message": "Voice not found: higgs-sable",
    "error_code": "VOICE_NOT_REGISTERED",
    "voice_id": "higgs-sable"
  }
}
```
Clients that catch `VOICE_NOT_REGISTERED` should call `POST /voices/clone` with the persona's reference WAV, then retry the original request.

**409 voice-checksum-mismatch error body**:
```json
{
  "detail": {
    "message": "Voice checksum mismatch for 'higgs-sable'",
    "error_code": "VOICE_CHECKSUM_MISMATCH",
    "voice_id": "higgs-sable",
    "expected": "a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f0"
  }
}
```

**Minimal example** (no voice):
```json
{
  "model": "chatterbox",
  "text": "Hello, how are you today?"
}
```

**Example with voice**:
```json
{
  "model": "chatterbox",
  "text": "Hello, how are you today?",
  "voice": "kronimi7030",
  "voice_checksum": "a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f0"
}
```

---

### POST /voices/clone

Clone a voice from reference audio.

**Request** (multipart/form-data):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string (required) | -- | Voice name (1–200 chars). Slugified to create voice_id. |
| `reference_audio` | file (required) | -- | Reference audio (WAV, FLAC, OGG, AIFF, or any FFmpeg-decodable format). Max 50 MB, min 3s, min 16kHz. |
| `reference_text` | string \| null | null | Transcript of reference audio. **Required for Higgs compatibility.** |
| `target_model` | string \| null | null | Hint for which model this voice targets. |
| `max_duration_s` | float | 300.0 | Max allowed reference duration (3–7200s). |

**Response** (201 Created):
```json
{
  "voice_id": "my-voice",
  "name": "My Voice",
  "wpm": null,
  "wav_sha256": "a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f0"
}
```

**Voice compatibility rules**:
- `reference_text` provided → compatible with `higgs`, `qwen3`, `chatterbox`, and `chatterbox_full`
- `reference_text` omitted → compatible with `chatterbox` and `chatterbox_full`

**Status codes**: 201 created, 409 voice already exists, 422 validation error.

---

### POST /voices/blend

Blend two existing voices into a new voice. **Chatterbox (`chatterbox` and `chatterbox_full`) only.**

**Request** (multipart/form-data):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string (required) | -- | Name for the blended voice |
| `voice_a` | string (required) | -- | First voice ID (contributes rhythm/prosody) |
| `voice_b` | string (required) | -- | Second voice ID |
| `texture_mix` | int | 50 | Texture blend: 0 = pure voice_a, 100 = pure voice_b |
| `model` | string | `"chatterbox"` | Which Chatterbox variant to use: `"chatterbox"` or `"chatterbox_full"` |

**Response** (201 Created):
```json
{
  "voice_id": "blend-name",
  "name": "Blend Name",
  "wpm": null
}
```

Blended voices are Chatterbox-only (`chatterbox` and `chatterbox_full`). The blend interpolates speaker identity embeddings while keeping rhythm/prosody from `voice_a`.

**Status codes**: 201 created, 404 source voice not found, 409 already exists.

---

### GET /voices

List registered voices.

**Query parameters**:

| Param | Type | Description |
|---|---|---|
| `model` | string (optional) | Filter to voices compatible with this model |

**Response** (200):
```json
{
  "voices": [
    {
      "voice_id": "kronimi7030",
      "name": "kronimi7030",
      "original_filename": "kroniivoice_15s.wav",
      "created_at": "2026-03-17T00:00:00Z",
      "duration_s": 15.0,
      "sample_rate": 48000,
      "wpm": 142.5,
      "compatible_models": ["chatterbox", "higgs"],
      "wav_sha256": "a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f012a3f2c1b0d9e7f0"
    }
  ]
}
```

---

### GET /voices/{voice_id}

Get details for a single voice.

**Response** (200): Same shape as a single entry in `/voices`.

**Status codes**: 200 OK, 404 not found.

---

### DELETE /voices/{voice_id}

Delete a registered voice and all its files.

**Status codes**: 204 deleted, 404 not found.

---

### GET /health

Server health and engine status.

**Response** (200):
```json
{
  "status": "ok",
  "active_model": "chatterbox",
  "engines": {
    "chatterbox":      { "loaded": true,  "estimated_vram_mb": 4700, "sample_rate": 24000 },
    "chatterbox_full": { "loaded": false, "estimated_vram_mb": 4700, "sample_rate": 24000 },
    "higgs":           { "loaded": false, "estimated_vram_mb": 9000, "sample_rate": 24000 },
    "qwen3":           { "loaded": false, "estimated_vram_mb": 5500, "sample_rate": 24000 }
  },
  "available_vram_mb": 10000,
  "voices": 3
}
```

---

### GET /models

List available models with load status.

**Response** (200):
```json
[
  { "name": "chatterbox",      "available": true, "loaded": true,  "active": true,  "estimated_vram_mb": 4700, "sample_rate": 24000 },
  { "name": "chatterbox_full", "available": true, "loaded": false, "active": false, "estimated_vram_mb": 4700, "sample_rate": 24000 },
  { "name": "higgs",           "available": true, "loaded": false, "active": false, "estimated_vram_mb": 9000, "sample_rate": 24000 },
  { "name": "qwen3",           "available": true, "loaded": false, "active": false, "estimated_vram_mb": 5500, "sample_rate": 24000 }
]
```

---

## Model Switching Behavior

- **Lazy loading**: No model is loaded at startup; loading happens on first request.
- **One-at-a-time**: Requesting a different model triggers unload of the current engine, then load of the new one.
- **Idle auto-unload**: After 60s with no requests the active engine is unloaded to free VRAM.
- **Swap time**: Chatterbox ~3s, Chatterbox Full ~3s (estimated), Higgs 8-bit ~12s, Higgs 4-bit ~21s, Qwen3 1.7B ~8s (estimated).

## Voice Compatibility Matrix

| Feature | Chatterbox | Chatterbox Full | Higgs | Qwen3 |
|---|---|---|---|---|
| Reference audio only | Yes | Yes | No | No |
| Reference audio + transcript | Yes | Yes | Yes (required) | Yes (Base model) |
| Pre-computed conditionals | Yes | No | No | No |
| Voice blending | Yes | Yes | No | No |
| Description-only (no voice) | No | No | Yes | Yes (VoiceDesign/instruct) |
| Preset speakers | No | No | No | Yes (CustomVoice model) |
| Tone/emotion instruct | No | No | No | Yes |
| Emotion intensity (`exaggeration`) | No | Yes | No | No |
| `top_k` sampling | Yes | No | No | No |
| Sample rate | 24 kHz | 24 kHz | 24 kHz | 24 kHz |

## .env Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `AVAILABLE_VRAM_MB` | `12000` | VRAM budget in MB. Models exceeding this are excluded from `/models`. Recommended: `10000`. |
| `TTS_VOICES_DIR` | `./voices` | Directory for voice reference files and metadata |
| `HIGGS_QUANT_BITS` | `8` | Higgs quantization: `4`, `8`, or `0`/`none` for bf16 |
| `HIGGS_REPO_PATH` | `/tmp/faster-higgs-audio` | Path to faster-higgs-audio source repo |
| `HIGGS_MODEL_ID` | `bosonai/higgs-audio-v2-generation-3B-base` | HuggingFace model ID |
| `HIGGS_TOKENIZER_ID` | `bosonai/higgs-audio-v2-tokenizer` | HuggingFace tokenizer ID |
| `CB_FULL_VRAM_MB` | `4700` | Override VRAM budget estimate for chatterbox_full |
| `QWEN3_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen3-TTS HuggingFace model ID. Variants: `Base` (cloning), `CustomVoice` (preset speakers), `VoiceDesign` (description). 0.6B variants also available. |
| `QWEN3_DTYPE` | `bfloat16` | Qwen3 weight dtype: `bfloat16` or `float16` |
| `QWEN3_VRAM_MB` | `5500` | Override Qwen3 VRAM budget estimate |
| `HF_TOKEN` | -- | HuggingFace token for gated model access |
