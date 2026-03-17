# TTS Server API Reference

Base URL: `http://localhost:8000`

## Endpoints

### POST /tts

Synthesize speech from text. The `model` field selects which TTS engine to use; the server lazy-loads it into VRAM (unloading any previously active engine).

**Request body** (JSON):

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string (required) | -- | Text to synthesize (1-5000 chars, non-whitespace) |
| `model` | string | `"chatterbox"` | Engine: `"chatterbox"` or `"higgs"` |
| `voice` | string \| null | `null` | Voice ID. Must be compatible with the selected model. |
| `temperature` | float | chatterbox: 0.8, higgs: 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.95 | Nucleus sampling threshold |
| `top_k` | int | chatterbox: 1000, higgs: 50 | Top-k sampling |
| `repetition_penalty` | float | 1.2 | **Chatterbox only.** Repetition penalty (1.0-3.0). |
| `seed` | int \| null | null | **Higgs only.** Random seed for deterministic generation. |
| `max_new_tokens` | int | 2048 | **Higgs only.** Max generated audio tokens (256-8192). |
| `scene_description` | string \| null | null | **Higgs only.** Environment/tone description injected into system prompt. |
| `speaker_description` | string \| null | null | **Higgs only.** Text-based voice profile (gender, accent, tone, pace). |
| `ras_win_len` | int \| null | 7 | **Higgs only.** Repetition-aware sampling window length. 0 to disable (recommended for humming/singing). |
| `ras_win_max_num_repeat` | int | 2 | **Higgs only.** Max repeated tokens within RAS window. |
| `force_audio_gen` | bool | true | **Higgs only.** Force audio token generation. |

**Response**: `audio/wav` (PCM 16-bit, mono)

**Response headers**:

| Header | Description |
|---|---|
| `X-Audio-Duration-S` | Duration in seconds |
| `X-Sample-Rate` | Sample rate (Hz) |
| `X-Audio-Frames` | Total audio frames |
| `X-Voice-WPM` | Voice words-per-minute (if calibrated) |

**Status codes**: 200 OK, 400 model unavailable or voice incompatible, 404 voice not found, 422 validation error.

**Example**:
```json
{
  "model": "chatterbox",
  "text": "Hello, how are you today?",
  "voice": "kronimi7030",
  "temperature": 0.8
}
```

---

### POST /voices/clone

Clone a voice from reference audio.

**Request** (multipart/form-data):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string (required) | -- | Voice name (1-200 chars). Slugified to create voice_id. |
| `reference_audio` | file (required) | -- | Reference audio file (WAV, FLAC, OGG, AIFF, or any format FFmpeg can decode). Max 50 MB, min 3s, min 16kHz. |
| `reference_text` | string \| null | null | Transcript of reference audio. **Required for higgs compatibility.** |
| `target_model` | string \| null | null | Hint for which model this voice targets. |
| `max_duration_s` | float | 300.0 | Max allowed reference duration (3-7200s). |

**Response** (201 Created):
```json
{
  "voice_id": "my-voice",
  "name": "My Voice",
  "wpm": null
}
```

**Voice compatibility rules**:
- If `reference_text` is provided: voice is compatible with both `higgs` and `chatterbox`
- If `reference_text` is omitted: voice is compatible with `chatterbox` only
- Higgs requires a transcript for voice cloning (it uses the text+audio pair as in-context examples)

**Status codes**: 201 created, 409 voice already exists, 422 validation error.

---

### POST /voices/blend

Blend two existing voices into a new voice. **Chatterbox only.**

**Request** (multipart/form-data):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string (required) | -- | Name for the blended voice |
| `voice_a` | string (required) | -- | First voice ID (supplies rhythm/prosody) |
| `voice_b` | string (required) | -- | Second voice ID |
| `texture_mix` | int | 50 | Texture blend: 0 = pure voice_a, 100 = pure voice_b |

**Response** (201 Created):
```json
{
  "voice_id": "blend-name",
  "name": "Blend Name",
  "wpm": null
}
```

Blended voices are compatible with `chatterbox` only. The blending interpolates speaker identity embeddings while keeping rhythm/prosody from `voice_a`.

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
      "compatible_models": ["chatterbox", "higgs"]
    }
  ]
}
```

---

### GET /voices/{voice_id}

Get details for a single voice.

**Response** (200): Same shape as a single entry in the `/voices` list.

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
    "chatterbox": {
      "loaded": true,
      "estimated_vram_mb": 3500,
      "sample_rate": 24000
    },
    "higgs": {
      "loaded": false,
      "estimated_vram_mb": 9000,
      "sample_rate": 24000
    }
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
  {
    "name": "chatterbox",
    "available": true,
    "loaded": true,
    "active": true,
    "estimated_vram_mb": 3500,
    "sample_rate": 24000
  },
  {
    "name": "higgs",
    "available": true,
    "loaded": false,
    "active": false,
    "estimated_vram_mb": 9000,
    "sample_rate": 24000
  }
]
```

---

## Model Switching Behavior

- **Lazy loading**: Models are loaded into VRAM on first request. No model is loaded at startup.
- **One-at-a-time**: Only one engine occupies VRAM. Requesting a different model triggers unload of the current engine, then load of the new one.
- **Idle auto-unload**: After 60 seconds with no requests, the active engine is automatically unloaded to free VRAM.
- **Swap time**: Chatterbox loads in ~3s, Higgs 8-bit in ~12s, Higgs 4-bit in ~21s.

## Voice Compatibility Matrix

| Feature | Chatterbox | Higgs |
|---|---|---|
| Reference audio only | Yes | No |
| Reference audio + transcript | Yes | Yes (required) |
| Pre-computed conditionals | Yes | No |
| Voice blending | Yes | No |
| Description-only generation | No | Yes (omit voice) |
| Sample rate | 24000 Hz | 24000 Hz |

## Paralinguistic Tag Comparison

Tags can be embedded in the `text` field to inject non-speech sounds.

| Tag | Chatterbox | Higgs |
|---|---|---|
| `[laugh]` | Yes | Yes |
| `[cough]` | Yes | Yes |
| `[chuckle]` | Yes | No |
| `[sigh]` | Yes | No |
| `[gasp]` | Yes | No |
| `[groan]` | Yes | No |
| `[sniff]` | Yes | No |
| `[shush]` | Yes | No |
| `[clear throat]` | Yes | No |
| `[applause]` | No | Yes |
| `[cheering]` | No | Yes |
| `[humming start]` / `[humming end]` | No | Yes |
| `[music start]` / `[music end]` | No | Yes |
| `[sing start]` / `[sing end]` | No | Yes |

**Overlap**: `[laugh]`, `[cough]` work with both engines.

## .env Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `AVAILABLE_VRAM_MB` | `12000` | VRAM budget in MB. Models exceeding this are excluded from `/models`. Recommended: `10000`. |
| `TTS_VOICES_DIR` | `./voices` | Directory for voice reference files and metadata |
| `HIGGS_QUANT_BITS` | `8` | Higgs quantization: `4`, `8`, or `0`/`none` for bf16 |
| `HIGGS_REPO_PATH` | `/tmp/faster-higgs-audio` | Path to faster-higgs-audio source repo |
| `HIGGS_MODEL_ID` | `bosonai/higgs-audio-v2-generation-3B-base` | HuggingFace model ID |
| `HIGGS_TOKENIZER_ID` | `bosonai/higgs-audio-v2-tokenizer` | HuggingFace tokenizer ID |
| `HF_TOKEN` | -- | HuggingFace token for gated model access |
