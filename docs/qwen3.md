# Qwen3-TTS Engine

**Source**: https://github.com/QwenLM/Qwen3-TTS
**Package**: `pip install qwen-tts`
**Default model**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
**Architecture**: Discrete multi-codebook language model (12 Hz token rate)
**Sample rate**: 24 kHz
**Estimated VRAM**: ~5.5 GB (1.7B bfloat16); use 0.6B variants to reduce to ~2.5 GB

---

## Model Variants

The `QWEN3_MODEL_ID` env var selects the variant. Each unlocks different generation modes:

| Variant | HuggingFace ID | Generation mode | Use case |
|---------|---------------|-----------------|----------|
| **Base** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning via `ref_audio` + `ref_text` | Consistent custom voice identity |
| **CustomVoice** | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset speakers + instruct | Fast preset voices with tone control |
| **VoiceDesign** | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Natural language voice description | One-off voices via text prompt |

0.6B variants (`Qwen3-TTS-12Hz-0.6B-*`) are also available at roughly half the VRAM cost.

The engine auto-detects the variant from `QWEN3_MODEL_ID` (checks for "customvoice" / "voicedesign" / default Base).

---

## Parameters

### Common (all variants)

| `/tts` field | Type | Default | Description |
|---|---|---|---|
| `qwen3_language` | string | `"Auto"` | Language: `"English"`, `"Chinese"`, `"Japanese"`, `"Korean"`, `"German"`, `"French"`, `"Russian"`, `"Portuguese"`, `"Spanish"`, `"Italian"`, or `"Auto"` |
| `temperature` | float | model default | Sampling temperature |
| `top_p` | float | model default | Nucleus sampling threshold |
| `top_k` | int | model default | Top-k sampling |
| `max_new_tokens` | int | model default | Max generated audio tokens |
| `seed` | int | null | Reproducibility seed (sets `torch.manual_seed`) |

### Base model — voice cloning

Pass a voice with `reference_text` (required):

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "text": "Hello, welcome to the show.",
    "voice": "my-cloned-voice",
    "qwen3_language": "English"
  }' -o output.wav
```

Voice must have been cloned with `reference_text` (sets `qwen3` in `compatible_models`).

### CustomVoice model — preset speakers

| `/tts` field | Type | Default | Description |
|---|---|---|---|
| `qwen3_speaker` | string | `"Ryan"` | Preset speaker name (see table below) |
| `qwen3_instruct` | string | null | Tone/emotion instruction. E.g. `"Speak with warm enthusiasm"` |

Available preset speakers:

| Speaker | Description | Primary Language |
|---------|-------------|-----------------|
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese |
| Eric | Lively Chengdu male, husky brightness | Chinese |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "text": "Let me tell you something fascinating.",
    "qwen3_speaker": "Ryan",
    "qwen3_instruct": "Speak with warm, engaging enthusiasm",
    "qwen3_language": "English"
  }' -o output.wav
```

### VoiceDesign model — description-only

Use `speaker_description` (shared with Higgs) or `qwen3_instruct`:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "text": "Welcome to the podcast.",
    "speaker_description": "A warm, slightly husky middle-aged female voice with calm authority",
    "qwen3_language": "English"
  }' -o output.wav
```

---

## Voice Cloning Setup

Clone a voice with `reference_text` to enable Qwen3 compatibility:

```bash
curl -X POST http://localhost:8000/voices/clone \
  -F "name=my-voice" \
  -F "reference_audio=@reference.wav" \
  -F "reference_text=Transcript of the reference audio." \
  -F "target_model=qwen3"
```

The cloned voice will appear with `"compatible_models": ["chatterbox", "higgs", "qwen3"]`.

---

## Podcast Quality Tips

- **Language**: Always set `qwen3_language` explicitly for non-English content. `"Auto"` works but explicit is more reliable.
- **VoiceDesign descriptions**: Include pace, pitch, and emotional character. E.g. `"A warm, clear male voice with moderate pace and calm authority"`.
- **CustomVoice instruct**: Describe delivery, not character. E.g. `"Speak with measured gravity"`, `"Deliver with light irony"`.
- **Seed**: Set `seed` for reproducible output during development.
- **VRAM tradeoff**: The 0.6B variants halve VRAM at some quality cost. Use Base/1.7B for production.

---

## VRAM Profiles

| Model | Estimated VRAM | Notes |
|-------|---------------|-------|
| `Qwen3-TTS-12Hz-1.7B-*` | ~5.5 GB | Default. bfloat16. |
| `Qwen3-TTS-12Hz-0.6B-*` | ~2.5 GB | Lightweight. Same API. |

Override with `QWEN3_VRAM_MB` if measured values differ.
