# Higgs Audio Engine

Higgs Audio v2 (3B) is a language-model-based TTS engine. It takes text-and-audio pairs as in-context examples for voice cloning, and supports system-prompt-style control over scene and speaker characteristics. It is heavier than Chatterbox but offers unique capabilities: description-only voice generation and structured scene framing.

## Additional /tts Parameters

These fields extend the base `/tts` request when `"model": "higgs"`:

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | float | `0.7` | Sampling temperature. Lower = more stable. Higgs is generally more sensitive to temperature than Chatterbox; stay in `0.6`–`0.8` for podcast use. |
| `top_k` | int | `50` | Top-k sampling. |
| `seed` | int \| null | `null` | Random seed for deterministic output. Set for reproducible takes. |
| `max_new_tokens` | int | `2048` | Max generated audio tokens (256–8192). Increase for longer passages. |
| `scene_description` | string \| null | `null` | Free-text scene context injected into the system prompt. Sets the environment and tone. |
| `speaker_description` | string \| null | `null` | Free-text voice profile. Describes gender, accent, tone, and pacing. Used when no voice clone is provided. |
| `ras_win_len` | int | `7` | Repetition-aware sampling window. Prevents looping audio tokens. Set to `0` for humming or singing content. |
| `ras_win_max_num_repeat` | int | `2` | Max repeated tokens within the RAS window. |
| `force_audio_gen` | bool | `true` | Force the model to generate audio tokens rather than text. |

## Voice Cloning

Higgs uses a transcript-paired approach: it sees the reference audio alongside its text as an in-context example. This means:

- **`reference_text` is required** when cloning for Higgs. Pass it when calling `/voices/clone`.
- The transcript should closely match what is spoken in the reference audio.
- Reference length: 5–20s works well. Longer references do not necessarily improve quality.

## Description-Only Voice Generation

Higgs can generate speech without any reference audio by using `speaker_description` alone:

```json
{
  "model": "higgs",
  "text": "Welcome back to the show.",
  "speaker_description": "A warm, mid-30s American woman with a calm, authoritative tone and a measured pace. Clear enunciation, slight vocal fry on sentence endings."
}
```

This is useful for rapid prototyping of a new host persona before committing to a reference recording.

## Scene Description

`scene_description` frames the acoustic environment and delivery style. It is injected as context before the text, so it influences prosody and energy level.

Examples for podcast use:
- `"A relaxed, studio-quality podcast. The host is speaking directly to a single listener."` — calm, intimate delivery
- `"A lively interview podcast. The host is engaged and upbeat, wrapping up an exciting segment."` — higher energy
- `"A solo narration podcast. Thoughtful pacing, slight pauses for emphasis."` — measured, reflective

Combining `scene_description` with a voice clone gives the most control over output character.

## Paralinguistic Tags

| Tag | Effect |
|---|---|
| `[laugh]` | Natural laugh |
| `[cough]` | Single cough |
| `[applause]` | Audience applause |
| `[cheering]` | Crowd cheering |
| `[humming start]` / `[humming end]` | Wrap text for humming delivery |
| `[music start]` / `[music end]` | Wrap text for musical/sung delivery |
| `[sing start]` / `[sing end]` | Wrap text for singing |

For podcast host use, `[laugh]` and `[cough]` are the most applicable. The singing/music/humming tags are for specialised content.

**Note**: When using `[humming start]`/`[humming end]` or the singing tags, set `ras_win_len: 0` to disable repetition-aware sampling, which can otherwise suppress the intentionally repeated tonal patterns.

## Quality Tips for Podcast Hosts

- **Scene description is a major lever**: Getting it right often has more impact than temperature tuning. Describe the recording environment and the emotional register of the host.
- **Temperature 0.65–0.75**: Higgs is more volatile than Chatterbox at high temperatures. Stay conservative.
- **Use seed for A/B takes**: Set `seed` to generate multiple deterministic variants and pick the best one.
- **Transcript accuracy for cloning**: Small mismatches between `reference_text` and the actual audio degrade voice fidelity significantly.
- **`max_new_tokens`**: Default 2048 is sufficient for ~30s of speech at typical podcast pace. For longer passages increase to 4096.

## VRAM

~6.7 GB at 8-bit quantization, ~4.2 GB at 4-bit. Loads in ~12s (8-bit) or ~21s (4-bit). Controlled via `HIGGS_QUANT_BITS` env var.
