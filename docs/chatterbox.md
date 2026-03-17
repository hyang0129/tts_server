# Chatterbox Engine

Chatterbox Turbo is the default TTS engine. It uses pre-computed speaker embeddings from reference audio for zero-shot voice cloning and supports the richest set of paralinguistic tags.

## Additional /tts Parameters

These fields extend the base `/tts` request when `"model": "chatterbox"`:

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | float | `0.8` | Sampling temperature. Lower = more stable/consistent; higher = more expressive/varied. For a podcast host, `0.7`–`0.85` is a good range. |
| `top_k` | int | `1000` | Top-k sampling. Higher values give the model more freedom. |
| `repetition_penalty` | float | `1.2` | Penalises repeated audio tokens (1.0–3.0). Increase if the output loops or stutters. |

## Voice Cloning

Chatterbox clones from reference audio alone — no transcript needed. Providing a transcript via `reference_text` also enables Higgs compatibility for the same voice.

**Recommended reference audio**: 10–30s of clean speech, consistent pace, no background noise. Quality of the reference directly drives output naturalness.

## Voice Blending

Unique to Chatterbox. Blend two cloned voices to create a hybrid character:

- `voice_a` — contributes rhythm and prosody (how the voice "moves")
- `voice_b` — contributes tonal texture
- `texture_mix` (0–100) — 0 = pure voice_a texture, 100 = pure voice_b texture

Useful for:
- Softening an overly intense reference voice
- Creating a consistent podcast persona that doesn't exist as a single reference clip
- Exploring the space between two different speaker styles

## Paralinguistic Tags

Embed these tags in the `text` field to inject non-speech sounds mid-sentence. Supported tags:

| Tag | Effect |
|---|---|
| `[laugh]` | Natural laugh |
| `[chuckle]` | Quieter, more restrained laugh |
| `[cough]` | Single cough |
| `[sigh]` | Audible sigh |
| `[gasp]` | Sharp intake of breath |
| `[groan]` | Low groan |
| `[sniff]` | Sniff |
| `[shush]` | Hushing sound |
| `[clear throat]` | Throat clear |

**Podcast host usage**: `[laugh]`, `[chuckle]`, `[sigh]`, and `[clear throat]` are the most natural in conversational speech. Place tags between sentences or at natural pause points.

```json
{
  "model": "chatterbox",
  "text": "And that's when it hit me. [chuckle] I had been thinking about it completely backwards.",
  "voice": "my-host-voice",
  "temperature": 0.78,
  "repetition_penalty": 1.2
}
```

## Quality Tips for Podcast Hosts

- **Temperature 0.75–0.82**: Sweet spot for expressive-but-coherent delivery. Above 0.9 introduces instability.
- **Repetition penalty 1.15–1.25**: Keeps speech from looping without making it sound robotic.
- **Reference quality matters most**: A clean 15–20s clip of the target voice will outperform any parameter tuning on a noisy reference.
- **Voice blending for consistency**: If your reference clips have variable pacing, blend the best-paced clip (`voice_a`) with the best-texture clip (`voice_b`) at `texture_mix: 40`.

## VRAM

~4.2 GB (RTX 5070 Ti). Loads in ~3s.
