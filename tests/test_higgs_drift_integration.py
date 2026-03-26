#!/usr/bin/env python3
"""Integration test: Higgs tail-audio continuation vs. baseline on Hanno Chapter 1.

Reproduces the voice identity drift scenario documented in
hyang0129/video_agent_long#158 and verifies that tail-audio continuation
(hyang0129/tts_server#19) mitigates it.

Uses the Akashic Archives persona voice (higgs-sable) — the exact voice used in
the Hanno the Navigator render where drift was observed. The fixture WAV matches
config/personas/akashic_archives/voice_ref.wav from video_agent_long.

Requires:
  - Running tts_server on TTS_SERVER_URL (default http://localhost:8765)
  - higgs-sable voice fixture installed (run `python tests/setup_test_voices.py`)
  - Higgs model available in the server's engine list

Audio output is saved to tests/samples/ for human review:
  hanno_ch1_baseline.wav                 — all 20 blocks, NO continuation (drift visible)
  hanno_ch1_continuation.wav             — all 20 blocks, WITH continuation (drift suppressed)
  hanno_ch1_drift_window_baseline.wav    — blocks 13-19 only, baseline
  hanno_ch1_drift_window_continuation.wav — blocks 13-19 only, continuation

The test does not assert perceptual speaker similarity (that requires human
listening). It asserts:
  - All blocks render successfully and return valid WAV audio
  - Both continuation fields must be provided together (422 otherwise)
  - Concatenated WAVs are valid and saved for review

Usage:
    pytest tests/test_higgs_drift_integration.py -v
    pytest tests/test_higgs_drift_integration.py -v -k baseline
    pytest tests/test_higgs_drift_integration.py -v -k continuation
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VOICE_FIXTURES_DIR = REPO_ROOT / "tests" / "voice_fixtures"
SAMPLES_DIR = REPO_ROOT / "tests" / "samples"

BASE_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8765")

# ---------------------------------------------------------------------------
# Sable / Akashic Archives persona voice
# Matches config/personas/akashic_archives.json from video_agent_long.
# higgs_voice_id: "higgs-sable"
# wav_sha256 of voice_ref.wav: 2e0563b2123b4d18ab7742e5073af4f97ad51ee3da028b366d8063ae6c3e924e
# ---------------------------------------------------------------------------

SABLE_VOICE_ID = "higgs-sable"
SABLE_VOICE_SHA256 = "2e0563b2123b4d18ab7742e5073af4f97ad51ee3da028b366d8063ae6c3e924e"
SABLE_REF_TEXT = (
    "History has a way of preserving the people it cannot explain. "
    "The details blur, but the shape of them remains."
)

# From akashic_archives.json → higgs_speaker_description
SABLE_SPEAKER_DESCRIPTION = (
    "Measured, dry female narrator with quiet authority. "
    "Unhurried delivery — the voice of someone who has seen everything "
    "and is still somehow surprised by humans. "
    "Academic pace with deliberate emphasis on key revelations."
)

# ---------------------------------------------------------------------------
# Hanno Chapter 1 — block definitions
# Sourced from hyang0129/video_agent_long#158 (confirmed human audio review,
# run ID vl_2026-03-25_hanno_the_navigator_15d16f).
#
# scene_description is composed from vocal_direction by HiggsTtsServerAdapter:
#   volume=soft  → "soft volume"
#   volume=normal, pace=moderate → None (no scene hint sent)
#
# Drift-risk blocks (True) are the ones flagged in issue #158:
#   block 15 — "Great start." — CLEARLY DIFFERENT SPEAKER
#   block 17 — "That's not a rounding error." — AUDIBLE DRIFT
#   block 19 — "And the reason..." — BORDERLINE
# ---------------------------------------------------------------------------

# (block_num, text, scene_description, is_drift_risk)
HANNO_CH1_BLOCKS: list[tuple[int, str, str | None, bool]] = [
    (0,  "Okay so the word 'gorilla' — it's not Latin.", "soft volume", False),
    (1,  "It's not Greek.", "soft volume", False),
    (2,  "It's Carthaginian.", "soft volume", False),
    (3,  "As in ancient Carthage.", "soft volume", False),
    (4,  "As in a civilization that got wiped off the map over two thousand years ago.", "soft volume", False),
    (5,  "Somehow, a dead empire named one of the largest primates on Earth.", None, False),
    (6,  "Here's how.", "soft volume", False),
    (7,  "Around 500 BC, a Carthaginian commander named Hanno sailed down the west coast of Africa.", None, False),
    (8,  "At the far end of his voyage, his expedition ran into large, hairy, hostile creatures.", None, False),
    (9,  "His men killed three, skinned them, brought the hides back to Carthage.", None, False),
    (10, "Twenty-three centuries later, in 1847, a scientist described the great ape "
         "species we now call the gorilla, read Hanno's old report, and just... borrowed the word.", None, False),
    (11, "Didn't even confirm they were the same animal.", None, False),
    (12, "A sailor's field note from the fifth century BC, recycled by a nineteenth-century "
         "taxonomist who basically shrugged and said 'close enough.'", None, False),
    (13, "And that sailor's report — the one the whole word traces back to — is a "
         "hundred-and-one-line abridged Greek translation of a lost original.", None, False),
    (14, "So like.", "soft volume", False),
    (15, "Great start.", "soft volume", True),     # CLEARLY DIFFERENT SPEAKER in baseline
    (16, "Because we're working from this tiny, mangled fragment, after two and a half "
         "thousand years of scholarship, experts still cannot agree.", None, False),
    (17, "That's not a rounding error.", None, True),  # AUDIBLE DRIFT in baseline
    (18, "That's the difference between barely leaving Morocco and reaching the equator.", None, False),
    (19, "And the reason we can't resolve it is honestly kind of heartbreaking.", "soft volume", True),
]

# Blocks 13-19: focused drift window for A/B comparison
DRIFT_WINDOW = [b for b in HANNO_CH1_BLOCKS if b[0] >= 13]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_is_running() -> bool:
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
        return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _post_tts(payload: dict, timeout: int = 300) -> bytes:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read()


def _get_voice(voice_id: str) -> dict | None:
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/voices/{voice_id}", timeout=10)
        return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _clone_voice_higgs(
    name: str, audio_path: Path, reference_text: str
) -> dict:
    """POST /voices/clone with target_model=higgs. Returns parsed response dict.

    On 409 (already exists), fetches and returns the existing metadata.
    """
    boundary = "----HiggsTestBoundary"
    audio_bytes = audio_path.read_bytes()

    def _field(fname: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{fname}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = (
        _field("name", name)
        + _field("reference_text", reference_text)
        + _field("target_model", "higgs")
        + (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="reference_audio"; '
            f'filename="{audio_path.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        f"{BASE_URL}/voices/clone",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            get_resp = urllib.request.urlopen(f"{BASE_URL}/voices/{slug}", timeout=10)
            return json.loads(get_resp.read())
        raise


def _higgs_available() -> bool:
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
        health = json.loads(resp.read())
        engines = health.get("engines", {})
        higgs = engines.get("higgs", {})
        return bool(higgs.get("deps_available", False))
    except (urllib.error.URLError, OSError):
        return False


def _concat_wavs(wav_list: list[bytes]) -> bytes:
    """Concatenate a list of WAV byte strings into one WAV."""
    all_frames = b""
    nchannels = sampwidth = framerate = None
    for wav_bytes in wav_list:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            if nchannels is None:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
            all_frames += wf.readframes(wf.getnframes())
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(nchannels or 1)
        wf.setsampwidth(sampwidth or 2)
        wf.setframerate(framerate or 24000)
        wf.writeframes(all_frames)
    return out.getvalue()


def _render_block_sequence(
    blocks: list[tuple[int, str, str | None, bool]],
    voice_id: str,
    voice_checksum: str,
    use_continuation: bool,
) -> list[tuple[int, str, bytes]]:
    """Render a block sequence with or without continuation audio.

    Returns list of (block_num, text, wav_bytes).
    """
    results: list[tuple[int, str, bytes]] = []
    prev_wav_bytes: bytes | None = None
    prev_text: str | None = None

    for block_num, text, scene_desc, _is_drift_risk in blocks:
        payload: dict = {
            "model": "higgs",
            "text": text,
            "voice": voice_id,
            "voice_checksum": voice_checksum,
            "speaker_description": SABLE_SPEAKER_DESCRIPTION,
        }
        if scene_desc is not None:
            payload["scene_description"] = scene_desc
        if use_continuation and prev_wav_bytes is not None and prev_text is not None:
            payload["continuation_audio_base64"] = base64.b64encode(prev_wav_bytes).decode()
            payload["continuation_audio_text"] = prev_text

        wav = _post_tts(payload, timeout=300)
        results.append((block_num, text, wav))
        prev_wav_bytes = wav
        prev_text = text

    return results


# ---------------------------------------------------------------------------
# Session-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def require_server():
    if not _server_is_running():
        pytest.skip(f"TTS server not running at {BASE_URL}")


@pytest.fixture(scope="session")
def require_higgs():
    if not _higgs_available():
        pytest.skip("Higgs engine deps not available on this server")


@pytest.fixture(scope="session")
def samples_dir():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return SAMPLES_DIR


@pytest.fixture(scope="session")
def sable_voice():
    """Ensure higgs-sable is registered on the server. Returns (voice_id, wav_sha256).

    Resolution order:
    1. Already registered with a matching wav_sha256 — use it directly.
    2. Already registered but no stored sha256 (pre-backfill install) —
       use the known SABLE_VOICE_SHA256 constant. VoiceStore._scan() now
       backfills sha256 for all WAV-backed voices on startup, so this branch
       is only reachable on a server that has not yet been restarted since
       the voice was installed.
    3. Not registered — clone from the tracked fixture WAV.
    """
    existing = _get_voice(SABLE_VOICE_ID)
    if existing is not None:
        sha256 = existing.get("wav_sha256") or SABLE_VOICE_SHA256
        return {"voice_id": SABLE_VOICE_ID, "wav_sha256": sha256}

    ref_wav = VOICE_FIXTURES_DIR / SABLE_VOICE_ID / "reference.wav"
    if not ref_wav.exists():
        pytest.skip(
            f"higgs-sable fixture not found at {ref_wav}. "
            "Run: python tests/setup_test_voices.py"
        )
    data = _clone_voice_higgs(SABLE_VOICE_ID, ref_wav, SABLE_REF_TEXT)
    sha256 = data.get("wav_sha256") or SABLE_VOICE_SHA256
    return {"voice_id": data["voice_id"], "wav_sha256": sha256}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHiggsContinuationAudio:
    """Regression test for voice identity drift (video_agent_long#158).

    Renders Hanno Chapter 1 blocks with the Sable voice (higgs-sable) in two
    passes — baseline and with tail-audio continuation — and saves audio for
    human review.

    Drift-risk blocks confirmed by human review of run vl_2026-03-25_hanno_the_navigator_15d16f:
      block 15: "Great start."                        — CLEARLY DIFFERENT SPEAKER
      block 17: "That's not a rounding error."        — AUDIBLE DRIFT
      block 19: "And the reason ... heartbreaking."   — BORDERLINE
    """

    def test_baseline_all_blocks(self, require_higgs, sable_voice, samples_dir):
        """Render all 20 Hanno blocks WITHOUT continuation.

        Expected: blocks 15, 17, 19 exhibit audible speaker drift vs. block 7.
        Output: tests/samples/hanno_ch1_baseline.wav
        """
        results = _render_block_sequence(
            HANNO_CH1_BLOCKS,
            sable_voice["voice_id"],
            sable_voice["wav_sha256"],
            use_continuation=False,
        )

        assert len(results) == len(HANNO_CH1_BLOCKS)
        for block_num, text, wav in results:
            assert wav[:4] == b"RIFF", f"Block {block_num} ({text[:30]!r}) not a valid WAV"
            assert len(wav) > 2000, f"Block {block_num} WAV suspiciously small"

        concatenated = _concat_wavs([wav for _, _, wav in results])
        out_path = samples_dir / "hanno_ch1_baseline.wav"
        out_path.write_bytes(concatenated)
        print(f"\n[OK] Baseline saved: {out_path}")
        print(f"     {len(results)} blocks — listen to blocks 15, 17, 19 vs. block 7 for drift.")

    def test_continuation_all_blocks(self, require_higgs, sable_voice, samples_dir):
        """Render all 20 Hanno blocks WITH tail-audio continuation.

        Each block receives the previous block's WAV + transcript as continuation
        context, providing a per-block speaker identity anchor.

        Expected: blocks 15, 17, 19 should match block 7's speaker identity
        more closely than the baseline.
        Output: tests/samples/hanno_ch1_continuation.wav
        """
        results = _render_block_sequence(
            HANNO_CH1_BLOCKS,
            sable_voice["voice_id"],
            sable_voice["wav_sha256"],
            use_continuation=True,
        )

        assert len(results) == len(HANNO_CH1_BLOCKS)
        for block_num, text, wav in results:
            assert wav[:4] == b"RIFF", f"Block {block_num} ({text[:30]!r}) not a valid WAV"
            assert len(wav) > 2000, f"Block {block_num} WAV suspiciously small"

        concatenated = _concat_wavs([wav for _, _, wav in results])
        out_path = samples_dir / "hanno_ch1_continuation.wav"
        out_path.write_bytes(concatenated)
        print(f"\n[OK] Continuation saved: {out_path}")
        print("     Compare blocks 15, 17, 19 against baseline for reduced drift.")

    def test_drift_window_baseline(self, require_higgs, sable_voice, samples_dir):
        """Render blocks 13-19 (drift window) WITHOUT continuation.

        Output: tests/samples/hanno_ch1_drift_window_baseline.wav
        """
        results = _render_block_sequence(
            DRIFT_WINDOW,
            sable_voice["voice_id"],
            sable_voice["wav_sha256"],
            use_continuation=False,
        )

        assert len(results) == len(DRIFT_WINDOW)
        for block_num, text, wav in results:
            assert wav[:4] == b"RIFF", f"Block {block_num} not a valid WAV"
            assert len(wav) > 2000

        out_path = samples_dir / "hanno_ch1_drift_window_baseline.wav"
        out_path.write_bytes(_concat_wavs([wav for _, _, wav in results]))
        print(f"\n[OK] Drift window baseline saved: {out_path}")

    def test_drift_window_continuation(self, require_higgs, sable_voice, samples_dir):
        """Render blocks 13-19 WITH continuation.

        Block 13 gets no continuation (first in window). Blocks 14-19 each
        receive the previous block's audio as tail-audio context.

        Output: tests/samples/hanno_ch1_drift_window_continuation.wav
        """
        results = _render_block_sequence(
            DRIFT_WINDOW,
            sable_voice["voice_id"],
            sable_voice["wav_sha256"],
            use_continuation=True,
        )

        assert len(results) == len(DRIFT_WINDOW)
        for block_num, text, wav in results:
            assert wav[:4] == b"RIFF", f"Block {block_num} not a valid WAV"
            assert len(wav) > 2000

        out_path = samples_dir / "hanno_ch1_drift_window_continuation.wav"
        out_path.write_bytes(_concat_wavs([wav for _, _, wav in results]))
        print(f"\n[OK] Drift window continuation saved: {out_path}")

    def test_continuation_payload_structure(self, require_higgs, sable_voice):
        """Verify continuation fields are accepted end-to-end.

        Synthesises block 14 ("So like.") as a continuation of block 13's output.
        This is the exact transition that triggers drift on block 15 in the
        Hanno baseline render.
        """
        voice_id = sable_voice["voice_id"]
        checksum = sable_voice["wav_sha256"]

        # Block 13 — anchor, no continuation
        block_13 = HANNO_CH1_BLOCKS[13]
        assert block_13[0] == 13, "block index/num mismatch — was a block inserted?"
        _num, text_13, scene_13, _ = block_13
        payload_13: dict = {
            "model": "higgs",
            "text": text_13,
            "voice": voice_id,
            "voice_checksum": checksum,
            "speaker_description": SABLE_SPEAKER_DESCRIPTION,
        }
        if scene_13:
            payload_13["scene_description"] = scene_13
        wav_13 = _post_tts(payload_13, timeout=300)
        assert wav_13[:4] == b"RIFF", "Block 13 (anchor) did not return a valid WAV"

        # Block 14 ("So like.") — with block 13 as continuation
        block_14 = HANNO_CH1_BLOCKS[14]
        assert block_14[0] == 14, "block index/num mismatch — was a block inserted?"
        _num, text_14, scene_14, _ = block_14
        payload_14: dict = {
            "model": "higgs",
            "text": text_14,
            "voice": voice_id,
            "voice_checksum": checksum,
            "speaker_description": SABLE_SPEAKER_DESCRIPTION,
            "continuation_audio_base64": base64.b64encode(wav_13).decode(),
            "continuation_audio_text": text_13,
        }
        if scene_14:
            payload_14["scene_description"] = scene_14
        wav_14 = _post_tts(payload_14, timeout=300)
        assert wav_14[:4] == b"RIFF", "Block 14 (with continuation) did not return a valid WAV"
        assert len(wav_14) > 2000

    def test_continuation_fields_paired_validation(self, require_higgs):
        """Server returns 422 when only one continuation field is provided.

        Verifies the Pydantic model_validator enforces both-or-neither at the
        API boundary before any synthesis is attempted.
        """
        # Only audio, no text
        body = json.dumps({
            "model": "higgs",
            "text": "Testing validation.",
            "speaker_description": "Calm narrator",
            "continuation_audio_base64": base64.b64encode(b"not-real-audio").decode(),
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=30)
        assert exc_info.value.code == 422, (
            f"Expected 422 for audio-only continuation, got {exc_info.value.code}"
        )

        # Only text, no audio
        body = json.dumps({
            "model": "higgs",
            "text": "Testing validation.",
            "speaker_description": "Calm narrator",
            "continuation_audio_text": "Some previous text.",
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=30)
        assert exc_info.value.code == 422, (
            f"Expected 422 for text-only continuation, got {exc_info.value.code}"
        )
