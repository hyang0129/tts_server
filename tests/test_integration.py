#!/usr/bin/env python3
"""Integration tests for the consolidated TTS server.

Requires a running server on port 8000 (or set TTS_SERVER_URL env var).

Usage:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k chatterbox
    pytest tests/test_integration.py -v -k higgs
    pytest tests/test_integration.py -v -k model_switching
    pytest tests/test_integration.py -v -k idle_unload
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

BASE_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def server_is_running() -> bool:
    """Check if the TTS server is reachable."""
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
        return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def get_health() -> dict:
    """GET /health and return parsed JSON."""
    resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
    return json.loads(resp.read())


def post_tts(payload: dict, timeout: int = 120) -> bytes:
    """POST to /tts and return raw WAV bytes."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read()


def clone_voice(name: str, audio_path: Path, reference_text: str) -> str:
    """Clone a voice and return the voice_id."""
    url = f"{BASE_URL}/voices/clone"
    boundary = "----TestBoundary"
    audio_bytes = audio_path.read_bytes()

    body_parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="name"\r\n\r\n'
        f"{name}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_text"\r\n\r\n'
        f"{reference_text}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"chatterbox\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_audio"; '
        f'filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n",
    ]
    body = (
        body_parts[0].encode()
        + body_parts[1].encode()
        + body_parts[2].encode()
        + body_parts[3].encode()
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data.get("voice_id") or data.get("id") or name
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return name  # already exists
        raise


def get_gpu_memory_used_mb() -> float | None:
    """Query nvidia-smi for current GPU memory usage. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def require_server():
    """Skip all tests if the server isn't running."""
    if not server_is_running():
        pytest.skip(f"TTS server not running at {BASE_URL}")


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self):
        resp = urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
        assert resp.status == 200

    def test_health_lists_models(self):
        health = get_health()
        assert "models" in health or "available_models" in health
        models = health.get("models") or health.get("available_models", [])
        assert len(models) >= 1


# ---------------------------------------------------------------------------
# Voice error tests
# ---------------------------------------------------------------------------


class TestVoiceErrors:
    def test_unknown_voice_returns_404_with_error_code(self):
        body = json.dumps({"model": "chatterbox", "text": "hello", "voice": "nonexistent-voice-xyz"}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=30)
        err = exc_info.value
        assert err.code == 404
        detail = json.loads(err.read())["detail"]
        assert detail["error_code"] == "VOICE_NOT_REGISTERED"
        assert detail["voice_id"] == "nonexistent-voice-xyz"


# ---------------------------------------------------------------------------
# Voice checksum tests
# ---------------------------------------------------------------------------


def clone_voice_full(name: str, audio_path: Path, reference_text: str) -> dict:
    """Clone a voice and return the full response dict (voice_id, wav_sha256, etc.)."""
    url = f"{BASE_URL}/voices/clone"
    boundary = "----TestBoundary"
    audio_bytes = audio_path.read_bytes()

    body_parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="name"\r\n\r\n'
        f"{name}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_text"\r\n\r\n'
        f"{reference_text}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"chatterbox\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_audio"; '
        f'filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n",
    ]
    body = (
        body_parts[0].encode()
        + body_parts[1].encode()
        + body_parts[2].encode()
        + body_parts[3].encode()
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # Already exists — fetch stored metadata so the caller gets wav_sha256.
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            get_resp = urllib.request.urlopen(f"{BASE_URL}/voices/{slug}", timeout=10)
            return json.loads(get_resp.read())
        raise


class TestVoiceChecksum:
    """Tests for the voice checksum (wav_sha256) feature."""

    @pytest.fixture(scope="class")
    def cloned_voice(self):
        """Clone a voice and return both voice_id and wav_sha256."""
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        data = clone_voice_full(
            "test_kronii_checksum",
            audio_path,
            "This is a sample of my voice for cloning purposes.",
        )
        voice_id = data.get("voice_id") or data.get("id")
        wav_sha256 = data.get("wav_sha256")
        assert voice_id, "clone response missing voice_id"
        return {"voice_id": voice_id, "wav_sha256": wav_sha256}

    def test_clone_returns_wav_sha256(self, cloned_voice):
        """POST /voices/clone response includes a non-empty 16-char hex wav_sha256."""
        wav_sha256 = cloned_voice["wav_sha256"]
        assert wav_sha256 is not None, "wav_sha256 missing from clone response"
        assert len(wav_sha256) == 64, (
            f"Expected 64-char hex string, got {len(wav_sha256)!r} chars: {wav_sha256!r}"
        )
        assert all(c in "0123456789abcdefABCDEF" for c in wav_sha256), (
            f"wav_sha256 is not a valid hex string: {wav_sha256!r}"
        )

    def test_tts_without_checksum_returns_422(self, cloned_voice):
        """POST /tts with voice set but voice_checksum omitted returns 422."""
        body = json.dumps({
            "model": "chatterbox",
            "text": "Checksum test.",
            "voice": cloned_voice["voice_id"],
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=30)
        assert exc_info.value.code == 422

    def test_tts_with_correct_checksum_returns_200(self, cloned_voice):
        """POST /tts with correct voice_checksum returns 200 audio/wav."""
        body = json.dumps({
            "model": "chatterbox",
            "text": "Checksum test.",
            "voice": cloned_voice["voice_id"],
            "voice_checksum": cloned_voice["wav_sha256"],
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=120)
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "audio/wav" in content_type, (
            f"Expected audio/wav response, got Content-Type: {content_type!r}"
        )
        wav_bytes = resp.read()
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_tts_with_wrong_checksum_returns_409(self, cloned_voice):
        """POST /tts with wrong voice_checksum returns 409 VOICE_CHECKSUM_MISMATCH."""
        body = json.dumps({
            "model": "chatterbox",
            "text": "Checksum test.",
            "voice": cloned_voice["voice_id"],
            "voice_checksum": "0" * 64,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=30)
        err = exc_info.value
        assert err.code == 409
        detail = json.loads(err.read())["detail"]
        assert detail["error_code"] == "VOICE_CHECKSUM_MISMATCH", (
            f"Expected error_code VOICE_CHECKSUM_MISMATCH, got: {detail.get('error_code')!r}"
        )
        assert detail["voice_id"] == cloned_voice["voice_id"], (
            f"Expected voice_id {cloned_voice['voice_id']!r}, got: {detail.get('voice_id')!r}"
        )

    def test_checksum_matches_wav_file(self, cloned_voice):
        """wav_sha256 must equal SHA-256 of the fixture WAV bytes (end-to-end hash integrity)."""
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        expected = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        assert cloned_voice["wav_sha256"] == expected, (
            f"Hash mismatch: server returned {cloned_voice['wav_sha256']!r}, "
            f"local SHA-256 of fixture is {expected!r}"
        )


# ---------------------------------------------------------------------------
# Chatterbox tests
# ---------------------------------------------------------------------------


class TestChatterbox:
    @pytest.fixture(scope="class")
    def voice_id(self):
        """Clone kronii voice for chatterbox tests."""
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        return clone_voice(
            "test_kronii_cb",
            audio_path,
            "This is a sample of my voice for cloning purposes.",
        )

    def test_chatterbox_clone_and_generate(self, voice_id):
        """Clone a voice and generate TTS with chatterbox model."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "chatterbox",
            "voice": voice_id,
        })
        # WAV files start with RIFF header
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_chatterbox_stt_validation(self, voice_id):
        """Generate chatterbox TTS and validate transcription quality."""
        expected_text = "Did you know? The hamburger was not actually invented in Hamburg."
        wav_bytes = post_tts({
            "text": expected_text,
            "model": "chatterbox",
            "voice": voice_id,
        })
        assert wav_bytes[:4] == b"RIFF"

        # Write to temp file for STT validation
        tmp_path = FIXTURES_DIR.parent / "artifacts" / "_test_cb_stt.wav"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(wav_bytes)

        try:
            from tests.stt_validate import validate_file
            result = validate_file(str(tmp_path), expected_text, use_llm=False)
            assert result["word_match_pct"] >= 70.0, (
                f"STT match too low: {result['word_match_pct']}% "
                f"(transcription: {result['transcription']!r})"
            )
        except ImportError:
            pytest.skip("stt_validate or faster-whisper not available")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Chatterbox Full tests
# ---------------------------------------------------------------------------


class TestChatterboxFull:
    @pytest.fixture(scope="class")
    def voice_id(self):
        """Clone kronii voice for chatterbox_full tests."""
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        return clone_voice(
            "test_kronii_cbfull",
            audio_path,
            "This is a sample of my voice for cloning purposes.",
        )

    def test_chatterbox_full_clone_and_generate(self, voice_id):
        """Clone a voice and generate TTS with chatterbox_full model."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "chatterbox_full",
            "voice": voice_id,
        })
        # WAV files start with RIFF header
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_chatterbox_full_exaggeration_param(self, voice_id):
        """Generate TTS with chatterbox_full model using exaggeration param."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "chatterbox_full",
            "voice": voice_id,
            "exaggeration": 0.8,
        })
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"

    def test_chatterbox_full_stt_validation(self, voice_id):
        """Generate chatterbox_full TTS and validate transcription quality."""
        expected_text = "Did you know? The hamburger was not actually invented in Hamburg."
        wav_bytes = post_tts({
            "text": expected_text,
            "model": "chatterbox_full",
            "voice": voice_id,
        })
        assert wav_bytes[:4] == b"RIFF"

        # Write to temp file for STT validation
        tmp_path = FIXTURES_DIR.parent / "artifacts" / "_test_cbfull_stt.wav"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(wav_bytes)

        try:
            from tests.stt_validate import validate_file
            result = validate_file(str(tmp_path), expected_text, use_llm=False)
            assert result["word_match_pct"] >= 70.0, (
                f"STT match too low: {result['word_match_pct']}% "
                f"(transcription: {result['transcription']!r})"
            )
        except ImportError:
            pytest.skip("stt_validate or faster-whisper not available")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Higgs tests
# ---------------------------------------------------------------------------


class TestHiggs:
    def test_higgs_description_generate(self):
        """Generate TTS with higgs model using speaker_description."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "higgs",
            "speaker_description": (
                "Male, moderate pitch, clear enunciation, "
                "neutral American accent, calm narration style"
            ),
        })
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_higgs_stt_validation(self):
        """Generate higgs TTS and validate transcription quality."""
        expected_text = "Did you know? The hamburger was not actually invented in Hamburg."
        wav_bytes = post_tts({
            "text": expected_text,
            "model": "higgs",
            "speaker_description": (
                "Female, moderate pitch, clear enunciation, "
                "neutral American accent, warm narration style"
            ),
        })
        assert wav_bytes[:4] == b"RIFF"

        tmp_path = FIXTURES_DIR.parent / "artifacts" / "_test_higgs_stt.wav"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(wav_bytes)

        try:
            from tests.stt_validate import validate_file
            result = validate_file(str(tmp_path), expected_text, use_llm=False)
            assert result["word_match_pct"] >= 70.0, (
                f"STT match too low: {result['word_match_pct']}% "
                f"(transcription: {result['transcription']!r})"
            )
        except ImportError:
            pytest.skip("stt_validate or faster-whisper not available")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Qwen3 tests
# ---------------------------------------------------------------------------


class TestQwen3:
    def test_qwen3_voice_design_generate(self):
        """Generate TTS with qwen3 model using speaker_description (voice design)."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "qwen3",
            "speaker_description": (
                "A warm, clear female voice with a calm narration style"
            ),
        })
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_qwen3_instruct_generate(self):
        """Generate TTS with qwen3 model using qwen3_instruct param."""
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "qwen3",
            "qwen3_language": "English",
            "qwen3_instruct": "Speak in a warm, clear tone with calm narration style",
        })
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_qwen3_clone_and_generate(self):
        """Clone a voice and generate TTS with qwen3 Base model."""
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        voice_id = clone_voice(
            "test_kronii_q3",
            audio_path,
            "This is a sample of my voice for cloning purposes.",
        )
        wav_bytes = post_tts({
            "text": "Did you know? The hamburger was not actually invented in Hamburg.",
            "model": "qwen3",
            "voice": voice_id,
        })
        assert wav_bytes[:4] == b"RIFF", "Response is not a valid WAV file"
        assert len(wav_bytes) > 1000, "WAV file suspiciously small"

    def test_qwen3_stt_validation(self):
        """Generate qwen3 TTS and validate transcription quality."""
        expected_text = "Did you know? The hamburger was not actually invented in Hamburg."
        wav_bytes = post_tts({
            "text": expected_text,
            "model": "qwen3",
            "speaker_description": "A warm, clear male voice with calm narration style",
        })
        assert wav_bytes[:4] == b"RIFF"

        tmp_path = FIXTURES_DIR.parent / "artifacts" / "_test_qwen3_stt.wav"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(wav_bytes)

        try:
            from tests.stt_validate import validate_file
            result = validate_file(str(tmp_path), expected_text, use_llm=False)
            assert result["word_match_pct"] >= 70.0, (
                f"STT match too low: {result['word_match_pct']}% "
                f"(transcription: {result['transcription']!r})"
            )
        except ImportError:
            pytest.skip("stt_validate or faster-whisper not available")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Model switching tests
# ---------------------------------------------------------------------------


class TestModelSwitching:
    """Verify that the server correctly swaps engines between requests."""

    @pytest.fixture(scope="class")
    def cb_voice_id(self):
        audio_path = FIXTURES_DIR / "kroniivoice_15s.wav"
        if not audio_path.exists():
            pytest.skip("kroniivoice_15s.wav fixture not found")
        return clone_voice(
            "test_switch_kronii",
            audio_path,
            "This is a sample of my voice for cloning purposes.",
        )

    def test_chatterbox_then_higgs_then_chatterbox(self, cb_voice_id):
        """Switch models: chatterbox -> higgs -> chatterbox."""
        short_text = "Testing model switching capabilities."

        # 1. Chatterbox request
        wav1 = post_tts({
            "text": short_text,
            "model": "chatterbox",
            "voice": cb_voice_id,
        })
        assert wav1[:4] == b"RIFF"

        # Verify health shows chatterbox active
        health = get_health()
        active = health.get("active_model") or health.get("loaded_model")
        if active:
            assert active == "chatterbox", f"Expected chatterbox active, got {active}"

        # 2. Higgs request (triggers model swap)
        wav2 = post_tts({
            "text": short_text,
            "model": "higgs",
            "speaker_description": "Male, moderate pitch, neutral accent",
        })
        assert wav2[:4] == b"RIFF"

        # Verify health shows higgs active
        health = get_health()
        active = health.get("active_model") or health.get("loaded_model")
        if active:
            assert active == "higgs", f"Expected higgs active, got {active}"

        # 3. Back to chatterbox
        wav3 = post_tts({
            "text": short_text,
            "model": "chatterbox",
            "voice": cb_voice_id,
        })
        assert wav3[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Idle unload tests
# ---------------------------------------------------------------------------


class TestIdleUnload:
    """Verify that the server unloads models after idle timeout (~60s)."""

    @pytest.mark.slow
    def test_idle_unload_frees_vram(self):
        """After idle timeout, verify no model is loaded and VRAM is freed."""
        # First, trigger a load so something is in VRAM
        try:
            post_tts({
                "text": "Loading model for idle test.",
                "model": "higgs",
                "speaker_description": "Male, moderate pitch, neutral accent",
            })
        except Exception:
            pytest.skip("Could not generate initial TTS for idle test")

        # Record VRAM usage right after generation
        vram_loaded = get_gpu_memory_used_mb()

        # Wait for idle timeout (60s) + buffer
        print("Waiting 75s for idle unload...")
        time.sleep(75)

        # Check health -- model should be unloaded
        health = get_health()
        active = health.get("active_model") or health.get("loaded_model")
        assert active is None, (
            f"Expected no model loaded after idle timeout, but got: {active}"
        )

        # If we could measure VRAM, check it dropped
        if vram_loaded is not None:
            vram_idle = get_gpu_memory_used_mb()
            if vram_idle is not None:
                freed_mb = vram_loaded - vram_idle
                print(
                    f"VRAM: {vram_loaded:.0f}MB loaded -> {vram_idle:.0f}MB idle "
                    f"(freed {freed_mb:.0f}MB)"
                )
                # Allow some baseline VRAM usage; just check meaningful drop
                assert freed_mb > 100, (
                    f"Expected significant VRAM freed after unload, "
                    f"but only {freed_mb:.0f}MB freed"
                )
