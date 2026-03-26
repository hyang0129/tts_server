#!/usr/bin/env python3
"""Unit tests for higgs_worker — no GPU or live server required.

These tests stub out torch, boson_multimodal, and examples.generation so the worker
module can be imported and exercised in any environment.
"""
from __future__ import annotations

import base64
import binascii
import os
import sys
import types
from unittest.mock import MagicMock
import numpy as np
import pytest


def _make_stub_modules():
    """Inject minimal stubs so higgs_worker imports without the higgs venv."""
    # torch stub
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = MagicMock()
    torch_mod.cuda.is_available = lambda: False
    sys.modules.setdefault("torch", torch_mod)

    # boson_multimodal stubs
    for name in [
        "boson_multimodal",
        "boson_multimodal.audio_processing",
        "boson_multimodal.audio_processing.higgs_audio_tokenizer",
        "boson_multimodal.data_types",
        "boson_multimodal.model",
        "boson_multimodal.model.higgs_audio",
        "boson_multimodal.model.higgs_audio.modeling_higgs_audio",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    # data_types: Message and AudioContent stubs
    dt = sys.modules["boson_multimodal.data_types"]
    if not hasattr(dt, "Message"):
        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content
        dt.Message = Message
    if not hasattr(dt, "AudioContent"):
        class AudioContent:
            def __init__(self, audio_url):
                self.audio_url = audio_url
        dt.AudioContent = AudioContent

    # load_higgs_audio_tokenizer stub
    tok_mod = sys.modules["boson_multimodal.audio_processing.higgs_audio_tokenizer"]
    if not hasattr(tok_mod, "load_higgs_audio_tokenizer"):
        tok_mod.load_higgs_audio_tokenizer = MagicMock()

    # examples.generation stub
    for name in ["examples", "examples.generation"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    gen_mod = sys.modules["examples.generation"]
    if not hasattr(gen_mod, "HiggsAudioModelClient"):
        gen_mod.HiggsAudioModelClient = MagicMock()


@pytest.fixture(scope="module")
def higgs_worker():
    """Import higgs_worker with all heavy deps stubbed out."""
    _make_stub_modules()
    # Remove any cached import so stubs take effect
    for key in list(sys.modules.keys()):
        if "higgs_worker" in key:
            del sys.modules[key]
    import workers.higgs_worker as hw
    return hw


def _make_wav_b64(samples: int = 2400, sr: int = 24000) -> str:
    """Return base64-encoded minimal WAV."""
    import struct
    # Minimal WAV header for PCM 16-bit mono
    data = b"\x00" * (samples * 2)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", len(data),
    )
    return base64.b64encode(header + data).decode()


class TestContinuationAudio:
    def test_continuation_populates_two_audio_ids(self, higgs_worker):
        """When continuation fields are supplied, audio_ids must have 2 entries."""
        hw = higgs_worker

        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        hw._audio_tokenizer = fake_tokenizer

        fake_waveform = np.zeros(24000, dtype=np.float32).tolist()
        fake_client = MagicMock()
        fake_client.generate.return_value = (fake_waveform, 24000, "")
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        cont_b64 = _make_wav_b64()

        req = {
            "text": "Hello world.",
            "voice_ref_path": "/fake/ref.wav",
            "voice_ref_text": "reference transcript",
            "params": {
                "continuation_audio_base64": cont_b64,
                "continuation_audio_text": "continuation transcript",
            },
        }

        hw._cmd_generate(req)

        # encode called twice: once for ref, once for continuation
        assert fake_tokenizer.encode.call_count == 2

        # audio_ids must have 2 entries
        call_kwargs = fake_client.generate.call_args[1]
        assert len(call_kwargs["audio_ids"]) == 2

    def test_no_continuation_leaves_single_audio_id(self, higgs_worker):
        """Without continuation fields, audio_ids must have exactly 1 entry."""
        hw = higgs_worker

        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        hw._audio_tokenizer = fake_tokenizer

        fake_waveform = np.zeros(24000, dtype=np.float32).tolist()
        fake_client = MagicMock()
        fake_client.generate.return_value = (fake_waveform, 24000, "")
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        req = {
            "text": "Hello world.",
            "voice_ref_path": "/fake/ref.wav",
            "voice_ref_text": "reference transcript",
            "params": {},
        }

        hw._cmd_generate(req)

        assert fake_tokenizer.encode.call_count == 1
        call_kwargs = fake_client.generate.call_args[1]
        assert len(call_kwargs["audio_ids"]) == 1

    def test_continuation_tempfile_cleaned_up(self, higgs_worker, tmp_path):
        """Continuation tempfile must be deleted even if generation raises."""
        hw = higgs_worker

        created_paths: list[str] = []

        import tempfile as _tempfile

        original_ntf = _tempfile.NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            f = original_ntf(*args, **kwargs)
            created_paths.append(f.name)
            return f

        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        hw._audio_tokenizer = fake_tokenizer

        fake_client = MagicMock()
        fake_client.generate.side_effect = RuntimeError("GPU OOM")
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        import unittest.mock as _mock
        with _mock.patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            with pytest.raises(RuntimeError, match="GPU OOM"):
                hw._cmd_generate({
                    "text": "test",
                    "voice_ref_path": "/fake/ref.wav",
                    "voice_ref_text": "ref",
                    "params": {
                        "continuation_audio_base64": _make_wav_b64(),
                        "continuation_audio_text": "cont",
                    },
                })

        # All created tempfiles must have been deleted
        for p in created_paths:
            assert not os.path.exists(p), f"Tempfile not cleaned up: {p}"

    def test_invalid_base64_raises(self, higgs_worker):
        """Malformed base64 in continuation_audio_base64 must raise an error."""
        hw = higgs_worker

        fake_tokenizer = MagicMock()
        hw._audio_tokenizer = fake_tokenizer

        fake_client = MagicMock()
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        with pytest.raises((binascii.Error, ValueError)):
            hw._cmd_generate({
                "text": "test",
                "voice_ref_path": None,
                "voice_ref_text": None,
                "params": {
                    "continuation_audio_base64": "this!is@not#valid$base64!!!",
                    "continuation_audio_text": "some text",
                },
            })

    def test_continuation_bad_wav_no_tempfile_leaked(self, higgs_worker):
        """Valid base64 that decodes to non-WAV bytes must raise before NamedTemporaryFile."""
        hw = higgs_worker

        fake_tokenizer = MagicMock()
        hw._audio_tokenizer = fake_tokenizer

        fake_client = MagicMock()
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        not_a_wav_b64 = base64.b64encode(b"not a wav file").decode()

        import unittest.mock as _mock
        import wave as _wave

        ntf_called = []

        def tracking_ntf(*args, **kwargs):
            ntf_called.append(True)
            import tempfile as _tempfile
            return _tempfile.NamedTemporaryFile(*args, **kwargs)

        with _mock.patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            with pytest.raises((ValueError, _wave.Error)):
                hw._cmd_generate({
                    "text": "test",
                    "voice_ref_path": None,
                    "voice_ref_text": None,
                    "params": {
                        "continuation_audio_base64": not_a_wav_b64,
                        "continuation_audio_text": "some text",
                    },
                })

        assert not ntf_called, "NamedTemporaryFile must not be called when WAV parsing fails"

    def test_continuation_without_voice_ref_has_single_audio_id(self, higgs_worker):
        """Continuation-only (no registered voice ref) must produce audio_ids of length 1."""
        hw = higgs_worker

        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        hw._audio_tokenizer = fake_tokenizer

        fake_waveform = np.zeros(24000, dtype=np.float32).tolist()
        fake_client = MagicMock()
        fake_client.generate.return_value = (fake_waveform, 24000, "")
        fake_client._max_new_tokens = 2048
        hw._client = fake_client

        req = {
            "text": "Hello world.",
            "voice_ref_path": None,
            "voice_ref_text": None,
            "params": {
                "continuation_audio_base64": _make_wav_b64(),
                "continuation_audio_text": "continuation only",
            },
        }

        hw._cmd_generate(req)

        # Only the continuation token — no voice ref
        assert fake_tokenizer.encode.call_count == 1
        call_kwargs = fake_client.generate.call_args[1]
        assert len(call_kwargs["audio_ids"]) == 1
