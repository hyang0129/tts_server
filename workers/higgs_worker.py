#!/usr/bin/env python3
"""Subprocess worker for the Higgs Audio TTS engine.

Venv: /workspaces/.venvs/tts_server-higgs/
IPC:  newline-delimited JSON over stdin/stdout.

Supported commands
------------------
ping              → {"status": "ok"}
load              → {"status": "ok", "sample_rate": <int>}
generate          → {"status": "ok", "audio": "<base64-float32>", "sample_rate": <int>}
unload            → {"status": "ok"}  then sys.exit(0)
"""
from __future__ import annotations

import base64
import gc
import os
import sys

# Redirect sys.stdout → stderr so library prints don't corrupt the JSON-RPC pipe.
# worker_protocol.send() uses os.write(1, ...) directly and is unaffected.
sys.stdout = sys.stderr

# Ensure the workers package is importable when the file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from workers.worker_protocol import read_request, send_error, send_ok

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HIGGS_REPO = os.environ.get("HIGGS_REPO_PATH") or os.path.join(
    os.environ.get("USERPROFILE", r"C:\Users\Default"), "tmp", "faster-higgs-audio"
)
MODEL_ID = os.environ.get("HIGGS_MODEL_ID", "bosonai/higgs-audio-v2-generation-3B-base")
TOKENIZER_ID = os.environ.get("HIGGS_TOKENIZER_ID", "bosonai/higgs-audio-v2-tokenizer")

_raw_quant = os.environ.get("HIGGS_QUANT_BITS", "8").strip().lower()
if _raw_quant in ("0", "none", "false", "off"):
    QUANTIZATION_BITS = 0
else:
    QUANTIZATION_BITS = int(_raw_quant)

# Attention implementation: "flash_attention_2" (default), "sdpa", or "eager".
# "flash_attention_2" requires flash_attn (installed by scripts/setup_venvs.sh).
# "sdpa" uses torch's built-in scaled_dot_product_attention — no extra packages.
_VALID_ATTN_IMPLS = {"flash_attention_2", "sdpa", "eager"}
ATTN_IMPL = os.environ.get("HIGGS_ATTN_IMPL", "flash_attention_2").strip().lower()
if ATTN_IMPL not in _VALID_ATTN_IMPLS:
    raise ValueError(
        f"HIGGS_ATTN_IMPL={ATTN_IMPL!r} is not valid. "
        f"Must be one of: {', '.join(sorted(_VALID_ATTN_IMPLS))}"
    )

DEFAULT_SCENE = "Audio is recorded from a quiet room."

_SAMPLE_RATE = 24000

_client = None
_audio_tokenizer = None
_attn_patched = False


def _ensure_higgs_path() -> None:
    if _HIGGS_REPO not in sys.path:
        sys.path.insert(0, _HIGGS_REPO)


def _cmd_ping() -> None:
    send_ok()


def _patch_attn_impl() -> None:
    """Monkey-patch HiggsAudioModel.from_pretrained to inject attn_implementation.

    HiggsAudioModelClient doesn't expose attn_implementation as a constructor arg,
    so we wrap from_pretrained to inject it before each call.  Only active when
    ATTN_IMPL differs from the transformers default ("eager").
    """
    global _attn_patched
    if _attn_patched or ATTN_IMPL == "eager":
        return
    from boson_multimodal.model.higgs_audio.modeling_higgs_audio import (  # type: ignore[import]
        HiggsAudioModel,
    )

    _orig_func = HiggsAudioModel.from_pretrained.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _patched(cls, *args, **kwargs):  # type: ignore[misc]
        kwargs.setdefault("attn_implementation", ATTN_IMPL)
        return _orig_func(cls, *args, **kwargs)

    HiggsAudioModel.from_pretrained = _patched  # type: ignore[method-assign]
    _attn_patched = True
    print(f"higgs_worker: attn_implementation={ATTN_IMPL!r}", file=sys.stderr, flush=True)


def _cmd_load() -> None:
    global _client, _audio_tokenizer
    _ensure_higgs_path()

    from boson_multimodal.audio_processing.higgs_audio_tokenizer import (  # type: ignore[import]
        load_higgs_audio_tokenizer,
    )
    from examples.generation import HiggsAudioModelClient  # type: ignore[import]

    _patch_attn_impl()
    _audio_tokenizer = load_higgs_audio_tokenizer(TOKENIZER_ID, device=DEVICE)
    _client = HiggsAudioModelClient(
        model_path=MODEL_ID,
        audio_tokenizer=_audio_tokenizer,
        use_quantization=QUANTIZATION_BITS > 0,
        quantization_bits=QUANTIZATION_BITS if QUANTIZATION_BITS > 0 else 4,
    )
    send_ok(sample_rate=_SAMPLE_RATE)


def _cmd_generate(req: dict) -> None:
    _ensure_higgs_path()

    from boson_multimodal.data_types import AudioContent, Message  # type: ignore[import]

    if _client is None or _audio_tokenizer is None:
        raise RuntimeError("Higgs engine not loaded")

    text: str = req["text"]
    voice_ref_path: str | None = req.get("voice_ref_path")
    voice_ref_text: str | None = req.get("voice_ref_text")
    params: dict = req.get("params") or {}

    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.95)
    top_k = params.get("top_k", 50)
    seed = params.get("seed")
    max_new_tokens = params.get("max_new_tokens", 2048)
    scene_description = params.get("scene_description")
    speaker_description = params.get("speaker_description")
    ras_win_len = params.get("ras_win_len", 7)
    ras_win_max_num_repeat = params.get("ras_win_max_num_repeat", 2)

    # Build system prompt
    scene = scene_description or DEFAULT_SCENE
    if speaker_description:
        scene = f"SPEAKER0: {speaker_description}\n{scene}"
    system_prompt = (
        "Generate audio following instruction.\n\n"
        f"<|scene_desc_start|>\n{scene}\n<|scene_desc_end|>"
    )

    messages: list[Message] = [
        Message(role="system", content=system_prompt),
    ]
    audio_ids: list = []

    # Voice cloning: reference audio + transcript
    if voice_ref_path and voice_ref_text:
        audio_tokens = _audio_tokenizer.encode(voice_ref_path)
        audio_ids.append(audio_tokens)
        messages.append(Message(role="user", content=voice_ref_text))
        messages.append(
            Message(
                role="assistant",
                content=AudioContent(audio_url=voice_ref_path),
            )
        )

    effective_ras = ras_win_len if ras_win_len and ras_win_len > 0 else 0

    original_max = _client._max_new_tokens
    _client._max_new_tokens = max_new_tokens
    try:
        waveform, sr, _text_output = _client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=[text],
            generation_chunk_buffer_size=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            ras_win_len=effective_ras,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed if seed is not None else 0,
        )
    finally:
        _client._max_new_tokens = original_max

    if waveform is None:
        raise RuntimeError("Higgs model returned no audio output")

    audio: np.ndarray = np.array(waveform, dtype=np.float32).flatten()
    audio_b64 = base64.b64encode(audio.tobytes()).decode()
    send_ok(audio=audio_b64, sample_rate=_SAMPLE_RATE)


def _cmd_unload() -> None:
    global _client, _audio_tokenizer
    if _client is not None:
        del _client
        _client = None
    if _audio_tokenizer is not None:
        del _audio_tokenizer
        _audio_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    send_ok()
    sys.exit(0)


_HANDLERS = {
    "ping": _cmd_ping,
    "load": _cmd_load,
    "generate": _cmd_generate,
    "unload": _cmd_unload,
}


def main() -> None:
    print("higgs_worker ready", file=sys.stderr, flush=True)
    while True:
        try:
            req = read_request()
        except BrokenPipeError:
            sys.exit(0)

        if req is None:
            # EOF — parent closed stdin
            sys.exit(0)

        cmd = req.get("cmd")
        handler = _HANDLERS.get(cmd)
        if handler is None:
            send_error(ValueError(f"Unknown command: {cmd!r}"), include_traceback=False)
            continue

        try:
            if cmd == "generate":
                handler(req)
            else:
                handler()
        except BrokenPipeError:
            sys.exit(0)
        except Exception as exc:
            send_error(exc, include_traceback=True)


if __name__ == "__main__":
    main()
