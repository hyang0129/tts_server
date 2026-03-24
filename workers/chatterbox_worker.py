#!/usr/bin/env python3
"""Subprocess worker for the ChatterboxTurbo TTS engine.

Venv: /workspaces/.venvs/tts_server-chatterbox/
IPC:  newline-delimited JSON over stdin/stdout.

Supported commands
------------------
ping              → {"status": "ok"}
load              → {"status": "ok", "sample_rate": <int>}
generate          → {"status": "ok", "audio": "<base64-float32>", "sample_rate": <int>}
blend_voices      → {"status": "ok", "sample_rate": <int>}
unload            → {"status": "ok"}  then sys.exit(0)
"""
from __future__ import annotations

import base64
import gc
import os
import sys

# Ensure the workers package is importable when the file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from workers.worker_protocol import read_request, send_error, send_ok

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_sample_rate: int = 24000


def _cmd_ping() -> None:
    send_ok()


def _cmd_load() -> None:
    global _model, _sample_rate
    from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore[import]

    _model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    _sample_rate = _model.sr
    send_ok(sample_rate=_sample_rate)


def _cmd_generate(req: dict) -> None:
    from chatterbox.tts_turbo import Conditionals  # type: ignore[import]

    if _model is None:
        raise RuntimeError("Chatterbox engine not loaded")

    text: str = req["text"]
    voice_ref_path: str | None = req.get("voice_ref_path")
    params: dict = req.get("params") or {}

    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.95)
    top_k = params.get("top_k", 1000)
    repetition_penalty = params.get("repetition_penalty", 1.2)
    conditionals_path = params.get("conditionals_path")

    if conditionals_path is not None:
        _model.conds = Conditionals.load(conditionals_path, map_location=DEVICE)
        wav = _model.generate(
            text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    else:
        wav = _model.generate(
            text,
            audio_prompt_path=voice_ref_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    # wav is (1, N) tensor — squeeze to 1D numpy float32
    audio: np.ndarray = wav[0].cpu().numpy().astype(np.float32)
    audio_b64 = base64.b64encode(audio.tobytes()).decode()
    send_ok(audio=audio_b64, sample_rate=_sample_rate)


def _cmd_blend_voices(req: dict) -> None:
    from chatterbox.models.t3.modules.cond_enc import T3Cond  # type: ignore[import]
    from chatterbox.tts_turbo import Conditionals  # type: ignore[import]

    if _model is None:
        raise RuntimeError("Chatterbox engine not loaded")

    path_a: str = req["path_a"]
    path_b: str = req["path_b"]
    texture_mix: int = req["texture_mix"]
    out_pt_path: str = req["out_pt_path"]

    _model.prepare_conditionals(path_a)
    conds_a = _model.conds

    _model.prepare_conditionals(path_b)
    conds_b = _model.conds

    alpha = texture_mix / 100.0

    emb_a = conds_a.t3.speaker_emb.float()
    emb_b = conds_b.t3.speaker_emb.float()
    blended_t3 = (1.0 - alpha) * emb_a + alpha * emb_b
    blended_t3 = blended_t3 / blended_t3.norm(p=2, dim=-1, keepdim=True)

    xvec_a = conds_a.gen["embedding"].float()
    xvec_b = conds_b.gen["embedding"].float()
    blended_xvec = (1.0 - alpha) * xvec_a + alpha * xvec_b
    blended_xvec = blended_xvec / blended_xvec.norm(p=2, dim=-1, keepdim=True)

    t3_cond = T3Cond(
        speaker_emb=blended_t3.to(dtype=emb_a.dtype),
        cond_prompt_speech_tokens=conds_a.t3.cond_prompt_speech_tokens,
        emotion_adv=conds_a.t3.emotion_adv,
    )

    gen_dict = dict(conds_a.gen)
    gen_dict["embedding"] = blended_xvec.to(dtype=xvec_a.dtype)

    result = Conditionals(t3_cond, gen_dict)
    result.save(out_pt_path)

    send_ok(sample_rate=_sample_rate)


def _cmd_unload() -> None:
    global _model
    if _model is not None:
        del _model
        _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    send_ok()
    sys.exit(0)


_HANDLERS = {
    "ping": _cmd_ping,
    "load": _cmd_load,
    "generate": _cmd_generate,
    "blend_voices": _cmd_blend_voices,
    "unload": _cmd_unload,
}


def main() -> None:
    print("chatterbox_worker ready", file=sys.stderr, flush=True)
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
            if cmd in ("generate", "blend_voices"):
                handler(req)
            else:
                handler()
        except BrokenPipeError:
            sys.exit(0)
        except Exception as exc:
            send_error(exc, include_traceback=True)


if __name__ == "__main__":
    main()
