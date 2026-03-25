#!/usr/bin/env python3
"""Subprocess worker for the Qwen3 TTS engine.

Venv: /workspaces/.venvs/tts_server-qwen3/
IPC:  newline-delimited JSON over stdin/stdout.

Supported commands
------------------
ping                → {"status": "ok"}
load                → {"status": "ok", "sample_rate": <int>}
generate            → {"status": "ok", "audio": "<base64-float32>", "sample_rate": <int>}
blend_voice_prompts → {"status": "ok"}
unload              → {"status": "ok"}  then sys.exit(0)
"""
from __future__ import annotations

import base64
import gc
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

# Redirect sys.stdout → stderr so library prints don't corrupt the JSON-RPC pipe.
# worker_protocol.send() uses os.write(1, ...) directly and is unaffected.
sys.stdout = sys.stderr

# Ensure the workers package is importable when the file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from workers.worker_protocol import read_request, send_error, send_ok

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_ID = os.environ.get("QWEN3_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

_raw_dtype = os.environ.get("QWEN3_DTYPE", "bfloat16").strip().lower()
DTYPE = torch.bfloat16 if _raw_dtype == "bfloat16" else torch.float16

# Voice-clone tuning constants (Base model only).
_MAX_REF_SECONDS: float = 8.0
# Cache version — bump when generation behaviour changes to invalidate stale pickles.
_CACHE_VERSION: int = 2
# Codec token rate (Hz) used to estimate max_new_tokens from text length.
_CODEC_HZ: int = 12
# Words-per-second assumed for estimating audio duration from text.
_WPS: float = 2.5

# Variant detection: dispatch generation method based on model ID suffix.
_model_id_lower = MODEL_ID.lower()
if "customvoice" in _model_id_lower:
    _MODEL_VARIANT = "customvoice"
elif "voicedesign" in _model_id_lower:
    _MODEL_VARIANT = "voicedesign"
else:
    _MODEL_VARIANT = "base"

_model = None
# Memory-level prompt cache: maps voice_ref_path -> prompt_items
_prompt_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Disk-cache helpers
# ---------------------------------------------------------------------------

def _disk_cache_path(voice_ref_path: str) -> Path:
    return Path(voice_ref_path).parent / "qwen3_prompt.pkl"


def _load_disk_cache(voice_ref_path: str) -> Any | None:
    """Try to load prompt_items from disk cache. Returns None on any failure."""
    cache_path = _disk_cache_path(voice_ref_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            stored = pickle.load(f)
        ref_path_obj = Path(voice_ref_path)
        if ref_path_obj.exists():
            ref_mtime = ref_path_obj.stat().st_mtime
            if stored.get("mtime") != ref_mtime:
                print(
                    f"Qwen3 disk cache stale for {voice_ref_path}, will recompute",
                    file=sys.stderr,
                    flush=True,
                )
                return None
        # else: blended voice — no reference.wav, skip mtime check
        if stored.get("version") != _CACHE_VERSION:
            print(
                f"Qwen3 disk cache version mismatch for {voice_ref_path}, will recompute",
                file=sys.stderr,
                flush=True,
            )
            return None
        return stored["prompt_items"]
    except Exception:
        print(
            f"Qwen3 disk cache load failed for {voice_ref_path}, will recompute",
            file=sys.stderr,
            flush=True,
        )
        return None


def _save_disk_cache(voice_ref_path: str, prompt_items: Any) -> None:
    """Try to save prompt_items to disk cache. Ignores any failure."""
    try:
        ref_mtime = Path(voice_ref_path).stat().st_mtime
        cache_path = _disk_cache_path(voice_ref_path)
        with cache_path.open("wb") as f:
            pickle.dump(
                {"version": _CACHE_VERSION, "mtime": ref_mtime, "prompt_items": prompt_items},
                f,
            )
        print(f"Qwen3 prompt cache saved to {cache_path}", file=sys.stderr, flush=True)
    except Exception:
        print(
            f"Qwen3 disk cache save failed for {voice_ref_path}",
            file=sys.stderr,
            flush=True,
        )


def _trim_ref_audio_path(voice_ref_path: str) -> str | None:
    """If the reference WAV exceeds _MAX_REF_SECONDS, write a trimmed copy to a
    temp file and return its path. Returns None if no trimming is needed."""
    try:
        import scipy.io.wavfile  # type: ignore[import]

        sr, data = scipy.io.wavfile.read(voice_ref_path)
        max_samples = int(_MAX_REF_SECONDS * sr)
        if data.shape[0] <= max_samples:
            return None  # already short enough
        trimmed = data[:max_samples]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        scipy.io.wavfile.write(tmp.name, sr, trimmed)
        print(
            f"Qwen3 trimmed ref audio from {data.shape[0] / sr:.1f}s "
            f"to {_MAX_REF_SECONDS:.1f}s -> {tmp.name}",
            file=sys.stderr,
            flush=True,
        )
        return tmp.name
    except Exception:
        print("Qwen3 ref audio trim failed, using original", file=sys.stderr, flush=True)
        return None


def _has_cached_prompt(voice_ref_path: str) -> bool:
    """Return True if a prompt is already cached (memory or disk) for this voice."""
    if voice_ref_path in _prompt_cache:
        return True
    return _disk_cache_path(voice_ref_path).exists()


def _get_voice_clone_prompt(model: Any, voice_ref_path: str, voice_ref_text: str | None) -> Any:
    """Return cached prompt_items, computing and caching if needed."""
    if voice_ref_path in _prompt_cache:
        return _prompt_cache[voice_ref_path]

    prompt_items = _load_disk_cache(voice_ref_path)
    if prompt_items is None:
        if not voice_ref_text:
            raise RuntimeError(
                f"No cached prompt and no reference_text for {voice_ref_path!r}. "
                "Re-clone the voice with reference_text to enable Qwen3 cloning."
            )
        tmp_path = _trim_ref_audio_path(voice_ref_path)
        ref_to_encode = tmp_path if tmp_path is not None else voice_ref_path
        try:
            prompt_items = model.create_voice_clone_prompt(ref_to_encode, voice_ref_text)
        finally:
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
        _save_disk_cache(voice_ref_path, prompt_items)

    _prompt_cache[voice_ref_path] = prompt_items
    return prompt_items


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_ping() -> None:
    send_ok()


def _cmd_load() -> None:
    global _model
    from qwen_tts import Qwen3TTSModel  # type: ignore[import]

    kwargs: dict = dict(device_map=DEVICE, dtype=DTYPE)
    try:
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_ID, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception:
        print(
            "Flash Attention 2 not available for Qwen3, using default attention",
            file=sys.stderr,
            flush=True,
        )
        _model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)

    print(
        f"Qwen3 worker loaded: {MODEL_ID} (variant={_MODEL_VARIANT})",
        file=sys.stderr,
        flush=True,
    )
    # sample_rate is reported after first generate(); use 24000 as placeholder.
    send_ok(sample_rate=24000)


def _cmd_generate(req: dict) -> None:  # noqa: C901 — complexity from variant dispatch
    global _model

    if _model is None:
        raise RuntimeError("Qwen3 engine not loaded")

    text: str = req["text"]
    voice_ref_path: str | None = req.get("voice_ref_path")
    voice_ref_text: str | None = req.get("voice_ref_text")
    params: dict = req.get("params") or {}

    language = params.get("qwen3_language") or "Auto"
    speaker = params.get("qwen3_speaker")
    instruct = params.get("qwen3_instruct") or params.get("speaker_description")

    # Forward standard generation params as HuggingFace generation kwargs.
    gen_kwargs: dict = {}
    if params.get("temperature") is not None:
        gen_kwargs["temperature"] = params["temperature"]
    if params.get("top_p") is not None:
        gen_kwargs["top_p"] = params["top_p"]
    if params.get("top_k") is not None:
        gen_kwargs["top_k"] = params["top_k"]
    if params.get("max_new_tokens") is not None:
        gen_kwargs["max_new_tokens"] = params["max_new_tokens"]
    if params.get("seed") is not None:
        torch.manual_seed(params["seed"])

    audio = None
    sr = 24000

    if _MODEL_VARIANT == "base" and voice_ref_path and (
        voice_ref_text or _has_cached_prompt(voice_ref_path)
    ):
        # Voice cloning: Base model with reference audio + transcript.
        clone_kwargs = dict(gen_kwargs)
        if "temperature" not in clone_kwargs:
            clone_kwargs["temperature"] = 0.5
        if "max_new_tokens" not in clone_kwargs:
            word_count = len(text.split())
            estimated_frames = int(word_count / _WPS * _CODEC_HZ)
            clone_kwargs["max_new_tokens"] = max(120, estimated_frames * 3)

        prompt_items = _get_voice_clone_prompt(_model, voice_ref_path, voice_ref_text)

        if not instruct:
            wavs, sr = _model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
                non_streaming_mode=True,
                **clone_kwargs,
            )
            audio = wavs[0]
        else:
            # instruct + voice clone: bypass the wrapper and call the core model directly.
            prompt_dict = _model._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts = [it.ref_text for it in prompt_items]
            input_ids = _model._tokenize_texts([_model._build_assistant_text(text)])
            ref_ids = [
                _model._tokenize_texts([_model._build_ref_text(rt)])[0]
                if rt else None
                for rt in ref_texts
            ]
            instruct_ids = [_model._tokenize_texts([_model._build_instruct_text(instruct)])[0]]
            gen_kw = _model._merge_generate_kwargs(**clone_kwargs)
            talker_codes, _ = _model.model.generate(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=prompt_dict,
                languages=[language],
                non_streaming_mode=True,
                **gen_kw,
            )
            wavs, sr = _model.model.speech_tokenizer.decode(
                [{"audio_codes": c} for c in talker_codes]
            )
            audio = wavs[0]

    elif _MODEL_VARIANT == "base":
        # Base model without voice reference: unconditioned generation.
        # generate_voice_design() is not supported on Base; use the low-level
        # model.generate() path (same as instruct+voice_clone but no ref audio).
        base_kwargs = dict(gen_kwargs)
        if "temperature" not in base_kwargs:
            base_kwargs["temperature"] = 0.5
        if "max_new_tokens" not in base_kwargs:
            word_count = len(text.split())
            estimated_frames = int(word_count / _WPS * _CODEC_HZ)
            base_kwargs["max_new_tokens"] = max(120, estimated_frames * 3)

        input_ids = _model._tokenize_texts([_model._build_assistant_text(text)])
        gen_kw = _model._merge_generate_kwargs(**base_kwargs)

        instruct_ids_list = None
        if instruct:
            instruct_ids_list = [
                _model._tokenize_texts([_model._build_instruct_text(instruct)])[0]
            ]

        talker_codes, _ = _model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids_list,
            ref_ids=None,
            voice_clone_prompt=None,
            languages=[language],
            non_streaming_mode=True,
            **gen_kw,
        )
        wavs, sr = _model.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes]
        )
        audio = wavs[0]

    elif _MODEL_VARIANT == "customvoice" or speaker is not None:
        # Preset speaker with optional instruct.
        effective_speaker = speaker or "Ryan"
        kwargs: dict = {}
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = _model.generate_custom_voice(
            text=text,
            language=language,
            speaker=effective_speaker,
            **kwargs,
            **gen_kwargs,
        )
        audio = wavs[0]

    else:
        # Voice design: natural language description via instruct parameter.
        kwargs = {}
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sr = _model.generate_voice_design(
            text=text,
            language=language,
            **kwargs,
            **gen_kwargs,
        )
        audio = wavs[0]

    if audio is None or len(audio) == 0:
        raise RuntimeError("Qwen3 model returned no audio output")

    # Ensure 1-D float32 numpy array.
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    audio = audio.flatten().astype(np.float32)

    audio_b64 = base64.b64encode(audio.tobytes()).decode()
    send_ok(audio=audio_b64, sample_rate=int(sr))


def _cmd_blend_voice_prompts(req: dict) -> None:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem  # type: ignore[import]

    pkl_path_a: str = req["pkl_path_a"]
    pkl_path_b: str = req["pkl_path_b"]
    alpha: float = float(req["alpha"])
    out_pkl_path: str = req["out_pkl_path"]

    def _load(path: str) -> Any:
        with open(path, "rb") as f:
            stored = pickle.load(f)
        items = stored["prompt_items"]
        return items[0] if isinstance(items, list) else items

    item_a = _load(pkl_path_a)
    item_b = _load(pkl_path_b)

    # Blend speaker embedding (L2-normalised interpolation).
    emb_a = item_a.ref_spk_embedding.cpu().float()
    emb_b = item_b.ref_spk_embedding.cpu().float()
    blended_emb = (1.0 - alpha) * emb_a + alpha * emb_b
    blended_emb = blended_emb / blended_emb.norm(p=2)
    blended_emb = blended_emb.to(item_a.ref_spk_embedding.dtype)

    # ref_code holds discrete codebook token IDs — keep voice_a's tokens.
    rc_a = item_a.ref_code
    blended_code = rc_a.cpu() if rc_a is not None else None

    blended_item = VoiceClonePromptItem(
        ref_code=blended_code,
        ref_spk_embedding=blended_emb,
        x_vector_only_mode=item_a.x_vector_only_mode,
        icl_mode=item_a.icl_mode,
        ref_text=item_a.ref_text,
    )

    # Write output pkl — no mtime field (blended voices have no reference.wav to track).
    Path(out_pkl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(
            {"version": _CACHE_VERSION, "prompt_items": [blended_item]},
            f,
        )

    send_ok()


def _cmd_unload() -> None:
    global _model
    _prompt_cache.clear()
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
    "blend_voice_prompts": _cmd_blend_voice_prompts,
    "unload": _cmd_unload,
}


def main() -> None:
    print("qwen3_worker ready", file=sys.stderr, flush=True)
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
            if cmd in ("generate", "blend_voice_prompts"):
                handler(req)
            else:
                handler()
        except BrokenPipeError:
            sys.exit(0)
        except Exception as exc:
            send_error(exc, include_traceback=True)


if __name__ == "__main__":
    main()
