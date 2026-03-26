from __future__ import annotations

import asyncio
import io
import json
import logging  # retained for InterceptHandler — do not remove
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.engine_chatterbox import ChatterboxEngine
from app.engine_chatterbox_full import ChatterboxFullEngine
from app.engine_higgs import HiggsEngine
from app.engine_qwen3 import Qwen3Engine
from app.model_manager import ModelManager
from app.voices import (
    DEFAULT_MAX_DURATION_S,
    VoiceListResponse,
    VoiceMetadata,
    VoiceStore,
    _slugify,
)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


MAX_TEXT_LEN = 5000
VOICES_DIR = os.environ.get("TTS_VOICES_DIR", "./voices")
DEFAULT_MODEL = "chatterbox"
AVAILABLE_VRAM_MB = int(os.environ.get("AVAILABLE_VRAM_MB", "12000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="{time:HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} — {message}",
    )
    logger.add(
        Path(__file__).parent.parent / "tts_server.log",
        level="DEBUG",
        rotation="20 MB",
        retention="7 days",
        serialize=False,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    manager = ModelManager(available_vram_mb=AVAILABLE_VRAM_MB)
    manager.register_engine(ChatterboxEngine())
    manager.register_engine(ChatterboxFullEngine())
    manager.register_engine(HiggsEngine())
    manager.register_engine(Qwen3Engine())

    available = manager.available_models()
    logger.info(f"TTS server starting. VRAM budget: {AVAILABLE_VRAM_MB}MB. Available models: {available}")

    app.state.manager = manager
    app.state.lock = asyncio.Lock()
    app.state.voice_store = VoiceStore(Path(VOICES_DIR))

    manager.start_idle_monitor()

    logger.info(f"Ready. {len(app.state.voice_store.list_voices())} registered voice(s).")
    yield

    await manager.shutdown()


app = FastAPI(title="TTS Server", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"{request.method} {request.url.path}")
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if response.status_code >= 400:
        # Re-read the response body to log the error detail, then re-stream it.
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        logger.warning(
            f"{request.method} {request.url.path} -> {response.status_code}"
            f" | {elapsed_ms:.1f}ms | {body.decode('utf-8', errors='replace')}"
        )
        from fastapi.responses import Response as _Response
        return _Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
    logger.debug(f"{request.method} {request.url.path} -> {response.status_code} | {elapsed_ms:.1f}ms")
    return response


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    model: str = Field(DEFAULT_MODEL, description="Engine to use")
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    voice: str | None = Field(None, description="Voice ID")
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # Chatterbox-specific
    repetition_penalty: float | None = None
    # Chatterbox Full-specific (also accepted by chatterbox for forward-compat)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    min_p: float | None = None
    # Higgs-specific
    seed: int | None = None
    max_new_tokens: int | None = None
    scene_description: str | None = None
    speaker_description: str | None = None
    ras_win_len: int | None = None
    ras_win_max_num_repeat: int | None = None
    force_audio_gen: bool | None = None
    # Qwen3-specific
    qwen3_language: str | None = None
    qwen3_speaker: str | None = None
    qwen3_instruct: str | None = None
    # Voice integrity
    voice_checksum: str | None = None

    @field_validator("text")
    @classmethod
    def no_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v

    @field_validator("voice_checksum")
    @classmethod
    def validate_checksum_format(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.lower()
            if len(v) != 64 or not all(c in "0123456789abcdef" for c in v):
                raise ValueError("voice_checksum must be a 64-character hex string (SHA-256)")
        return v


class VoiceCreateResponse(BaseModel):
    voice_id: str
    name: str
    wpm: float | None = None
    wav_sha256: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _audio_headers(audio: np.ndarray, sample_rate: int) -> dict:
    frame_count = len(audio)
    return {
        "X-Audio-Duration-S": f"{frame_count / sample_rate:.2f}",
        "X-Sample-Rate": str(sample_rate),
        "X-Audio-Frames": str(frame_count),
    }


# ---------------------------------------------------------------------------
# POST /tts
# ---------------------------------------------------------------------------


@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize(req: TTSRequest) -> Response:
    manager: ModelManager = app.state.manager
    voice_store: VoiceStore = app.state.voice_store

    model_name = req.model

    # Validate model is available
    if model_name not in manager.available_models():
        raise HTTPException(
            400,
            detail=f"Model '{model_name}' is not available. "
            f"Available: {manager.available_models()}",
        )

    # Resolve voice references
    voice_ref_path: str | None = None
    voice_ref_text: str | None = None
    conditionals_path: str | None = None

    if req.voice:
        meta = voice_store.get_voice(req.voice)
        if meta is None:
            raise HTTPException(
                404,
                detail={
                    "message": f"Voice not found: {req.voice}",
                    "error_code": "VOICE_NOT_REGISTERED",
                    "voice_id": req.voice,
                },
            )
        if model_name not in meta.compatible_models:
            raise HTTPException(
                400,
                detail=f"Voice '{req.voice}' is not compatible with model '{model_name}'. "
                f"Compatible models: {meta.compatible_models}",
            )

        # Voice checksum validation
        if req.voice_checksum is None:
            raise HTTPException(422, detail="voice_checksum is required when voice is specified")
        if meta.wav_sha256 and req.voice_checksum != meta.wav_sha256:
            raise HTTPException(
                409,
                detail={
                    "message": f"Voice checksum mismatch for '{req.voice}'",
                    "error_code": "VOICE_CHECKSUM_MISMATCH",
                    "voice_id": req.voice,
                    "expected": meta.wav_sha256,
                },
            )
        # If meta.wav_sha256 is "" (legacy voice), allow through without error.

        # For chatterbox, prefer conditionals if available
        if model_name == "chatterbox":
            cond_path = voice_store.get_conditionals_path(req.voice)
            if cond_path is not None:
                conditionals_path = str(cond_path)
            else:
                voice_ref_path = str(voice_store.get_reference_path(req.voice))
        elif model_name == "chatterbox_full":
            cond_path = voice_store.get_conditionals_path(req.voice)
            if cond_path is not None:
                conditionals_path = str(cond_path)
            else:
                voice_ref_path = str(voice_store.get_reference_path(req.voice))
        elif model_name == "qwen3":
            # Blended qwen3 voices have no reference.wav — their identity lives entirely
            # in qwen3_prompt.pkl.  We still pass the expected reference.wav path so the
            # engine can locate the sibling pkl via Path(voice_ref_path).parent.
            voice_ref_path = str(voice_store.get_reference_path(req.voice))
            voice_ref_text = voice_store.get_reference_text(req.voice)
        else:
            voice_ref_path = str(voice_store.get_reference_path(req.voice))
            voice_ref_text = voice_store.get_reference_text(req.voice)

    # Build engine-specific params
    params = {}
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.top_p is not None:
        params["top_p"] = req.top_p
    if req.top_k is not None:
        params["top_k"] = req.top_k

    if model_name == "chatterbox":
        if req.repetition_penalty is not None:
            params["repetition_penalty"] = req.repetition_penalty
        if conditionals_path is not None:
            params["conditionals_path"] = conditionals_path
    elif model_name == "chatterbox_full":
        if req.repetition_penalty is not None:
            params["repetition_penalty"] = req.repetition_penalty
        if req.exaggeration is not None:
            params["exaggeration"] = req.exaggeration
        if req.cfg_weight is not None:
            params["cfg_weight"] = req.cfg_weight
        if req.min_p is not None:
            params["min_p"] = req.min_p
        if conditionals_path is not None:
            params["conditionals_path"] = conditionals_path
    elif model_name == "higgs":
        if req.seed is not None:
            params["seed"] = req.seed
        if req.max_new_tokens is not None:
            params["max_new_tokens"] = req.max_new_tokens
        if req.scene_description is not None:
            params["scene_description"] = req.scene_description
        if req.speaker_description is not None:
            params["speaker_description"] = req.speaker_description
        if req.ras_win_len is not None:
            params["ras_win_len"] = req.ras_win_len
        if req.ras_win_max_num_repeat is not None:
            params["ras_win_max_num_repeat"] = req.ras_win_max_num_repeat
        if req.force_audio_gen is not None:
            params["force_audio_gen"] = req.force_audio_gen
    elif model_name == "qwen3":
        if req.qwen3_language is not None:
            params["qwen3_language"] = req.qwen3_language
        if req.qwen3_speaker is not None:
            params["qwen3_speaker"] = req.qwen3_speaker
        if req.qwen3_instruct is not None:
            params["qwen3_instruct"] = req.qwen3_instruct
        # speaker_description doubles as instruct for voice-design mode
        if req.speaker_description is not None:
            params["speaker_description"] = req.speaker_description
        if req.seed is not None:
            params["seed"] = req.seed
        if req.max_new_tokens is not None:
            params["max_new_tokens"] = req.max_new_tokens

    logger.info(
        f"TTS request: model={req.model}, voice_id={req.voice},"
        f" text_len={len(req.text)}, text={req.text[:80]!r}"
    )
    t0 = time.perf_counter()
    async with app.state.lock:
        engine = await manager.ensure_loaded(model_name)
        audio, sr = await engine.generate(
            text=req.text,
            voice_ref_path=voice_ref_path,
            voice_ref_text=voice_ref_text,
            **params,
        )
    elapsed = time.perf_counter() - t0
    logger.info(f"TTS complete: model={req.model}, generate_ms={elapsed * 1000:.1f}")

    headers = _audio_headers(audio, sr)
    if req.voice:
        voice_meta = voice_store.get_voice(req.voice)
        if voice_meta and voice_meta.wpm is not None:
            headers["X-Voice-WPM"] = f"{voice_meta.wpm:.1f}"

    return Response(
        content=_encode_wav(audio, sr),
        media_type="audio/wav",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# POST /voices/clone
# ---------------------------------------------------------------------------


@app.post("/voices/clone", response_model=VoiceCreateResponse, status_code=201)
async def clone_voice(
    name: str = Form(..., min_length=1, max_length=200),
    reference_audio: UploadFile = File(...),
    reference_text: str | None = Form(
        None,
        max_length=2000,
        description="Transcript of reference audio (required for higgs cloning)",
    ),
    target_model: str | None = Form(
        None,
        description="Target model for this voice (affects compatible_models)",
    ),
    max_duration_s: float = Form(DEFAULT_MAX_DURATION_S, ge=3.0, le=7200.0),
) -> VoiceCreateResponse:
    voice_store: VoiceStore = app.state.voice_store
    audio_bytes = await reference_audio.read()
    original_filename = reference_audio.filename or "unknown.wav"
    model = target_model or "chatterbox"
    logger.info(
        f"Voice clone: model={model}, ref_size={len(audio_bytes)}B,"
        f" ref_text_len={len(reference_text) if reference_text else 0}"
    )

    try:
        meta = voice_store.create_voice(
            name,
            audio_bytes,
            original_filename,
            reference_text=reference_text,
            target_model=target_model,
            max_duration_s=max_duration_s,
        )
    except FileExistsError as exc:
        raise HTTPException(409, detail=f"Voice already exists: {exc}")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    logger.info(f"Voice cloned: voice_id={meta.voice_id}")
    return VoiceCreateResponse(
        voice_id=meta.voice_id,
        name=meta.name,
        wav_sha256=meta.wav_sha256 if meta.wav_sha256 else None,
    )


# ---------------------------------------------------------------------------
# POST /voices/blend (chatterbox + qwen3)
# ---------------------------------------------------------------------------


@app.post("/voices/blend", response_model=VoiceCreateResponse, status_code=201)
async def blend_voices(
    name: str = Form(..., min_length=1, max_length=200),
    voice_a: str = Form(
        ..., description="Voice ID for the first source (supplies rhythm / ref_code)"
    ),
    voice_b: str = Form(..., description="Voice ID for the second source"),
    texture_mix: int = Form(
        50,
        ge=0,
        le=100,
        description="Texture blend: 0 = pure voice_a, 100 = pure voice_b",
    ),
    model: str = Form(
        "chatterbox",
        description="Which engine to blend for: 'chatterbox', 'chatterbox_full', or 'qwen3'",
    ),
) -> VoiceCreateResponse:
    manager: ModelManager = app.state.manager
    voice_store: VoiceStore = app.state.voice_store

    logger.info(f"Voice blend: a={voice_a}, b={voice_b}, weight={texture_mix}")
    for vid, label in [(voice_a, "voice_a"), (voice_b, "voice_b")]:
        meta = voice_store.get_voice(vid)
        if meta is None:
            raise HTTPException(404, detail=f"{label} not found: {vid}")
        if model not in meta.compatible_models:
            raise HTTPException(
                400,
                detail=f"{label} '{vid}' is not compatible with model '{model}'. "
                f"Compatible models: {meta.compatible_models}",
            )

    blend_config = {"voice_a": voice_a, "voice_b": voice_b, "texture_mix": texture_mix}

    if model == "qwen3":
        # Qwen3 blending runs through the subprocess worker.
        pkl_a = voice_store.get_qwen3_prompt_path(voice_a)
        pkl_b = voice_store.get_qwen3_prompt_path(voice_b)
        if pkl_a is None or pkl_b is None:
            missing = voice_a if pkl_a is None else voice_b
            raise HTTPException(
                422,
                detail=f"Voice '{missing}' has no Qwen3 prompt cache. "
                "Make one synthesis request with that voice first to build the cache.",
            )

        voice_id = _slugify(name)
        if voice_store.get_voice(voice_id) is not None:
            raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

        voice_dir = voice_store._base_dir / voice_id
        try:
            voice_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

        out_pkl_path = str(voice_dir / "qwen3_prompt.pkl")
        alpha = texture_mix / 100.0

        async with app.state.lock:
            engine = await manager.ensure_loaded("qwen3")
            await engine.blend_voice_prompts_ipc(
                str(pkl_a), str(pkl_b), alpha, out_pkl_path=out_pkl_path
            )

        sources = f"{voice_a}+{voice_b}"
        meta = VoiceMetadata(
            voice_id=voice_id,
            name=name,
            original_filename=f"blend:{sources}",
            created_at=datetime.now(timezone.utc),
            duration_s=0.0,
            sample_rate=24000,
            compatible_models=["qwen3"],
        )
        raw = json.loads(meta.model_dump_json())
        raw["blend_config"] = blend_config
        (voice_dir / "metadata.json").write_text(json.dumps(raw, indent=2))
        voice_store._index[voice_id] = meta
        logger.info(f"Voice blended: voice_id={meta.voice_id}")
        return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)

    if model == "chatterbox_full":
        # --- chatterbox_full blend ---
        voice_id = _slugify(name)
        if voice_store.get_voice(voice_id) is not None:
            raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

        voice_dir = voice_store._base_dir / voice_id
        try:
            voice_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

        path_a = str(voice_store.get_reference_path(voice_a))
        path_b = str(voice_store.get_reference_path(voice_b))
        out_pt_path = str(voice_dir / "conditionals.pt")

        async with app.state.lock:
            engine = await manager.ensure_loaded("chatterbox_full")
            if not hasattr(engine, "blend_voices"):
                raise HTTPException(500, detail="Blend requires chatterbox_full engine")
            await engine.blend_voices(path_a, path_b, texture_mix, out_pt_path=out_pt_path)

        sources = f"{voice_a}+{voice_b}"
        meta = VoiceMetadata(
            voice_id=voice_id,
            name=name,
            original_filename=f"blend:{sources}",
            created_at=datetime.now(timezone.utc),
            duration_s=0.0,
            sample_rate=engine.sample_rate,
            compatible_models=["chatterbox_full"],
        )
        raw = json.loads(meta.model_dump_json())
        raw["blend_config"] = blend_config
        (voice_dir / "metadata.json").write_text(json.dumps(raw, indent=2))
        voice_store._index[voice_id] = meta
        logger.info(f"Voice blended: voice_id={meta.voice_id}")
        return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)

    # --- chatterbox blend ---
    voice_id = _slugify(name)
    if voice_store.get_voice(voice_id) is not None:
        raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

    voice_dir = voice_store._base_dir / voice_id
    try:
        voice_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise HTTPException(409, detail=f"Voice already exists: {voice_id}")

    path_a = str(voice_store.get_reference_path(voice_a))
    path_b = str(voice_store.get_reference_path(voice_b))
    out_pt_path = str(voice_dir / "conditionals.pt")

    async with app.state.lock:
        engine = await manager.ensure_loaded("chatterbox")
        if not hasattr(engine, "blend_voices"):
            raise HTTPException(500, detail="Blend requires chatterbox engine")
        await engine.blend_voices(path_a, path_b, texture_mix, out_pt_path=out_pt_path)

    sources = f"{voice_a}+{voice_b}"
    meta = VoiceMetadata(
        voice_id=voice_id,
        name=name,
        original_filename=f"blend:{sources}",
        created_at=datetime.now(timezone.utc),
        duration_s=0.0,
        sample_rate=engine.sample_rate,
        compatible_models=["chatterbox"],
    )
    raw = json.loads(meta.model_dump_json())
    raw["blend_config"] = blend_config
    (voice_dir / "metadata.json").write_text(json.dumps(raw, indent=2))
    voice_store._index[voice_id] = meta
    logger.info(f"Voice blended: voice_id={meta.voice_id}")
    return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)


# ---------------------------------------------------------------------------
# POST /voices/design  (qwen3 VoiceDesign → save as cloneable reference)
# ---------------------------------------------------------------------------


class VoiceDesignRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language voice description, e.g. 'A warm, slightly husky male voice'",
    )
    language: str = Field("English", description="Language for the generated sample")
    seed: int | None = Field(None, description="Fixed seed for reproducible voice identity")
    sample_text: str = Field(
        "The quick brown fox jumps over the lazy dog near a quiet river bank. "
        "Bright yellow flowers bloom across the meadow.",
        min_length=10,
        max_length=500,
        description="Text synthesised to produce the reference clip (stored as reference_text)",
    )


@app.post("/voices/design", response_model=VoiceCreateResponse, status_code=201)
async def design_voice(req: VoiceDesignRequest) -> VoiceCreateResponse:
    """Generate a novel voice from a text description, save it, and register it as a
    cloneable Qwen3 voice.

    Requires the server to be running with a VoiceDesign or Base variant model.
    The generated WAV is saved as the voice reference; re-run /voices/design with the
    same description + seed to reproduce an acoustically similar voice.
    """
    import io as _io

    import soundfile as _sf

    manager: ModelManager = app.state.manager
    voice_store: VoiceStore = app.state.voice_store

    if "qwen3" not in manager.available_models():
        raise HTTPException(400, detail="Qwen3 model is not available")

    params: dict = {"qwen3_language": req.language}
    if req.seed is not None:
        params["seed"] = req.seed

    async with app.state.lock:
        engine = await manager.ensure_loaded("qwen3")
        audio, sr = await engine.generate(
            text=req.sample_text,
            voice_ref_path=None,
            voice_ref_text=None,
            speaker_description=req.description,
            **params,
        )

    # Encode to WAV bytes
    buf = _io.BytesIO()
    _sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    try:
        meta = voice_store.create_voice(
            name=req.name,
            audio_bytes=wav_bytes,
            original_filename="designed.wav",
            reference_text=req.sample_text,
            target_model="qwen3",
        )
    except FileExistsError as exc:
        raise HTTPException(409, detail=f"Voice already exists: {exc}")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))

    return VoiceCreateResponse(voice_id=meta.voice_id, name=meta.name)


# ---------------------------------------------------------------------------
# GET /voices
# ---------------------------------------------------------------------------


@app.get("/voices", response_model=VoiceListResponse)
async def list_voices(model: str | None = None) -> VoiceListResponse:
    voice_store: VoiceStore = app.state.voice_store
    return VoiceListResponse(voices=voice_store.list_voices(model=model))


# ---------------------------------------------------------------------------
# GET /voices/{voice_id}
# ---------------------------------------------------------------------------


@app.get("/voices/{voice_id}", response_model=VoiceMetadata)
async def get_voice(voice_id: str) -> VoiceMetadata:
    voice_store: VoiceStore = app.state.voice_store
    meta = voice_store.get_voice(voice_id)
    if meta is None:
        raise HTTPException(404, detail=f"Voice not found: {voice_id}")
    return meta


# ---------------------------------------------------------------------------
# DELETE /voices/{voice_id}
# ---------------------------------------------------------------------------


@app.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str) -> Response:
    voice_store: VoiceStore = app.state.voice_store
    if not voice_store.delete_voice(voice_id):
        raise HTTPException(404, detail=f"Voice not found: {voice_id}")
    logger.info(f"Voice deleted: voice_id={voice_id}")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    manager: ModelManager = app.state.manager
    voice_store: VoiceStore | None = getattr(app.state, "voice_store", None)

    engines_status = {}
    for name, engine in manager.all_engines().items():
        engines_status[name] = {
            "loaded": engine.is_loaded,
            "deps_available": engine.deps_available,
            "estimated_vram_mb": engine.estimated_vram_mb,
            "sample_rate": engine.sample_rate,
        }

    return {
        "status": "ok",
        "active_model": manager.active_engine_name,
        "engines": engines_status,
        "available_vram_mb": AVAILABLE_VRAM_MB,
        "voices": len(voice_store.list_voices()) if voice_store else 0,
    }


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


@app.get("/models")
def list_models() -> list[dict]:
    manager: ModelManager = app.state.manager
    available = manager.available_models()
    result = []
    for name, engine in manager.all_engines().items():
        result.append({
            "name": name,
            "available": name in available,
            "deps_available": engine.deps_available,
            "loaded": engine.is_loaded,
            "active": name == manager.active_engine_name,
            "estimated_vram_mb": engine.estimated_vram_mb,
            "sample_rate": engine.sample_rate,
        })
    return result
