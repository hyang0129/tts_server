from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
from loguru import logger
from pydantic import BaseModel

MAX_REFERENCE_SIZE = 50 * 1024 * 1024  # 50 MB
MIN_DURATION_S = 3.0
DEFAULT_MAX_DURATION_S = 300.0  # 5 minutes
MIN_SAMPLE_RATE = 16000

_SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog near a quiet river bank. "
    "Bright yellow flowers bloom across the meadow while gentle winds carry "
    "the scent of fresh pine through the valley below. A distant church bell "
    "rings twice, marking the hour as birds circle overhead in wide arcs."
)
CALIBRATION_WORD_COUNT = len(CALIBRATION_TEXT.split())


class VoiceMetadata(BaseModel):
    voice_id: str
    name: str
    original_filename: str
    created_at: datetime
    duration_s: float
    sample_rate: int
    wpm: float | None = None
    compatible_models: list[str] = []
    wav_sha256: str = ""


class VoiceListResponse(BaseModel):
    voices: list[VoiceMetadata]


def _slugify(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError("name must contain at least one alphanumeric character")
    return slug


def _convert_to_wav(audio_bytes: bytes, original_filename: str) -> bytes:
    ext = Path(original_filename).suffix.lower() if original_filename else ""
    if ext in _SOUNDFILE_EXTENSIONS:
        return audio_bytes

    with (
        tempfile.NamedTemporaryFile(suffix=ext or ".bin", delete=False) as src,
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst,
    ):
        src.write(audio_bytes)
        src_path, dst_path = src.name, dst.name

    try:
        logger.debug(f"Converting {original_filename!r} → WAV via ffmpeg")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ac", "1",
                "-acodec", "pcm_s16le",
                dst_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            logger.warning(f"ffmpeg returned non-zero for {original_filename!r}: {stderr!r}")
            raise ValueError(f"ffmpeg conversion failed: {stderr[:200]}")
        return Path(dst_path).read_bytes()
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(dst_path).unlink(missing_ok=True)


def _validate_reference_audio(
    audio_bytes: bytes,
    original_filename: str = "",
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> tuple[bytes, float, int]:
    if len(audio_bytes) > MAX_REFERENCE_SIZE:
        raise ValueError(
            f"Reference audio too large: {len(audio_bytes)} bytes "
            f"(max {MAX_REFERENCE_SIZE // (1024 * 1024)} MB)"
        )

    wav_bytes = _convert_to_wav(audio_bytes, original_filename)

    try:
        data, sample_rate = sf.read(io.BytesIO(wav_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid audio file: {exc}") from exc

    if sample_rate < MIN_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate too low: {sample_rate} Hz (min {MIN_SAMPLE_RATE} Hz)"
        )

    frame_count = data.shape[0]
    duration_s = frame_count / sample_rate
    if duration_s < MIN_DURATION_S:
        raise ValueError(
            f"Reference audio too short: {duration_s:.1f}s (min {MIN_DURATION_S}s)"
        )
    if duration_s > max_duration_s:
        raise ValueError(
            f"Reference audio too long: {duration_s:.1f}s (max {max_duration_s}s)"
        )

    logger.debug(f"Validated audio: {duration_s:.1f}s @ {sample_rate}Hz")
    return wav_bytes, duration_s, sample_rate


class VoiceStore:
    """Unified file-based voice reference storage.

    Each voice is stored as ``<base_dir>/<voice_id>/`` containing:
    - ``metadata.json`` (includes ``compatible_models`` list)
    - ``reference.wav`` — raw reference audio
    - ``reference.txt`` (optional) — transcript for higgs cloning
    - ``conditionals.pt`` (optional) — pre-computed chatterbox conditionals
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, VoiceMetadata] = {}
        self._scan()

    def _scan(self) -> None:
        logger.debug(f"Scanning voice store at {str(self._base_dir)!r}")
        self._index.clear()
        for meta_path in self._base_dir.glob("*/metadata.json"):
            try:
                raw = json.loads(meta_path.read_text())
                meta = VoiceMetadata(**raw)
                # Backfill wav_sha256 for legacy voices that were stored without it.
                if not meta.wav_sha256:
                    ref_wav = meta_path.parent / "reference.wav"
                    if ref_wav.exists():
                        sha = hashlib.sha256(ref_wav.read_bytes()).hexdigest()
                        meta = meta.model_copy(update={"wav_sha256": sha})
                        raw["wav_sha256"] = sha
                        tmp = meta_path.with_suffix(".tmp")
                        tmp.write_text(json.dumps(raw, indent=2))
                        os.replace(tmp, meta_path)
                        logger.info(
                            f"Backfilled wav_sha256 for legacy voice {meta.voice_id!r}: {sha[:16]}..."
                        )
                self._index[meta.voice_id] = meta
            except Exception:
                logger.warning(f"Skipping corrupt voice metadata: {meta_path}")
        logger.debug(f"Loaded {len(self._index)} voice(s) from disk")

    def list_voices(self, model: str | None = None) -> list[VoiceMetadata]:
        voices = sorted(self._index.values(), key=lambda v: v.name)
        if model:
            voices = [v for v in voices if model in v.compatible_models]
        return voices

    def get_voice(self, voice_id: str) -> VoiceMetadata | None:
        return self._index.get(voice_id)

    def get_reference_path(self, voice_id: str) -> Path:
        if voice_id not in self._index:
            raise KeyError(voice_id)
        return self._base_dir / voice_id / "reference.wav"

    def get_reference_text(self, voice_id: str) -> str | None:
        if voice_id not in self._index:
            raise KeyError(voice_id)
        txt_path = self._base_dir / voice_id / "reference.txt"
        if txt_path.exists():
            return txt_path.read_text().strip()
        return None

    def get_qwen3_prompt_path(self, voice_id: str) -> Path | None:
        """Return path to qwen3_prompt.pkl if it exists, else None."""
        if voice_id not in self._index:
            raise KeyError(voice_id)
        pkl_path = self._base_dir / voice_id / "qwen3_prompt.pkl"
        return pkl_path if pkl_path.exists() else None

    def get_conditionals_path(self, voice_id: str) -> Path | None:
        if voice_id not in self._index:
            raise KeyError(voice_id)
        pt_path = self._base_dir / voice_id / "conditionals.pt"
        return pt_path if pt_path.exists() else None

    def create_voice(
        self,
        name: str,
        audio_bytes: bytes,
        original_filename: str,
        *,
        reference_text: str | None = None,
        target_model: str | None = None,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
    ) -> VoiceMetadata:
        voice_id = _slugify(name)
        if voice_id in self._index:
            raise FileExistsError(voice_id)

        wav_bytes, duration_s, sample_rate = _validate_reference_audio(
            audio_bytes, original_filename, max_duration_s=max_duration_s
        )
        logger.debug(f"Computing SHA-256 for {original_filename!r}")
        wav_sha256 = hashlib.sha256(wav_bytes).hexdigest()
        logger.debug(f"SHA-256: {wav_sha256}")

        voice_dir = self._base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)

        ref_path = voice_dir / "reference.wav"
        ref_path.write_bytes(wav_bytes)

        # Determine compatible models based on what's provided.
        compatible = []
        if reference_text:
            txt_path = voice_dir / "reference.txt"
            txt_path.write_text(reference_text)
            compatible.append("higgs")
            # Qwen3 Base model also requires a transcript for voice cloning.
            compatible.append("qwen3")
        # Chatterbox and Chatterbox Full can use any reference WAV (no transcript needed).
        compatible.append("chatterbox")
        compatible.append("chatterbox_full")

        # If a target_model was specified, ensure it's in the list.
        if target_model and target_model not in compatible:
            compatible.append(target_model)

        meta = VoiceMetadata(
            voice_id=voice_id,
            name=name,
            original_filename=original_filename,
            created_at=datetime.now(timezone.utc),
            duration_s=round(duration_s, 2),
            sample_rate=sample_rate,
            compatible_models=sorted(set(compatible)),
            wav_sha256=wav_sha256,
        )
        meta_path = voice_dir / "metadata.json"
        meta_path.write_text(meta.model_dump_json(indent=2))

        self._index[voice_id] = meta
        logger.info(f"Registered voice {name!r} (id={voice_id}, {duration_s:.1f}s)")
        return meta

    def create_blended_voice(
        self,
        name: str,
        conditionals: object,
        blend_config: dict,
        sample_rate: int,
        compatible_model: str = "chatterbox",
    ) -> VoiceMetadata:
        voice_id = _slugify(name)
        if voice_id in self._index:
            raise FileExistsError(voice_id)

        voice_dir = self._base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)

        conditionals.save(str(voice_dir / "conditionals.pt"))

        sources = f"{blend_config['voice_a']}+{blend_config['voice_b']}"
        meta = VoiceMetadata(
            voice_id=voice_id,
            name=name,
            original_filename=f"blend:{sources}",
            created_at=datetime.now(timezone.utc),
            duration_s=0.0,
            sample_rate=sample_rate,
            compatible_models=[compatible_model],
        )
        meta_path = voice_dir / "metadata.json"
        raw = json.loads(meta.model_dump_json())
        raw["blend_config"] = blend_config
        meta_path.write_text(json.dumps(raw, indent=2))

        self._index[voice_id] = meta
        logger.info(f"Created blended voice {name!r} (id={voice_id})")
        return meta

    def create_blended_qwen3_voice(
        self,
        name: str,
        prompt_item: Any,
        blend_config: dict,
    ) -> VoiceMetadata:
        """Store a blended Qwen3 VoiceClonePromptItem as a new voice.

        No reference.wav is created — the voice is identified entirely by its
        qwen3_prompt.pkl, which holds the blended speaker embedding + ref_code.
        """
        voice_id = _slugify(name)
        if voice_id in self._index:
            raise FileExistsError(voice_id)

        voice_dir = self._base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)

        # Write pkl without mtime (no reference.wav to track).
        pkl_path = voice_dir / "qwen3_prompt.pkl"
        with pkl_path.open("wb") as f:
            # Import _CACHE_VERSION lazily to avoid a hard dependency at import time.
            try:
                from app.engine_qwen3 import _CACHE_VERSION
            except ImportError:
                _CACHE_VERSION = 2
            pickle.dump({"version": _CACHE_VERSION, "prompt_items": [prompt_item]}, f)

        sources = f"{blend_config['voice_a']}+{blend_config['voice_b']}"
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

        self._index[voice_id] = meta
        logger.info(f"Created blended Qwen3 voice {name!r} (id={voice_id})")
        return meta

    def update_wpm(self, voice_id: str, wpm: float) -> None:
        meta = self._index.get(voice_id)
        if meta is None:
            raise KeyError(voice_id)
        meta = meta.model_copy(update={"wpm": round(wpm, 1)})
        self._index[voice_id] = meta
        meta_path = self._base_dir / voice_id / "metadata.json"
        meta_path.write_text(meta.model_dump_json(indent=2))

    def delete_voice(self, voice_id: str) -> bool:
        if voice_id not in self._index:
            return False
        voice_dir = self._base_dir / voice_id
        shutil.rmtree(voice_dir, ignore_errors=True)
        del self._index[voice_id]
        logger.info(f"Deleted voice {voice_id}")
        return True
