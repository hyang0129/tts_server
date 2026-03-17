from __future__ import annotations

import asyncio
import gc
import logging
import os

import numpy as np
import torch

from app.engine_base import TTSEngine

logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_ID = os.environ.get("QWEN3_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

_raw_dtype = os.environ.get("QWEN3_DTYPE", "bfloat16").strip().lower()
DTYPE = torch.bfloat16 if _raw_dtype == "bfloat16" else torch.float16

# 1.7B model in bfloat16 ≈ 3.4 GB weights + KV cache/activations overhead
ESTIMATED_VRAM_MB = int(os.environ.get("QWEN3_VRAM_MB", "5500"))

# Variant detection: dispatch generation method based on model ID suffix.
# Base     → generate_voice_clone  (voice_ref_path + voice_ref_text required)
# CustomVoice → generate_custom_voice  (qwen3_speaker required)
# VoiceDesign → generate_voice_design  (qwen3_instruct / speaker_description)
_model_id_lower = MODEL_ID.lower()
if "customvoice" in _model_id_lower:
    _MODEL_VARIANT = "customvoice"
elif "voicedesign" in _model_id_lower:
    _MODEL_VARIANT = "voicedesign"
else:
    _MODEL_VARIANT = "base"


class Qwen3Engine(TTSEngine):
    name = "qwen3"
    sample_rate = 24000  # updated after first generate() call
    estimated_vram_mb = ESTIMATED_VRAM_MB

    def __init__(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def deps_available(self) -> bool:
        try:
            import qwen_tts  # noqa: F401

            return True
        except ImportError:
            return False

    async def load(self) -> None:
        from qwen_tts import Qwen3TTSModel

        loop = asyncio.get_running_loop()

        def _load():
            return Qwen3TTSModel.from_pretrained(
                MODEL_ID,
                device_map=DEVICE,
                dtype=DTYPE,
            )

        self._model = await loop.run_in_executor(None, _load)
        logger.info("Qwen3 engine loaded: %s (variant=%s)", MODEL_ID, _MODEL_VARIANT)

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def generate(
        self,
        text: str,
        voice_ref_path: str | None,
        voice_ref_text: str | None,
        **params,
    ) -> tuple[np.ndarray, int]:
        model = self._model
        if model is None:
            raise RuntimeError("Qwen3 engine not loaded")

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

        loop = asyncio.get_running_loop()

        if _MODEL_VARIANT == "base" and voice_ref_path and voice_ref_text:
            # Voice cloning: Base model with reference audio + transcript.
            def _run():
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=voice_ref_path,
                    ref_text=voice_ref_text,
                    **gen_kwargs,
                )
                return wavs[0], sr

        elif _MODEL_VARIANT == "customvoice" or speaker is not None:
            # Preset speaker with optional instruct.
            effective_speaker = speaker or "Ryan"

            def _run():
                kwargs = {}
                if instruct:
                    kwargs["instruct"] = instruct
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=effective_speaker,
                    **kwargs,
                    **gen_kwargs,
                )
                return wavs[0], sr

        else:
            # Voice design: natural language description via instruct parameter.
            def _run():
                kwargs = {}
                if instruct:
                    kwargs["instruct"] = instruct
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=language,
                    **kwargs,
                    **gen_kwargs,
                )
                return wavs[0], sr

        audio, sr = await loop.run_in_executor(None, _run)

        # Update sample_rate from first call (model reports the actual value).
        if sr != self.sample_rate:
            Qwen3Engine.sample_rate = sr

        if audio is None or len(audio) == 0:
            raise RuntimeError("Qwen3 model returned no audio output")

        # Ensure 1-D float32 numpy array.
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        audio = audio.flatten().astype(np.float32)

        return audio, sr
