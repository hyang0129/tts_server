from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys

import numpy as np
import torch

from app.engine_base import TTSEngine

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HIGGS_REPO = os.environ.get("HIGGS_REPO_PATH", "/tmp/faster-higgs-audio")
MODEL_ID = os.environ.get(
    "HIGGS_MODEL_ID", "bosonai/higgs-audio-v2-generation-3B-base"
)
TOKENIZER_ID = os.environ.get(
    "HIGGS_TOKENIZER_ID", "bosonai/higgs-audio-v2-tokenizer"
)
DEFAULT_SCENE = "Audio is recorded from a quiet room."

_raw_quant = os.environ.get("HIGGS_QUANT_BITS", "8").strip().lower()
if _raw_quant in ("0", "none", "false", "off"):
    QUANTIZATION_BITS = 0
else:
    QUANTIZATION_BITS = int(_raw_quant)

# VRAM estimates: ~5GB for 4-bit, ~9GB for 8-bit, ~12GB for bf16
_VRAM_BY_QUANT = {0: 12000, 4: 5000, 8: 9000}


def _ensure_higgs_path() -> None:
    if _HIGGS_REPO not in sys.path:
        sys.path.insert(0, _HIGGS_REPO)


class HiggsEngine(TTSEngine):
    name = "higgs"
    sample_rate = 24000
    estimated_vram_mb = _VRAM_BY_QUANT.get(QUANTIZATION_BITS, 9000)

    def __init__(self) -> None:
        self._client = None
        self._audio_tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None and self._audio_tokenizer is not None

    @property
    def deps_available(self) -> bool:
        try:
            _ensure_higgs_path()
            import boson_multimodal  # noqa: F401
            from examples import generation  # noqa: F401
            return True
        except ImportError:
            return False

    async def load(self) -> None:
        _ensure_higgs_path()

        from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
            load_higgs_audio_tokenizer,
        )
        from examples.generation import HiggsAudioModelClient

        loop = asyncio.get_running_loop()

        def _load():
            audio_tokenizer = load_higgs_audio_tokenizer(
                TOKENIZER_ID, device=DEVICE
            )
            client = HiggsAudioModelClient(
                model_path=MODEL_ID,
                audio_tokenizer=audio_tokenizer,
                use_quantization=QUANTIZATION_BITS > 0,
                quantization_bits=QUANTIZATION_BITS if QUANTIZATION_BITS > 0 else 4,
            )
            return client, audio_tokenizer

        self._client, self._audio_tokenizer = await loop.run_in_executor(
            None, _load
        )

    async def unload(self) -> None:
        if self._client is not None:
            del self._client
            self._client = None
        if self._audio_tokenizer is not None:
            del self._audio_tokenizer
            self._audio_tokenizer = None
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
        _ensure_higgs_path()

        from boson_multimodal.data_types import AudioContent, Message

        client = self._client
        audio_tokenizer = self._audio_tokenizer
        if client is None or audio_tokenizer is None:
            raise RuntimeError("Higgs engine not loaded")

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
            audio_tokens = audio_tokenizer.encode(voice_ref_path)
            audio_ids.append(audio_tokens)
            messages.append(Message(role="user", content=voice_ref_text))
            messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=voice_ref_path),
                )
            )

        effective_ras = ras_win_len if ras_win_len and ras_win_len > 0 else 0

        loop = asyncio.get_running_loop()

        def _run():
            original_max = client._max_new_tokens
            client._max_new_tokens = max_new_tokens
            try:
                waveform, sr, text_output = client.generate(
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
                client._max_new_tokens = original_max

            if waveform is None:
                raise RuntimeError("Higgs model returned no audio output")
            return waveform

        audio = await loop.run_in_executor(None, _run)
        return audio, self.sample_rate
