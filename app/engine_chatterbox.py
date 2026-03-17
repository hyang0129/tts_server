from __future__ import annotations

import asyncio
import gc
import logging

import numpy as np
import torch

from app.engine_base import TTSEngine

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    sample_rate = 24000  # updated after load from model.sr
    estimated_vram_mb = 4700

    def __init__(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def deps_available(self) -> bool:
        try:
            import chatterbox  # noqa: F401
            return True
        except ImportError:
            return False

    async def load(self) -> None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(
            None, lambda: ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        )
        self._model = model
        self.sample_rate = model.sr

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
        from chatterbox.tts_turbo import Conditionals

        model = self._model
        if model is None:
            raise RuntimeError("Chatterbox engine not loaded")

        temperature = params.get("temperature", 0.8)
        top_p = params.get("top_p", 0.95)
        top_k = params.get("top_k", 1000)
        repetition_penalty = params.get("repetition_penalty", 1.2)
        conditionals_path = params.get("conditionals_path")

        loop = asyncio.get_running_loop()

        def _run():
            if conditionals_path is not None:
                model.conds = Conditionals.load(
                    conditionals_path, map_location=DEVICE
                )
                return model.generate(
                    text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            return model.generate(
                text,
                audio_prompt_path=voice_ref_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        wav = await loop.run_in_executor(None, _run)
        # wav is a (1, N) tensor — squeeze to 1D numpy
        audio = wav[0].cpu().numpy()
        return audio, self.sample_rate

    async def blend_voices(
        self,
        path_a: str,
        path_b: str,
        texture_mix: int,
    ):
        """Blend two voice references into new Conditionals."""
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.tts_turbo import Conditionals

        model = self._model
        if model is None:
            raise RuntimeError("Chatterbox engine not loaded")

        loop = asyncio.get_running_loop()

        def _run():
            model.prepare_conditionals(path_a)
            conds_a = model.conds

            model.prepare_conditionals(path_b)
            conds_b = model.conds

            alpha = texture_mix / 100.0

            emb_a = conds_a.t3.speaker_emb.float()
            emb_b = conds_b.t3.speaker_emb.float()
            blended_t3 = (1.0 - alpha) * emb_a + alpha * emb_b
            blended_t3 = blended_t3 / blended_t3.norm(p=2, dim=-1, keepdim=True)

            xvec_a = conds_a.gen["embedding"].float()
            xvec_b = conds_b.gen["embedding"].float()
            blended_xvec = (1.0 - alpha) * xvec_a + alpha * xvec_b
            blended_xvec = blended_xvec / blended_xvec.norm(
                p=2, dim=-1, keepdim=True
            )

            t3_cond = T3Cond(
                speaker_emb=blended_t3.to(dtype=emb_a.dtype),
                cond_prompt_speech_tokens=conds_a.t3.cond_prompt_speech_tokens,
                emotion_adv=conds_a.t3.emotion_adv,
            )

            gen_dict = dict(conds_a.gen)
            gen_dict["embedding"] = blended_xvec.to(dtype=xvec_a.dtype)

            return Conditionals(t3_cond, gen_dict)

        return await loop.run_in_executor(None, _run)
