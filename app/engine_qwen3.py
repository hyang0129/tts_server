from __future__ import annotations

import asyncio
import gc
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

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

# Voice-clone tuning constants (Base model only).
# Optimal reference audio length is 3-10s; longer refs increase EOS instability.
_MAX_REF_SECONDS: float = 8.0
# Cache version — bump when generation behaviour changes to invalidate stale pickles.
_CACHE_VERSION: int = 2
# Codec token rate (Hz) used to estimate max_new_tokens from text length.
_CODEC_HZ: int = 12
# Words-per-second assumed for estimating audio duration from text.
_WPS: float = 2.5

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
        self._prompt_cache: dict[str, Any] = {}

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
            kwargs = dict(device_map=DEVICE, dtype=DTYPE)
            try:
                return Qwen3TTSModel.from_pretrained(
                    MODEL_ID, attn_implementation="flash_attention_2", **kwargs
                )
            except Exception:
                logger.warning(
                    "Flash Attention 2 not available for Qwen3, using default attention"
                )
                return Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)

        self._model = await loop.run_in_executor(None, _load)
        logger.info("Qwen3 engine loaded: %s (variant=%s)", MODEL_ID, _MODEL_VARIANT)

    async def unload(self) -> None:
        self._prompt_cache.clear()
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _disk_cache_path(self, voice_ref_path: str) -> Path:
        return Path(voice_ref_path).parent / "qwen3_prompt.pkl"

    def _load_disk_cache(self, voice_ref_path: str) -> Any | None:
        """Try to load prompt_items from disk cache. Returns None on any failure."""
        cache_path = self._disk_cache_path(voice_ref_path)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as f:
                stored = pickle.load(f)
            ref_path_obj = Path(voice_ref_path)
            if ref_path_obj.exists():
                ref_mtime = ref_path_obj.stat().st_mtime
                if stored.get("mtime") != ref_mtime:
                    logger.debug("Qwen3 disk cache stale for %s, will recompute", voice_ref_path)
                    return None
            # else: blended voice — no reference.wav, skip mtime check
            if stored.get("version") != _CACHE_VERSION:
                logger.debug("Qwen3 disk cache version mismatch for %s, will recompute", voice_ref_path)
                return None
            return stored["prompt_items"]
        except Exception:
            logger.debug("Qwen3 disk cache load failed for %s, will recompute", voice_ref_path)
            return None

    def _save_disk_cache(self, voice_ref_path: str, prompt_items: Any) -> None:
        """Try to save prompt_items to disk cache. Ignores any failure."""
        try:
            ref_mtime = Path(voice_ref_path).stat().st_mtime
            cache_path = self._disk_cache_path(voice_ref_path)
            with cache_path.open("wb") as f:
                pickle.dump(
                    {"version": _CACHE_VERSION, "mtime": ref_mtime, "prompt_items": prompt_items},
                    f,
                )
            logger.debug("Qwen3 prompt cache saved to %s", cache_path)
        except Exception:
            logger.debug("Qwen3 disk cache save failed for %s", voice_ref_path)

    @staticmethod
    def _trim_ref_audio_path(voice_ref_path: str) -> str | None:
        """If the reference WAV exceeds _MAX_REF_SECONDS, write a trimmed copy to a
        temp file and return its path. Returns None if no trimming is needed."""
        try:
            import scipy.io.wavfile

            sr, data = scipy.io.wavfile.read(voice_ref_path)
            max_samples = int(_MAX_REF_SECONDS * sr)
            if data.shape[0] <= max_samples:
                return None  # already short enough
            trimmed = data[:max_samples]
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            scipy.io.wavfile.write(tmp.name, sr, trimmed)
            logger.debug(
                "Qwen3 trimmed ref audio from %.1fs to %.1fs -> %s",
                data.shape[0] / sr,
                _MAX_REF_SECONDS,
                tmp.name,
            )
            return tmp.name
        except Exception:
            logger.debug("Qwen3 ref audio trim failed, using original")
            return None

    def _has_cached_prompt(self, voice_ref_path: str) -> bool:
        """Return True if a prompt is already cached (memory or disk) for this voice."""
        if voice_ref_path in self._prompt_cache:
            return True
        return self._disk_cache_path(voice_ref_path).exists()

    def _get_voice_clone_prompt(self, model: Any, voice_ref_path: str, voice_ref_text: str | None) -> Any:
        """Return cached prompt_items, computing and caching if needed.
        Trims reference audio to _MAX_REF_SECONDS before encoding if necessary."""
        if voice_ref_path in self._prompt_cache:
            return self._prompt_cache[voice_ref_path]

        prompt_items = self._load_disk_cache(voice_ref_path)
        if prompt_items is None:
            if not voice_ref_text:
                raise RuntimeError(
                    f"No cached prompt and no reference_text for {voice_ref_path!r}. "
                    "Re-clone the voice with reference_text to enable Qwen3 cloning."
                )
            tmp_path = self._trim_ref_audio_path(voice_ref_path)
            ref_to_encode = tmp_path if tmp_path is not None else voice_ref_path
            try:
                prompt_items = model.create_voice_clone_prompt(ref_to_encode, voice_ref_text)
            finally:
                if tmp_path is not None:
                    try:
                        Path(tmp_path).unlink()
                    except Exception:
                        pass
            self._save_disk_cache(voice_ref_path, prompt_items)

        self._prompt_cache[voice_ref_path] = prompt_items
        return prompt_items

    @staticmethod
    def blend_voice_prompts(pkl_path_a: str, pkl_path_b: str, alpha: float) -> Any:
        """Blend two Qwen3 voice pkl files into a new VoiceClonePromptItem.

        alpha=0 → pure voice_a, alpha=1 → pure voice_b.
        ref_spk_embedding is linearly interpolated then L2-normalised.
        ref_code is averaged when both have the same shape; otherwise voice_a is used.
        All tensors returned on CPU so the result can be pickled without a live GPU context.
        """
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

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

        # ref_code holds discrete codebook token IDs (int64) — averaging integer indices
        # produces meaningless tokens, so we never blend ref_code. Keep voice_a's tokens,
        # which serve as the ICL reference audio pattern alongside the blended embedding.
        rc_a = item_a.ref_code
        blended_code = rc_a.cpu() if rc_a is not None else None

        return VoiceClonePromptItem(
            ref_code=blended_code,
            ref_spk_embedding=blended_emb,
            x_vector_only_mode=item_a.x_vector_only_mode,
            icl_mode=item_a.icl_mode,
            ref_text=item_a.ref_text,
        )

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

        if _MODEL_VARIANT == "base" and voice_ref_path and (
            voice_ref_text or self._has_cached_prompt(voice_ref_path)
        ):
            # Voice cloning: Base model with reference audio + transcript.
            # _get_voice_clone_prompt is blocking (may call model.create_voice_clone_prompt),
            # so it must run inside the executor, not on the event loop thread.
            _ref_path = voice_ref_path
            _ref_text = voice_ref_text

            # Apply voice-clone-specific generation defaults unless caller overrode them.
            # temperature=0.5 stabilises EOS behaviour (0.9 default causes premature/runaway).
            clone_kwargs = dict(gen_kwargs)
            if "temperature" not in clone_kwargs:
                clone_kwargs["temperature"] = 0.5
            # Dynamic max_new_tokens: 3× estimated speech duration in codec frames.
            # Prevents runaway generation while still leaving headroom for slow delivery.
            if "max_new_tokens" not in clone_kwargs:
                word_count = len(text.split())
                estimated_frames = int(word_count / _WPS * _CODEC_HZ)
                clone_kwargs["max_new_tokens"] = max(120, estimated_frames * 3)

            _instruct = instruct

            def _run():
                prompt_items = self._get_voice_clone_prompt(model, _ref_path, _ref_text)
                if not _instruct:
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=prompt_items,
                        non_streaming_mode=True,
                        **clone_kwargs,
                    )
                    return wavs[0], sr
                # instruct + voice clone: bypass the wrapper and call the core model
                # directly so we can pass both voice_clone_prompt and instruct_ids.
                # ICL mode is kept for generation (ref_code as prefix conditions the
                # generated tokens on the reference voice). However we decode the
                # generated codes directly without prepending ref_code — this avoids
                # the silence gap that appears when the codec decoder sees the reference
                # prefix and the subsequent trim calculation lands in a silence region.
                prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts = [it.ref_text for it in prompt_items]
                input_ids = model._tokenize_texts([model._build_assistant_text(text)])
                ref_ids = [
                    model._tokenize_texts([model._build_ref_text(rt)])[0]
                    if rt else None
                    for rt in ref_texts
                ]
                instruct_ids = [model._tokenize_texts([model._build_instruct_text(_instruct)])[0]]
                gen_kw = model._merge_generate_kwargs(**clone_kwargs)
                talker_codes, _ = model.model.generate(
                    input_ids=input_ids,
                    instruct_ids=instruct_ids,
                    ref_ids=ref_ids,
                    voice_clone_prompt=prompt_dict,
                    languages=[language],
                    non_streaming_mode=True,
                    **gen_kw,
                )
                # Decode generated codes directly — no ref_code prepend, no trim.
                wavs, sr = model.model.speech_tokenizer.decode(
                    [{"audio_codes": c} for c in talker_codes]
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
