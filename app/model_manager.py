from __future__ import annotations

import asyncio
import gc
import time

from loguru import logger

from app.engine_base import TTSEngine

IDLE_TIMEOUT_S = 60
IDLE_CHECK_INTERVAL_S = 10


def _vram_free_mb() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (total - reserved) // (1024 ** 2)
    except Exception:
        pass
    return None


class ModelManager:
    """Manages lazy-loading and swapping of TTS engines in VRAM.

    Only one engine is loaded at a time. Engines are swapped on demand and
    automatically unloaded after IDLE_TIMEOUT_S seconds of inactivity.
    """

    def __init__(self, available_vram_mb: int) -> None:
        self._engines: dict[str, TTSEngine] = {}
        self._available_vram_mb = available_vram_mb
        self._active_engine: str | None = None
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None

    def register_engine(self, engine: TTSEngine) -> None:
        self._engines[engine.name] = engine

    def available_models(self) -> list[str]:
        """Return engine names that fit within the VRAM budget and have deps installed."""
        return [
            name
            for name, engine in self._engines.items()
            if engine.estimated_vram_mb <= self._available_vram_mb
            and engine.deps_available
        ]

    def get_engine(self, name: str) -> TTSEngine | None:
        return self._engines.get(name)

    @property
    def active_engine_name(self) -> str | None:
        return self._active_engine

    def all_engines(self) -> dict[str, TTSEngine]:
        return dict(self._engines)

    async def ensure_loaded(self, model_name: str) -> TTSEngine:
        """Ensure the requested engine is loaded, swapping if needed."""
        async with self._lock:
            self._last_request_time = time.monotonic()

            if model_name not in self._engines:
                raise ValueError(f"Unknown model: {model_name}")

            if model_name not in self.available_models():
                raise ValueError(
                    f"Model {model_name} requires "
                    f"{self._engines[model_name].estimated_vram_mb}MB VRAM "
                    f"but only {self._available_vram_mb}MB available"
                )

            if self._active_engine == model_name:
                engine = self._engines[model_name]
                if engine.is_loaded:
                    logger.debug(f"Engine {model_name} already loaded — cache hit")
                    return engine

            # Unload current engine if one is loaded.
            if self._active_engine is not None:
                await self._unload_current()

            engine = self._engines[model_name]
            vram_before = _vram_free_mb()
            logger.debug(f"VRAM before load: {vram_before} MB free")
            t0 = time.perf_counter()
            await engine.load()
            load_ms = time.perf_counter() - t0
            self._active_engine = model_name
            vram_after = _vram_free_mb()
            logger.debug(f"VRAM after load: {vram_after} MB free")
            logger.info(f"Engine {model_name} loaded in {load_ms:.3f}s")
            return engine

    async def _unload_current(self) -> None:
        """Unload the currently active engine and free VRAM."""
        if self._active_engine is None:
            return
        name = self._active_engine
        engine = self._engines[name]
        vram_before = _vram_free_mb()
        logger.debug(f"VRAM before unload: {vram_before} MB free")
        t0 = time.perf_counter()
        await engine.unload()
        unload_ms = time.perf_counter() - t0
        gc.collect()  # harmless, keep
        self._active_engine = None
        vram_after = _vram_free_mb()
        logger.debug(f"VRAM after unload: {vram_after} MB free")
        logger.info(f"Engine {name} unloaded in {unload_ms:.3f}s")

    async def shutdown(self) -> None:
        """Unload any loaded engine. Called at server shutdown."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            self._idle_task = None

        async with self._lock:
            await self._unload_current()

    def start_idle_monitor(self) -> None:
        """Start the background task that unloads idle models."""
        self._idle_task = asyncio.create_task(self._idle_monitor())

    async def _idle_monitor(self) -> None:
        """Periodically check if the active engine has been idle too long."""
        try:
            while True:
                await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
                async with self._lock:
                    if self._active_engine is None:
                        continue
                    if self._last_request_time == 0.0:
                        continue
                    elapsed = time.monotonic() - self._last_request_time
                    logger.debug(f"Idle monitor: engine={self._active_engine}, idle_s={elapsed:.1f}")
                    if elapsed > IDLE_TIMEOUT_S:
                        logger.info(
                            "Engine %s idle for %.0fs, unloading",
                            self._active_engine,
                            elapsed,
                        )
                        await self._unload_current()
        except asyncio.CancelledError:
            return
