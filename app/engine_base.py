from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    """Abstract base class for TTS engine backends."""

    name: str
    sample_rate: int
    estimated_vram_mb: int

    @abstractmethod
    async def load(self) -> None:
        """Load model weights into VRAM."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Free all VRAM held by this engine."""
        ...

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice_ref_path: str | None,
        voice_ref_text: str | None,
        **params,
    ) -> tuple[np.ndarray, int]:
        """Generate speech. Returns (waveform_1d, sample_rate)."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @property
    def deps_available(self) -> bool:
        """Check if this engine's Python dependencies are importable."""
        return True
