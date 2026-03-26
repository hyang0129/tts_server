from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger


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


class SubprocessEngine(TTSEngine):
    """Base class for TTS engines that run as subprocess workers."""

    # Subclasses must set these:
    name: str
    sample_rate: int
    estimated_vram_mb: int
    _worker_script: Path  # absolute path to worker .py file
    _worker_python: str   # absolute path to venv python binary

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._is_loaded: bool = False
        self._deps_available: bool | None = None  # None = not yet probed

    @property
    def is_loaded(self) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._is_loaded
        )

    @property
    def deps_available(self) -> bool:
        if self._deps_available is None:
            self._deps_available = self._probe_deps()
        return self._deps_available

    def _probe_deps(self) -> bool:
        """One-time probe: try to import the engine package in the worker venv."""
        # Subclasses override this to set what to import
        raise NotImplementedError

    async def load(self) -> None:
        """Spawn worker subprocess and send load command."""
        if self.is_loaded:
            return

        self._proc = await asyncio.create_subprocess_exec(
            self._worker_python, str(self._worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit host stderr so worker logs go to server logs
            limit=256 * 1024 * 1024,  # 256 MB — base64 audio can be large
        )

        # Send load command
        response = await self._send_command({"cmd": "load"})
        self.sample_rate = response.get("sample_rate", self.sample_rate)
        self._is_loaded = True

    async def unload(self) -> None:
        """Send unload command and wait for process to exit."""
        if self._proc is None:
            return

        try:
            await self._send_command({"cmd": "unload"})
        except Exception:
            pass

        # Wait for process to exit; escalate to SIGKILL if needed
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass  # best effort — clear refs regardless

        self._proc = None
        self._is_loaded = False

    async def generate(
        self,
        text: str,
        voice_ref_path: str | None,
        voice_ref_text: str | None,
        **params,
    ) -> tuple[np.ndarray, int]:
        """Send generate command to worker. Retries once if worker crashes mid-generation."""
        cmd = {
            "cmd": "generate",
            "text": text,
            "voice_ref_path": voice_ref_path,
            "voice_ref_text": voice_ref_text,
            "params": params,
        }
        t0 = time.perf_counter()
        try:
            response = await self._send_command(cmd)
        except RuntimeError as exc:
            if "closed stdout unexpectedly" not in str(exc):
                raise
            # Worker crashed mid-generation (e.g. VRAM fragmentation after model swap).
            # Clear state and reload before retrying once.
            logger.warning("Worker %s crashed during generate — reloading and retrying", self.name)
            self._proc = None
            self._is_loaded = False
            await self.load()
            response = await self._send_command(cmd)
        elapsed = time.perf_counter() - t0
        logger.info(f"generate completed in {elapsed:.3f}s for engine {self.name}")

        audio = np.frombuffer(base64.b64decode(response["audio"]), dtype=np.float32)
        sample_rate = response.get("sample_rate", self.sample_rate)
        return audio, sample_rate

    async def _send_command(self, cmd: dict) -> dict:
        """Send one JSON command to worker stdin, read one JSON response from stdout."""
        if self._proc is None or self._proc.returncode is not None:
            raise RuntimeError(f"Worker process for {self.name} is not running")

        logger.debug(f"Sending command {cmd['cmd']} to worker {self.name}")
        line = (json.dumps(cmd) + "\n").encode()
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

        response_line = await self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError(f"Worker process for {self.name} closed stdout unexpectedly")

        try:
            response = json.loads(response_line.decode())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Worker {self.name} sent non-JSON response: {response_line!r}"
            ) from exc
        if response.get("status") == "error":
            if response.get("traceback"):
                logger.error(
                    "Worker %s error traceback:\n%s", self.name, response["traceback"]
                )
            raise RuntimeError(f"{response.get('error', 'WorkerError')}: {response.get('message', '')}")

        logger.debug(f"Worker {self.name} responded OK")
        return response
