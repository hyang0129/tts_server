from __future__ import annotations

import os
import platform
import subprocess as _sp
from pathlib import Path

from loguru import logger

from app.engine_base import SubprocessEngine

MODEL_ID = os.environ.get("QWEN3_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# 1.7B model in bfloat16 ≈ 3.4 GB weights + KV cache/activations overhead
ESTIMATED_VRAM_MB = int(os.environ.get("QWEN3_VRAM_MB", "5500"))

# Cache version — bump when generation behaviour changes to invalidate stale pickles.
# voices.py imports this constant; keep it here so that import still works.
_CACHE_VERSION: int = 2

_REPO_ROOT = Path(__file__).parent.parent
if platform.system() == "Windows":
    _QWEN3_PYTHON = str(_REPO_ROOT / ".venvs" / "qwen3" / "Scripts" / "python.exe")
else:
    _QWEN3_PYTHON = str(Path("/workspaces/.venvs/tts_server-qwen3/bin/python"))


class Qwen3Engine(SubprocessEngine):
    name = "qwen3"
    sample_rate = 24000
    estimated_vram_mb = ESTIMATED_VRAM_MB
    _worker_script = _REPO_ROOT / "workers" / "qwen3_worker.py"
    _worker_python = _QWEN3_PYTHON

    def _probe_deps(self) -> bool:
        logger.debug(f"Probing qwen3 deps at {self._worker_python!r}")
        try:
            result = _sp.run(
                [self._worker_python, "-c", "import qwen_tts"],
                capture_output=True, timeout=30
            ).returncode == 0
        except Exception:
            result = False
        logger.debug(f"qwen3 deps available: {result}")
        return result

    async def blend_voice_prompts_ipc(
        self,
        pkl_path_a: str,
        pkl_path_b: str,
        alpha: float,
        out_pkl_path: str,
    ) -> None:
        """Send blend_voice_prompts to worker. Worker writes pkl to out_pkl_path."""
        logger.debug(f"blend_voice_prompts_ipc → worker {self.name}")
        await self._send_command({
            "cmd": "blend_voice_prompts",
            "pkl_path_a": pkl_path_a,
            "pkl_path_b": pkl_path_b,
            "alpha": alpha,
            "out_pkl_path": out_pkl_path,
        })
        logger.debug(f"blend_voice_prompts_ipc ← worker {self.name}: done")
