from __future__ import annotations

import os
import platform
import subprocess as _sp
from pathlib import Path

from loguru import logger

from app.engine_base import SubprocessEngine

_REPO_ROOT = Path(__file__).parent.parent
_CB_PYTHON = os.environ.get("CB_WORKER_PYTHON") or (
    str(_REPO_ROOT / ".venvs" / "chatterbox" / "Scripts" / "python.exe")
    if platform.system() == "Windows"
    else str(Path("/workspaces/.venvs/tts_server-chatterbox/bin/python"))
)


class ChatterboxEngine(SubprocessEngine):
    name = "chatterbox"
    sample_rate = 24000
    estimated_vram_mb = 4700
    _worker_script = _REPO_ROOT / "workers" / "chatterbox_worker.py"
    _worker_python = _CB_PYTHON

    def _probe_deps(self) -> bool:
        logger.debug(f"Probing chatterbox deps: python={self._worker_python!r}")
        try:
            result = _sp.run(
                [self._worker_python, "-c", "import chatterbox"],
                capture_output=True, timeout=30
            ).returncode == 0
        except Exception:
            result = False
        logger.debug(f"chatterbox deps available: {result}")
        return result

    async def blend_voices(
        self,
        path_a: str,
        path_b: str,
        texture_mix: int,
        out_pt_path: str,
    ) -> None:
        """Send blend_voices command to worker. Worker writes .pt directly to out_pt_path."""
        logger.debug(f"blend_voices → worker {self.name}: a={path_a!r} b={path_b!r} mix={texture_mix}")
        await self._send_command({
            "cmd": "blend_voices",
            "path_a": path_a,
            "path_b": path_b,
            "texture_mix": texture_mix,
            "out_pt_path": out_pt_path,
        })
        logger.debug(f"blend_voices ← worker {self.name}: wrote {out_pt_path!r}")
