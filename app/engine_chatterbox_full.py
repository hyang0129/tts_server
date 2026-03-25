from __future__ import annotations

import os
import subprocess as _sp
from pathlib import Path

from app.engine_base import SubprocessEngine


class ChatterboxFullEngine(SubprocessEngine):
    name = "chatterbox_full"
    sample_rate = 24000
    estimated_vram_mb = int(os.environ.get("CB_FULL_VRAM_MB", "4700"))
    _worker_script = Path(__file__).parent.parent / "workers" / "chatterbox_full_worker.py"
    _worker_python = "/workspaces/.venvs/tts_server-chatterbox/bin/python"

    def _probe_deps(self) -> bool:
        try:
            return _sp.run(
                [self._worker_python, "-c", "import chatterbox"],
                capture_output=True, timeout=10
            ).returncode == 0
        except Exception:
            return False

    async def blend_voices(
        self,
        path_a: str,
        path_b: str,
        texture_mix: int,
        out_pt_path: str,
    ) -> None:
        """Send blend_voices command to worker. Worker writes .pt directly to out_pt_path."""
        await self._send_command({
            "cmd": "blend_voices",
            "path_a": path_a,
            "path_b": path_b,
            "texture_mix": texture_mix,
            "out_pt_path": out_pt_path,
        })
