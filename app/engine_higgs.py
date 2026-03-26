from __future__ import annotations

import os
import subprocess as _sp
from pathlib import Path

from loguru import logger

from app.engine_base import SubprocessEngine

_raw_quant = os.environ.get("HIGGS_QUANT_BITS", "8").strip().lower()
if _raw_quant in ("0", "none", "false", "off"):
    QUANTIZATION_BITS = 0
else:
    QUANTIZATION_BITS = int(_raw_quant)

# VRAM estimates: ~5GB for 4-bit, ~9GB for 8-bit, ~12GB for bf16
_VRAM_BY_QUANT = {0: 12000, 4: 5000, 8: 9000}


class HiggsEngine(SubprocessEngine):
    name = "higgs"
    sample_rate = 24000
    estimated_vram_mb = _VRAM_BY_QUANT.get(QUANTIZATION_BITS, 9000)
    _worker_script = Path(__file__).parent.parent / "workers" / "higgs_worker.py"
    _worker_python = str(Path(__file__).parent.parent / ".venvs" / "higgs" / "Scripts" / "python.exe")

    def _probe_deps(self) -> bool:
        logger.debug(f"Probing higgs deps at {self._worker_python!r}")
        higgs_repo = os.environ.get("HIGGS_REPO_PATH") or os.path.join(
            os.environ.get("USERPROFILE", r"C:\Users\Default"), "tmp", "faster-higgs-audio"
        )
        probe = (
            f"import sys; sys.path.insert(0, {higgs_repo!r}); import boson_multimodal"
        )
        try:
            result = _sp.run(
                [self._worker_python, "-c", probe],
                capture_output=True, timeout=30
            ).returncode == 0
        except Exception:
            result = False
        logger.debug(f"higgs deps available: {result}")
        return result
