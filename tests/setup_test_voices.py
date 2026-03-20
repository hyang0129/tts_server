"""Install tracked voice fixtures into the runtime voices/ directory.

Run this once after a fresh clone or container rebuild, before running
integration tests that depend on named voices (e.g. ragnar-narrator).

Usage:
    python tests/setup_test_voices.py

The voice fixtures live in tests/voice_fixtures/<voice_id>/ and are committed
to git. This script copies them into the runtime voices/ directory that the
tts_server reads at startup.

Why voice fixtures are not committed directly under voices/:
    The voices/ directory is gitignored (runtime data, user-cloned voices).
    Fixtures are tracked under tests/voice_fixtures/ and installed on demand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "voice_fixtures"
VOICES_DIR = Path(__file__).parent.parent / "voices"


def install_voice(voice_id: str, overwrite: bool = False) -> None:
    src = FIXTURES_DIR / voice_id
    dst = VOICES_DIR / voice_id

    if dst.exists():
        if not overwrite:
            print(f"[SKIP] {voice_id} already installed at {dst}")
            return
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, dst / f.name)

    meta = dst / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        print(f"[OK]   {voice_id} installed — {data.get('duration_s', '?')}s reference, "
              f"compatible: {data.get('compatible_models', [])}")
    else:
        print(f"[OK]   {voice_id} installed")


def main() -> None:
    if not FIXTURES_DIR.exists():
        print(f"[ERROR] No voice_fixtures directory at {FIXTURES_DIR}")
        return

    voices = sorted(FIXTURES_DIR.iterdir())
    if not voices:
        print("[INFO] No voice fixtures to install.")
        return

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice_dir in voices:
        if voice_dir.is_dir():
            install_voice(voice_dir.name)


if __name__ == "__main__":
    main()
