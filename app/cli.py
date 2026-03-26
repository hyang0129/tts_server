"""CLI entry point for the TTS server.

Installed as the `tts-server` console script via pyproject.toml.
Reads configuration from environment variables (and .env in the repo root).

Usage:
    tts-server                               # port 8765
    TTS_PORT=8080 tts-server
    AVAILABLE_VRAM_MB=10000 tts-server       # set VRAM budget explicitly (recommended)
"""
from __future__ import annotations

import os


def serve() -> None:
    import uvicorn

    port = int(os.environ.get("TTS_PORT", "8765"))
    host = os.environ.get("TTS_HOST", "127.0.0.1")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
    )
