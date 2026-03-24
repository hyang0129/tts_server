#!/usr/bin/env python3
"""Stdlib-only JSON protocol helpers for newline-delimited IPC over stdin/stdout."""
from __future__ import annotations

import json
import sys
import traceback


def send(payload: dict) -> None:
    """Write a JSON payload to stdout followed by a newline, then flush."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def send_ok(**payload) -> None:
    """Write a success response to stdout.

    All keyword arguments are merged into the response alongside ``"status": "ok"``.
    """
    send({"status": "ok", **payload})


def send_error(exc: BaseException, include_traceback: bool = True) -> None:
    """Write an error response to stdout.

    Parameters
    ----------
    exc:
        The exception that was caught.
    include_traceback:
        When *True* (default), ``traceback.format_exc()`` is included in the
        response.  Pass *False* for expected validation errors where the full
        traceback is noise.
    """
    tb = traceback.format_exc() if include_traceback else None
    send(
        {
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        }
    )


def read_request() -> dict | None:
    """Read and parse one JSON line from stdin.

    Returns the parsed dict, or *None* on EOF.  Exits cleanly on
    ``BrokenPipeError``.
    """
    try:
        line = sys.stdin.readline()
    except BrokenPipeError:
        sys.exit(0)
    if not line:
        return None
    return json.loads(line)
