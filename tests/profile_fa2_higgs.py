#!/usr/bin/env python3
"""Profile Flash Attention 2 vs SDPA on 4-bit Higgs.

Spawns the higgs worker directly (no HTTP server needed) with each attention
implementation, measures load + generation time, and saves both WAV files to
tests/samples/ for human listening comparison.

Usage (from repo root, host venv active):
    python tests/profile_fa2_higgs.py

Output:
    tests/samples/higgs_4bit_sdpa.wav
    tests/samples/higgs_4bit_fa2.wav
    Timing table printed to stdout.

Requirements:
    - .venvs/higgs/  set up (run scripts/setup_venvs.ps1 first)
    - flash-attn installed in higgs venv (setup_venvs.ps1 Step 4)
    - .env present with HIGGS_REPO_PATH / HIGGS_MODEL_ID
"""
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "samples"
WORKER_SCRIPT = REPO_ROOT / "workers" / "higgs_worker.py"
WORKER_PYTHON = REPO_ROOT / ".venvs" / "higgs" / "Scripts" / "python.exe"

# ~10 second fixture text (≈130 words/min rate → ~25 words ≈ 10-12 s of audio)
FIXTURE_TEXT = (
    "The old lighthouse stood at the edge of the rocky coast, "
    "its beam sweeping steadily across the dark waters as storm clouds "
    "gathered on the horizon. Sailors had relied on its light for generations."
)

ATTN_RUNS = [
    ("sdpa",              "higgs_4bit_sdpa.wav"),
    ("flash_attention_2", "higgs_4bit_fa2.wav"),
]


# ---------------------------------------------------------------------------
# .env loader (stdlib only — host venv has no python-dotenv)
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


# ---------------------------------------------------------------------------
# Worker IPC helpers
# ---------------------------------------------------------------------------

def _send(proc: subprocess.Popen, payload: dict) -> None:
    line = (json.dumps(payload) + "\n").encode()
    proc.stdin.write(line)
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("Worker did not respond within timeout")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker stdout closed unexpectedly")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("status") == "error":
            raise RuntimeError(
                f"Worker error: {msg.get('message')} | {msg.get('traceback', '')[:500]}"
            )
        return msg


def _save_wav(path: Path, audio_b64: str, sample_rate: int) -> None:
    audio_bytes = base64.b64decode(audio_b64)
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    # scipy expects int16 or float32; write as float32 WAV
    wavfile.write(str(path), sample_rate, audio)


# ---------------------------------------------------------------------------
# Single profiling run
# ---------------------------------------------------------------------------

def _run_one(attn_impl: str, out_path: Path, env: dict[str, str]) -> dict[str, float]:
    """Spawn the higgs worker with the given attn impl, run one generate, return timings."""
    worker_env = {**os.environ, **env, "HIGGS_ATTN_IMPL": attn_impl, "HIGGS_QUANT_BITS": "4"}

    print(f"\n[{attn_impl}] Starting worker...")
    t_start = time.monotonic()

    proc = subprocess.Popen(
        [str(WORKER_PYTHON), str(WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,  # stream worker logs to our stderr
        env=worker_env,
        cwd=str(REPO_ROOT),
    )

    # Load model
    print(f"[{attn_impl}] Loading model (4-bit)...")
    t_load_start = time.monotonic()
    _send(proc, {"cmd": "load"})
    _recv(proc, timeout=300.0)
    t_load_end = time.monotonic()
    load_time = t_load_end - t_load_start
    print(f"[{attn_impl}] Load done in {load_time:.1f}s")

    # Generate
    print(f"[{attn_impl}] Generating audio...")
    t_gen_start = time.monotonic()
    _send(proc, {"cmd": "generate", "text": FIXTURE_TEXT, "params": {}})
    resp = _recv(proc, timeout=300.0)
    t_gen_end = time.monotonic()
    gen_time = t_gen_end - t_gen_start
    print(f"[{attn_impl}] Generate done in {gen_time:.1f}s")

    # Save WAV
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _save_wav(out_path, resp["audio"], resp["sample_rate"])
    audio_duration = len(base64.b64decode(resp["audio"])) / 4 / resp["sample_rate"]
    print(f"[{attn_impl}] Saved {out_path.name}  ({audio_duration:.1f}s audio)")

    # Unload + shutdown
    _send(proc, {"cmd": "unload"})
    proc.wait(timeout=30)

    return {
        "load_s": load_time,
        "gen_s": gen_time,
        "audio_s": audio_duration,
        "rtf": gen_time / audio_duration if audio_duration > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dotenv = _load_dotenv(REPO_ROOT / ".env")

    if not WORKER_PYTHON.exists():
        sys.exit(f"ERROR: higgs venv not found at {WORKER_PYTHON}\nRun scripts/setup_venvs.ps1 first.")

    results: list[tuple[str, str, dict]] = []
    for attn_impl, filename in ATTN_RUNS:
        out_path = SAMPLES_DIR / filename
        try:
            timings = _run_one(attn_impl, out_path, dotenv)
            results.append((attn_impl, str(out_path), timings))
        except Exception as exc:
            print(f"\n[ERROR] {attn_impl} run failed: {exc}", file=sys.stderr)
            results.append((attn_impl, str(out_path), {"error": str(exc)}))

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Attention':<22} {'Load':>7} {'Generate':>10} {'Audio':>7} {'RTF':>6}")
    print("-" * 65)
    for attn_impl, path, t in results:
        if "error" in t:
            print(f"{attn_impl:<22}  ERROR: {t['error']}")
        else:
            print(
                f"{attn_impl:<22} {t['load_s']:>6.1f}s {t['gen_s']:>9.1f}s "
                f"{t['audio_s']:>6.1f}s {t['rtf']:>5.2f}x"
            )
    print("=" * 65)
    print("RTF = generate_time / audio_duration  (lower is faster)")
    print()
    print("Output WAVs for human review:")
    for _, path, t in results:
        if "error" not in t:
            print(f"  {path}")


if __name__ == "__main__":
    main()
