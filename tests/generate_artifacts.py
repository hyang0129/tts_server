#!/usr/bin/env python3
"""Generate TTS audio artifacts from the consolidated TTS server.

Generates WAVs for chatterbox, higgs, and qwen3 models via the unified /tts endpoint.

Usage:
    python tests/generate_artifacts.py [--base-url http://localhost:8000] [--model both] [--dry-run] [--validate]

Artifact plan:
    Chatterbox:   6 texts x 2 clone voices (kronii_cb, nimi_cb) = 12 WAVs
    Higgs:        6 texts x 2 description voices (default_male, default_female) = 12 WAVs
    Qwen3:        5 texts x 2 preset speakers (qwen3_ryan, qwen3_serena) = 10 WAVs
                  Requires QWEN3_MODEL_ID=...CustomVoice on server.
    Qwen3 (Base): 4 texts x 3 clone voices (kronii-q3, nimi-q3, kronimi) = 12 WAVs
                  Requires QWEN3_MODEL_ID=...Base on server; long+ten_second fixtures excluded.
    Total:        46 WAVs (or 34 without qwen3_base)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "tests" / "artifacts"
MANIFEST_PATH = REPO_ROOT / "tests" / "manifest.json"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

DEFAULT_BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Voice definitions
# ---------------------------------------------------------------------------

# Chatterbox voices: clone from reference audio
CHATTERBOX_VOICES = {
    "kronii_cb": {
        "audio_path": FIXTURES_DIR / "kroniivoice_15s.wav",
        "reference_text": "This is a sample of my voice for cloning purposes.",
    },
    "nimi_cb": {
        "audio_path": FIXTURES_DIR / "nimivoice_15s.wav",
        "reference_text": "This is a sample of my voice for cloning purposes.",
    },
}

# Higgs voices: description-only (skip cloning -- unreliable)
HIGGS_VOICES = {
    "default_male": (
        "Male, moderate pitch, clear enunciation, "
        "neutral American accent, calm narration style"
    ),
    "default_female": (
        "Female, moderate pitch, clear enunciation, "
        "neutral American accent, warm narration style"
    ),
}

# Qwen3 voices: preset speakers via CustomVoice model (no reference audio needed).
# Set QWEN3_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice on the server.
# long fixture is excluded (known STT edge case with military proper nouns).
QWEN3_VOICES = {
    "qwen3_ryan": "Ryan",
    "qwen3_serena": "Serena",
}
QWEN3_SKIP_LABELS = {"long"}

# Qwen3 Base voice-clone voices: must already exist in the voice store.
# Requires QWEN3_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base on the server.
# Clone voices are registered via /voices/clone or /voices/blend; the voice_id is used directly.
QWEN3_BASE_CLONE_VOICES = ["kronii-q3", "nimi-q3", "kronimi"]
# ten_second excluded: [chuckle] is a Chatterbox-only paralinguistic tag; Qwen3 speaks it literally.
QWEN3_BASE_SKIP_LABELS = {"long", "ten_second"}

# ---------------------------------------------------------------------------
# Text samples -- 6 per voice, diverse content
# ---------------------------------------------------------------------------

TEXTS = {
    "short": {
        "text": (
            "Did you know? The hamburger wasn't actually invented "
            "in Hamburg, Germany."
        ),
        "source": "hamburger scene_01",
    },
    "medium": {
        "text": (
            "The first hamburger bun wasn't introduced until the "
            "1904 World\u2019s Fair in St. Louis \u2014 before that, burgers "
            "were eaten with bread slices."
        ),
        "source": "hamburger scene_06",
    },
    "long": {
        "text": (
            "Did you know Allied soldiers called any enemy tank a Tiger "
            "\u2014 regardless of its actual type \u2014 because the Tiger I was so "
            "feared? The American M4 Sherman was produced over 49,000 times "
            "\u2014 winning through sheer quantity over German armor quality. "
            "The Soviet T-34 shocked German engineers so much they considered "
            "copying its sloped armor for their own future tank designs."
        ),
        "source": "ww2_tanks segments 1-3",
    },
    "ten_second": {
        "text": (
            "Welcome to Chatterbox Turbo. [chuckle] This is a quick "
            "demonstration of natural, high-quality text to speech synthesis. "
            "We hope you enjoy the result."
        ),
        "source": "ten_second_script",
    },
    "numbers": {
        "text": (
            "Germany\u2019s Panzer 8 Maus weighed 188 tonnes \u2014 so heavy "
            "it could only cross rivers by driving along the submerged "
            "riverbed."
        ),
        "source": "ww2_tanks segment 4",
    },
    "expressive": {
        "text": (
            "I can\u2019t believe they actually did it! [laugh] That was "
            "the most incredible thing I\u2019ve ever seen, and honestly, "
            "I\u2019m still in shock."
        ),
        "source": "custom expressive content",
    },
}

TEXT_LABELS = list(TEXTS.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_server(base_url: str, timeout: float = 30.0) -> bool:
    """Poll the server health endpoint until it responds or timeout."""
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/health"
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(health_url, timeout=5)
            if resp.status == 200:
                print(f"Server is ready at {base_url}")
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def clone_voice(
    base_url: str, voice_name: str, audio_path: Path, reference_text: str
) -> str | None:
    """Clone a voice via multipart POST to /voices/clone.

    Returns the voice_id on success.
    """
    url = f"{base_url}/voices/clone"
    boundary = "----TTSServerBoundary"
    audio_bytes = audio_path.read_bytes()
    content_type_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }
    mime = content_type_map.get(
        audio_path.suffix.lower(), "application/octet-stream"
    )

    body_parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="name"\r\n\r\n'
        f"{voice_name}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_text"\r\n\r\n'
        f"{reference_text}\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"chatterbox\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="reference_audio"; '
        f'filename="{audio_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n",
    ]
    body = (
        body_parts[0].encode()
        + body_parts[1].encode()
        + body_parts[2].encode()
        + body_parts[3].encode()
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        voice_id = data.get("voice_id") or data.get("id") or voice_name
        print(f"  Cloned voice '{voice_name}' -> id={voice_id}")
        return voice_id
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # Server slugifies the name (underscores → dashes); return the slug.
            slug = re.sub(r"[^a-z0-9]+", "-", voice_name.lower()).strip("-")
            print(f"  Voice '{voice_name}' already exists, reusing (id={slug})")
            return slug
        print(f"  ERROR cloning voice '{voice_name}': {exc}")
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR cloning voice '{voice_name}': {exc}")
        return None


def synthesize(
    base_url: str,
    text: str,
    model: str,
    voice_id: str | None = None,
    speaker_description: str | None = None,
    qwen3_speaker: str | None = None,
) -> bytes | None:
    """Call the /tts endpoint. Returns WAV bytes on success.

    The consolidated server requires a 'model' field to select the backend.
    For chatterbox, pass voice_id (cloned voice).
    For higgs, pass speaker_description.
    """
    url = f"{base_url}/tts"
    payload: dict = {"text": text, "model": model}
    if voice_id is not None:
        payload["voice"] = voice_id
    if speaker_description is not None:
        payload["speaker_description"] = speaker_description
    if qwen3_speaker is not None:
        payload["qwen3_speaker"] = qwen3_speaker

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"  ERROR synthesizing: {exc}")
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode(errors="replace")[:200]
                print(f"    Detail: {detail}")
            except Exception:
                pass
        return None


def artifact_filename(text_label: str, voice_name: str, model: str) -> str:
    """Build the WAV path for a (text, voice) pair, under a per-model subdir."""
    return f"{model}/{text_label}_{voice_name}.wav"


def load_manifest() -> dict:
    """Load the manifest.json file."""
    return json.loads(MANIFEST_PATH.read_text())


def build_plan(model_filter: str) -> list[tuple[str, str, str, str]]:
    """Build generation plan: list of (filename, text, voice_name, model).

    model_filter: 'chatterbox', 'higgs', 'qwen3', or 'both'
    """
    plan: list[tuple[str, str, str, str]] = []
    for text_label, text_info in TEXTS.items():
        if model_filter in ("chatterbox", "both"):
            for voice_name in CHATTERBOX_VOICES:
                fname = artifact_filename(text_label, voice_name, "chatterbox")
                plan.append((fname, text_info["text"], voice_name, "chatterbox"))
        if model_filter in ("higgs", "both"):
            for voice_name in HIGGS_VOICES:
                fname = artifact_filename(text_label, voice_name, "higgs")
                plan.append((fname, text_info["text"], voice_name, "higgs"))
        if model_filter in ("qwen3", "both"):
            if text_label not in QWEN3_SKIP_LABELS:
                for voice_name in QWEN3_VOICES:
                    fname = artifact_filename(text_label, voice_name, "qwen3")
                    plan.append((fname, text_info["text"], voice_name, "qwen3"))
        if model_filter in ("qwen3_base",):
            if text_label not in QWEN3_BASE_SKIP_LABELS:
                for voice_name in QWEN3_BASE_CLONE_VOICES:
                    fname = artifact_filename(text_label, voice_name, "qwen3_base")
                    plan.append((fname, text_info["text"], voice_name, "qwen3_base"))
    return plan


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def run(
    base_url: str,
    model_filter: str = "both",
    dry_run: bool = False,
    validate: bool = False,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("chatterbox", "higgs", "qwen3", "qwen3_base"):
        (ARTIFACTS_DIR / subdir).mkdir(exist_ok=True)

    plan = build_plan(model_filter)

    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Would generate {len(plan)} artifacts:\n")
        for fname, text, voice, model in plan:
            print(f"  {fname}  [{model}/{voice}]")
            print(f"    text: {text[:80]}{'...' if len(text) > 80 else ''}")
        return

    # Wait for server
    if not wait_for_server(base_url):
        print(f"ERROR: server not reachable at {base_url} after 30s")
        print(
            "Start the server with: "
            "uvicorn app.main:app --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)

    # Clone chatterbox voices that have reference audio
    voice_ids: dict[str, str] = {}
    if model_filter in ("chatterbox", "both"):
        print("\n--- Cloning chatterbox voices ---")
        for vname, vconfig in CHATTERBOX_VOICES.items():
            audio_path = vconfig["audio_path"]
            if not audio_path.exists():
                print(f"  SKIP {vname}: file not found at {audio_path}")
                continue
            vid = clone_voice(
                base_url, vname, audio_path, vconfig["reference_text"]
            )
            if vid:
                voice_ids[vname] = vid

    # Higgs description voices don't need cloning
    for vname in HIGGS_VOICES:
        voice_ids[vname] = vname

    # Qwen3 uses preset speakers (CustomVoice model) — no cloning needed.
    for vname in QWEN3_VOICES:
        voice_ids[vname] = vname

    # Synthesize
    generated: list[str] = []
    failed: list[str] = []

    print(f"\n--- Generating {len(plan)} artifacts ---")
    for fname, text, voice_name, model in plan:
        out_path = ARTIFACTS_DIR / fname
        print(f"  {fname} [{model}] ...", end=" ", flush=True)

        if model == "chatterbox":
            vid = voice_ids.get(voice_name)
            if vid is None:
                print("SKIPPED (voice not cloned)")
                failed.append(fname)
                continue
            wav_bytes = synthesize(base_url, text, model=model, voice_id=vid)
        elif model == "higgs":
            desc = HIGGS_VOICES[voice_name]
            wav_bytes = synthesize(
                base_url, text, model=model, speaker_description=desc
            )
        elif model == "qwen3":
            speaker = QWEN3_VOICES.get(voice_name)
            wav_bytes = synthesize(
                base_url, text, model="qwen3", qwen3_speaker=speaker
            )
        elif model == "qwen3_base":
            wav_bytes = synthesize(base_url, text, model="qwen3", voice_id=voice_name)
        else:
            print("SKIPPED (unknown model)")
            failed.append(fname)
            continue

        if wav_bytes:
            out_path.write_bytes(wav_bytes)
            generated.append(fname)
            print(f"OK ({len(wav_bytes)} bytes)")
        else:
            failed.append(fname)
            print("FAILED")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Model filter: {model_filter}")
    print(f"  Generated:    {len(generated)}")
    print(f"  Failed:       {len(failed)}")
    print(f"  Output:       {ARTIFACTS_DIR}")
    if generated:
        print("\nGenerated files:")
        for f in generated:
            print(f"    {f}")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"    {f}")

    # Validation
    if validate:
        _run_validation(generated)


def _run_validation(generated_files: list[str]) -> None:
    """Run STT validation on generated artifacts."""
    print("\n--- Running STT validation ---")

    stt_validate_path = REPO_ROOT / "tests" / "stt_validate.py"
    if not stt_validate_path.exists():
        print("WARNING: stt_validate.py not found -- skipping validation.")
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stt_validate", str(stt_validate_path)
    )
    stt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stt_mod)

    if not MANIFEST_PATH.exists():
        print("ERROR: manifest.json not found, cannot validate")
        return

    manifest = load_manifest()
    artifacts = manifest.get("artifacts", [])

    results = []
    all_pass = True

    for entry in artifacts:
        fname = entry["filename"]
        if fname not in generated_files:
            continue

        wav_path = ARTIFACTS_DIR / fname
        if not wav_path.exists():
            results.append(
                {
                    "filename": fname,
                    "status": "missing",
                    "word_match_pct": 0.0,
                }
            )
            all_pass = False
            continue

        expected = entry["expected_text"]
        try:
            result = stt_mod.validate_file(str(wav_path), expected)
            pct = result.get("word_match_pct", 0.0)
            llm = result.get("llm_adjudication")

            if pct >= 95.0:
                passed = True
            elif llm and llm.get("verdict") == "pass":
                passed = True
            else:
                passed = False

            status = "pass" if passed else "fail"
            if not passed:
                all_pass = False

            entry_result = {
                "filename": fname,
                "status": status,
                "word_match_pct": pct,
                "transcription": result.get("transcription", ""),
            }
            if llm:
                entry_result["llm_adjudication"] = llm
            results.append(entry_result)

            llm_note = ""
            if llm:
                llm_note = f" (LLM: {llm['verdict'].upper()})"
            print(f"  {fname}: {pct:.1f}% match{llm_note} -- {status.upper()}")
        except Exception as exc:
            results.append(
                {
                    "filename": fname,
                    "status": "error",
                    "word_match_pct": 0.0,
                    "error": str(exc),
                }
            )
            all_pass = False
            print(f"  {fname}: ERROR -- {exc}")

    # Save validation report
    report = {
        "overall_pass": all_pass,
        "threshold_pct": 95.0,
        "total_files": len(results),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    report_path = ARTIFACTS_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nValidation report saved to {report_path}")
    print(
        f"Overall: {'PASS' if all_pass else 'FAIL'} "
        f"({report['passed']}/{report['total_files']} passed)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS artifacts (24 WAVs: 6 texts x 4 voices, 2 models)"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"TTS server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        choices=["chatterbox", "higgs", "qwen3", "qwen3_base", "both"],
        default="both",
        help=(
            "Which model(s) to generate for (default: both). "
            "qwen3=CustomVoice preset speakers; "
            "qwen3_base=Base voice-clone artifacts (kronii-q3, nimi-q3, kronimi)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without calling the server",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run STT validation on generated artifacts",
    )
    args = parser.parse_args()
    run(args.base_url, args.model, args.dry_run, args.validate)


if __name__ == "__main__":
    main()
