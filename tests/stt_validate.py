#!/usr/bin/env python3
"""Validate TTS audio output quality using Speech-to-Text transcription.

Uses faster-whisper (CTranslate2-optimized Whisper) to transcribe WAV files
and compare against expected text using word-level sequence matching.

Single-file mode:
    python tests/stt_validate.py --wav path.wav --expected "the expected text"

Batch mode (reads manifest mapping filenames to expected texts):
    python tests/stt_validate.py --artifacts-dir tests/artifacts/ --manifest tests/manifest.json

Exit codes:
    0 = all files pass (>= threshold)
    1 = one or more files fail
    2 = usage / runtime error
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# Lazy-loaded to avoid import cost when just checking --help.
_model = None
_anthropic_client = None

# Auto-load .env from repo root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_THRESHOLD = 95.0


def _get_model():
    """Load the faster-whisper model (cached across calls)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


_CONTRACTIONS = {
    "wasn't": "was not", "weren't": "were not", "isn't": "is not",
    "aren't": "are not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not", "can't": "cannot",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "he's": "he is", "she's": "she is", "what's": "what is",
    "let's": "let us", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "world's": "worlds",
}


def normalize_text(text: str) -> list[str]:
    """Lowercase, expand contractions, strip punctuation, split into words."""
    t = text.lower()
    for contraction, expansion in _CONTRACTIONS.items():
        t = t.replace(contraction, expansion)
    return re.sub(r"[^a-z0-9 ]", "", t).split()


def word_match_rate(expected: str, transcribed: str) -> float:
    """Compute word-level match percentage using longest common subsequence.

    Returns a float 0-100. Uses difflib.SequenceMatcher on normalized word
    lists, which finds the longest contiguous matching subsequences. The
    score is (matched words / expected words) * 100.
    """
    e_words = normalize_text(expected)
    t_words = normalize_text(transcribed)
    if not e_words:
        return 100.0 if not t_words else 0.0
    matcher = SequenceMatcher(None, e_words, t_words)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (matched / len(e_words)) * 100.0


def _get_anthropic_client():
    """Lazy-load the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic()
        except (ImportError, Exception):
            _anthropic_client = False  # sentinel: unavailable
    return _anthropic_client if _anthropic_client is not False else None


def llm_adjudicate(expected: str, transcribed: str) -> dict:
    """Ask Claude Haiku whether the STT mismatch is a real quality issue.

    Returns {"verdict": "pass"|"fail", "reason": str} or None if unavailable.
    Only call this on mismatches (below threshold) to minimize cost.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    prompt = (
        "You are judging whether a speech-to-text transcription is a reasonable "
        "match for the original script that was spoken. Minor differences due to "
        "STT limitations are acceptable:\n"
        "- Proper noun spelling variations (Louis' vs Louie's)\n"
        "- Contraction differences (wasn't vs was not)\n"
        "- Minor punctuation or filler words\n"
        "- Singular/plural variations (fact vs facts)\n\n"
        "However, these indicate REAL quality problems:\n"
        "- Missing or garbled words/sentences\n"
        "- Completely wrong words that change meaning\n"
        "- Incoherent fragments or syllable soup\n"
        "- Large sections of text missing or added\n\n"
        f"ORIGINAL SCRIPT:\n{expected}\n\n"
        f"STT TRANSCRIPTION:\n{transcribed}\n\n"
        "Is this a reasonable transcription of the original script? "
        "Reply with exactly one line: PASS or FAIL, followed by a brief reason."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        verdict = "pass" if text.upper().startswith("PASS") else "fail"
        return {"verdict": verdict, "reason": text}
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc)}


def transcribe_wav(wav_path: str | Path) -> str:
    """Transcribe a WAV file and return the full text."""
    model = _get_model()
    segments, _info = model.transcribe(str(wav_path), language="en")
    return " ".join(seg.text.strip() for seg in segments)


def validate_file(
    wav_path: str | Path,
    expected_text: str,
    threshold: float = DEFAULT_THRESHOLD,
    use_llm: bool = True,
) -> dict:
    """Validate a single WAV file and return a result dict.

    This is the public API used by generate_artifacts.py's --validate flag.
    When word match is below threshold and use_llm=True, asks Claude Haiku
    to adjudicate whether the mismatch is a real quality issue or STT noise.

    Returns:
        {"word_match_pct": float, "transcription": str,
         "llm_adjudication": {...} | None}
    """
    transcription = transcribe_wav(wav_path)
    match_pct = word_match_rate(expected_text, transcription)

    result = {
        "word_match_pct": round(match_pct, 1),
        "transcription": transcription,
        "llm_adjudication": None,
    }

    # Only call LLM on mismatches to minimize cost
    if match_pct < threshold and use_llm:
        adjudication = llm_adjudicate(expected_text, transcription)
        result["llm_adjudication"] = adjudication

    return result


def validate_single(
    wav_path: Path,
    expected_text: str,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    verbose: bool = False,
) -> tuple[bool, float, str]:
    """Validate a single WAV file.

    Returns (passed, match_pct, transcription).
    """
    transcription = transcribe_wav(wav_path)
    match_pct = word_match_rate(expected_text, transcription)
    passed = match_pct >= threshold

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"{status:4s} {match_pct:5.1f}% {wav_path.name}")
        if not passed:
            print(f"  Expected:     {expected_text[:120]}")
            print(f"  Transcribed:  {transcription[:120]}")

    return passed, match_pct, transcription


def _normalize_manifest(manifest: dict | list) -> dict[str, str]:
    """Accept either format and return {filename: expected_text}.

    Supports:
      - {"artifacts": [{"filename": "...", "expected_text": "..."}, ...]}
      - {"filename": "expected text", ...}
    """
    if isinstance(manifest, dict) and "artifacts" in manifest:
        return {
            entry["filename"]: entry["expected_text"]
            for entry in manifest["artifacts"]
        }
    return manifest


def validate_batch(
    artifacts_dir: Path,
    manifest: dict,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    verbose: bool = False,
) -> dict:
    """Validate all files in a manifest.

    Accepts both manifest formats (see _normalize_manifest).
    Returns a summary dict with per-file results.
    """
    file_map = _normalize_manifest(manifest)
    results = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for filename, expected_text in sorted(file_map.items()):
        wav_path = artifacts_dir / filename
        if not wav_path.exists():
            if verbose:
                print(f"SKIP       {filename} (not found)")
            skip_count += 1
            results.append(
                {
                    "file": filename,
                    "status": "skip",
                    "reason": "file not found",
                }
            )
            continue

        passed, match_pct, transcription = validate_single(
            wav_path, expected_text, threshold, verbose=verbose
        )

        llm_result = None
        if not passed:
            llm_result = llm_adjudicate(expected_text, transcription)
            if llm_result and llm_result.get("verdict") == "pass":
                passed = True
                if verbose:
                    print(f"  LLM override → PASS  ({llm_result['reason'][:80]})")

        if passed:
            pass_count += 1
        else:
            fail_count += 1

        entry = {
            "file": filename,
            "status": "pass" if passed else "fail",
            "match_pct": round(match_pct, 1),
            "expected": expected_text,
            "transcribed": transcription,
        }
        if llm_result:
            entry["llm_adjudication"] = llm_result
        results.append(entry)

    total = pass_count + fail_count
    summary = {
        "total": total,
        "passed": pass_count,
        "failed": fail_count,
        "skipped": skip_count,
        "pass_rate": round((pass_count / total * 100) if total else 0, 1),
        "threshold": threshold,
        "results": results,
    }

    if verbose:
        print()
        print(f"{'=' * 60}")
        print(
            f"Total: {total} | Pass: {pass_count} | Fail: {fail_count} | "
            f"Skip: {skip_count} | Rate: {summary['pass_rate']:.1f}%"
        )
        print(f"Threshold: {threshold}%")
        print(f"{'=' * 60}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate TTS output quality via STT transcription"
    )
    parser.add_argument("--wav", type=Path, help="Single WAV file to validate")
    parser.add_argument("--expected", type=str, help="Expected text for --wav mode")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory containing WAV artifacts (batch mode)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest mapping filenames to expected texts (batch mode)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Word match %% threshold for pass/fail (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (batch mode)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-file results",
    )
    args = parser.parse_args()

    # Single file mode
    if args.wav:
        if not args.expected:
            parser.error("--expected is required with --wav")
        if not args.wav.exists():
            print(f"ERROR: WAV file not found: {args.wav}", file=sys.stderr)
            sys.exit(2)
        passed, match_pct, transcription = validate_single(
            args.wav, args.expected, args.threshold, verbose=True
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "file": str(args.wav),
                        "passed": passed,
                        "match_pct": round(match_pct, 1),
                        "transcribed": transcription,
                    },
                    indent=2,
                )
            )
        sys.exit(0 if passed else 1)

    # Batch mode
    if args.artifacts_dir:
        if not args.artifacts_dir.is_dir():
            print(
                f"ERROR: artifacts dir not found: {args.artifacts_dir}",
                file=sys.stderr,
            )
            sys.exit(2)

        if args.manifest:
            if not args.manifest.exists():
                print(
                    f"ERROR: manifest not found: {args.manifest}",
                    file=sys.stderr,
                )
                sys.exit(2)
            manifest = json.loads(args.manifest.read_text())
        else:
            parser.error("--manifest is required with --artifacts-dir")

        summary = validate_batch(
            args.artifacts_dir,
            manifest,
            args.threshold,
            verbose=not args.json,
        )

        if args.json:
            print(json.dumps(summary, indent=2))

        sys.exit(0 if summary["failed"] == 0 else 1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
