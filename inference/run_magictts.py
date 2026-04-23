#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from release_utils import build_prompt_prefix_context
from release_utils import build_subprocess_env
from release_utils import build_track_from_preset
from release_utils import DEFAULT_SYNTH_PYTHON
from release_utils import save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_ROOTS = {
    "zh": REPO_ROOT / "assets" / "default_prompt",
    "en": REPO_ROOT / "assets" / "default_prompt_en",
}
RUN_SPONTANEOUS_DEMO = REPO_ROOT / "inference" / "run_spontaneous_demo.py"
RUN_EDIT_FROM_JSON = REPO_ROOT / "inference" / "run_edit_from_json.py"
DEFAULT_EN_INLINE_CONTENT_MS = 55.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-text-file", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--control-json", default=None)
    parser.add_argument("--mfa-jobs", type=int, default=8)
    parser.add_argument("--keep-mfa-workdir", action="store_true")
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--content-ms", type=float, default=170.0)
    parser.add_argument("--punct-ms", type=float, default=50.0)
    return parser.parse_args()


def parse_inline_target_text_zh(text: str):
    plain_chars = []
    duration_ms = {}
    pause_ms = {}
    occurrence = {}
    last_char_key = None
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "[":
            end = text.find("]", i + 1)
            if end == -1:
                raise ValueError("unclosed pause marker '[' in --target-text")
            if last_char_key is None:
                raise ValueError("pause marker must appear after at least one visible character")
            payload = text[i + 1 : end].strip()
            if not payload:
                raise ValueError("empty pause marker [] is not allowed")
            pause_ms[last_char_key] = float(payload)
            i = end + 1
            continue

        if ch in "{}]":
            raise ValueError(f"unexpected control character '{ch}' in --target-text")

        plain_chars.append(ch)
        occurrence[ch] = occurrence.get(ch, 0) + 1
        last_char_key = f"{ch}#{occurrence[ch]}"
        i += 1

        if i < len(text) and text[i] == "{":
            end = text.find("}", i + 1)
            if end == -1:
                raise ValueError("unclosed duration marker '{' in --target-text")
            payload = text[i + 1 : end].strip()
            if not payload:
                raise ValueError("empty duration marker {} is not allowed")
            duration_ms[last_char_key] = float(payload)
            i = end + 1

    return {
        "plain_text": "".join(plain_chars),
        "duration_ms": duration_ms,
        "pause_ms": pause_ms,
        "has_control": bool(duration_ms or pause_ms),
        "default_content_ms": 170.0,
    }


def _is_en_word_char(ch: str) -> bool:
    return bool(re.match(r"[A-Za-z']", ch))


def parse_inline_target_text_en(text: str):
    plain_chars = []
    duration_ms = {}
    pause_ms = {}
    occurrence = {}
    current_word_keys = []
    last_visible_key = None
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == "[":
            end = text.find("]", i + 1)
            if end == -1:
                raise ValueError("unclosed pause marker '[' in --target-text")
            if last_visible_key is None:
                raise ValueError("pause marker must appear after at least one visible character")
            payload = text[i + 1 : end].strip()
            if not payload:
                raise ValueError("empty pause marker [] is not allowed")
            pause_ms[last_visible_key] = float(payload)
            i = end + 1
            continue

        if ch in "{}]":
            raise ValueError(f"unexpected control character '{ch}' in --target-text")

        plain_chars.append(ch)
        occurrence[ch] = occurrence.get(ch, 0) + 1
        char_key = f"{ch}#{occurrence[ch]}"
        if not ch.isspace():
            last_visible_key = char_key
        i += 1

        if _is_en_word_char(ch):
            current_word_keys.append(char_key)
        else:
            current_word_keys = []

        if i < len(text) and text[i] == "{":
            end = text.find("}", i + 1)
            if end == -1:
                raise ValueError("unclosed duration marker '{' in --target-text")
            payload = text[i + 1 : end].strip()
            if not payload:
                raise ValueError("empty duration marker {} is not allowed")
            if not current_word_keys:
                raise ValueError("English duration marker must appear after an English word")
            total_duration_ms = float(payload)
            per_char_ms = total_duration_ms / len(current_word_keys)
            for key in current_word_keys:
                duration_ms[key] = per_char_ms
            i = end + 1

    return {
        "plain_text": "".join(plain_chars),
        "duration_ms": duration_ms,
        "pause_ms": pause_ms,
        "has_control": bool(duration_ms or pause_ms),
        "default_content_ms": DEFAULT_EN_INLINE_CONTENT_MS,
    }


def parse_inline_target_text(text: str, language: str):
    if language == "en":
        return parse_inline_target_text_en(text)
    return parse_inline_target_text_zh(text)


def get_default_prompt_assets(language: str):
    asset_root = DEFAULT_PROMPT_ROOTS[language]
    audio_candidates = [asset_root / "prompt.wav", asset_root / "prompt.mp3"]
    for path in audio_candidates:
        if path.exists():
            return {
                "root": asset_root,
                "audio": path,
                "text": asset_root / "prompt.txt",
                "track": asset_root / "prompt_track.json",
                "alignment_raw": asset_root / "prompt_alignment_raw.json",
                "alignment_debug": asset_root / "prompt_alignment_debug.json",
            }
    raise FileNotFoundError(f"no built-in prompt audio found under {asset_root}")


def maybe_load_builtin_prompt_prefix_context(
    prompt_audio: Path,
    prompt_text: str,
    prompt_language: str,
    output_dir: Path,
):
    assets = get_default_prompt_assets(prompt_language)
    if prompt_audio.resolve() != assets["audio"].resolve():
        return None
    if not assets["track"].exists() or not assets["text"].exists():
        return None
    builtin_prompt_text = assets["text"].read_text(encoding="utf-8").strip()
    if prompt_text.strip() != builtin_prompt_text:
        return None

    prompt_track = json.loads(assets["track"].read_text(encoding="utf-8"))
    save_json(output_dir / "prompt_track.json", prompt_track)
    if assets["alignment_raw"].exists():
        save_json(output_dir / "prompt_alignment_raw.json", json.loads(assets["alignment_raw"].read_text(encoding="utf-8")))
    if assets["alignment_debug"].exists():
        save_json(output_dir / "prompt_alignment_debug.json", json.loads(assets["alignment_debug"].read_text(encoding="utf-8")))

    return {
        "audio_path": str(prompt_audio.resolve()),
        "prefix_tokens": list(prompt_track["prefix_tokens"]),
        "prefix_durations": [list(pair) for pair in prompt_track["prefix_durations"]],
        "prefix_token_count": len(prompt_track["prefix_tokens"]),
        "source_sample_dir": str(prompt_audio.resolve().parent),
        "source_prompt_frames": float(prompt_track["prompt_frames"]),
        "source_total_frames": float(prompt_track["prompt_frames"]),
        "prompt_text_source": prompt_track["prompt_text"],
        "prompt_id": prompt_track.get("prompt_id", prompt_audio.stem),
        "prompt_metadata": None,
    }


def load_control_variant(path: Path, target_text: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "slug": payload.get("slug", "controlled"),
        "text": target_text,
        "seed": payload.get("seed"),
        "notes": payload.get("notes", ""),
        "duration_ms": payload.get("duration_ms", {}),
        "pause_ms": payload.get("pause_ms", {}),
    }


def run_spontaneous(args, prompt_text: str, output_dir: Path):
    cmd = [
        sys.executable,
        str(RUN_SPONTANEOUS_DEMO),
        "--prompt-audio",
        str(Path(args.prompt_audio).resolve()),
        "--prompt-text",
        prompt_text,
        "--target-text",
        args.target_text,
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--output-prefix",
        "gen_target_only",
        "--steps",
        str(args.steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--sway-sampling-coef",
        str(args.sway_sampling_coef),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=build_subprocess_env())


def run_controlled(args, prompt_text: str, output_dir: Path):
    prefix_context = build_prompt_prefix_context(
        prompt_audio=Path(args.prompt_audio).resolve(),
        prompt_text=prompt_text,
        prompt_language=args.language,
        output_dir=output_dir,
        prompt_id=None,
        mfa_jobs=args.mfa_jobs,
        keep_mfa_workdir=args.keep_mfa_workdir,
    )
    variant = load_control_variant(Path(args.control_json).resolve(), args.target_text)
    track_payload = build_track_from_preset(
        prefix_context=prefix_context,
        variant=variant,
        content_ms=args.content_ms,
        punct_ms=args.punct_ms,
    )
    track_path = output_dir / "custom_track.json"
    save_json(track_path, track_payload)
    save_json(output_dir / "request.json", variant)
    cmd = [
        sys.executable,
        str(RUN_EDIT_FROM_JSON),
        "--audio",
        str(Path(args.prompt_audio).resolve()),
        "--track-json",
        str(track_path.resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--synth-python",
        args.synth_python,
        "--steps",
        str(args.steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--sway-sampling-coef",
        str(args.sway_sampling_coef),
    ]
    if variant.get("seed") is not None:
        cmd.extend(["--seed", str(variant["seed"])])
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=build_subprocess_env())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    default_assets = get_default_prompt_assets(args.language)
    if args.prompt_audio is None:
        args.prompt_audio = str(default_assets["audio"])
    if args.prompt_text_file is None:
        args.prompt_text_file = str(default_assets["text"])
    prompt_text = args.prompt_text or Path(args.prompt_text_file).read_text(encoding="utf-8").strip()
    raw_target_text = args.target_text
    inline_control = parse_inline_target_text(args.target_text, args.language)

    if args.control_json and inline_control["has_control"]:
        raise ValueError("use either inline control markers in --target-text or --control-json, not both")

    plain_target_text = inline_control["plain_text"]

    if args.control_json:
        args.target_text = plain_target_text
        run_controlled(args, prompt_text, output_dir)
        mode = "controlled"
        control_source = "control_json"
    elif inline_control["has_control"]:
        args.target_text = plain_target_text
        variant = {
            "slug": "controlled_inline",
            "text": plain_target_text,
            "seed": None,
            "notes": "Controlled by inline markers in --target-text.",
            "duration_ms": inline_control["duration_ms"],
            "pause_ms": inline_control["pause_ms"],
        }
        prompt_audio_path = Path(args.prompt_audio).resolve()
        prefix_context = maybe_load_builtin_prompt_prefix_context(
            prompt_audio=prompt_audio_path,
            prompt_text=prompt_text,
            prompt_language=args.language,
            output_dir=output_dir,
        )
        if prefix_context is None:
            prefix_context = build_prompt_prefix_context(
                prompt_audio=prompt_audio_path,
                prompt_text=prompt_text,
                prompt_language=args.language,
                output_dir=output_dir,
                prompt_id=None,
                mfa_jobs=args.mfa_jobs,
                keep_mfa_workdir=args.keep_mfa_workdir,
            )
        track_payload = build_track_from_preset(
            prefix_context=prefix_context,
            variant=variant,
            content_ms=inline_control["default_content_ms"],
            punct_ms=args.punct_ms,
        )
        track_path = output_dir / "custom_track.json"
        save_json(track_path, track_payload)
        save_json(output_dir / "request.json", variant)
        cmd = [
            sys.executable,
            str(RUN_EDIT_FROM_JSON),
            "--audio",
            str(Path(args.prompt_audio).resolve()),
            "--track-json",
            str(track_path.resolve()),
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--synth-python",
            args.synth_python,
            "--steps",
            str(args.steps),
            "--cfg-strength",
            str(args.cfg_strength),
            "--sway-sampling-coef",
            str(args.sway_sampling_coef),
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=build_subprocess_env())
        mode = "controlled"
        control_source = "inline_target_text"
    else:
        args.target_text = plain_target_text
        run_spontaneous(args, prompt_text, output_dir)
        mode = "spontaneous"
        control_source = None

    summary = {
        "mode": mode,
        "prompt_audio": str(Path(args.prompt_audio).resolve()),
        "prompt_text": prompt_text,
        "target_text": plain_target_text,
        "raw_target_text": raw_target_text,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "control_json": str(Path(args.control_json).resolve()) if args.control_json else None,
        "control_source": control_source,
        "default_content_ms_for_unmarked_chars": (
            inline_control["default_content_ms"] if inline_control["has_control"] else args.content_ms
        ),
    }
    save_json(output_dir / "entry_summary.json", summary)


if __name__ == "__main__":
    main()
