#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

from release_utils import build_prompt_prefix_context
from release_utils import build_subprocess_env
from release_utils import build_track_from_preset
from release_utils import DEFAULT_SYNTH_PYTHON
from release_utils import save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "assets" / "default_prompt"
RUN_SPONTANEOUS_DEMO = REPO_ROOT / "inference" / "run_spontaneous_demo.py"
RUN_EDIT_FROM_JSON = REPO_ROOT / "inference" / "run_edit_from_json.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-audio", default=str(ASSET_ROOT / "prompt.wav"))
    parser.add_argument("--prompt-text-file", default=str(ASSET_ROOT / "prompt.txt"))
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


def parse_inline_target_text(text: str):
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
    prompt_text = args.prompt_text or Path(args.prompt_text_file).read_text(encoding="utf-8").strip()
    inline_control = parse_inline_target_text(args.target_text)

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
        prefix_context = build_prompt_prefix_context(
            prompt_audio=Path(args.prompt_audio).resolve(),
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
        "raw_target_text": args.target_text,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "control_json": str(Path(args.control_json).resolve()) if args.control_json else None,
        "control_source": control_source,
        "default_content_ms_for_unmarked_chars": args.content_ms,
    }
    save_json(output_dir / "entry_summary.json", summary)


if __name__ == "__main__":
    main()
