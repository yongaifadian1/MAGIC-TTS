#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from release_utils import build_subprocess_env
from release_utils import DEFAULT_SYNTH_PYTHON


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EDIT_DEMO = REPO_ROOT / "inference" / "run_edit_demo.py"
PRESET_ROOT = REPO_ROOT / "presets"
ASSET_ROOT = REPO_ROOT / "assets" / "default_prompt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument(
        "--prompt-audio",
        default=str(ASSET_ROOT / "prompt.wav"),
        help="Default uses the built-in distributable prompt.",
    )
    parser.add_argument(
        "--prompt-text-file",
        default=str(ASSET_ROOT / "prompt.txt"),
        help="Default uses the built-in distributable prompt text.",
    )
    parser.add_argument(
        "--variant-spec",
        default=(
            "navigation_turn=v1_baseline_eqdur,v2_pause_only_boundary,v3_pause_plus_char_turn;"
            "kids_reading=v1_baseline_eqdur,v2_pause_only_syllable,v3_pause_plus_char_syllable;"
            "accessibility_code=v1_baseline_eqdur,v2_pause_only_grouped,v3_pause_plus_char_digits;"
            "station_wushanzhan=v1_baseline_eqdur,v2_pause_only_station_boundary,v3_pause_plus_char_station_name"
        ),
    )
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--content-ms", type=float, default=170.0)
    parser.add_argument("--punct-ms", type=float, default=50.0)
    parser.add_argument("--mfa-jobs", type=int, default=8)
    return parser.parse_args()


def parse_variant_spec(spec: str):
    mapping = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        scene, variants = item.split("=", 1)
        mapping[scene.strip()] = variants.strip()
    return mapping


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = Path(args.prompt_text_file).read_text(encoding="utf-8").strip()
    variant_map = parse_variant_spec(args.variant_spec)

    preset_paths = {
        "navigation_turn": PRESET_ROOT / "navigation_turn.json",
        "kids_reading": PRESET_ROOT / "kids_reading.json",
        "accessibility_code": PRESET_ROOT / "accessibility_code.json",
        "station_wushanzhan": PRESET_ROOT / "station_wushanzhan.json",
    }

    for scene, preset_path in preset_paths.items():
        cmd = [
            sys.executable,
            str(RUN_EDIT_DEMO),
            "--prompt-audio",
            str(Path(args.prompt_audio).resolve()),
            "--prompt-text",
            prompt_text,
            "--language",
            args.language,
            "--preset",
            str(preset_path.resolve()),
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--output-dir",
            str(output_dir),
            "--variant-slug",
            variant_map[scene],
            "--synth-python",
            args.synth_python,
            "--steps",
            str(args.steps),
            "--cfg-strength",
            str(args.cfg_strength),
            "--sway-sampling-coef",
            str(args.sway_sampling_coef),
            "--content-ms",
            str(args.content_ms),
            "--punct-ms",
            str(args.punct_ms),
            "--mfa-jobs",
            str(args.mfa_jobs),
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=build_subprocess_env())


if __name__ == "__main__":
    main()
