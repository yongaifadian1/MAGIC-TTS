#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

from release_utils import build_subprocess_env


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SPONTANEOUS_DEMO = REPO_ROOT / "inference" / "run_spontaneous_demo.py"
PRESET_ROOT = REPO_ROOT / "presets"
ASSET_ROOT = REPO_ROOT / "assets" / "default_prompt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
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
        "--scene-spec",
        default=(
            "navigation_turn=前方路口，左转。;"
            "kids_reading=请跟我读，苹果。;"
            "accessibility_code=验证码是三七九，二一八。;"
            "station_wushanzhan=前方到站，五山站。"
        ),
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    return parser.parse_args()


def parse_scene_spec(spec: str):
    mapping = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        scene, text = item.split("=", 1)
        mapping[scene.strip()] = text.strip()
    return mapping


def load_scene_slug_map():
    preset_paths = {
        "navigation_turn": PRESET_ROOT / "navigation_turn.json",
        "kids_reading": PRESET_ROOT / "kids_reading.json",
        "accessibility_code": PRESET_ROOT / "accessibility_code.json",
        "station_wushanzhan": PRESET_ROOT / "station_wushanzhan.json",
    }
    mapping = {}
    for key, path in preset_paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping[key] = data["slug"]
    return mapping


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = Path(args.prompt_text_file).read_text(encoding="utf-8").strip()
    scene_text_map = parse_scene_spec(args.scene_spec)
    scene_slug_map = load_scene_slug_map()

    for scene_key, target_text in scene_text_map.items():
        scene_output_dir = output_dir / scene_slug_map[scene_key] / "spontaneous"
        cmd = [
            sys.executable,
            str(RUN_SPONTANEOUS_DEMO),
            "--prompt-audio",
            str(Path(args.prompt_audio).resolve()),
            "--prompt-text",
            prompt_text,
            "--target-text",
            target_text,
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--output-dir",
            str(scene_output_dir),
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


if __name__ == "__main__":
    main()
