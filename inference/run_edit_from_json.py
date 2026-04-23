#!/usr/bin/env python3

import argparse
from pathlib import Path

from release_utils import DEFAULT_SYNTH_PYTHON
from release_utils import synthesize_with_track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument("--track-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesize_with_track(
        prompt_audio=Path(args.audio).resolve() if args.audio else None,
        track_json=Path(args.track_json).resolve(),
        output_dir=output_dir,
        checkpoint=Path(args.checkpoint).resolve(),
        synth_python=args.synth_python,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
