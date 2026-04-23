#!/usr/bin/env python3

import argparse
from pathlib import Path

from release_utils import build_prompt_prefix_context
from release_utils import build_track_from_preset
from release_utils import DEFAULT_SYNTH_PYTHON
from release_utils import load_preset
from release_utils import save_json
from release_utils import select_variants
from release_utils import synthesize_with_track
from release_utils import write_release_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--variant-slug", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--mfa-jobs", type=int, default=8)
    parser.add_argument("--keep-mfa-workdir", action="store_true")
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--content-ms", type=float, default=170.0)
    parser.add_argument("--punct-ms", type=float, default=50.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = load_preset(Path(args.preset).resolve())
    variants = select_variants(preset, args.variant_slug)
    if not variants:
        raise ValueError("no variants selected")

    prefix_context = build_prompt_prefix_context(
        prompt_audio=Path(args.prompt_audio).resolve(),
        prompt_text=args.prompt_text,
        prompt_language=args.language,
        output_dir=output_dir,
        prompt_id=args.prompt_id,
        mfa_jobs=args.mfa_jobs,
        keep_mfa_workdir=args.keep_mfa_workdir,
    )

    variant_dirs = []

    for variant in variants:
        variant_dir = output_dir / preset["slug"] / variant["slug"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        track_payload = build_track_from_preset(
            prefix_context=prefix_context,
            variant=variant,
            content_ms=args.content_ms,
            punct_ms=args.punct_ms,
        )
        track_path = variant_dir / "custom_track.json"
        save_json(track_path, track_payload)
        save_json(variant_dir / "request.json", variant)
        synthesize_with_track(
            prompt_audio=Path(args.prompt_audio).resolve(),
            track_json=track_path,
            output_dir=variant_dir,
            checkpoint=Path(args.checkpoint).resolve(),
            synth_python=args.synth_python,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            seed=variant.get("seed"),
        )
        variant_dirs.append(variant_dir)

    write_release_summary(
        output_dir=output_dir,
        preset=preset,
        prompt_audio=Path(args.prompt_audio).resolve(),
        prompt_text=args.prompt_text,
        checkpoint=Path(args.checkpoint).resolve(),
        variant_dirs=variant_dirs,
    )


if __name__ == "__main__":
    main()
