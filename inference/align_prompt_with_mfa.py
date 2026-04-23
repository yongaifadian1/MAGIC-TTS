#!/usr/bin/env python3

import argparse
from pathlib import Path

from release_utils import build_prompt_prefix_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--mfa-jobs", type=int, default=8)
    parser.add_argument("--keep-mfa-workdir", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_prompt_prefix_context(
        prompt_audio=Path(args.prompt_audio).resolve(),
        prompt_text=args.prompt_text,
        prompt_language=args.language,
        output_dir=output_dir,
        prompt_id=args.prompt_id,
        mfa_jobs=args.mfa_jobs,
        keep_mfa_workdir=args.keep_mfa_workdir,
    )


if __name__ == "__main__":
    main()
