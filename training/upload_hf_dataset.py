#!/usr/bin/env python3
"""Upload a prepared MAGIC-TTS dataset folder to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Prepared dataset directory, for example data/b150_public_eval_smoke_100_pkg",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo id, for example username/b150-public-eval-smoke-100",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload MAGIC-TTS dataset package")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path inside the dataset repo where files will be uploaded.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repo if it does not exist yet.",
    )
    return parser.parse_args()


def validate_dataset_dir(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "README.md",
        dataset_dir / "duration.json",
        dataset_dir / "raw",
        dataset_dir / "raw" / "state.json",
        dataset_dir / "raw" / "dataset_info.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "dataset package is incomplete; missing required paths:\n" + "\n".join(missing)
        )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir does not exist: {dataset_dir}")
    validate_dataset_dir(dataset_dir)

    from huggingface_hub import HfApi

    api = HfApi()
    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        path_in_repo=args.repo_path,
        revision=args.revision,
        commit_message=args.commit_message,
    )

    print(f"Uploaded {dataset_dir} to hf://datasets/{args.repo_id}/{args.repo_path}")


if __name__ == "__main__":
    main()
