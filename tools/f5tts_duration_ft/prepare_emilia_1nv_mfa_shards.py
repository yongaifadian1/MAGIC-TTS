#!/usr/bin/env python3
"""Shard the current Emilia 1nv training jsonl for large-scale MFA alignment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL: list[str] = []
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "mfa_shards"
DEFAULT_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"
DEFAULT_TEXT_FIELD = "ground_truth"
DEFAULT_SAMPLES_PER_SHARD = 10000
AUDIO_ROOT = DEFAULT_AUDIO_ROOT


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def infer_language(audio_path: str, text: str) -> tuple[str, str]:
    audio = Path(audio_path)
    bucket = "UNK"
    try:
        rel_audio = audio.relative_to(AUDIO_ROOT)
        bucket = rel_audio.parts[0]
    except ValueError:
        rel_audio = None

    upper_bucket = bucket.upper()
    if "ZH" in upper_bucket:
        return "zh", bucket
    if "EN" in upper_bucket:
        return "en", bucket
    return ("zh" if contains_cjk(text) else "en"), bucket


def iter_jsonl(paths: list[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield path, line


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", nargs="+", required=True, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    global AUDIO_ROOT
    audio_root = Path(args.audio_root).resolve()
    AUDIO_ROOT = audio_root
    output_root = Path(args.output_root).resolve()
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    current_shard_index = defaultdict(int)
    current_shard_size = defaultdict(int)
    current_handles = {}
    shard_counts = Counter()
    lang_counts = Counter()
    bucket_counts = Counter()
    kept = 0
    skipped = 0
    shard_paths: list[Path] = []

    def open_handle(lang: str):
        shard_dir = shard_root / lang
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{current_shard_index[lang]:05d}.jsonl"
        handle = shard_path.open("w", encoding="utf-8")
        current_handles[lang] = handle
        shard_paths.append(shard_path)
        return handle

    for source_path, line in iter_jsonl(args.input_jsonl):
        try:
            obj = json.loads(line)
        except Exception:
            skipped += 1
            continue

        audio_path = str(obj.get("audio_path", "")).strip()
        text = str(obj.get(args.text_field, "")).strip()
        if not audio_path or not text:
            skipped += 1
            continue

        audio = Path(audio_path)
        if not audio.is_absolute():
            audio = (audio_root / audio).resolve()

        try:
            rel_audio = audio.relative_to(audio_root)
        except ValueError:
            skipped += 1
            continue

        lang, bucket = infer_language(audio_path, text)
        if lang not in {"en", "zh"}:
            skipped += 1
            continue

        if lang not in current_handles:
            open_handle(lang)
        elif current_shard_size[lang] >= args.samples_per_shard:
            current_handles[lang].close()
            current_shard_index[lang] += 1
            current_shard_size[lang] = 0
            open_handle(lang)

        utt_id = f"{lang}_{current_shard_index[lang]:05d}_{current_shard_size[lang]:06d}"
        record = {
            "utt_id": utt_id,
            "audio_path": str(audio),
            "rel_audio_path": rel_audio.as_posix(),
            "text": text,
            "text_field": args.text_field,
            "language": lang,
            "bucket": bucket,
            "source_jsonl": source_path,
        }
        current_handles[lang].write(json.dumps(record, ensure_ascii=False) + "\n")
        current_shard_size[lang] += 1
        shard_counts[str(shard_root / lang / f"shard_{current_shard_index[lang]:05d}.jsonl")] += 1
        lang_counts[lang] += 1
        bucket_counts[bucket] += 1
        kept += 1

    for handle in current_handles.values():
        handle.close()

    shard_list_path = output_root / "shard_list.txt"
    with shard_list_path.open("w", encoding="utf-8") as f:
        for path in sorted(shard_paths):
            f.write(str(path) + "\n")

    summary = {
        "input_jsonl": args.input_jsonl,
        "text_field": args.text_field,
        "audio_root": str(audio_root),
        "samples_per_shard": args.samples_per_shard,
        "kept": kept,
        "skipped": skipped,
        "language_counts": dict(lang_counts),
        "bucket_counts": dict(bucket_counts),
        "shard_counts": dict(sorted(shard_counts.items())),
        "shard_list": str(shard_list_path),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Emilia 1nv MFA shards\n\n")
        f.write(f"- text field: `{args.text_field}`\n")
        f.write(f"- samples per shard: `{args.samples_per_shard}`\n")
        f.write(f"- kept: `{kept}`\n")
        f.write(f"- skipped: `{skipped}`\n")
        f.write(f"- shard list: `{shard_list_path}`\n")
        f.write("\n## Language counts\n")
        for lang, count in sorted(lang_counts.items()):
            f.write(f"- `{lang}`: {count}\n")
        f.write("\n## Bucket counts\n")
        for bucket, count in sorted(bucket_counts.items()):
            f.write(f"- `{bucket}`: {count}\n")

    print(f"wrote shard metadata to: {output_root}")
    print(f"shard list: {shard_list_path}")
    print(f"kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()
