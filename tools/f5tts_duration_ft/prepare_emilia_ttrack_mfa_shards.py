#!/usr/bin/env python3
"""Shard the full T-Track Emilia align TSVs for large-scale MFA alignment."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EN_ALIGN_TSV = REPO_ROOT / "data" / "emilia_en_align.tsv"
DEFAULT_ZH_ALIGN_TSV = REPO_ROOT / "data" / "emilia_zh_align.tsv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "mfa_shards_ttrack"
DEFAULT_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"
DEFAULT_SAMPLES_PER_SHARD = 36000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en-align-tsv", default=str(DEFAULT_EN_ALIGN_TSV))
    parser.add_argument("--zh-align-tsv", default=str(DEFAULT_ZH_ALIGN_TSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT))
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    return parser.parse_args()


def iter_align_tsv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw:
                continue
            try:
                audio_path, text = raw.split("\t", 1)
            except ValueError:
                yield line_no, None, None
                continue
            audio_path = audio_path.strip()
            text = text.strip()
            if not audio_path or not text:
                yield line_no, None, None
                continue
            yield line_no, audio_path, text


def main():
    args = parse_args()
    audio_root = Path(args.audio_root).resolve()
    output_root = Path(args.output_root).resolve()
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    current_shard_index = {"en": 0, "zh": 0}
    current_shard_size = {"en": 0, "zh": 0}
    current_handles: Dict[str, object] = {}
    shard_counts = Counter()
    lang_counts = Counter()
    bucket_counts = Counter()
    skipped = 0
    shard_paths: List[Path] = []

    def open_handle(lang: str):
        shard_dir = shard_root / lang
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{current_shard_index[lang]:05d}.jsonl"
        handle = shard_path.open("w", encoding="utf-8")
        current_handles[lang] = handle
        shard_paths.append(shard_path)
        return handle

    input_map = {
        "en": Path(args.en_align_tsv).resolve(),
        "zh": Path(args.zh_align_tsv).resolve(),
    }

    for lang, input_path in input_map.items():
        for line_no, audio_path, text in iter_align_tsv(input_path):
            if not audio_path or not text:
                skipped += 1
                continue

            audio = Path(audio_path)
            try:
                rel_audio = audio.relative_to(audio_root)
            except ValueError:
                skipped += 1
                continue

            if lang not in current_handles:
                open_handle(lang)
            elif current_shard_size[lang] >= args.samples_per_shard:
                current_handles[lang].close()
                current_shard_index[lang] += 1
                current_shard_size[lang] = 0
                open_handle(lang)

            bucket = rel_audio.parts[0] if rel_audio.parts else "UNK"
            utt_id = f"{lang}_{current_shard_index[lang]:05d}_{current_shard_size[lang]:06d}"
            shard_path = shard_root / lang / f"shard_{current_shard_index[lang]:05d}.jsonl"
            record = {
                "utt_id": utt_id,
                "audio_path": audio_path,
                "rel_audio_path": rel_audio.as_posix(),
                "text": text,
                "text_field": "align_tsv",
                "language": lang,
                "bucket": bucket,
                "source_align_tsv": str(input_path),
                "source_line_no": line_no,
            }
            current_handles[lang].write(json.dumps(record, ensure_ascii=False) + "\n")
            current_shard_size[lang] += 1
            shard_counts[str(shard_path)] += 1
            lang_counts[lang] += 1
            bucket_counts[bucket] += 1

    for handle in current_handles.values():
        handle.close()

    shard_list_path = output_root / "shard_list.txt"
    with shard_list_path.open("w", encoding="utf-8") as f:
        for path in sorted(shard_paths):
            f.write(str(path) + "\n")

    summary = {
        "input_align_tsv": {lang: str(path) for lang, path in input_map.items()},
        "audio_root": str(audio_root),
        "samples_per_shard": args.samples_per_shard,
        "kept": sum(lang_counts.values()),
        "skipped": skipped,
        "language_counts": dict(lang_counts),
        "bucket_counts": dict(bucket_counts),
        "shard_counts": dict(sorted(shard_counts.items())),
        "shard_list": str(shard_list_path),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Emilia T-Track MFA shards\n\n")
        f.write(f"- en align tsv: `{input_map['en']}`\n")
        f.write(f"- zh align tsv: `{input_map['zh']}`\n")
        f.write(f"- samples per shard: `{args.samples_per_shard}`\n")
        f.write(f"- kept: `{summary['kept']}`\n")
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
    print(f"kept={summary['kept']} skipped={skipped}")


if __name__ == "__main__":
    main()
