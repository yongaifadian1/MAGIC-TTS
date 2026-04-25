#!/usr/bin/env python3
"""Export the 100-sample B@150 public-eval smoke set into release format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "b150_public_eval_smoke_100_pkg"
DEFAULT_SPLIT_NAME = "b150_public_eval_smoke_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-samples",
        required=True,
        help="Path to the maintainer-side selected_samples.jsonl source file.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-name", default=DEFAULT_SPLIT_NAME)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def convert_char_to_pinyin(text_list, polyphone: bool = True):
    import jieba
    from pypinyin import Style, lazy_pinyin

    final_text_list = []
    zh_quote_trans = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    custom_trans = str.maketrans({";": ","})
    for text in text_list:
        char_list = []
        text = text.translate(zh_quote_trans)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for item in seg:
                    if item not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(item)
            else:
                for char in seg:
                    if ord(char) < 256:
                        char_list.extend(char)
                    else:
                        if char not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(char, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(char)
        final_text_list.append(char_list)
    return final_text_list


def tokenize_word(word_text: str):
    if contains_cjk(word_text):
        return convert_char_to_pinyin([word_text], polyphone=True)[0]
    return list(word_text)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sidecar_words(sidecar_path: Path) -> list[dict]:
    with sidecar_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    words = []
    for segment in obj.get("segments", []):
        for word_info in segment.get("words", []):
            word = str(word_info.get("word", "")).strip()
            if not word:
                continue
            start = float(word_info.get("start", 0.0) or 0.0)
            end = float(word_info.get("end", start) or start)
            if end <= start:
                continue
            words.append({"word": word, "start": start, "end": end})
    return words


def build_token_track_from_words(words: list[dict]) -> tuple[list[str], list[list[float]]]:
    tokens = []
    token_durations = []
    for word_idx, word_info in enumerate(words):
        word_text = str(word_info["word"])
        start = float(word_info["start"])
        end = float(word_info["end"])
        duration_frames = max(0.0, end - start) * TARGET_SAMPLE_RATE / HOP_LENGTH
        next_start = end
        if word_idx + 1 < len(words):
            next_start = float(words[word_idx + 1]["start"])
        pause_after_frames = max(0.0, next_start - end) * TARGET_SAMPLE_RATE / HOP_LENGTH
        word_tokens = tokenize_word(word_text)
        if not word_tokens:
            continue
        content_positions = [idx for idx, token in enumerate(word_tokens) if str(token).strip()]
        denom = max(1, len(content_positions))
        per_token_content = duration_frames / denom
        last_content_idx = content_positions[-1] if content_positions else -1
        for local_idx, token in enumerate(word_tokens):
            tokens.append(str(token).lower())
            if local_idx in content_positions:
                content_value = per_token_content
                pause_value = pause_after_frames if local_idx == last_content_idx else 0.0
            else:
                content_value = 0.0
                pause_value = 0.0
            token_durations.append([float(content_value), float(pause_value)])
    return tokens, token_durations


def infer_rel_audio_path(audio_path: Path) -> Path:
    parts = list(audio_path.parts)
    if "Emilia_decoded" in parts:
        idx = parts.index("Emilia_decoded")
        return Path("audio").joinpath(*parts[idx + 1 :])
    return Path("audio") / audio_path.name


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    selected_samples_path = Path(args.selected_samples).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_empty_dir(output_dir, overwrite=args.overwrite)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(selected_samples_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    dataset_rows = []
    durations = []
    export_manifest_rows = []
    rel_audio_paths = [infer_rel_audio_path(Path(row["audio_path"]).resolve()) for row in rows]

    for parent in sorted({path.parent for path in rel_audio_paths}):
        os.makedirs(raw_dir / parent, exist_ok=True)

    for row, rel_audio_path in zip(rows, rel_audio_paths):
        audio_path = Path(row["audio_path"]).resolve()
        sidecar_path = Path(row["mfa_words_json"]).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio: {audio_path}")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"missing sidecar: {sidecar_path}")

        words = load_sidecar_words(sidecar_path)
        text_tokens, token_durations = build_token_track_from_words(words)
        if len(text_tokens) != len(token_durations):
            raise RuntimeError(
                f"text/token_duration mismatch for {row['sample_id']}: "
                f"{len(text_tokens)} vs {len(token_durations)}"
        )

        dst_audio_path = raw_dir / rel_audio_path
        subprocess.run(["cp", str(audio_path), str(dst_audio_path)], check=True)

        duration_sec = float(row["duration_sec"])
        prompt_boundary_index = int(row["prefix_token_count"])
        if prompt_boundary_index <= 0 or prompt_boundary_index >= len(text_tokens):
            raise RuntimeError(
                f"invalid prompt boundary for {row['sample_id']}: "
                f"{prompt_boundary_index} not in (0, {len(text_tokens)})"
            )

        dataset_rows.append(
            {
                "sample_id": row["sample_id"],
                "utt_id": row["utt_id"],
                "language": row["language"],
                "audio_path": str(rel_audio_path),
                "duration": duration_sec,
                "text": text_tokens,
                "raw_text": row["text"],
                "token_durations": token_durations,
                "prompt_boundary_index": prompt_boundary_index,
                "prompt_sec": float(row["prompt_sec"]),
                "target_text": row["target_text"],
            }
        )
        durations.append(duration_sec)
        export_manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "utt_id": row["utt_id"],
                "language": row["language"],
                "audio_path": str(rel_audio_path),
                "duration": duration_sec,
                "prompt_boundary_index": prompt_boundary_index,
                "target_text": row["target_text"],
            }
        )

    from datasets import Dataset

    with tempfile.TemporaryDirectory(prefix="magictts_eval_export_", dir="/tmp") as tmp_dir:
        raw_hf_dir = Path(tmp_dir) / "raw"
        Dataset.from_list(dataset_rows).save_to_disk(str(raw_hf_dir))
        for path in raw_hf_dir.iterdir():
            shutil.move(str(path), str(raw_dir / path.name))

    with (output_dir / "duration.json").open("w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False, indent=2)

    with (output_dir / "selected_samples.exported.jsonl").open("w", encoding="utf-8") as f:
        for row in export_manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "export_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split_name": args.split_name,
                "purpose": "user smoke-test fine-tuning and pipeline validation",
                "num_samples": len(dataset_rows),
                "selected_samples_filename": selected_samples_path.name,
                "format": "raw/ + duration.json",
                "audio_path_base": "raw/",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# B150 Public Eval Smoke 100\n\n"
            "This dataset is a 100-sample public-eval smoke split packaged in the\n"
            "same `raw + duration.json` format as the user-facing training data.\n\n"
            "It is intended for low-cost pipeline bring-up and fine-tuning smoke\n"
            "tests. It is not the recommended split for formal experiments.\n\n"
            "This package is self-contained: audio files live under `raw/audio/`\n"
            "and row-level `audio_path` entries are relative to `raw/`.\n"
        )

    print(f"Exported {len(dataset_rows)} samples to {output_dir}")


if __name__ == "__main__":
    main()
