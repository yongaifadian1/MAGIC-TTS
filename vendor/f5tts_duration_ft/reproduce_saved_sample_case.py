import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import jieba
from pypinyin import Style, lazy_pinyin

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
DEFAULT_SYNTH_PYTHON = os.environ.get("MAGICTTS_SYNTH_PYTHON", sys.executable)
SCRIPT_DIR = Path(__file__).resolve().parent
SYNTH_SCRIPT = SCRIPT_DIR / "ttrack_edit_synthesize.py"


def convert_char_to_pinyin(text_list, polyphone=True):
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


def tokenize_text(text):
    return convert_char_to_pinyin([text], polyphone=True)[0]


def build_token_duration_track(audio_path):
    words_path = Path(audio_path).with_suffix(".words.json")
    with words_path.open("r", encoding="utf-8") as f:
        align_obj = json.load(f)

    segments = align_obj.get("segments", [])
    flat_words = []
    for segment in segments:
        for word_info in segment.get("words", []):
            if word_info.get("word", ""):
                flat_words.append(word_info)

    tokens = []
    token_duration_features = []
    for word_idx, word_info in enumerate(flat_words):
        word_text = word_info.get("word", "")
        start = float(word_info.get("start", 0.0) or 0.0)
        end = float(word_info.get("end", start) or start)
        duration_frames = max(0.0, end - start) * TARGET_SAMPLE_RATE / HOP_LENGTH
        next_start = end
        if word_idx + 1 < len(flat_words):
            next_start = float(flat_words[word_idx + 1].get("start", end) or end)
        pause_after_frames = max(0.0, next_start - end) * TARGET_SAMPLE_RATE / HOP_LENGTH
        word_tokens = tokenize_text(word_text)
        if not word_tokens:
            continue
        content_positions = [idx for idx, token in enumerate(word_tokens) if str(token).strip()]
        denom = max(1, len(content_positions))
        per_token_content = duration_frames / denom
        last_content_idx = content_positions[-1] if content_positions else -1
        for idx, token in enumerate(word_tokens):
            tokens.append(token)
            if idx in content_positions:
                content_value = per_token_content
                pause_value = pause_after_frames if idx == last_content_idx else 0.0
            else:
                content_value = 0.0
                pause_value = 0.0
            token_duration_features.append([content_value, pause_value])
    return tokens, token_duration_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument(
        "--set-token-indices-ms",
        default=None,
        help="Pairs like '123:500;125:500' to set specific token content durations in milliseconds.",
    )
    parser.add_argument(
        "--set-token-pause-indices-ms",
        default=None,
        help="Pairs like '125:500' to set specific token pause_after durations in milliseconds.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional duration-finetuned F5 checkpoint passed to the synthesis script.",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def parse_token_index_ms_pairs(spec):
    if not spec:
        return []
    pairs = []
    for chunk in re.split(r"[;,]+", spec):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError("invalid token-duration pair '{}'".format(chunk))
        index_text, ms_text = chunk.split(":", 1)
        token_index = int(index_text.strip())
        duration_ms = float(ms_text.strip())
        if token_index < 0:
            raise ValueError("token index must be non-negative")
        if duration_ms <= 0:
            raise ValueError("duration ms must be positive")
        pairs.append((token_index, duration_ms))
    return pairs


def main():
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (sample_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with (sample_dir / "converted_text.json").open("r", encoding="utf-8") as f:
        converted_text = json.load(f)

    audio_path = metadata["audio_path"]
    prompt_frames = float(metadata["prompt_frames"])
    total_frames = int(metadata["total_frames"])
    rebuilt_tokens, rebuilt_token_durations = build_token_duration_track(audio_path)

    if rebuilt_tokens != converted_text:
        raise ValueError("reconstructed tokens do not match converted_text.json")

    edited_token_durations = [[float(x), float(y)] for x, y in rebuilt_token_durations]
    token_edits = []
    for token_index, duration_ms in parse_token_index_ms_pairs(args.set_token_indices_ms):
        if token_index >= len(edited_token_durations):
            raise ValueError("token index {} out of range {}".format(token_index, len(edited_token_durations)))
        target_frames = duration_ms * TARGET_SAMPLE_RATE / 1000.0 / HOP_LENGTH
        edited_token_durations[token_index][0] = target_frames
        token_edits.append(
            {
                "token_index": token_index,
                "token": rebuilt_tokens[token_index],
                "duration_ms": duration_ms,
                "duration_frames": target_frames,
            }
        )
    pause_edits = []
    for token_index, pause_ms in parse_token_index_ms_pairs(args.set_token_pause_indices_ms):
        if token_index >= len(edited_token_durations):
            raise ValueError("token index {} out of range {}".format(token_index, len(edited_token_durations)))
        target_frames = pause_ms * TARGET_SAMPLE_RATE / 1000.0 / HOP_LENGTH
        edited_token_durations[token_index][1] = target_frames
        pause_edits.append(
            {
                "token_index": token_index,
                "token": rebuilt_tokens[token_index],
                "pause_ms": pause_ms,
                "pause_frames": target_frames,
            }
        )

    original_track_total_frames = sum(sum(pair) for pair in rebuilt_token_durations)
    edited_track_total_frames = sum(sum(pair) for pair in edited_token_durations)
    residual_frames = max(0.0, float(total_frames) - original_track_total_frames)
    edited_total_frames = int(round(edited_track_total_frames + residual_frames))

    track_payload = {
        "audio_path": audio_path,
        "raw_text": metadata.get("raw_text"),
        "text_tokens": converted_text,
        "token_durations": rebuilt_token_durations,
        "edited_token_durations": edited_token_durations,
        "original_total_frames": float(total_frames),
        "edited_total_frames": edited_total_frames,
        "original_track_total_frames": original_track_total_frames,
        "edited_track_total_frames": edited_track_total_frames,
        "track_residual_frames": residual_frames,
        "token_edits": token_edits,
        "pause_edits": pause_edits,
        "source_sample_dir": str(sample_dir),
        "source_prompt_frames": prompt_frames,
        "source_total_frames": total_frames,
        "reproduction_mode": "training_consistent_prompt_completion",
    }
    track_json = output_dir / "repro_track.json"
    with track_json.open("w", encoding="utf-8") as f:
        json.dump(track_payload, f, ensure_ascii=False, indent=2)

    synth_cmd = [
        args.synth_python,
        str(SYNTH_SCRIPT),
        "--audio",
        audio_path,
        "--track-json",
        str(track_json),
        "--output-dir",
        str(output_dir),
        "--prompt-frames",
        str(prompt_frames),
        "--steps",
        str(args.steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--sway-sampling-coef",
        str(args.sway_sampling_coef),
        "--preserve-cond-audio-in-output",
    ]
    if args.checkpoint:
        synth_cmd.extend(["--checkpoint", args.checkpoint])
    if args.seed is not None:
        synth_cmd.extend(["--seed", str(args.seed)])

    subprocess.run(synth_cmd, check=True, cwd=SCRIPT_DIR.parents[1])


if __name__ == "__main__":
    main()
