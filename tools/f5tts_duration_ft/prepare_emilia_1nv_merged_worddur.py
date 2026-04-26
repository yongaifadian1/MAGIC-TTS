import json
import os
import argparse
import wave
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets.arrow_writer import ArrowWriter
import jieba
from pypinyin import Style, lazy_pinyin
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL_PATHS: list[str] = []
DEFAULT_SAVE_DIR = str(REPO_ROOT / "data" / "b150_public")
DEFAULT_TOKENIZER = "pinyin"  # "char" | "pinyin"
DEFAULT_POLYPHONE = True
DEFAULT_MAX_WORKERS = 16
DEFAULT_CHUNKSIZE = 256
DEFAULT_ALIGNMENT_AUDIO_ROOT = str(REPO_ROOT / "data" / "raw_audio")
GLOBALS = {
    "tokenizer": DEFAULT_TOKENIZER,
    "polyphone": DEFAULT_POLYPHONE,
    "text_field": "ground_truth",
    "sample_rate": 24000,
    "hop_length": 256,
    "audio_root": None,
    "alignment_root": None,
    "alignment_audio_root": DEFAULT_ALIGNMENT_AUDIO_ROOT,
    "require_alignments": False,
}


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
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for count in pattern_count.values():
        if count > tolerance:
            return True
    return False


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.translate(str.maketrans({",": "，", "!": "！", "?": "？"}))
    return text


def tokenize_text(text: str):
    if GLOBALS["tokenizer"] == "pinyin":
        return convert_char_to_pinyin([text], polyphone=GLOBALS["polyphone"])[0]
    return list(text)


def words_json_path_from_audio(audio_path: str) -> Path:
    audio = Path(audio_path)
    alignment_root = GLOBALS["alignment_root"]
    if not alignment_root:
        return audio.with_suffix(".words.json")

    alignment_audio_root = Path(GLOBALS["alignment_audio_root"])
    try:
        rel_audio = audio.relative_to(alignment_audio_root)
    except ValueError:
        rel_audio = Path(*audio.parts[1:]) if audio.is_absolute() else audio
    return Path(alignment_root) / rel_audio.with_suffix(".words.json")


def build_token_duration_track(audio_path: str):
    words_path = words_json_path_from_audio(audio_path)
    try:
        with words_path.open("r", encoding="utf-8") as f:
            align_obj = json.load(f)
    except Exception:
        return None

    segments = align_obj.get("segments", [])
    if not segments:
        return None

    flat_words = []
    for segment in segments:
        for word_info in segment.get("words", []):
            if word_info.get("word", ""):
                flat_words.append(word_info)

    if not flat_words:
        return None

    tokens = []
    token_duration_features = []
    for word_idx, word_info in enumerate(flat_words):
            word_text = word_info.get("word", "")
            start = float(word_info.get("start", 0.0) or 0.0)
            end = float(word_info.get("end", start) or start)
            duration_frames = max(0.0, end - start) * GLOBALS["sample_rate"] / GLOBALS["hop_length"]
            next_start = end
            if word_idx + 1 < len(flat_words):
                next_start = float(flat_words[word_idx + 1].get("start", end) or end)
            pause_after_frames = max(0.0, next_start - end) * GLOBALS["sample_rate"] / GLOBALS["hop_length"]
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

    if not tokens:
        return None
    return tokens, token_duration_features


def process_line(line: str):
    try:
        obj = json.loads(line)
    except Exception:
        return None

    audio_path = obj.get("audio_path", "").strip()
    raw_text = obj.get(GLOBALS["text_field"], "").strip()
    if not audio_path or not raw_text:
        return None

    audio = Path(audio_path)
    if not audio.is_absolute():
        audio_root = GLOBALS["audio_root"]
        if not audio_root:
            return None
        audio = (Path(audio_root) / audio).resolve()
    audio_path = str(audio)

    raw_text = normalize_text(raw_text)
    if repetition_found(raw_text):
        return None

    duration = obj.get("duration")
    if duration is not None:
        try:
            duration = float(duration)
        except Exception:
            duration = None
    if duration is None:
        meta_path = Path(audio_path).with_suffix(".json")
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            duration = float(meta["duration"])
        except Exception:
            try:
                with wave.open(audio_path, "rb") as wf:
                    duration = float(wf.getnframes()) / float(wf.getframerate())
            except Exception:
                return None

    text = raw_text
    token_duration_track = build_token_duration_track(audio_path)
    if token_duration_track is not None:
        text, token_durations = token_duration_track
    else:
        if GLOBALS["require_alignments"]:
            return None
        text = tokenize_text(text)
        token_durations = [[0.0, 0.0] for _ in text]

    return {
        "audio_path": audio_path,
        "text": text,
        "raw_text": raw_text,
        "duration": duration,
        "token_durations": token_durations,
    }, duration, text


def iter_lines(paths: list[str], max_lines: int | None = None):
    seen = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line
                    seen += 1
                    if max_lines is not None and seen >= max_lines:
                        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--input-jsonl", nargs="+", required=True, default=DEFAULT_INPUT_JSONL_PATHS)
    parser.add_argument(
        "--text-field",
        default="ground_truth",
        help="JSONL text field to use, e.g. ground_truth, MNV_output, NVASR_output, or NVASR_no_nv.",
    )
    parser.add_argument("--tokenizer", choices=["char", "pinyin"], default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--no-polyphone", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional root used to resolve relative audio_path entries in the input JSONL.",
    )
    parser.add_argument(
        "--alignment-root",
        default=None,
        help="Optional root directory that mirrors audio relative paths and stores *.words.json sidecars.",
    )
    parser.add_argument(
        "--alignment-audio-root",
        default=DEFAULT_ALIGNMENT_AUDIO_ROOT,
        help="Audio root used to map audio paths into alignment-root.",
    )
    parser.add_argument(
        "--require-alignments",
        action="store_true",
        help="Skip samples whose token-duration sidecar is missing or unreadable.",
    )
    return parser.parse_args()


def make_metadata_path_writable(path: Path):
    if path.exists():
        try:
            path.chmod(0o664)
        except OSError as exc:
            print(f"Warning: could not chmod existing metadata file {path}: {exc}")


def precreate_metadata_path(path: Path):
    old_umask = os.umask(0o002)
    try:
        if not path.exists():
            with path.open("w", encoding="utf-8"):
                pass
        try:
            path.chmod(0o664)
        except OSError as exc:
            print(f"Warning: could not chmod precreated metadata file {path}: {exc}")
    finally:
        os.umask(old_umask)


def main():
    args = parse_args()
    GLOBALS["tokenizer"] = args.tokenizer
    GLOBALS["polyphone"] = not args.no_polyphone
    GLOBALS["text_field"] = args.text_field
    GLOBALS["sample_rate"] = args.sample_rate
    GLOBALS["hop_length"] = args.hop_length
    GLOBALS["audio_root"] = args.audio_root
    GLOBALS["alignment_root"] = args.alignment_root
    GLOBALS["alignment_audio_root"] = args.alignment_audio_root
    GLOBALS["require_alignments"] = args.require_alignments

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    vocab_set = set()
    kept = 0
    skipped = 0

    arrow_path = save_dir / "raw.arrow"
    duration_path = save_dir / "duration.json"
    vocab_path = save_dir / "vocab.txt"
    precreate_metadata_path(duration_path)
    precreate_metadata_path(vocab_path)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        with ArrowWriter(path=str(arrow_path), writer_batch_size=1) as writer:
            for result in tqdm(
                executor.map(process_line, iter_lines(args.input_jsonl, args.max_lines), chunksize=args.chunksize),
                desc="Processing merged Emilia 1nv",
            ):
                if result is None:
                    skipped += 1
                    continue
                sample, duration, text = result
                writer.write(sample)
                durations.append(duration)
                vocab_set.update(list(text))
                kept += 1

    # Some parallel filesystems can inherit a restrictive umask after the
    # multiprocessing ArrowWriter phase, leaving zero-permission metadata files.
    old_umask = os.umask(0o002)
    try:
        make_metadata_path_writable(duration_path)
        with duration_path.open("w", encoding="utf-8") as f:
            json.dump({"duration": durations}, f, ensure_ascii=False)
        try:
            duration_path.chmod(0o664)
        except OSError as exc:
            print(f"Warning: could not chmod {duration_path}: {exc}")

        make_metadata_path_writable(vocab_path)
        with vocab_path.open("w", encoding="utf-8") as f:
            for vocab in sorted(vocab_set):
                f.write(vocab + "\n")
        try:
            vocab_path.chmod(0o664)
        except OSError as exc:
            print(f"Warning: could not chmod {vocab_path}: {exc}")
    finally:
        os.umask(old_umask)

    print(f"Saved to: {save_dir}")
    print(f"Input jsonl: {args.input_jsonl}")
    print(f"Text field: {args.text_field}")
    print(f"Alignment root: {args.alignment_root}")
    print(f"Alignment audio root: {args.alignment_audio_root}")
    print(f"Require alignments: {args.require_alignments}")
    print(f"Kept samples: {kept}")
    print(f"Skipped samples: {skipped}")
    print(f"Total hours: {sum(durations) / 3600:.2f}")
    print(f"Vocab size: {len(vocab_set)}")


if __name__ == "__main__":
    main()
