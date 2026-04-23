#!/usr/bin/env python3
"""Run Timing Control Accuracy evaluation on random b150 samples."""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import jieba
import numpy as np
import torchaudio
from pypinyin import Style, lazy_pinyin


TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    REPO_ROOT / "outputs" / "ttrack_b150_mfa_duration_dataset_v1_fast" / "manifests" / "b150_filtered_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "timing_control_accuracy_b150_seed2026"
DEFAULT_SYNTH_PYTHON = os.environ.get("MAGICTTS_SYNTH_PYTHON", sys.executable)
SYNTH_SCRIPT = SCRIPT_DIR / "ttrack_edit_synthesize.py"
MFA_BIN = os.environ.get("MAGICTTS_MFA_BIN")
MFA_ROOT = Path(os.environ.get("MAGICTTS_MFA_ROOT", str((REPO_ROOT / ".mfa_root").resolve())))
PAUSE_F1_THRESHOLDS_MS = (50.0, 100.0)

LANGUAGE_MODELS = {
    "en": {
        "dictionary": MFA_ROOT / "pretrained_models" / "dictionary" / "english_mfa.dict",
        "acoustic": MFA_ROOT / "pretrained_models" / "acoustic" / "english_mfa.zip",
    },
    "zh": {
        "dictionary": MFA_ROOT / "pretrained_models" / "dictionary" / "mandarin_mfa.dict",
        "acoustic": MFA_ROOT / "pretrained_models" / "acoustic" / "mandarin_mfa.zip",
    },
}


class SampleSpec(NamedTuple):
    sample_id: str
    sample_index: int
    record: dict
    duration_sec: float
    prompt_sec: float
    prompt_samples: int
    prompt_frames: float
    boundary_word_index: int
    prefix_token_count: int
    total_token_count: int
    target_text: str
    target_word_count: int
    language: str
    sample_dir: Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-duration-sec", type=float, default=3.0)
    parser.add_argument("--max-duration-sec", type=float, default=10.0)
    parser.add_argument("--prompt-frac", type=float, default=0.3)
    parser.add_argument("--min-prompt-sec", type=float, default=0.5)
    parser.add_argument("--min-target-sec", type=float, default=0.5)
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--sample-seed-base", type=int, default=910000)
    parser.add_argument("--mfa-jobs", type=int, default=8)
    parser.add_argument("--keep-mfa-workdir", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-mfa", action="store_true")
    parser.add_argument("--only-eval-existing", action="store_true")
    parser.add_argument("--output-prefix", default="gen")
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--omit-token-durations", action="store_true")
    parser.add_argument(
        "--use-full-text-with-prompt-duration",
        action="store_true",
        help="Use full prompt+target text and full token-duration track at inference time.",
    )
    return parser.parse_args()


def resolve_mfa_bin() -> str:
    if MFA_BIN:
        return MFA_BIN
    path_mfa = shutil.which("mfa")
    if path_mfa:
        return path_mfa
    raise FileNotFoundError(
        "mfa not found in PATH; install montreal-forced-aligner or set MAGICTTS_MFA_BIN"
    )


def tagged_name(base_name, result_tag):
    if not result_tag:
        return base_name
    return "%s_%s" % (result_tag, base_name)


def load_manifest(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


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


def tokenize_word(word_text):
    if contains_cjk(word_text):
        return convert_char_to_pinyin([word_text], polyphone=True)[0]
    return list(word_text)


def normalize_language(language: str) -> str:
    language = str(language or "").lower()
    if language.startswith("zh"):
        return "zh"
    return "en"


def load_audio_duration_sec(audio_path: Path) -> float:
    info = torchaudio.info(str(audio_path))
    if info.sample_rate <= 0:
        raise ValueError(f"invalid sample rate for {audio_path}")
    return float(info.num_frames) / float(info.sample_rate)


def load_audio_mono(audio_path: Path):
    audio, sample_rate = torchaudio.load(str(audio_path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio


def load_sidecar_words(sidecar_path):
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


def choose_boundary_word_index(
    words,
    total_duration_sec,
    prompt_frac,
    min_prompt_sec,
    min_target_sec,
):
    if len(words) < 2:
        return None
    target_prompt_sec = total_duration_sec * prompt_frac
    best_index = None
    best_score = None
    for boundary_word_index in range(1, len(words)):
        prompt_sec = float(words[boundary_word_index]["start"])
        target_sec = total_duration_sec - prompt_sec
        if prompt_sec < min_prompt_sec or target_sec < min_target_sec:
            continue
        score = abs(prompt_sec - target_prompt_sec)
        if best_score is None or score < best_score:
            best_score = score
            best_index = boundary_word_index
    return best_index


def join_words_for_transcript(words, language):
    word_texts = [str(item["word"]).strip() for item in words if str(item["word"]).strip()]
    if normalize_language(language) == "zh":
        return "".join(word_texts)
    return " ".join(word_texts)


def build_token_track_from_words(words):
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
            tokens.append(token)
            if local_idx in content_positions:
                content_value = per_token_content
                pause_value = pause_after_frames if local_idx == last_content_idx else 0.0
            else:
                content_value = 0.0
                pause_value = 0.0
            token_durations.append([float(content_value), float(pause_value)])
    return tokens, token_durations


def add_residual_to_last_pause(token_durations, residual_frames):
    if not token_durations:
        return
    if abs(residual_frames) < 1e-6:
        return
    updated_pause = float(token_durations[-1][1]) + float(residual_frames)
    if updated_pause < 0.0 and abs(updated_pause) <= 1e-4:
        updated_pause = 0.0
    if updated_pause < -1e-6:
        raise ValueError(
            f"last-token pause becomes negative after residual correction: "
            f"pause={token_durations[-1][1]}, residual={residual_frames}"
        )
    token_durations[-1][1] = max(0.0, updated_pause)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sample_spec(record, sample_index, output_dir, args):
    audio_path = Path(record["audio_path"]).resolve()
    if not audio_path.exists():
        return None
    mfa_sidecar_path = Path(record["mfa_words_json"]).resolve()
    if not mfa_sidecar_path.exists():
        return None

    duration_sec = load_audio_duration_sec(audio_path)
    if not (args.min_duration_sec < duration_sec < args.max_duration_sec):
        return None

    words = load_sidecar_words(mfa_sidecar_path)
    boundary_word_index = choose_boundary_word_index(
        words=words,
        total_duration_sec=duration_sec,
        prompt_frac=args.prompt_frac,
        min_prompt_sec=args.min_prompt_sec,
        min_target_sec=args.min_target_sec,
    )
    if boundary_word_index is None:
        return None

    full_tokens, _ = build_token_track_from_words(words)
    prefix_tokens, _ = build_token_track_from_words(words[:boundary_word_index])
    suffix_words = words[boundary_word_index:]
    suffix_tokens, _ = build_token_track_from_words(suffix_words)
    if not full_tokens or not suffix_tokens:
        return None

    prompt_sec = float(words[boundary_word_index]["start"])
    prompt_samples = int(round(prompt_sec * TARGET_SAMPLE_RATE))
    prompt_frames = prompt_samples / float(HOP_LENGTH)
    sample_id = f"sample_{sample_index:03d}_{record['utt_id']}"
    sample_dir = output_dir / sample_id
    language = normalize_language(record.get("language", "en"))
    target_text = join_words_for_transcript(suffix_words, language=language)
    if not target_text.strip():
        return None

    return SampleSpec(
        sample_id=sample_id,
        sample_index=sample_index,
        record=record,
        duration_sec=duration_sec,
        prompt_sec=prompt_sec,
        prompt_samples=prompt_samples,
        prompt_frames=prompt_frames,
        boundary_word_index=boundary_word_index,
        prefix_token_count=len(prefix_tokens),
        total_token_count=len(full_tokens),
        target_text=target_text,
        target_word_count=len(suffix_words),
        language=language,
        sample_dir=sample_dir,
    )


def select_samples(records, output_dir, args):
    rng = random.Random(args.seed)
    records = list(records)
    rng.shuffle(records)
    selected = []
    scanned = 0
    for record in records:
        scanned += 1
        spec = build_sample_spec(record=record, sample_index=len(selected), output_dir=output_dir, args=args)
        if spec is None:
            continue
        selected.append(spec)
        if len(selected) >= args.num_samples:
            break
    if len(selected) < args.num_samples:
        raise SystemExit(
            f"only selected {len(selected)} usable samples after scanning {scanned} manifest records; "
            f"wanted {args.num_samples}"
        )
    return selected


def prepare_sample_artifacts(spec, args):
    spec.sample_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(spec.record["audio_path"]).resolve()
    mfa_sidecar_path = Path(spec.record["mfa_words_json"]).resolve()
    words = load_sidecar_words(mfa_sidecar_path)
    full_tokens, full_token_durations = build_token_track_from_words(words)
    suffix_words = words[spec.boundary_word_index :]
    suffix_tokens, suffix_token_durations = build_token_track_from_words(suffix_words)

    if len(full_tokens) != spec.total_token_count:
        raise ValueError(f"{spec.sample_id}: full token count mismatch")
    if len(full_tokens[: spec.prefix_token_count]) != spec.prefix_token_count:
        raise ValueError(f"{spec.sample_id}: prefix token count mismatch")

    total_frames = float(spec.duration_sec * TARGET_SAMPLE_RATE / HOP_LENGTH)
    target_only_frames = total_frames - float(spec.prompt_frames)
    target_track_frames = float(sum(sum(pair) for pair in suffix_token_durations))
    target_residual_frames = target_only_frames - target_track_frames
    add_residual_to_last_pause(suffix_token_durations, target_residual_frames)
    add_residual_to_last_pause(full_token_durations[spec.prefix_token_count :], target_residual_frames)

    control_track = {
        "audio_path": str(audio_path),
        "raw_text": spec.record["text"],
        "text_tokens": full_tokens,
        "token_durations": full_token_durations,
        "edited_token_durations": full_token_durations,
        "original_total_frames": total_frames,
        "edited_total_frames": int(round(total_frames)),
        "prefix_token_count": spec.prefix_token_count,
        "source_prompt_frames": spec.prompt_frames,
        "source_total_frames": total_frames,
        "prompt_frac": args.prompt_frac,
        "target_text": spec.target_text,
        "target_word_count": spec.target_word_count,
        "reproduction_mode": "timing_control_accuracy_prompt_completion",
    }
    gt_track = {
        "audio_path": str(audio_path),
        "mfa_words_json": str(mfa_sidecar_path),
        "target_text": spec.target_text,
        "text_tokens": suffix_tokens,
        "token_durations": suffix_token_durations,
        "edited_token_durations": suffix_token_durations,
        "edited_total_frames": int(round(target_only_frames)),
        "target_only_frames": target_only_frames,
        "language": spec.language,
        "prompt_sec": spec.prompt_sec,
        "prompt_frames": spec.prompt_frames,
        "boundary_word_index": spec.boundary_word_index,
    }
    sample_meta = {
        "sample_id": spec.sample_id,
        "sample_index": spec.sample_index,
        "utt_id": spec.record["utt_id"],
        "audio_path": str(audio_path),
        "mfa_words_json": str(mfa_sidecar_path),
        "text": spec.record["text"],
        "language": spec.language,
        "duration_sec": spec.duration_sec,
        "prompt_frac": args.prompt_frac,
        "prompt_sec": spec.prompt_sec,
        "prompt_samples": spec.prompt_samples,
        "prompt_frames": spec.prompt_frames,
        "boundary_word_index": spec.boundary_word_index,
        "prefix_token_count": spec.prefix_token_count,
        "total_token_count": spec.total_token_count,
        "target_word_count": spec.target_word_count,
        "target_text": spec.target_text,
    }

    write_json(spec.sample_dir / "sample_metadata.json", sample_meta)
    write_json(spec.sample_dir / "control_track.json", control_track)
    write_json(spec.sample_dir / "gt_target_only_track.json", gt_track)
    (spec.sample_dir / "target_target_only.txt").write_text(spec.target_text + "\n", encoding="utf-8")

    target_audio_path = spec.sample_dir / "target_target_only.wav"
    if not target_audio_path.exists() or not args.skip_existing:
        audio = load_audio_mono(audio_path)
        target_audio = audio[:, spec.prompt_samples :]
        torchaudio.save(str(target_audio_path), target_audio, TARGET_SAMPLE_RATE)

    return sample_meta


def run_command(cmd, env=None, cwd=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def synthesize_sample(spec, args):
    gen_audio_path = spec.sample_dir / ("%s_target_only.wav" % args.output_prefix)
    if args.skip_existing and gen_audio_path.exists():
        return

    cmd = [
        args.synth_python,
        str(SYNTH_SCRIPT),
        "--audio",
        str(Path(spec.record["audio_path"]).resolve()),
        "--track-json",
        str(spec.sample_dir / "control_track.json"),
        "--output-dir",
        str(spec.sample_dir),
        "--output-prefix",
        str(args.output_prefix),
        "--prompt-frames",
        str(spec.prompt_frames),
        "--steps",
        str(args.steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--sway-sampling-coef",
        str(args.sway_sampling_coef),
        "--seed",
        str(args.sample_seed_base + spec.sample_index),
        "--preserve-cond-audio-in-output",
    ]
    if not args.use_full_text_with_prompt_duration:
        cmd.append("--use-target-only-text")
    if args.omit_token_durations:
        cmd.append("--omit-token-durations")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    run_command(cmd, cwd=REPO_ROOT)


def mfa_env(runtime_root):
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(runtime_root)
    env["PKUSEG_HOME"] = str(MFA_ROOT / "pkuseg")
    return env


def run_mfa_for_language(samples, output_dir, language, num_jobs, keep_workdir, args):
    if not samples:
        return {
            "language": language,
            "num_requested": 0,
            "num_submitted": 0,
            "num_aligned": 0,
            "missing_generation": [],
            "missing_mfa_output": [],
        }
    if language not in LANGUAGE_MODELS:
        raise SystemExit(f"unsupported language for MFA: {language}")

    work_base = output_dir / "_mfa_work"
    work_base.mkdir(parents=True, exist_ok=True)
    shard_dir = work_base / language
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    corpus_dir = shard_dir / "corpus"
    aligned_dir = shard_dir / "aligned"
    temp_dir = shard_dir / "temp"
    runtime_root = shard_dir / "runtime"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    submitted_specs = []
    missing_generation = []

    try:
        for spec in samples:
            wav_path = spec.sample_dir / ("%s_target_only.wav" % args.output_prefix)
            if not wav_path.exists():
                missing_generation.append(
                    {
                        "sample_id": spec.sample_id,
                        "language": spec.language,
                        "reason": "missing_generated_audio",
                        "path": str(wav_path),
                    }
                )
                continue
            link_path = corpus_dir / f"{spec.sample_id}.wav"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(wav_path)
            (corpus_dir / f"{spec.sample_id}.lab").write_text(spec.target_text + "\n", encoding="utf-8")
            submitted_specs.append(spec)

        if not submitted_specs:
            summary = {
                "language": language,
                "num_requested": len(samples),
                "num_submitted": 0,
                "num_aligned": 0,
                "missing_generation": missing_generation,
                "missing_mfa_output": [],
            }
            write_json(output_dir / tagged_name("mfa_summary_%s.json" % language, args.result_tag), summary)
            return summary

        model_info = LANGUAGE_MODELS[language]
        cmd = [
            resolve_mfa_bin(),
            "align",
            str(corpus_dir),
            str(model_info["dictionary"]),
            str(model_info["acoustic"]),
            str(aligned_dir),
            "--single_speaker",
            "--output_format",
            "json",
            "--clean",
            "-j",
            str(num_jobs),
            "-t",
            str(temp_dir),
        ]
        # Keep MFA's working directory on a stable path. Using the shard directory as cwd
        # can break its multiprocessing export step if that directory is removed or invalidated.
        run_command(cmd, env=mfa_env(runtime_root), cwd=REPO_ROOT)

        missing_mfa_output = []
        aligned_count = 0
        for spec in submitted_specs:
            aligned_json_path = aligned_dir / f"{spec.sample_id}.json"
            if not aligned_json_path.exists():
                missing_mfa_output.append(
                    {
                        "sample_id": spec.sample_id,
                        "language": spec.language,
                        "reason": "missing_mfa_output",
                        "path": str(aligned_json_path),
                    }
                )
                continue
            with aligned_json_path.open("r", encoding="utf-8") as f:
                aligned_obj = json.load(f)
            write_json(
                spec.sample_dir / tagged_name("%s_target_only_mfa_raw.json" % args.output_prefix, args.result_tag),
                aligned_obj,
            )
            aligned_count += 1

        summary = {
            "language": language,
            "num_requested": len(samples),
            "num_submitted": len(submitted_specs),
            "num_aligned": aligned_count,
            "missing_generation": missing_generation,
            "missing_mfa_output": missing_mfa_output,
        }
        write_json(output_dir / tagged_name("mfa_summary_%s.json" % language, args.result_tag), summary)
        return summary
    finally:
        if not keep_workdir and shard_dir.exists():
            shutil.rmtree(shard_dir, ignore_errors=True)


def load_words_from_mfa_output(aligned_obj):
    entries = aligned_obj.get("tiers", {}).get("words", {}).get("entries", [])
    words = []
    for start, end, text in entries:
        word = str(text).strip()
        if not word:
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        words.append({"word": word, "start": start, "end": end})
    return words


def safe_pearson(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def pause_f1(target_pause, pred_pause, threshold_ms):
    threshold_frames = threshold_ms * TARGET_SAMPLE_RATE / 1000.0 / HOP_LENGTH
    tp = fp = fn = 0
    for gt_value, pred_value in zip(target_pause, pred_pause):
        gt_flag = gt_value >= threshold_frames
        pred_flag = pred_value >= threshold_frames
        if gt_flag and pred_flag:
            tp += 1
        elif (not gt_flag) and pred_flag:
            fp += 1
        elif gt_flag and (not pred_flag):
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold_ms": threshold_ms,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def ms_from_frames(value):
    return value * HOP_LENGTH * 1000.0 / TARGET_SAMPLE_RATE


def evaluate_sample(spec, args):
    gt_track_path = spec.sample_dir / "gt_target_only_track.json"
    gen_mfa_path = spec.sample_dir / tagged_name("%s_target_only_mfa_raw.json" % args.output_prefix, args.result_tag)
    if not gt_track_path.exists():
        raise FileNotFoundError(f"missing gt track for {spec.sample_id}")
    if not gen_mfa_path.exists():
        raise FileNotFoundError(f"missing MFA alignment for {spec.sample_id}")

    with gt_track_path.open("r", encoding="utf-8") as f:
        gt_track = json.load(f)
    with gen_mfa_path.open("r", encoding="utf-8") as f:
        gen_mfa = json.load(f)

    gen_words = load_words_from_mfa_output(gen_mfa)
    gen_tokens, gen_token_durations = build_token_track_from_words(gen_words)
    gen_track = {
        "target_text": spec.target_text,
        "text_tokens": gen_tokens,
        "token_durations": gen_token_durations,
        "language": spec.language,
        "num_words": len(gen_words),
    }
    write_json(
        spec.sample_dir / tagged_name("%s_target_only_track.json" % args.output_prefix, args.result_tag),
        gen_track,
    )

    gt_tokens = gt_track["text_tokens"]
    gt_token_durations = gt_track["token_durations"]
    if gt_tokens != gen_tokens:
        raise ValueError(f"{spec.sample_id}: token mismatch between GT and generated MFA track")
    if len(gt_token_durations) != len(gen_token_durations):
        raise ValueError(f"{spec.sample_id}: token duration length mismatch")

    gt_content = [float(pair[0]) for pair in gt_token_durations]
    gt_pause = [float(pair[1]) for pair in gt_token_durations]
    pred_content = [float(pair[0]) for pair in gen_token_durations]
    pred_pause = [float(pair[1]) for pair in gen_token_durations]

    content_errors = [abs(a - b) for a, b in zip(pred_content, gt_content)]
    pause_errors = [abs(a - b) for a, b in zip(pred_pause, gt_pause)]
    content_sq_errors = [(a - b) ** 2 for a, b in zip(pred_content, gt_content)]
    pause_sq_errors = [(a - b) ** 2 for a, b in zip(pred_pause, gt_pause)]

    sample_metrics = {
        "sample_id": spec.sample_id,
        "utt_id": spec.record["utt_id"],
        "language": spec.language,
        "num_tokens": len(gt_tokens),
        "num_words": spec.target_word_count,
        "content_mae_frames": float(np.mean(content_errors)),
        "content_mae_ms": ms_from_frames(float(np.mean(content_errors))),
        "pause_mae_frames": float(np.mean(pause_errors)),
        "pause_mae_ms": ms_from_frames(float(np.mean(pause_errors))),
        "content_rmse_frames": math.sqrt(float(np.mean(content_sq_errors))),
        "content_rmse_ms": ms_from_frames(math.sqrt(float(np.mean(content_sq_errors)))),
        "pause_rmse_frames": math.sqrt(float(np.mean(pause_sq_errors))),
        "pause_rmse_ms": ms_from_frames(math.sqrt(float(np.mean(pause_sq_errors)))),
        "content_pearson": safe_pearson(gt_content, pred_content),
        "pause_pearson": safe_pearson(gt_pause, pred_pause),
        "gt_nonzero_pause_tokens": int(sum(value > 0 for value in gt_pause)),
        "pred_nonzero_pause_tokens": int(sum(value > 0 for value in pred_pause)),
    }
    for threshold_ms in PAUSE_F1_THRESHOLDS_MS:
        pause_stats = pause_f1(gt_pause, pred_pause, threshold_ms=threshold_ms)
        sample_metrics[f"pause_f1_{int(threshold_ms)}ms"] = pause_stats["f1"]
        sample_metrics[f"pause_precision_{int(threshold_ms)}ms"] = pause_stats["precision"]
        sample_metrics[f"pause_recall_{int(threshold_ms)}ms"] = pause_stats["recall"]

    write_json(spec.sample_dir / tagged_name("sample_metrics.json", args.result_tag), sample_metrics)
    return sample_metrics


def aggregate_metrics(sample_metrics, samples, args):
    if not sample_metrics:
        raise SystemExit("no sample metrics available for aggregation")

    def mean_key(key):
        values = [item[key] for item in sample_metrics if item.get(key) is not None]
        if not values:
            return None
        return float(np.mean(values))

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "seed": args.seed,
        "num_requested_samples": args.num_samples,
        "num_completed_samples": len(sample_metrics),
        "num_total_selected_samples": len(samples),
        "duration_range_sec": [args.min_duration_sec, args.max_duration_sec],
        "prompt_frac": args.prompt_frac,
        "min_prompt_sec": args.min_prompt_sec,
        "min_target_sec": args.min_target_sec,
        "checkpoint": args.checkpoint,
        "content_mae_frames": mean_key("content_mae_frames"),
        "content_mae_ms": mean_key("content_mae_ms"),
        "pause_mae_frames": mean_key("pause_mae_frames"),
        "pause_mae_ms": mean_key("pause_mae_ms"),
        "content_rmse_frames": mean_key("content_rmse_frames"),
        "content_rmse_ms": mean_key("content_rmse_ms"),
        "pause_rmse_frames": mean_key("pause_rmse_frames"),
        "pause_rmse_ms": mean_key("pause_rmse_ms"),
        "content_pearson": mean_key("content_pearson"),
        "pause_pearson": mean_key("pause_pearson"),
        "pause_f1_50ms": mean_key("pause_f1_50ms"),
        "pause_f1_100ms": mean_key("pause_f1_100ms"),
        "languages": sorted({spec.language for spec in samples}),
    }
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    records = load_manifest(manifest_path)
    samples = select_samples(records=records, output_dir=output_dir, args=args)

    selected_rows = []
    for spec in samples:
        selected_rows.append(prepare_sample_artifacts(spec=spec, args=args))
    write_jsonl(output_dir / "selected_samples.jsonl", selected_rows)

    if not args.only_eval_existing and not args.skip_generation:
        for spec in samples:
            synthesize_sample(spec=spec, args=args)

    mfa_summaries = []
    if not args.skip_mfa:
        for language in sorted({spec.language for spec in samples}):
            language_samples = [spec for spec in samples if spec.language == language]
            mfa_summaries.append(run_mfa_for_language(
                samples=language_samples,
                output_dir=output_dir,
                language=language,
                num_jobs=args.mfa_jobs,
                keep_workdir=args.keep_mfa_workdir,
                args=args,
            ))
    else:
        print(json.dumps({"stage": "prepared_or_generated_only", "num_samples": len(samples)}, ensure_ascii=False, indent=2))
        return

    write_json(output_dir / tagged_name("mfa_summaries.json", args.result_tag), {"summaries": mfa_summaries})

    sample_metrics = []
    failed_samples = []
    for spec in samples:
        try:
            sample_metrics.append(evaluate_sample(spec, args))
        except Exception as exc:
            failed_samples.append(
                {
                    "sample_id": spec.sample_id,
                    "utt_id": spec.record["utt_id"],
                    "language": spec.language,
                    "reason": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
    write_jsonl(output_dir / tagged_name("sample_metrics.jsonl", args.result_tag), sample_metrics)
    write_jsonl(output_dir / tagged_name("failed_samples.jsonl", args.result_tag), failed_samples)
    summary = aggregate_metrics(sample_metrics=sample_metrics, samples=samples, args=args)
    summary["num_failed_samples"] = len(failed_samples)
    write_json(output_dir / tagged_name("metrics_summary.json", args.result_tag), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
