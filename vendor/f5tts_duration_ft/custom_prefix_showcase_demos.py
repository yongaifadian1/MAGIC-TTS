import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torchaudio

from reproduce_saved_sample_case import HOP_LENGTH, TARGET_SAMPLE_RATE, tokenize_text

SCRIPT_DIR = Path(__file__).resolve().parent
SYNTH_SCRIPT = SCRIPT_DIR / "ttrack_edit_synthesize.py"
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_SYNTH_PYTHON = os.environ.get("MAGICTTS_SYNTH_PYTHON", sys.executable)
DEFAULT_CHECKPOINT = str((REPO_ROOT / "checkpoints" / "magictts_36k.pt").resolve())
DEFAULT_TEMPLATE_TRACK = (
    REPO_ROOT / "outputs" / "control_track_example" / "custom_track.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "controlled_demos"

DEFAULT_CONTENT_MS = 170.0
DEFAULT_PUNCT_MS = 50.0

PUNCTUATION_CHARS = set("，。？！；：、,.!?;:")

DEMO_SPECS = [
    {
        "slug": "01_navigation_turn",
        "title": "导航播报",
        "variants": [
            {
                "slug": "v1_baseline_eqdur",
                "text": "前方路口，左转。",
                "seed": 1101,
                "notes": "基线版：目标句子里的每个内容字都设为 170ms，标点默认 50ms。",
            },
            {
                "slug": "v2_pause_only_boundary",
                "text": "前方路口，左转。",
                "seed": 1101,
                "pause_ms": {"，": 260.0},
                "notes": "第一步先只加停顿：把“左转”前的逗号停顿拉到 260ms，突出动作切入前的边界，但“左转”这个动作词本身还不够醒。",
            },
            {
                "slug": "v3_pause_plus_char_turn",
                "text": "前方路口，左转。",
                "seed": 1101,
                "pause_ms": {"，": 260.0},
                "duration_ms": {"左": 225.0, "转": 225.0},
                "notes": "第二步继续加字 duration：保留逗号停顿，再把“左”“转”都拉到 225ms，让关键动作词自己也更稳、更突出。",
            },
        ],
    },
    {
        "slug": "02_kids_reading",
        "title": "儿童教学朗读",
        "variants": [
            {
                "slug": "v1_baseline_eqdur",
                "text": "请跟我读，苹果。",
                "seed": 1202,
                "notes": "基线版：目标句子里的每个内容字都设为 170ms，标点默认 50ms。",
            },
            {
                "slug": "v2_pause_only_syllable",
                "text": "请跟我读，苹果。",
                "seed": 1202,
                "pause_ms": {"，": 260.0},
                "notes": "第一步先只加停顿：在“请跟我读”后加 260ms 停顿，形成老师领读的节奏，但“苹”“果”两个字本身还是偏短。",
            },
            {
                "slug": "v3_pause_plus_char_syllable",
                "text": "请跟我读，苹果。",
                "seed": 1202,
                "pause_ms": {"，": 260.0},
                "duration_ms": {"苹": 225.0, "果": 225.0},
                "notes": "第二步继续加字 duration：保留“请跟我读”后的 260ms 停顿，再把“苹”“果”都拉到 225ms，让老师式带读更清楚。",
            },
        ],
    },
    {
        "slug": "03_accessibility_code",
        "title": "无障碍验证码播报",
        "variants": [
            {
                "slug": "v1_baseline_eqdur",
                "text": "验证码是三七九，二一八。",
                "seed": 1303,
                "notes": "基线版：目标句子里的每个内容字都设为 170ms，标点默认 50ms。",
            },
            {
                "slug": "v2_pause_only_grouped",
                "text": "验证码是三七九，二一八。",
                "seed": 1303,
                "pause_ms": {"，": 260.0},
                "notes": "第一步先只加停顿：在 3+3 分组边界处加 260ms 停顿，但每个数字本身还是略快。",
            },
            {
                "slug": "v3_pause_plus_char_digits",
                "text": "验证码是三七九，二一八。",
                "seed": 1303,
                "pause_ms": {"，": 260.0},
                "duration_ms": {
                    "三": 225.0,
                    "七": 225.0,
                    "九": 225.0,
                    "二": 225.0,
                    "一": 225.0,
                    "八": 225.0,
                },
                "notes": "第二步继续加字 duration：保留 3+3 分组处的 260ms 停顿，同时把六码都拉到 225ms，让无障碍播报更稳更清晰。",
            },
        ],
    },
    {
        "slug": "04_dialogue_emotion",
        "title": "有声书对白情绪",
        "variants": [
            {
                "slug": "v1_baseline_eqdur",
                "text": "你，真的要去吗？",
                "seed": 1404,
                "notes": "基线版：目标句子里的每个内容字都设为 170ms，标点默认 50ms。",
            },
            {
                "slug": "v2_pause_only_hesitation",
                "text": "你，真的要去吗？",
                "seed": 1404,
                "pause_ms": {"，": 260.0, "的": 260.0},
                "notes": "第一步先只加停顿：把“你”后的逗号和“的”后停顿都设为 260ms，能听出迟疑，但情绪还不够挂在词上。",
            },
            {
                "slug": "v3_pause_plus_char_emotion",
                "text": "你，真的要去吗？",
                "seed": 1404,
                "pause_ms": {"，": 260.0, "的": 260.0},
                "duration_ms": {"真": 225.0, "吗": 225.0},
                "notes": "第二步继续加字 duration：保留迟疑停顿，再把“真”“吗”拉到 225ms，让质疑和不舍更落在词上。",
            },
        ],
    },
    {
        "slug": "05_station_name_wushanzhan",
        "title": "站名播报",
        "variants": [
            {
                "slug": "v1_baseline_eqdur",
                "text": "下一站，五山站。",
                "seed": 1505,
                "notes": "基线版：目标句子里的每个内容字都设为 170ms，标点默认 50ms。",
            },
            {
                "slug": "v2_pause_only_station_boundary",
                "text": "下一站，五山站。",
                "seed": 1505,
                "pause_ms": {"，": 260.0},
                "notes": "第一步先只加停顿：把站名前的逗号拉到 260ms，让前缀和站名边界更清楚，但“五山站”三个字本身还是有点容易连在一起。",
            },
            {
                "slug": "v3_pause_plus_char_station_name",
                "text": "下一站，五山站。",
                "seed": 1505,
                "pause_ms": {"，": 260.0},
                "duration_ms": {"五": 235.0, "山": 235.0, "站#2": 235.0},
                "notes": "第二步继续加字 duration：保留站名前的停顿，同时把“五”“山”和站名里的第二个“站”都拉到 235ms，让站名更清楚、更像正式播报。",
            },
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--template-track", default=DEFAULT_TEMPLATE_TRACK)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--demo-slug", default=None)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--prompt-metadata", default=None)
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--prompt-seconds", type=float, default=3.0)
    parser.add_argument("--synth-python", default=DEFAULT_SYNTH_PYTHON)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--content-ms", type=float, default=DEFAULT_CONTENT_MS)
    parser.add_argument("--punct-ms", type=float, default=DEFAULT_PUNCT_MS)
    return parser.parse_args()


def frames_from_ms(value_ms: float) -> float:
    return value_ms * TARGET_SAMPLE_RATE / 1000.0 / HOP_LENGTH


def frames_from_seconds(value_sec: float) -> float:
    return value_sec * TARGET_SAMPLE_RATE / HOP_LENGTH


def load_template_track(track_path: Path) -> dict:
    with track_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_target_tokens_and_durations(target_text: str, content_frames: float, punct_frames: float):
    text_tokens = []
    token_durations = []
    char_to_token = []
    occurrence_counter = {}

    for char in target_text:
        occurrence_counter[char] = occurrence_counter.get(char, 0) + 1
        char_occurrence = occurrence_counter[char]
        char_tokens = tokenize_text(char)

        if not char_tokens:
            continue

        main_token_index = None
        for token in char_tokens:
            token_index = len(text_tokens)
            text_tokens.append(token)
            if token == " ":
                token_durations.append([0.0, 0.0])
                continue

            if token in PUNCTUATION_CHARS:
                token_durations.append([punct_frames, 0.0])
            else:
                token_durations.append([content_frames, 0.0])
            main_token_index = token_index

        if main_token_index is not None:
            char_to_token.append(
                {
                    "char": char,
                    "occurrence": char_occurrence,
                    "full_token_index": main_token_index,
                    "token": text_tokens[main_token_index],
                }
            )

    return text_tokens, token_durations, char_to_token


def build_prompt_prefix_tokens_and_durations(prompt_text: str, prompt_frames: float):
    text_tokens = []
    non_space_indices = []

    for char in prompt_text:
        for token in tokenize_text(char):
            token_index = len(text_tokens)
            text_tokens.append(token)
            if token != " ":
                non_space_indices.append(token_index)

    token_durations = [[0.0, 0.0] for _ in text_tokens]
    if non_space_indices:
        per_token_frames = prompt_frames / len(non_space_indices)
        for token_index in non_space_indices:
            token_durations[token_index][0] = per_token_frames

    return text_tokens, token_durations


def load_prompt_override(args):
    if args.prompt_metadata:
        metadata_path = Path(args.prompt_metadata).resolve()
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        prompt_audio = metadata.get("prompt_local_path") or metadata.get("prompt_audio_path")
        prompt_text = metadata.get("prompt_text", "")
        prompt_id = args.prompt_id or metadata.get("speaker") or metadata_path.parent.name
        return {
            "prompt_audio": prompt_audio,
            "prompt_text": prompt_text,
            "prompt_id": prompt_id,
            "prompt_metadata": str(metadata_path),
        }

    if args.prompt_audio:
        if not args.prompt_text:
            raise ValueError("--prompt-text is required when --prompt-audio is provided")
        return {
            "prompt_audio": args.prompt_audio,
            "prompt_text": args.prompt_text,
            "prompt_id": args.prompt_id or Path(args.prompt_audio).stem,
            "prompt_metadata": None,
        }

    return None


def build_prefix_context(args, template_track: dict) -> dict:
    prompt_override = load_prompt_override(args)
    if prompt_override is None:
        prefix_token_count = int(template_track["prefix_token_count"])
        return {
            "audio_path": template_track["audio_path"],
            "prefix_tokens": list(template_track["text_tokens"][:prefix_token_count]),
            "prefix_durations": [
                list(pair) for pair in template_track["edited_token_durations"][:prefix_token_count]
            ],
            "prefix_token_count": prefix_token_count,
            "source_sample_dir": template_track["source_sample_dir"],
            "source_prompt_frames": template_track["source_prompt_frames"],
            "source_total_frames": template_track["source_total_frames"],
            "prompt_text_source": template_track["prompt_text_source"],
            "prompt_id": "template_sample01",
            "prompt_metadata": None,
        }

    prompt_frames = frames_from_seconds(args.prompt_seconds)
    prefix_tokens, prefix_durations = build_prompt_prefix_tokens_and_durations(
        prompt_override["prompt_text"], prompt_frames
    )
    return {
        "audio_path": str(Path(prompt_override["prompt_audio"]).resolve()),
        "prefix_tokens": prefix_tokens,
        "prefix_durations": prefix_durations,
        "prefix_token_count": len(prefix_tokens),
        "source_sample_dir": str(Path(prompt_override["prompt_audio"]).resolve().parent),
        "source_prompt_frames": prompt_frames,
        "source_total_frames": prompt_frames,
        "prompt_text_source": prompt_override["prompt_text"],
        "prompt_id": prompt_override["prompt_id"],
        "prompt_metadata": prompt_override["prompt_metadata"],
    }


def make_char_lookup(char_to_token):
    lookup = {}
    for item in char_to_token:
        bare_key = item["char"]
        keyed = f"{item['char']}#{item['occurrence']}"
        lookup.setdefault(bare_key, []).append(item)
        lookup[keyed] = item
    return lookup


def resolve_edit_target(key: str, lookup: dict):
    if "#" in key:
        if key not in lookup:
            raise KeyError(f"cannot resolve edit target '{key}'")
        item = lookup[key]
        return item["full_token_index"], item

    candidates = lookup.get(key, [])
    if len(candidates) != 1:
        raise KeyError(f"edit target '{key}' is missing or ambiguous; use occurrence syntax like '{key}#1'")
    item = candidates[0]
    return item["full_token_index"], item


def build_custom_track(prefix_context: dict, variant: dict, content_frames: float, punct_frames: float) -> dict:
    prefix_token_count = int(prefix_context["prefix_token_count"])
    prefix_tokens = list(prefix_context["prefix_tokens"])
    prefix_durations = [list(pair) for pair in prefix_context["prefix_durations"]]

    target_tokens, target_durations, char_to_token = build_target_tokens_and_durations(
        variant["text"], content_frames, punct_frames
    )
    edited_target_durations = [list(pair) for pair in target_durations]
    lookup = make_char_lookup(char_to_token)

    duration_edits = []
    for key, duration_ms in variant.get("duration_ms", {}).items():
        token_index, item = resolve_edit_target(key, lookup)
        duration_frames = frames_from_ms(float(duration_ms))
        edited_target_durations[token_index][0] = duration_frames
        duration_edits.append(
            {
                "key": key,
                "char": item["char"],
                "occurrence": item["occurrence"],
                "full_token_index": prefix_token_count + token_index,
                "duration_ms": float(duration_ms),
                "duration_frames": duration_frames,
            }
        )

    pause_edits = []
    for key, pause_ms in variant.get("pause_ms", {}).items():
        token_index, item = resolve_edit_target(key, lookup)
        pause_frames = frames_from_ms(float(pause_ms))
        edited_target_durations[token_index][1] = pause_frames
        pause_edits.append(
            {
                "key": key,
                "char": item["char"],
                "occurrence": item["occurrence"],
                "full_token_index": prefix_token_count + token_index,
                "pause_ms": float(pause_ms),
                "pause_frames": pause_frames,
            }
        )

    full_text_tokens = prefix_tokens + target_tokens
    full_token_durations = prefix_durations + target_durations
    full_edited_token_durations = prefix_durations + edited_target_durations
    edited_total_frames = int(round(sum(sum(pair) for pair in full_edited_token_durations)))

    full_char_to_token = []
    for item in char_to_token:
        full_char_to_token.append(
            {
                **item,
                "full_token_index": prefix_token_count + item["full_token_index"],
            }
        )

    return {
        "audio_path": prefix_context["audio_path"],
        "raw_text": f"{prefix_context['prompt_text_source']} || {variant['text']}",
        "text_tokens": full_text_tokens,
        "token_durations": full_token_durations,
        "edited_token_durations": full_edited_token_durations,
        "original_total_frames": float(edited_total_frames),
        "edited_total_frames": edited_total_frames,
        "original_track_total_frames": float(sum(sum(pair) for pair in full_token_durations)),
        "edited_track_total_frames": float(sum(sum(pair) for pair in full_edited_token_durations)),
        "track_residual_frames": 0.0,
        "token_edits": duration_edits,
        "pause_edits": pause_edits,
        "source_sample_dir": prefix_context["source_sample_dir"],
        "source_prompt_frames": prefix_context["source_prompt_frames"],
        "source_total_frames": prefix_context["source_total_frames"],
        "prompt_text_source": prefix_context["prompt_text_source"],
        "prompt_id": prefix_context["prompt_id"],
        "prompt_metadata": prefix_context["prompt_metadata"],
        "custom_text": variant["text"],
        "target_only_text": variant["text"],
        "prefix_token_count": prefix_token_count,
        "target_token_offset": prefix_token_count,
        "char_to_token": full_char_to_token,
        "demo_slug": variant["slug"],
        "demo_notes": variant["notes"],
    }


def synthesize_variant(args, track_payload: dict, variant_dir: Path, seed: int):
    track_json = variant_dir / "custom_track.json"
    with track_json.open("w", encoding="utf-8") as f:
        json.dump(track_payload, f, ensure_ascii=False, indent=2)

    synth_cmd = [
        args.synth_python,
        str(SYNTH_SCRIPT),
        "--audio",
        track_payload["audio_path"],
        "--track-json",
        str(track_json),
        "--output-dir",
        str(variant_dir),
        "--prompt-frames",
        str(track_payload["source_prompt_frames"]),
        "--prompt-seconds",
        str(args.prompt_seconds),
        "--steps",
        str(args.steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--sway-sampling-coef",
        str(args.sway_sampling_coef),
        "--seed",
        str(seed),
        "--checkpoint",
        args.checkpoint,
        "--preserve-cond-audio-in-output",
    ]
    subprocess.run(synth_cmd, check=True, cwd=SCRIPT_DIR.parents[1])

    prompt_samples = int(round(float(track_payload["source_prompt_frames"]) * HOP_LENGTH))
    gen_audio, sample_rate = torchaudio.load(str(variant_dir / "gen.wav"))
    cropped_audio = gen_audio[:, min(prompt_samples, gen_audio.shape[-1]) :]
    torchaudio.save(str(variant_dir / "gen_target_only.wav"), cropped_audio, sample_rate)

    crop_metadata = {
        "prompt_samples_removed": min(prompt_samples, gen_audio.shape[-1]),
        "sample_rate": sample_rate,
        "target_only_seconds": cropped_audio.shape[-1] / sample_rate,
    }
    with (variant_dir / "crop_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(crop_metadata, f, ensure_ascii=False, indent=2)


def describe_variant_controls(variant: dict) -> str:
    parts = [f"text={variant['text']}", f"seed={variant['seed']}"]
    if variant.get("duration_ms"):
        duration_desc = ", ".join(f"{k}={v:.0f}ms" for k, v in variant["duration_ms"].items())
        parts.append(f"字时长控制: {duration_desc}")
    else:
        parts.append("字时长控制: 无，沿用基线 170ms/字")
    if variant.get("pause_ms"):
        pause_desc = ", ".join(f"{k}={v:.0f}ms" for k, v in variant["pause_ms"].items())
        parts.append(f"停顿控制: {pause_desc}")
    else:
        parts.append("停顿控制: 无额外停顿")
    return "；".join(parts)


def select_demos(demo_slug: str | None, max_demos: int | None):
    if demo_slug:
        requested = [item.strip() for item in demo_slug.split(",") if item.strip()]
        demos = []
        seen = set()
        available = {demo["slug"]: demo for demo in DEMO_SPECS}
        for slug in requested:
            if slug not in available:
                raise ValueError(f"unknown demo slug: {slug}")
            if slug in seen:
                continue
            demos.append(available[slug])
            seen.add(slug)
        if not demos:
            raise ValueError(f"unknown demo slug: {demo_slug}")
    else:
        demos = list(DEMO_SPECS)

    if max_demos is not None:
        if max_demos <= 0:
            raise ValueError("--max-demos must be positive")
        demos = demos[:max_demos]

    return demos


def build_root_summary(output_root: Path, args, demos: list[dict]) -> dict:
    summary = {
        "output_root": str(output_root),
        "template_track": str(Path(args.template_track).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_id": args.prompt_id,
        "prompt_metadata": args.prompt_metadata,
        "prompt_audio": args.prompt_audio,
        "prompt_text": args.prompt_text,
        "prompt_seconds": args.prompt_seconds,
        "content_ms": args.content_ms,
        "punct_ms": args.punct_ms,
        "demos": [],
    }
    for demo in demos:
        demo_summary = {"slug": demo["slug"], "title": demo["title"], "variants": []}
        for variant in demo["variants"]:
            variant_dir = output_root / demo["slug"] / variant["slug"]
            demo_summary["variants"].append(
                {
                    "slug": variant["slug"],
                    "text": variant["text"],
                    "notes": variant["notes"],
                    "dir": str(variant_dir),
                    "gen_target_only_wav": str(variant_dir / "gen_target_only.wav"),
                }
            )
        summary["demos"].append(demo_summary)
    return summary


def write_root_artifacts(output_root: Path, args, demos: list[dict]) -> None:
    summary = build_root_summary(output_root, args, demos)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_root / "README.txt").open("w", encoding="utf-8") as f:
        f.write("T-Track 四个应用场景：从停顿控制到字时长控制 demo\n")
        f.write("\n")
        f.write(f"输出根目录: {output_root}\n")
        f.write(f"使用 checkpoint: {Path(args.checkpoint).resolve()}\n")
        f.write(f"使用模板 track: {Path(args.template_track).resolve()}\n")
        if args.prompt_metadata or args.prompt_audio:
            f.write(f"使用 prompt: {args.prompt_id or args.prompt_metadata or args.prompt_audio}\n")
            f.write(f"prompt 截取长度: {args.prompt_seconds:.2f}s\n")
        f.write(
            f"统一基线设置: 内容字默认 {args.content_ms:.0f}ms/字，标点默认 {args.punct_ms:.0f}ms，"
            f"steps={args.steps}，cfg={args.cfg_strength}，sway={args.sway_sampling_coef}\n"
        )
        f.write("故事线: 先只加停顿 duration，验证边界和节奏是否更清楚；如果效果还不够，再继续加关键字的 content duration。\n")
        f.write("说明: 所有 gen_target_only.wav 都已经裁掉 prompt，只保留目标句子的生成部分。\n")
        f.write("\n")
        for demo in demos:
            f.write(f"{demo['slug']} - {demo['title']}\n")
            for variant in demo["variants"]:
                variant_dir = output_root / demo["slug"] / variant["slug"]
                f.write(f"  {variant['slug']}: {variant['text']}\n")
                f.write(f"  设定: {describe_variant_controls(variant)}\n")
                f.write(f"  说明: {variant['notes']}\n")
                f.write(f"  目录: {variant_dir}\n")
            f.write("\n")

    with (output_root / "README.md").open("w", encoding="utf-8") as f:
        f.write("# T-Track 四个应用场景：从停顿控制到字时长控制 Demo\n\n")
        f.write("## 总设置\n")
        f.write(f"- 输出根目录: `{output_root}`\n")
        f.write(f"- 使用 checkpoint: `{Path(args.checkpoint).resolve()}`\n")
        f.write(f"- 使用模板 track: `{Path(args.template_track).resolve()}`\n")
        if args.prompt_metadata or args.prompt_audio:
            f.write(f"- 使用 prompt: `{args.prompt_id or args.prompt_metadata or args.prompt_audio}`\n")
            f.write(f"- prompt 截取长度: `{args.prompt_seconds:.2f}s`\n")
        f.write(
            f"- 统一基线设置: 内容字默认 `{args.content_ms:.0f}ms/字`，标点默认 `{args.punct_ms:.0f}ms`，"
            f"`steps={args.steps}`，`cfg={args.cfg_strength}`，`sway={args.sway_sampling_coef}`\n"
        )
        f.write("- 故事线: 先只加停顿 duration，看看边界和节奏是否更清楚；如果效果还不够，再继续加关键字的 content duration。\n")
        f.write("- 所有 `gen_target_only.wav` 都已经裁掉 prompt，只保留目标句子的生成部分。\n\n")
        for demo in demos:
            f.write(f"## {demo['slug']} {demo['title']}\n")
            for variant in demo["variants"]:
                variant_dir = output_root / demo["slug"] / variant["slug"]
                rel_audio = f"./{demo['slug']}/{variant['slug']}/gen_target_only.wav"
                rel_track = f"./{demo['slug']}/{variant['slug']}/custom_track.json"
                f.write(f"- `{variant['slug']}`: `{variant['text']}`\n")
                f.write(f"- 设定: {describe_variant_controls(variant)}\n")
                f.write(f"- 说明: {variant['notes']}\n")
                f.write(f"- 音频: [{rel_audio}]({rel_audio})\n")
                f.write(f"- Track: [{rel_track}]({rel_track})\n")
                f.write(f"- 目录: `{variant_dir}`\n")
            f.write("\n")


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_demos = select_demos(args.demo_slug, args.max_demos)

    template_track = load_template_track(Path(args.template_track))
    prefix_context = build_prefix_context(args, template_track)
    content_frames = frames_from_ms(args.content_ms)
    punct_frames = frames_from_ms(args.punct_ms)

    for demo in selected_demos:
        demo_dir = output_root / demo["slug"]
        demo_dir.mkdir(parents=True, exist_ok=True)

        for variant in demo["variants"]:
            variant_dir = demo_dir / variant["slug"]
            variant_dir.mkdir(parents=True, exist_ok=True)
            with (variant_dir / "request.json").open("w", encoding="utf-8") as f:
                json.dump(variant, f, ensure_ascii=False, indent=2)

            track_payload = build_custom_track(prefix_context, variant, content_frames, punct_frames)
            synthesize_variant(args, track_payload, variant_dir, variant["seed"])

    write_root_artifacts(output_root, args, selected_demos)

if __name__ == "__main__":
    main()
