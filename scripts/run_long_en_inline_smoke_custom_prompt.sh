#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/checkpoints/magictts_36k.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${1:-$REPO_ROOT/outputs/long_en_inline_smoke_custom_prompt}"
PROMPT_AUDIO="${PROMPT_AUDIO:?PROMPT_AUDIO is required}"
PROMPT_TEXT_FILE="${PROMPT_TEXT_FILE:?PROMPT_TEXT_FILE is required}"
PROMPT_WORDS_JSON="${PROMPT_WORDS_JSON:-}"
export MAGICTTS_PRETRAINED_ROOT="${MAGICTTS_PRETRAINED_ROOT:-$REPO_ROOT/pretrained}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"

mkdir -p "$OUT_ROOT"

PROMPT_DIR="$OUT_ROOT/prompt_prefix"
SPONT_DIR="$OUT_ROOT/spontaneous_long_en"
CTRL_DIR="$OUT_ROOT/controlled_inline_long_en"
PROMPT_TRACK_JSON="$PROMPT_DIR/prompt_track.json"

PLAIN_TEXT="After the meeting, please review the budget, call Michael before Friday afternoon, and send the final delivery schedule to the client."
INLINE_TEXT="After the meeting[220], please review the budget[220], call Michael{360} before Friday afternoon[260], and send the final delivery{320} schedule{320} to the client."

echo "[smoke] repo_root=$REPO_ROOT"
echo "[smoke] checkpoint=$CHECKPOINT"
echo "[smoke] out_root=$OUT_ROOT"
echo "[smoke] prompt_audio=$PROMPT_AUDIO"
echo "[smoke] prompt_text_file=$PROMPT_TEXT_FILE"
echo "[smoke] prompt_words_json=${PROMPT_WORDS_JSON:-<none>}"

if [ -n "$PROMPT_WORDS_JSON" ]; then
  "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

repo_root = Path("'"$REPO_ROOT"'")
prompt_audio = Path("'"$PROMPT_AUDIO"'").resolve()
prompt_text = Path("'"$PROMPT_TEXT_FILE"'").read_text(encoding="utf-8").strip()
prompt_words_json = Path("'"$PROMPT_WORDS_JSON"'").resolve()
prompt_dir = Path("'"$PROMPT_DIR"'").resolve()
prompt_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str((repo_root / "inference").resolve()))
sys.path.insert(0, str((repo_root / "vendor" / "f5tts_duration_ft").resolve()))
from release_utils import prompt_num_frames  # type: ignore
from run_timing_control_accuracy_b150 import add_residual_to_last_pause, build_token_track_from_words, load_sidecar_words  # type: ignore

words = load_sidecar_words(prompt_words_json)
prefix_tokens, prefix_durations = build_token_track_from_words(words)
actual_prompt_frames = prompt_num_frames(prompt_audio)
residual_frames = actual_prompt_frames - float(sum(sum(pair) for pair in prefix_durations))
add_residual_to_last_pause(prefix_durations, residual_frames)

prompt_track = {
    "prompt_audio": str(prompt_audio),
    "prompt_text": prompt_text,
    "language": "en",
    "prompt_id": prompt_audio.stem,
    "prefix_tokens": prefix_tokens,
    "prefix_durations": prefix_durations,
    "prompt_frames": actual_prompt_frames,
    "num_words": len(words),
    "residual_frames_added_to_last_pause": residual_frames,
    "source_words_json": str(prompt_words_json),
}
(prompt_dir / "prompt_track.json").write_text(json.dumps(prompt_track, ensure_ascii=False, indent=2), encoding="utf-8")
(prompt_dir / "prompt_alignment_raw.json").write_text(prompt_words_json.read_text(encoding="utf-8"), encoding="utf-8")
(prompt_dir / "prompt_alignment_debug.json").write_text(json.dumps({
    "source_words_json": str(prompt_words_json),
    "build_mode": "reuse_existing_mfa_words_json",
}, ensure_ascii=False, indent=2), encoding="utf-8")
'
else
  "$PYTHON_BIN" inference/align_prompt_with_mfa.py \
    --prompt-audio "$PROMPT_AUDIO" \
    --prompt-text "$(cat "$PROMPT_TEXT_FILE")" \
    --language en \
    --output-dir "$PROMPT_DIR"
fi

"$PYTHON_BIN" inference/run_spontaneous_demo.py \
  --prompt-audio "$PROMPT_AUDIO" \
  --prompt-text "$(cat "$PROMPT_TEXT_FILE")" \
  --target-text "$PLAIN_TEXT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$SPONT_DIR" \
  --output-prefix gen_target_only

"$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

repo_root = Path("'"$REPO_ROOT"'")
out_dir = Path("'"$CTRL_DIR"'")
prompt_audio = Path("'"$PROMPT_AUDIO"'").resolve()
prompt_track_json = Path("'"$PROMPT_TRACK_JSON"'").resolve()
inline_text = "'"$INLINE_TEXT"'"

sys.path.insert(0, str((repo_root / "inference").resolve()))
sys.path.insert(0, str((repo_root / "vendor" / "f5tts_duration_ft").resolve()))
from run_magictts import parse_inline_target_text  # type: ignore
from custom_prefix_showcase_demos import build_custom_track, frames_from_ms  # type: ignore

out_dir.mkdir(parents=True, exist_ok=True)
prompt_track = json.loads(prompt_track_json.read_text(encoding="utf-8"))
inline_control = parse_inline_target_text(inline_text, "en")
plain_text = inline_control["plain_text"]

prefix_context = {
    "audio_path": str(prompt_audio),
    "prefix_tokens": list(prompt_track["prefix_tokens"]),
    "prefix_durations": [list(pair) for pair in prompt_track["prefix_durations"]],
    "prefix_token_count": len(prompt_track["prefix_tokens"]),
    "source_sample_dir": str(prompt_audio.parent.resolve()),
    "source_prompt_frames": float(prompt_track["prompt_frames"]),
    "source_total_frames": float(prompt_track["prompt_frames"]),
    "prompt_text_source": prompt_track["prompt_text"],
    "prompt_id": prompt_track.get("prompt_id", prompt_audio.stem),
    "prompt_metadata": None,
}
variant = {
    "slug": "controlled_inline_long_en",
    "text": plain_text,
    "seed": None,
    "notes": "Controlled by inline markers in long English smoke with custom prompt.",
    "duration_ms": inline_control["duration_ms"],
    "pause_ms": inline_control["pause_ms"],
}
track = build_custom_track(
    prefix_context=prefix_context,
    variant=variant,
    content_frames=frames_from_ms(float(inline_control["default_content_ms"])),
    punct_frames=frames_from_ms(50.0),
)
(out_dir / "custom_track.json").write_text(json.dumps(track, ensure_ascii=False, indent=2), encoding="utf-8")
(out_dir / "request.json").write_text(json.dumps({
    "raw_inline_text": inline_text,
    "plain_text": plain_text,
    "default_content_ms": inline_control["default_content_ms"],
    "duration_ms": inline_control["duration_ms"],
    "pause_ms": inline_control["pause_ms"],
}, ensure_ascii=False, indent=2), encoding="utf-8")
'

"$PYTHON_BIN" inference/run_edit_from_json.py \
  --audio "$PROMPT_AUDIO" \
  --track-json "$CTRL_DIR/custom_track.json" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$CTRL_DIR"
