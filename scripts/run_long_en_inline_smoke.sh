#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/checkpoints/magictts_36k.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${1:-$REPO_ROOT/outputs/long_en_inline_smoke}"
PROMPT_AUDIO="$REPO_ROOT/assets/default_prompt_en/prompt.mp3"
PROMPT_TEXT_FILE="$REPO_ROOT/assets/default_prompt_en/prompt.txt"
PROMPT_TRACK_JSON="$REPO_ROOT/assets/default_prompt_en/prompt_track.json"
export MAGICTTS_PRETRAINED_ROOT="${MAGICTTS_PRETRAINED_ROOT:-$REPO_ROOT/pretrained}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"

mkdir -p "$OUT_ROOT"

SPONT_DIR="$OUT_ROOT/spontaneous_long_en"
CTRL_DIR="$OUT_ROOT/controlled_inline_long_en"

PLAIN_TEXT="After the meeting, please review the budget, call Michael before Friday afternoon, and send the final delivery schedule to the client."
INLINE_TEXT="After the meeting[220], please review the budget[220], call Michael{360} before Friday afternoon[260], and send the final delivery{320} schedule{320} to the client."

echo "[smoke] repo_root=$REPO_ROOT"
echo "[smoke] checkpoint=$CHECKPOINT"
echo "[smoke] out_root=$OUT_ROOT"
echo "[smoke] pretrained_root=$MAGICTTS_PRETRAINED_ROOT"

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
    "notes": "Controlled by inline markers in long English smoke.",
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
