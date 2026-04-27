#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MFA_JOBS="${MFA_JOBS:-2}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-10000}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_finetune_dataset.sh \
    --input-jsonl /path/to/manifest.jsonl \
    --audio-root /path/to/raw_audio_root \
    --output-dir data/b150_public

Required:
  --input-jsonl PATH [PATH ...]
  --audio-root PATH
  --output-dir PATH

Optional:
  --text-field FIELD
  --sidecar-root PATH
  --shard-root PATH
  --num-jobs N
  --samples-per-shard N
  --keep-workdir
  --skip-existing
EOF
}

INPUT_JSONL=()
TEXT_FIELD="${MAGICTTS_TEXT_FIELD:-target_text}"
AUDIO_ROOT=""
OUTPUT_DIR=""
SIDECAR_ROOT=""
SHARD_ROOT=""
NUM_JOBS="$MFA_JOBS"
KEEP_WORKDIR=0
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-jsonl)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        INPUT_JSONL+=("$1")
        shift
      done
      ;;
    --text-field)
      TEXT_FIELD="${2:-}"
      shift 2
      ;;
    --audio-root)
      AUDIO_ROOT="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --sidecar-root)
      SIDECAR_ROOT="${2:-}"
      shift 2
      ;;
    --shard-root)
      SHARD_ROOT="${2:-}"
      shift 2
      ;;
    --num-jobs)
      NUM_JOBS="${2:-}"
      shift 2
      ;;
    --samples-per-shard)
      SAMPLES_PER_SHARD="${2:-}"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#INPUT_JSONL[@]} -eq 0 || -z "$AUDIO_ROOT" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$SHARD_ROOT" ]]; then
  SHARD_ROOT="$REPO_ROOT/data/mfa_shards"
fi
if [[ -z "$SIDECAR_ROOT" ]]; then
  SIDECAR_ROOT="$REPO_ROOT/data/prompt_sidecars"
fi

mkdir -p "$SHARD_ROOT" "$SIDECAR_ROOT" "$OUTPUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/tools/f5tts_duration_ft/prepare_emilia_1nv_mfa_shards.py" \
  --input-jsonl "${INPUT_JSONL[@]}" \
  --text-field "$TEXT_FIELD" \
  --audio-root "$AUDIO_ROOT" \
  --output-root "$SHARD_ROOT" \
  --samples-per-shard "$SAMPLES_PER_SHARD"

while IFS= read -r manifest; do
  [[ -n "$manifest" ]] || continue
  cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/tools/f5tts_duration_ft/run_mfa_alignment_shard.py"
    --manifest "$manifest"
    --sidecar-root "$SIDECAR_ROOT"
    --audio-root "$AUDIO_ROOT"
    --num-jobs "$NUM_JOBS"
  )
  if [[ "$KEEP_WORKDIR" == "1" ]]; then
    cmd+=(--keep-workdir)
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  "${cmd[@]}"
done < "$SHARD_ROOT/shard_list.txt"

"$PYTHON_BIN" "$REPO_ROOT/tools/f5tts_duration_ft/prepare_emilia_1nv_merged_worddur.py" \
  --input-jsonl "${INPUT_JSONL[@]}" \
  --text-field "$TEXT_FIELD" \
  --save-dir "$OUTPUT_DIR" \
  --audio-root "$AUDIO_ROOT" \
  --alignment-root "$SIDECAR_ROOT" \
  --alignment-audio-root "$AUDIO_ROOT" \
  --require-alignments
