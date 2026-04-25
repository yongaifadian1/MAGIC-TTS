#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$REPO_ROOT/outputs/long_en_inline_smoke_custom_prompt_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="$REPO_ROOT/$(basename "$OUT_ROOT").slurm.out"
PROMPT_AUDIO="${PROMPT_AUDIO:?PROMPT_AUDIO is required}"
PROMPT_TEXT_FILE="${PROMPT_TEXT_FILE:?PROMPT_TEXT_FILE is required}"
PROMPT_WORDS_JSON="${PROMPT_WORDS_JSON:-}"

sbatch \
  -J mtrel_longen \
  -p gznetA800 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --output="$LOG_PATH" \
  --chdir="$REPO_ROOT" \
  --export=ALL,PROMPT_AUDIO,PROMPT_TEXT_FILE,PROMPT_WORDS_JSON \
  --wrap="bash $REPO_ROOT/scripts/run_long_en_inline_smoke_custom_prompt.sh $OUT_ROOT"

echo "$OUT_ROOT"
echo "$LOG_PATH"
