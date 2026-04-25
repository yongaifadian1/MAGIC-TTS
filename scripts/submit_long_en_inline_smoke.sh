#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$REPO_ROOT/outputs/long_en_inline_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="$REPO_ROOT/$(basename "$OUT_ROOT").slurm.out"

sbatch \
  -J mtrel_longen \
  -p gznetA800 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --output="$LOG_PATH" \
  --chdir="$REPO_ROOT" \
  --wrap="bash $REPO_ROOT/scripts/run_long_en_inline_smoke.sh $OUT_ROOT"

echo "$OUT_ROOT"
echo "$LOG_PATH"
