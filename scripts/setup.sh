#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED_ROOT="${MAGICTTS_PRETRAINED_ROOT:-$REPO_ROOT/pretrained}"
LOCAL_F5R_ROOT="${MAGICTTS_SETUP_LOCAL_F5R_ROOT:-}"
DOWNLOAD_BASE_CKPT="${MAGICTTS_DOWNLOAD_BASE_CKPT:-0}"
SKIP_PIP_INSTALL="${MAGICTTS_SKIP_PIP_INSTALL:-0}"
TORCH_VERSION="${MAGICTTS_TORCH_VERSION:-2.3.0}"
TORCH_WHL_INDEX="${MAGICTTS_TORCH_WHL_INDEX:-https://download.pytorch.org/whl/cu118}"

mkdir -p "$PRETRAINED_ROOT/F5TTS_Base" "$PRETRAINED_ROOT/vocos-mel-24khz"

if [[ "$SKIP_PIP_INSTALL" != "1" ]]; then
  "$PYTHON_BIN" -m pip install \
    --index-url "$TORCH_WHL_INDEX" \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCH_VERSION}"
  "$PYTHON_BIN" -m pip install -e "$REPO_ROOT"
fi

if [[ -n "$LOCAL_F5R_ROOT" ]]; then
  cp "$LOCAL_F5R_ROOT/pretrained_model/F5-TTS-official/F5TTS_Base/vocab.txt" "$PRETRAINED_ROOT/F5TTS_Base/vocab.txt"
  cp "$LOCAL_F5R_ROOT/pretrained_model/vocos-mel-24khz/config.yaml" "$PRETRAINED_ROOT/vocos-mel-24khz/config.yaml"
  cp "$LOCAL_F5R_ROOT/pretrained_model/vocos-mel-24khz/pytorch_model.bin" "$PRETRAINED_ROOT/vocos-mel-24khz/pytorch_model.bin"
  if [[ "$DOWNLOAD_BASE_CKPT" == "1" ]]; then
    cp "$LOCAL_F5R_ROOT/pretrained_model/F5-TTS-official/F5TTS_Base/model_1200000.safetensors" "$PRETRAINED_ROOT/F5TTS_Base/model_1200000.safetensors"
  fi
else
  export REPO_ROOT
  export PRETRAINED_ROOT
  export DOWNLOAD_BASE_CKPT
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

pretrained_root = Path(os.environ["PRETRAINED_ROOT"])
download_base = os.environ.get("DOWNLOAD_BASE_CKPT", "0") == "1"

hf_hub_download(
    repo_id="SWivid/F5-TTS",
    filename="F5TTS_Base/vocab.txt",
    repo_type="model",
    local_dir=str(pretrained_root),
    local_dir_use_symlinks=False,
)

snapshot_download(
    repo_id="charactr/vocos-mel-24khz",
    local_dir=str(pretrained_root / "vocos-mel-24khz"),
    local_dir_use_symlinks=False,
    allow_patterns=["config.yaml", "pytorch_model.bin"],
)

if download_base:
    hf_hub_download(
        repo_id="SWivid/F5-TTS",
        filename="F5TTS_Base/model_1200000.safetensors",
        repo_type="model",
        local_dir=str(pretrained_root),
        local_dir_use_symlinks=False,
    )
PY
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe not found in PATH; install ffmpeg before running inference." >&2
fi

echo "MAGIC-TTS setup complete."
