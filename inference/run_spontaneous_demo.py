#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "vendor" / "f5tts_duration_ft"
PRETRAINED_ROOT = Path(os.environ.get("MAGICTTS_PRETRAINED_ROOT", str((REPO_ROOT / "pretrained").resolve())))


def ensure_imports():
    preferred = [str(REPO_ROOT), str(TOOL_ROOT)]
    existing = [path for path in sys.path if path not in preferred]
    sys.path[:] = preferred + existing


ensure_imports()

try:
    import rjieba  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import jieba as _jieba

    sys.modules["rjieba"] = _jieba

from duration_cfm import CFM  # noqa: E402
from duration_dit import DiT  # noqa: E402
from f5_tts.infer.utils_infer import infer_process  # noqa: E402
from f5_tts.infer.utils_infer import load_vocoder  # noqa: E402
from f5_tts.infer.utils_infer import preprocess_ref_audio_text  # noqa: E402
from f5_tts.model.utils import get_tokenizer  # noqa: E402


TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
DEFAULT_TOKENIZER_PATH = str((PRETRAINED_ROOT / "F5TTS_Base" / "vocab.txt").resolve())
DEFAULT_VOCODER_PATH = str((PRETRAINED_ROOT / "vocos-mel-24khz").resolve())
DEFAULT_CHECKPOINT = str((REPO_ROOT / "checkpoints" / "magictts_36k.pt").resolve())


def build_model(tokenizer_path: str) -> CFM:
    n_mel_channels = 100
    win_length = 1024
    n_fft = 1024
    mel_spec_type = "vocos"

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, "custom")
    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=TARGET_SAMPLE_RATE,
        mel_spec_type=mel_spec_type,
    )

    return CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
            duration_condition=True,
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )


def load_training_checkpoint(model: CFM, ckpt_path: str) -> None:
    import torch

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocoder-path", default=DEFAULT_VOCODER_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="spontaneous")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--remove-silence", action="store_true")
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.tokenizer_path)
    load_training_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=args.vocoder_path,
        device=device,
    )

    prompt_audio, prompt_text = preprocess_ref_audio_text(args.prompt_audio, args.prompt_text, device=device)
    wav, sr, spect = infer_process(
        prompt_audio,
        prompt_text,
        args.target_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed,
        device=device,
    )

    out_wav = output_dir / f"{args.output_prefix}.wav"
    sf.write(str(out_wav), wav, sr)

    metadata = {
        "prompt_audio": str(Path(args.prompt_audio).resolve()),
        "prompt_text": args.prompt_text,
        "target_text": args.target_text,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tokenizer_path": args.tokenizer_path,
        "vocoder_path": args.vocoder_path,
        "output_wav": str(out_wav.resolve()),
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "sway_sampling_coef": args.sway_sampling_coef,
        "speed": args.speed,
        "seed": args.seed,
        "device": device,
        "mode": "spontaneous_no_text_duration",
        "sample_rate": sr,
        "spectrogram_shape": list(spect.shape),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
