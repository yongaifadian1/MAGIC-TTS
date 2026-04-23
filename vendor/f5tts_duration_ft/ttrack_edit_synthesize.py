import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torchaudio
from safetensors.torch import load_file

warnings.filterwarnings("ignore", category=UserWarning, message=".*cudnn.*")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRETRAINED_ROOT = Path(os.environ.get("MAGICTTS_PRETRAINED_ROOT", str((REPO_ROOT / "pretrained").resolve())))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
DEFAULT_CHECKPOINT = str((REPO_ROOT / "checkpoints" / "magictts_36k.pt").resolve())
DEFAULT_PRETRAINED_CKPT = str((PRETRAINED_ROOT / "F5TTS_Base" / "model_1200000.safetensors").resolve())
DEFAULT_TOKENIZER_PATH = str((PRETRAINED_ROOT / "F5TTS_Base" / "vocab.txt").resolve())
DEFAULT_VOCODER_PATH = str((PRETRAINED_ROOT / "vocos-mel-24khz").resolve())


def ensure_imports() -> None:
    preferred = [str(REPO_ROOT), str(SCRIPT_DIR)]
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
from f5_tts.infer.utils_infer import load_vocoder  # noqa: E402
from f5_tts.model.utils import get_tokenizer  # noqa: E402


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


def load_pretrained_base(model: CFM, ckpt_path: str) -> None:
    state = load_file(ckpt_path)
    filtered_state = {}
    for key, value in state.items():
        if key.startswith("ema_model."):
            stripped = key.replace("ema_model.", "", 1)
            if stripped not in {"initted", "step"}:
                filtered_state[stripped] = value
        else:
            filtered_state[key] = value
    model.load_state_dict(filtered_state, strict=False)


def load_training_checkpoint(model: CFM, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    del checkpoint


def load_audio(audio_path: str) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--track-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="gen")
    parser.add_argument("--checkpoint", default=os.environ.get("F5TTS_EDIT_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--pretrained-ckpt", default=os.environ.get("F5TTS_FT_PRETRAINED_CKPT", DEFAULT_PRETRAINED_CKPT))
    parser.add_argument("--tokenizer-path", default=os.environ.get("F5TTS_FT_TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--sample-vocoder-name", default="vocos")
    parser.add_argument("--sample-vocoder-local-path", default=DEFAULT_VOCODER_PATH)
    parser.add_argument("--prompt-seconds", type=float, default=3.0)
    parser.add_argument("--prompt-frames", type=float, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preserve-cond-audio-in-output", action="store_true")
    parser.add_argument("--omit-token-durations", action="store_true")
    parser.add_argument("--use-target-only-text", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    track_path = Path(args.track_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    with track_path.open("r", encoding="utf-8") as f:
        track = json.load(f)

    audio = load_audio(args.audio)
    if args.prompt_frames is not None:
        prompt_samples = min(audio.shape[-1], int(round(args.prompt_frames * HOP_LENGTH)))
    else:
        prompt_samples = min(audio.shape[-1], int(round(args.prompt_seconds * TARGET_SAMPLE_RATE)))
    prompt_audio = audio[:, :prompt_samples]

    model = build_model(args.tokenizer_path)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        load_training_checkpoint(model, str(checkpoint_path))
        checkpoint_source = str(checkpoint_path)
    else:
        load_pretrained_base(model, args.pretrained_ckpt)
        checkpoint_source = args.pretrained_ckpt
    model.eval().to(device)

    vocoder = load_vocoder(
        vocoder_name=args.sample_vocoder_name,
        is_local=True,
        local_path=args.sample_vocoder_local_path,
        device=device,
    )

    prefix_token_count = int(track.get("prefix_token_count", 0) or 0)
    use_target_only_text = args.use_target_only_text and prefix_token_count > 0

    text_tokens = track["text_tokens"]
    edited_token_durations = track["edited_token_durations"]
    total_frames = int(track["edited_total_frames"])
    target_only_frames = None

    if use_target_only_text:
        text_tokens = text_tokens[prefix_token_count:]
        edited_token_durations = edited_token_durations[prefix_token_count:]
        target_only_frames = float(sum(sum(pair) for pair in edited_token_durations))
        total_frames = int(round(prompt_samples / HOP_LENGTH + target_only_frames))

    token_durations = None
    if not args.omit_token_durations:
        token_durations = torch.tensor(edited_token_durations, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        generated_audio, _ = model.sample(
            cond=prompt_audio.to(device),
            text=[text_tokens],
            duration=total_frames,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            token_durations=token_durations,
            seed=args.seed,
            vocoder=vocoder.decode,
            preserve_cond_audio_in_output=args.preserve_cond_audio_in_output,
        )

    generated_audio = generated_audio.to(torch.float32).cpu()
    input_audio_name = "input.wav" if args.output_prefix == "gen" else f"{args.output_prefix}_input.wav"
    prompt_audio_name = "prompt.wav" if args.output_prefix == "gen" else f"{args.output_prefix}_prompt.wav"
    torchaudio.save(str(output_dir / input_audio_name), audio.cpu(), TARGET_SAMPLE_RATE)
    torchaudio.save(str(output_dir / prompt_audio_name), prompt_audio.cpu(), TARGET_SAMPLE_RATE)
    torchaudio.save(str(output_dir / f"{args.output_prefix}.wav"), generated_audio, TARGET_SAMPLE_RATE)
    if args.preserve_cond_audio_in_output and prompt_samples < generated_audio.shape[-1]:
        target_only_audio = generated_audio[:, prompt_samples:]
        torchaudio.save(
            str(output_dir / f"{args.output_prefix}_target_only.wav"),
            target_only_audio,
            TARGET_SAMPLE_RATE,
        )

    metadata = {
        "audio_path": str(Path(args.audio).resolve()),
        "track_json": str(track_path),
        "output_prefix": args.output_prefix,
        "prompt_seconds": args.prompt_seconds,
        "prompt_samples": prompt_samples,
        "prompt_frames": prompt_samples / HOP_LENGTH,
        "edited_total_frames": total_frames,
        "preserve_cond_audio_in_output": args.preserve_cond_audio_in_output,
        "omit_token_durations": args.omit_token_durations,
        "use_target_only_text": use_target_only_text,
        "target_only_frames": target_only_frames,
        "checkpoint_source": checkpoint_source,
        "device": str(device),
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "sway_sampling_coef": args.sway_sampling_coef,
    }
    synthesis_metadata_name = (
        "synthesis_metadata.json"
        if args.output_prefix == "gen"
        else f"{args.output_prefix}_synthesis_metadata.json"
    )
    with (output_dir / synthesis_metadata_name).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del vocoder
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
