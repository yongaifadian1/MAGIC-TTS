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


def ensure_imports() -> None:
    preferred = [str(SCRIPT_DIR), str(REPO_ROOT)]
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


TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256
DEFAULT_PRETRAINED_CKPT = str((PRETRAINED_ROOT / "F5TTS_Base" / "model_1200000.safetensors").resolve())


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.payload, weights_only=False, map_location="cpu")
    tokenizer_path = payload["tokenizer_path"]
    checkpoint_source = payload["checkpoint_source"]
    pretrained_ckpt = payload.get("pretrained_ckpt", DEFAULT_PRETRAINED_CKPT)

    model = build_model(tokenizer_path)
    if checkpoint_source:
        load_training_checkpoint(model, checkpoint_source)
    else:
        load_pretrained_base(model, pretrained_ckpt)
    model.eval().to(device)

    vocoder = load_vocoder(
        vocoder_name=payload["sample_vocoder_name"],
        is_local=True,
        local_path=payload["sample_vocoder_local_path"],
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    mel = payload["mel"]
    mel_spec = mel.permute(0, 2, 1)
    mel_lengths = payload["mel_lengths"]
    text_inputs = payload["text_inputs"]
    audio_paths = payload.get("audio_paths", [])
    raw_texts = payload.get("raw_texts", [])
    token_durations = payload.get("token_durations")
    token_duration_mask = payload.get("token_duration_mask")
    sample_prompt_frac = float(payload["sample_prompt_frac"])
    target_sample_rate = int(payload["target_sample_rate"])
    sample_cfg_strength = float(payload["sample_cfg_strength"])
    sample_nfe_step = int(payload["sample_nfe_step"])
    sample_sway_sampling_coef = float(payload["sample_sway_sampling_coef"])

    for sample_idx, sample_text in enumerate(text_inputs):
        total_frames = int(mel_lengths[sample_idx])
        if total_frames <= 1:
            prompt_frames = 1
        else:
            prompt_frames = max(1, min(total_frames - 1, int(total_frames * sample_prompt_frac)))

        prompt_mel = mel[sample_idx : sample_idx + 1, :, :prompt_frames].to(device)
        target_mel = mel[sample_idx : sample_idx + 1, :, :total_frames].to(device)
        sample_text_input = [sample_text]
        sample_token_durations = None
        sample_token_duration_mask = None
        if torch.is_tensor(token_durations):
            sample_token_durations = token_durations[sample_idx : sample_idx + 1].to(device)
        if torch.is_tensor(token_duration_mask):
            sample_token_duration_mask = token_duration_mask[sample_idx : sample_idx + 1].to(device)

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=mel_spec[sample_idx : sample_idx + 1, :prompt_frames, :].to(device),
                text=sample_text_input,
                duration=total_frames,
                steps=sample_nfe_step,
                cfg_strength=sample_cfg_strength,
                sway_sampling_coef=sample_sway_sampling_coef,
                token_durations=sample_token_durations,
                token_duration_mask=sample_token_duration_mask,
                preserve_cond_audio_in_output=True,
            )

            generated_no_token_duration, _ = model.sample(
                cond=mel_spec[sample_idx : sample_idx + 1, :prompt_frames, :].to(device),
                text=sample_text_input,
                duration=total_frames,
                steps=sample_nfe_step,
                cfg_strength=sample_cfg_strength,
                sway_sampling_coef=sample_sway_sampling_coef,
                token_durations=sample_token_durations,
                token_duration_mask=(
                    torch.zeros_like(sample_token_duration_mask)
                    if sample_token_duration_mask is not None
                    else (
                        torch.zeros_like(sample_token_durations)
                        if sample_token_durations is not None
                        else None
                    )
                ),
                preserve_cond_audio_in_output=True,
            )

        generated = generated[:, :total_frames, :].to(torch.float32)
        generated_mel = generated.permute(0, 2, 1).to(device)
        generated_no_token_duration = generated_no_token_duration[:, :total_frames, :].to(torch.float32)
        generated_no_token_duration_mel = generated_no_token_duration.permute(0, 2, 1).to(device)

        prompt_audio = vocoder.decode(prompt_mel)
        target_audio = vocoder.decode(target_mel)
        gen_audio = vocoder.decode(generated_mel)
        gen_no_token_duration_audio = vocoder.decode(generated_no_token_duration_mel)

        sample_dir = Path(args.output_dir) / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(sample_dir / "prompt.wav"), prompt_audio.cpu(), target_sample_rate)
        torchaudio.save(str(sample_dir / "target.wav"), target_audio.cpu(), target_sample_rate)
        torchaudio.save(str(sample_dir / "gen.wav"), gen_audio.cpu(), target_sample_rate)
        torchaudio.save(
            str(sample_dir / "gen_no_token_duration.wav"),
            gen_no_token_duration_audio.cpu(),
            target_sample_rate,
        )

        with (sample_dir / "text.txt").open("w", encoding="utf-8") as f:
            if isinstance(sample_text, str):
                f.write(sample_text)
            else:
                f.write("".join(sample_text))

        with (sample_dir / "converted_text.json").open("w", encoding="utf-8") as f:
            json.dump(sample_text_input[0], f, ensure_ascii=False, indent=2)

        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "audio_path": audio_paths[sample_idx] if sample_idx < len(audio_paths) else None,
                    "raw_text": raw_texts[sample_idx] if sample_idx < len(raw_texts) else None,
                    "prompt_frames": prompt_frames,
                    "total_frames": total_frames,
                    "saved_variants": ["gen.wav", "gen_no_token_duration.wav"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        del prompt_mel
        del target_mel
        del generated
        del generated_mel
        del generated_no_token_duration
        del generated_no_token_duration_mel
        del prompt_audio
        del target_audio
        del gen_audio
        del gen_no_token_duration_audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del vocoder
    del model
    del payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
