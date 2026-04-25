#!/usr/bin/env python3

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, message=".*cudnn.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor" / "f5tts_duration_ft"
PRETRAINED_ROOT = Path(os.environ.get("MAGICTTS_PRETRAINED_ROOT", str((REPO_ROOT / "pretrained").resolve())))
DEFAULT_DATASET = REPO_ROOT / "data" / "b150_public"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "finetune_runs"
DEFAULT_INIT_MODEL_CKPT = REPO_ROOT / "checkpoints" / "magictts_36k.pt"
DEFAULT_PRETRAINED_CKPT = PRETRAINED_ROOT / "F5TTS_Base" / "model_1200000.safetensors"
DEFAULT_TOKENIZER = PRETRAINED_ROOT / "F5TTS_Base" / "vocab.txt"
DEFAULT_VOCODER = PRETRAINED_ROOT / "vocos-mel-24khz"


def ensure_imports() -> None:
    preferred = [str(VENDOR_ROOT), str(REPO_ROOT)]
    existing = [path for path in sys.path if path not in preferred]
    sys.path[:] = preferred + existing


ensure_imports()

CFM = None
DiT = None
Trainer = None
load_dataset = None
get_tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MAGIC-TTS with a local token-duration dataset."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--dataset-type",
        default="CustomDatasetPath",
        choices=["CustomDatasetPath", "CustomDataset", "HFDataset"],
    )
    parser.add_argument("--run-name", default="magictts_finetune")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--pretrained-ckpt", default=str(DEFAULT_PRETRAINED_CKPT))
    parser.add_argument("--init-model-ckpt", default=str(DEFAULT_INIT_MODEL_CKPT))
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--sample-vocoder-path", default=str(DEFAULT_VOCODER))
    parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--lr", type=float, default=7.5e-5)
    parser.add_argument("--batch-size", type=int, default=30000)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--save-updates", type=int, default=1000)
    parser.add_argument("--last-updates", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--duration-dropout-prob", type=float, default=0.2)
    parser.add_argument("--prompt-duration-mask-prob", type=float, default=0.0)
    parser.add_argument("--prompt-text-mask-prob", type=float, default=0.0)
    parser.add_argument("--prompt-text-target-only-prob", type=float, default=0.0)
    parser.add_argument("--prompt-text-mask-token", default="")
    parser.add_argument("--lr-decay-updates", type=int, default=0)
    parser.add_argument("--reset-train-state", action="store_true")
    return parser.parse_args()


def import_training_modules() -> None:
    global CFM, DiT, Trainer, load_dataset, get_tokenizer

    import torch  # noqa: F401
    from safetensors.torch import load_file  # noqa: F401

    try:
        import rjieba  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        import jieba as _jieba

        sys.modules["rjieba"] = _jieba

    from duration_cfm import CFM as imported_cfm
    from duration_dataset import load_dataset as imported_load_dataset
    from duration_dit import DiT as imported_dit
    from duration_trainer import Trainer as imported_trainer
    from f5_tts.model.utils import get_tokenizer as imported_get_tokenizer

    CFM = imported_cfm
    DiT = imported_dit
    Trainer = imported_trainer
    load_dataset = imported_load_dataset
    get_tokenizer = imported_get_tokenizer


def build_model(tokenizer_path: str):
    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    mel_spec_type = "vocos"

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, "custom")

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
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
    return model


def load_pretrained_base(model, ckpt_path: str) -> None:
    from safetensors.torch import load_file

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


def load_training_checkpoint(model, ckpt_path: str) -> None:
    import torch

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "ema_model_state_dict" in checkpoint:
        state = checkpoint["ema_model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=False)


def configure_env(args, checkpoint_dir: Path) -> None:
    os.environ["F5TTS_FT_DATASET_PATH"] = str(Path(args.dataset).resolve())
    os.environ["F5TTS_FT_CKPT_DIR"] = str(checkpoint_dir)
    os.environ["F5TTS_FT_RUN_NAME"] = args.run_name
    os.environ["F5TTS_FT_PRETRAINED_CKPT"] = str(Path(args.pretrained_ckpt).resolve())
    os.environ["F5TTS_FT_TOKENIZER_PATH"] = str(Path(args.tokenizer_path).resolve())
    os.environ["F5TTS_FT_SAMPLE_VOCODER_PATH"] = str(Path(args.sample_vocoder_path).resolve())
    os.environ["F5TTS_FT_LR"] = str(args.lr)
    os.environ["F5TTS_FT_BATCH_SIZE"] = str(args.batch_size)
    os.environ["F5TTS_FT_MAX_SAMPLES"] = str(args.max_samples)
    os.environ["F5TTS_FT_GRAD_ACCUM"] = str(args.grad_accum)
    os.environ["F5TTS_FT_MAX_GRAD_NORM"] = str(args.max_grad_norm)
    os.environ["F5TTS_FT_EPOCHS"] = str(args.epochs)
    os.environ["F5TTS_FT_MAX_UPDATES"] = str(args.max_updates)
    os.environ["F5TTS_FT_WARMUP_UPDATES"] = str(args.warmup_updates)
    os.environ["F5TTS_FT_SAVE_UPDATES"] = str(args.save_updates)
    os.environ["F5TTS_FT_LAST_UPDATES"] = str(args.last_updates)
    os.environ["F5TTS_FT_NUM_WORKERS"] = str(args.num_workers)
    os.environ["F5TTS_FT_DURATION_DROPOUT_PROB"] = str(args.duration_dropout_prob)
    os.environ["F5TTS_FT_PROMPT_DURATION_MASK_PROB"] = str(args.prompt_duration_mask_prob)
    os.environ["F5TTS_FT_PROMPT_TEXT_MASK_PROB"] = str(args.prompt_text_mask_prob)
    os.environ["F5TTS_FT_PROMPT_TEXT_TARGET_ONLY_PROB"] = str(args.prompt_text_target_only_prob)
    os.environ["F5TTS_FT_PROMPT_TEXT_MASK_TOKEN"] = args.prompt_text_mask_token
    os.environ["F5TTS_FT_LR_DECAY_UPDATES"] = str(args.lr_decay_updates)
    os.environ["F5TTS_FT_LOG_SAMPLES"] = "1" if args.log_samples else "0"
    os.environ["F5TTS_FT_RESET_TRAIN_STATE"] = "1" if args.reset_train_state else "0"


def main():
    args = parse_args()
    import_training_modules()

    if args.init_model_ckpt is not None and str(args.init_model_ckpt).strip().lower() in {"", "none", "null"}:
        args.init_model_ckpt = None

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset path does not exist: {dataset_path}\n"
            "Expected a prepared dataset directory such as data/b150_public."
        )

    checkpoint_dir = (
        Path(args.checkpoint_dir).resolve()
        if args.checkpoint_dir
        else (DEFAULT_CHECKPOINT_ROOT / args.run_name).resolve()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configure_env(args, checkpoint_dir)

    model = build_model(args.tokenizer_path)
    resumable_ckpt = checkpoint_dir / "model_last.pt"
    if resumable_ckpt.exists():
        print(f"Found resumable checkpoint at {resumable_ckpt}, trainer will resume from it.")
    elif args.init_model_ckpt:
        load_training_checkpoint(model, str(Path(args.init_model_ckpt).resolve()))
        print(f"Initialized from training checkpoint: {Path(args.init_model_ckpt).resolve()}")
    else:
        load_pretrained_base(model, str(Path(args.pretrained_ckpt).resolve()))
        print(f"Initialized from base checkpoint: {Path(args.pretrained_ckpt).resolve()}")

    trainer = Trainer(
        model,
        args.epochs,
        args.lr,
        num_warmup_updates=args.warmup_updates,
        save_per_updates=args.save_updates,
        checkpoint_path=str(checkpoint_dir),
        batch_size=args.batch_size,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        logger=None if args.logger == "none" else args.logger,
        wandb_project="MAGIC-TTS",
        wandb_run_name=args.run_name,
        wandb_resume_id=None,
        last_per_steps=args.last_updates,
        max_updates=args.max_updates if args.max_updates > 0 else None,
        log_samples=args.log_samples,
        mel_spec_type="vocos",
    )

    train_dataset = load_dataset(
        str(dataset_path),
        "custom",
        dataset_type=args.dataset_type,
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
    )
    trainer.train(train_dataset, num_workers=args.num_workers, resumable_with_seed=666)


if __name__ == "__main__":
    main()
