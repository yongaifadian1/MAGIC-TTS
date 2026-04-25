from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from duration_cfm import CFM
from duration_dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists

# trainer


def mask_prompt_text_prefix(text_inputs, prompt_boundary_index, apply_prompt_mask, mask_token=""):
    masked = list(text_inputs)
    masked_prefix_ratios = []

    for sample_idx, sample_text in enumerate(masked):
        if not bool(apply_prompt_mask[sample_idx].item()):
            continue

        if sample_text is None:
            continue

        boundary = int(prompt_boundary_index[sample_idx].item())

        if isinstance(sample_text, str):
            sample_tokens = list(sample_text)
            boundary = max(0, min(boundary, len(sample_tokens)))
            if boundary <= 0:
                continue
            sample_tokens[:boundary] = [mask_token] * boundary
            masked[sample_idx] = sample_tokens
            masked_prefix_ratios.append(boundary / max(1, len(sample_tokens)))
            continue

        sample_tokens = list(sample_text)
        boundary = max(0, min(boundary, len(sample_tokens)))
        if boundary <= 0:
            continue
        sample_tokens[:boundary] = [mask_token] * boundary
        masked[sample_idx] = sample_tokens
        masked_prefix_ratios.append(boundary / max(1, len(sample_tokens)))

    return masked, masked_prefix_ratios


def slice_to_target_text_prefix(
    text_inputs,
    token_durations,
    token_duration_mask,
    prompt_boundary_index,
    apply_target_only,
):
    sliced_text = []
    sliced_token_duration_list = []
    sliced_token_duration_mask_list = []
    sliced_prefix_ratios = []

    has_token_durations = token_durations is not None

    for sample_idx, sample_text in enumerate(text_inputs):
        if isinstance(sample_text, str):
            sample_tokens = list(sample_text)
        else:
            sample_tokens = list(sample_text)

        valid_text_len = len(sample_tokens)
        boundary = int(prompt_boundary_index[sample_idx].item())
        boundary = max(0, min(boundary, valid_text_len))
        use_target_only = bool(apply_target_only[sample_idx].item()) and boundary > 0

        if use_target_only:
            sample_tokens = sample_tokens[boundary:]
            sliced_prefix_ratios.append(boundary / max(1, valid_text_len))

        sliced_text.append(sample_tokens)

        if not has_token_durations:
            continue

        valid_duration_len = valid_text_len
        sample_token_durations = token_durations[sample_idx, :valid_duration_len, :]
        if token_duration_mask is not None:
            sample_token_duration_mask = token_duration_mask[sample_idx, :valid_duration_len, :]
        else:
            sample_token_duration_mask = torch.ones_like(sample_token_durations)

        if use_target_only:
            sample_token_durations = sample_token_durations[boundary:, :]
            sample_token_duration_mask = sample_token_duration_mask[boundary:, :]

        sliced_token_duration_list.append(sample_token_durations)
        sliced_token_duration_mask_list.append(sample_token_duration_mask)

    if not has_token_durations:
        return sliced_text, token_durations, token_duration_mask, sliced_prefix_ratios

    max_text_len = max(len(tokens) for tokens in sliced_text)
    padded_token_durations = []
    padded_token_duration_masks = []
    for sample_token_durations, sample_token_duration_mask in zip(
        sliced_token_duration_list, sliced_token_duration_mask_list
    ):
        cur_len = sample_token_durations.shape[0]
        if cur_len < max_text_len:
            pad_rows = max_text_len - cur_len
            sample_token_durations = torch.nn.functional.pad(sample_token_durations, (0, 0, 0, pad_rows), value=0.0)
            sample_token_duration_mask = torch.nn.functional.pad(
                sample_token_duration_mask, (0, 0, 0, pad_rows), value=0.0
            )
        padded_token_durations.append(sample_token_durations)
        padded_token_duration_masks.append(sample_token_duration_mask)

    return (
        sliced_text,
        torch.stack(padded_token_durations, dim=0),
        torch.stack(padded_token_duration_masks, dim=0),
        sliced_prefix_ratios,
    )


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_steps=None,
        max_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        print(f"Using logger: {logger}")
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.max_updates = max_updates
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.vocoder_name = mel_spec_type

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        saved_path = None
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                saved_path = f"{self.checkpoint_path}/model_last.pt"
                self.accelerator.save(checkpoint, saved_path)
                print(f"Saved last checkpoint at step {step}")
            else:
                saved_path = f"{self.checkpoint_path}/model_{step}.pt"
                self.accelerator.save(checkpoint, saved_path)
            del checkpoint
            gc.collect()
        return saved_path

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        checkpoint_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")]
        if not checkpoint_files:
            return 0

        if "model_last.pt" in checkpoint_files:
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                checkpoint_files,
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        if self.is_main:
            # Allow loading checkpoint with mismatched keys (e.g., from different model versions)
            ema_state_dict = checkpoint["ema_model_state_dict"]
            # Filter out unexpected keys
            model_state_keys = set(self.ema_model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in ema_state_dict.items() if k in model_state_keys}
            missing_keys = model_state_keys - set(ema_state_dict.keys())
            unexpected_keys = set(ema_state_dict.keys()) - model_state_keys
            if missing_keys:
                print(f"Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint (will be ignored): {len(unexpected_keys)}")
                for key in sorted(unexpected_keys)[:5]:  # Show first 5 unexpected keys
                    print(f"  - {key}")
            self.ema_model.load_state_dict(filtered_state_dict, strict=False)

        if "step" in checkpoint:
            # Filter out unexpected keys for model state dict as well
            model_state_dict = checkpoint["model_state_dict"]
            model_keys = set(self.accelerator.unwrap_model(self.model).state_dict().keys())
            filtered_model_state_dict = {k: v for k, v in model_state_dict.items() if k in model_keys}
            missing_model_keys = model_keys - set(model_state_dict.keys())
            unexpected_model_keys = set(model_state_dict.keys()) - model_keys
            if missing_model_keys:
                print(f"Missing keys in model checkpoint: {len(missing_model_keys)}")
            if unexpected_model_keys:
                print(f"Unexpected keys in model checkpoint (will be ignored): {len(unexpected_model_keys)}")
                for key in sorted(unexpected_model_keys)[:5]:
                    print(f"  - {key}")
            self.accelerator.unwrap_model(self.model).load_state_dict(filtered_model_state_dict, strict=False)
            reset_train_state = os.environ.get("F5TTS_FT_RESET_TRAIN_STATE", "0").lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if reset_train_state:
                if self.is_main:
                    print("Loaded model weights from checkpoint and reset optimizer/scheduler/global step")
                step = 0
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0

        del checkpoint
        gc.collect()
        return step

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            sample_cfg_strength = float(os.environ.get("F5TTS_FT_SAMPLE_CFG_STRENGTH", "2.0"))
            sample_nfe_step = int(os.environ.get("F5TTS_FT_SAMPLE_STEPS", "32"))
            sample_sway_sampling_coef = float(os.environ.get("F5TTS_FT_SAMPLE_SWAY", "-1.0"))
            sample_count = int(os.environ.get("F5TTS_FT_SAMPLE_COUNT", "10"))
            sample_prompt_frac = float(os.environ.get("F5TTS_FT_SAMPLE_PROMPT_FRAC", "0.3"))
            target_sample_rate = int(os.environ.get("F5TTS_FT_SAMPLE_RATE", "24000"))
            sample_vocoder_name = os.environ.get("F5TTS_FT_SAMPLE_VOCODER", self.vocoder_name)
            sample_vocoder_local_path = os.environ.get(
                "F5TTS_FT_SAMPLE_VOCODER_PATH",
                str(Path(__file__).resolve().parents[2] / "pretrained" / "vocos-mel-24khz"),
            )
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)
            payload_dir = f"{self.checkpoint_path}/sample_payloads"
            os.makedirs(payload_dir, exist_ok=True)
            sample_worker_script = Path(__file__).resolve().parent / "duration_finetune_sample_logger_worker.py"
            project_root = Path(__file__).resolve().parents[2]

            def save_logged_samples(global_step, batch, mel_lengths, text_inputs, checkpoint_source):
                if not self.accelerator.is_local_main_process:
                    return None

                step_log_path = f"{log_samples_path}/step_{global_step}"
                os.makedirs(step_log_path, exist_ok=True)
                actual_sample_count = min(sample_count, len(text_inputs))
                if actual_sample_count <= 0:
                    return None

                batch_mel = batch["mel"][:actual_sample_count].detach().cpu().contiguous()
                if torch.is_tensor(mel_lengths):
                    sample_mel_lengths = mel_lengths[:actual_sample_count].detach().cpu().contiguous()
                else:
                    sample_mel_lengths = torch.tensor(mel_lengths[:actual_sample_count], dtype=torch.long)
                sample_text_inputs = list(text_inputs[:actual_sample_count])
                sample_raw_texts = list(batch.get("raw_text", [])[:actual_sample_count])
                sample_audio_paths = list(batch.get("audio_path", [])[:actual_sample_count])
                batch_token_durations = batch.get("token_durations")
                batch_token_duration_mask = batch.get("token_duration_mask")
                sample_token_durations = (
                    batch_token_durations[:actual_sample_count].detach().cpu().contiguous()
                    if torch.is_tensor(batch_token_durations)
                    else None
                )
                sample_token_duration_mask = (
                    batch_token_duration_mask[:actual_sample_count].detach().cpu().contiguous()
                    if torch.is_tensor(batch_token_duration_mask)
                    else None
                )

                payload = {
                    "global_step": global_step,
                    "mel": batch_mel,
                    "mel_lengths": sample_mel_lengths,
                    "text_inputs": sample_text_inputs,
                    "raw_texts": sample_raw_texts,
                    "audio_paths": sample_audio_paths,
                    "token_durations": sample_token_durations,
                    "token_duration_mask": sample_token_duration_mask,
                    "sample_prompt_frac": sample_prompt_frac,
                    "target_sample_rate": target_sample_rate,
                    "sample_cfg_strength": sample_cfg_strength,
                    "sample_nfe_step": sample_nfe_step,
                    "sample_sway_sampling_coef": sample_sway_sampling_coef,
                    "sample_vocoder_name": sample_vocoder_name,
                    "sample_vocoder_local_path": sample_vocoder_local_path,
                    "tokenizer_path": os.environ["F5TTS_FT_TOKENIZER_PATH"],
                    "pretrained_ckpt": os.environ["F5TTS_FT_PRETRAINED_CKPT"],
                    "checkpoint_source": checkpoint_source or "",
                }

                fd, payload_path = tempfile.mkstemp(
                    prefix=f"step_{global_step:07d}_",
                    suffix=".pt",
                    dir=payload_dir,
                )
                os.close(fd)
                try:
                    torch.save(payload, payload_path)
                    worker_env = os.environ.copy()
                    worker_env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
                    try:
                        subprocess.run(
                            [sys.executable, str(sample_worker_script), "--payload", payload_path, "--output-dir", step_log_path],
                            check=True,
                            env=worker_env,
                            cwd=project_root,
                        )
                    except subprocess.CalledProcessError as exc:
                        warning_path = Path(step_log_path) / "sample_worker_error.txt"
                        warning_path.write_text(
                            f"Sample worker failed at step {global_step} with exit code {exc.returncode}.\n"
                            "Training continued; inspect the slurm log for the worker traceback.\n",
                            encoding="utf-8",
                        )
                        print(
                            f"Warning: sample worker failed at step {global_step} "
                            f"with exit code {exc.returncode}; continuing training.",
                            flush=True,
                        )
                    return None
                finally:
                    if os.path.exists(payload_path):
                        os.remove(payload_path)
                    del payload
                    del batch_mel
                    del sample_mel_lengths
                    del sample_text_inputs
                    del sample_raw_texts
                    del sample_audio_paths
                    del sample_token_durations
                    del sample_token_duration_mask
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None
        duration_dropout_prob = float(os.environ.get("F5TTS_FT_DURATION_DROPOUT_PROB", "0.0"))
        prompt_duration_mask_prob = float(os.environ.get("F5TTS_FT_PROMPT_DURATION_MASK_PROB", "0.0"))
        prompt_text_mask_prob = float(os.environ.get("F5TTS_FT_PROMPT_TEXT_MASK_PROB", "0.0"))
        prompt_text_target_only_prob = float(os.environ.get("F5TTS_FT_PROMPT_TEXT_TARGET_ONLY_PROB", "0.0"))
        prompt_text_mask_token = os.environ.get("F5TTS_FT_PROMPT_TEXT_MASK_TOKEN", "")

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_residual=False
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        scheduler_step_scale = 1 if getattr(self.accelerator, "split_batches", False) else self.accelerator.num_processes
        warmup_steps = int(
            self.num_warmup_updates * scheduler_step_scale
        )  # accelerate may step the scheduler once per process when split_batches=False
        total_updates = int(len(train_dataloader) * self.epochs / self.grad_accumulation_steps)
        if exists(self.max_updates):
            total_updates = min(total_updates, self.max_updates)
        decay_updates_override = int(os.environ.get("F5TTS_FT_LR_DECAY_UPDATES", "0"))
        if decay_updates_override > 0:
            decay_steps = max(1, int(decay_updates_override * scheduler_step_scale))
            scheduler_total_steps = warmup_steps + decay_steps
        else:
            scheduler_total_steps = int(total_updates * scheduler_step_scale)
            decay_steps = max(1, scheduler_total_steps - warmup_steps)
        if self.is_main:
            print(
                "Scheduler setup: "
                f"process_scale={scheduler_step_scale}, "
                f"warmup_updates={self.num_warmup_updates}, "
                f"train_total_updates={total_updates}, "
                f"decay_updates_override={decay_updates_override}, "
                f"warmup_steps={warmup_steps}, "
                f"decay_steps={decay_steps}, "
                f"scheduler_total_steps={scheduler_total_steps}"
            )
        if self.is_main and scheduler_total_steps <= warmup_steps:
            print(
                "Warning: scheduler_total_steps "
                f"({scheduler_total_steps}) <= warmup_steps ({warmup_steps}); "
                f"clamping decay_steps to {decay_steps}."
            )
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus
        start_step = self.load_checkpoint()
        global_step = start_step
        reset_dataloader_state = os.environ.get("F5TTS_FT_RESET_DATALOADER_STATE", "0").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            if reset_dataloader_state and start_step > 0:
                skipped_epoch = 0
                skipped_batch = 0
                use_skipped_dataloader = False
                if self.is_main:
                    print(
                        "F5TTS_FT_RESET_DATALOADER_STATE=1: "
                        f"loaded checkpoint step {start_step}, keeping global_step but restarting dataloader epoch"
                    )
            else:
                skipped_epoch = int(start_step // orig_epoch_step)
                skipped_batch = start_step % orig_epoch_step
                skipped_dataloader = self.accelerator.skip_first_batches(
                    train_dataloader, num_batches=skipped_batch
                )
                use_skipped_dataloader = True
        else:
            skipped_epoch = 0
            skipped_batch = 0
            use_skipped_dataloader = False

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            
            # Save initial samples at the start of training (step 0)
            if self.log_samples and epoch == skipped_epoch and start_step == 0:
                # Get first batch for initial sample logging
                first_batch = next(iter(train_dataloader))
                first_text_inputs = first_batch["text"]
                first_mel_lengths = first_batch["mel_lengths"]
                save_logged_samples(
                    0,
                    first_batch,
                    first_mel_lengths,
                    first_text_inputs,
                    None,
                )
                del first_batch
                del first_text_inputs
                del first_mel_lengths
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if exists(resumable_with_seed) and epoch == skipped_epoch and use_skipped_dataloader:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    token_durations = batch.get("token_durations")
                    token_duration_mask = batch.get("token_duration_mask")
                    prompt_boundary_index = batch.get("prompt_boundary_index")

                    duration_dropout_applied = 0.0
                    prompt_duration_mask_applied = 0.0
                    prompt_duration_masked_prefix_ratio = 0.0
                    prompt_text_mask_applied = 0.0
                    prompt_text_masked_prefix_ratio = 0.0
                    prompt_text_target_only_applied = 0.0
                    prompt_text_target_only_prefix_ratio = 0.0
                    conditioned_ratio = 0.0

                    if prompt_text_target_only_prob > 0 and prompt_boundary_index is not None:
                        apply_target_text_only = (
                            torch.rand(prompt_boundary_index.shape[0], device=prompt_boundary_index.device)
                            < prompt_text_target_only_prob
                        )
                        text_inputs, token_durations, token_duration_mask, sliced_prefix_ratios = (
                            slice_to_target_text_prefix(
                                text_inputs=text_inputs,
                                token_durations=token_durations,
                                token_duration_mask=token_duration_mask,
                                prompt_boundary_index=prompt_boundary_index,
                                apply_target_only=apply_target_text_only,
                            )
                        )
                        prompt_text_target_only_applied = float(apply_target_text_only.float().mean().item())
                        if sliced_prefix_ratios:
                            prompt_text_target_only_prefix_ratio = float(
                                sum(sliced_prefix_ratios) / len(sliced_prefix_ratios)
                            )

                    if prompt_text_mask_prob > 0 and prompt_boundary_index is not None:
                        apply_prompt_text_mask = (
                            torch.rand(prompt_boundary_index.shape[0], device=prompt_boundary_index.device)
                            < prompt_text_mask_prob
                        )
                        text_inputs, masked_text_prefix_ratios = mask_prompt_text_prefix(
                            text_inputs=text_inputs,
                            prompt_boundary_index=prompt_boundary_index,
                            apply_prompt_mask=apply_prompt_text_mask,
                            mask_token=prompt_text_mask_token,
                        )
                        prompt_text_mask_applied = float(apply_prompt_text_mask.float().mean().item())
                        if masked_text_prefix_ratios:
                            prompt_text_masked_prefix_ratio = float(
                                sum(masked_text_prefix_ratios) / len(masked_text_prefix_ratios)
                            )

                    if token_duration_mask is not None:
                        token_duration_mask = token_duration_mask.clone()
                        if (
                            prompt_duration_mask_prob > 0
                            and prompt_boundary_index is not None
                            and token_durations is not None
                        ):
                            apply_prompt_mask = (
                                torch.rand(token_duration_mask.shape[0], device=token_duration_mask.device)
                                < prompt_duration_mask_prob
                            )
                            masked_prefix_ratios = []
                            for sample_idx in range(token_duration_mask.shape[0]):
                                if not bool(apply_prompt_mask[sample_idx].item()):
                                    continue
                                boundary = int(prompt_boundary_index[sample_idx].item())
                                boundary = max(0, min(boundary, token_duration_mask.shape[1]))
                                if boundary <= 0:
                                    continue
                                token_duration_mask[sample_idx, :boundary, :] = 0.0
                                masked_prefix_ratios.append(boundary / max(1, token_duration_mask.shape[1]))
                            prompt_duration_mask_applied = float(apply_prompt_mask.float().mean().item())
                            if masked_prefix_ratios:
                                prompt_duration_masked_prefix_ratio = float(
                                    sum(masked_prefix_ratios) / len(masked_prefix_ratios)
                                )
                        if duration_dropout_prob > 0 and token_durations is not None:
                            drop_samples = (
                                torch.rand(token_duration_mask.shape[0], device=token_duration_mask.device)
                                < duration_dropout_prob
                            )
                            if drop_samples.any():
                                token_duration_mask[drop_samples] = 0.0
                            duration_dropout_applied = float(drop_samples.float().mean().item())
                        conditioned_ratio = float((token_duration_mask.sum(dim=(1, 2)) > 0).float().mean().item())

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_step)

                    loss, cond, pred = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
                        noise_scheduler=self.noise_scheduler,
                        token_durations=token_durations,
                        token_duration_mask=token_duration_mask,
                    )
                    duration_metrics = self.accelerator.unwrap_model(self.model).transformer.get_duration_metrics()
                    duration_metrics["duration_condition/dropout_applied_ratio"] = duration_dropout_applied
                    duration_metrics["duration_condition/prompt_mask_applied_ratio"] = prompt_duration_mask_applied
                    duration_metrics["duration_condition/prompt_masked_prefix_ratio"] = prompt_duration_masked_prefix_ratio
                    duration_metrics["text_condition/prompt_mask_applied_ratio"] = prompt_text_mask_applied
                    duration_metrics["text_condition/prompt_masked_prefix_ratio"] = prompt_text_masked_prefix_ratio
                    duration_metrics["text_condition/target_only_applied_ratio"] = prompt_text_target_only_applied
                    duration_metrics["text_condition/target_only_prefix_ratio"] = prompt_text_target_only_prefix_ratio
                    duration_metrics["duration_condition/conditioned_sample_ratio"] = conditioned_ratio
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    scalar_logs = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                    for key, value in duration_metrics.items():
                        scalar_logs[key] = float(value.item() if torch.is_tensor(value) else value)
                    self.accelerator.log(scalar_logs, step=global_step)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_step)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)
                        for key, value in duration_metrics.items():
                            self.writer.add_scalar(key, float(value.item() if torch.is_tensor(value) else value), global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())


                did_save_checkpoint = False
                sample_checkpoint_source = None

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    sample_checkpoint_source = self.save_checkpoint(global_step)
                    did_save_checkpoint = True

                if global_step % self.last_per_steps == 0:
                    last_checkpoint_source = self.save_checkpoint(global_step, last=True)
                    if sample_checkpoint_source is None:
                        sample_checkpoint_source = last_checkpoint_source
                    did_save_checkpoint = True

                if self.log_samples and did_save_checkpoint:
                    save_logged_samples(global_step, batch, mel_lengths, text_inputs, sample_checkpoint_source)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if exists(self.max_updates) and global_step >= self.max_updates:
                    if not did_save_checkpoint:
                        sample_checkpoint_source = self.save_checkpoint(global_step, last=True)
                        if self.log_samples:
                            save_logged_samples(global_step, batch, mel_lengths, text_inputs, sample_checkpoint_source)
                    self.accelerator.end_training()
                    return

                del cond
                del pred
                del loss
                del mel_spec
                del mel_lengths
                del text_inputs
                del token_durations
                del token_duration_mask
                del prompt_boundary_index
                del batch

        self.save_checkpoint(global_step, last=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.accelerator.end_training()
