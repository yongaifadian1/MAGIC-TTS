# Fine-Tuning MAGIC-TTS

This repository includes a local fine-tuning entrypoint and does not depend on
external source trees such as a separate `F5R-TTS` checkout.

## What You Need

1. A prepared dataset directory.
2. Local pretrained assets under `pretrained/`.
3. A downloaded MAGIC-TTS checkpoint under `checkpoints/`.
4. A CUDA-capable PyTorch environment.

The default expected main training dataset location is:

```text
data/b150_public
```

For a lightweight smoke-test split, the repository already includes:

```text
data/b150_public_eval_smoke_100_pkg
```

The default training initialization checkpoint is:

```text
checkpoints/magictts_36k.pt
```

An optional F5-TTS base checkpoint can still be used through:

```text
pretrained/F5TTS_Base/model_1200000.safetensors
```

and passed explicitly with `--pretrained-ckpt` when needed.

## Dataset Format

The fine-tuning script expects the same prepared dataset format used by
`duration_dataset.py` with `dataset_type=CustomDatasetPath`.

At minimum, your dataset directory should contain:

```text
data/b150_public/
  raw/
  duration.json
```

or:

```text
data/b150_public/
  raw.arrow
  duration.json
```

`duration.json` must contain a `duration` array aligned with the dataset rows.
Each row should include:

- `audio_path`
- `text`
- `duration`
- `token_durations`

Optional fields:

- `raw_text`

## Recommended Public Split

For the user-facing B@150 release, use:

```text
public_train = full_b150_high_confidence - official_b150_test_100
```

This repository does not hardcode the held-out 100-sample list yet. Instead, it
expects the released public dataset to already exclude that official test set.

For user bring-up, a separate `public_eval` smoke split can be published in the
same format. That split is useful for verifying that data loading, checkpoint
init, logging, and a short fine-tune run all work end to end, but it should not
be treated as the main public training split.

Maintainers can rebuild the current 100-sample smoke package from an internal
`selected_samples.jsonl` source file with:

```bash
python training/export_public_eval_smoke.py \
  --selected-samples /path/to/selected_samples.jsonl \
  --output-dir data/b150_public_eval_smoke_100_pkg \
  --overwrite
```

## Setup

```bash
conda create -n magictts python=3.10 -y
conda activate magictts
bash scripts/setup.sh
```

## Run Fine-Tuning

Minimal example:

```bash
bash scripts/run_finetune.sh \
  --dataset data/b150_public \
  --run-name b150_public_sft \
  --batch-size 30000 \
  --warmup-updates 1000 \
  --save-updates 1000 \
  --last-updates 500
```

Smoke-test example:

```bash
bash scripts/run_finetune.sh \
  --dataset data/b150_public_eval_smoke_100_pkg \
  --run-name smoke_eval100 \
  --max-updates 50 \
  --save-updates 50 \
  --last-updates 50
```

Resume automatically from:

```text
checkpoints/finetune_runs/<run-name>/model_last.pt
```

Initialize from a different MAGIC-TTS training checkpoint:

```bash
bash scripts/run_finetune.sh \
  --dataset data/b150_public \
  --run-name b150_public_plus10k \
  --init-model-ckpt checkpoints/magictts_36k.pt
```

Initialize from the F5-TTS base checkpoint instead:

```bash
bash scripts/run_finetune.sh \
  --dataset data/b150_public \
  --run-name b150_public_from_base \
  --init-model-ckpt "" \
  --pretrained-ckpt pretrained/F5TTS_Base/model_1200000.safetensors
```

## Useful Flags

- `--max-updates`
- `--duration-dropout-prob`
- `--prompt-duration-mask-prob`
- `--prompt-text-mask-prob`
- `--prompt-text-target-only-prob`
- `--logger tensorboard|wandb|none`
- `--log-samples`

## Outputs

Checkpoints:

```text
checkpoints/finetune_runs/<run-name>/
```

TensorBoard logs:

```text
runs/<run-name>/
```

## Notes

- This fine-tuning path uses only code vendored inside this repository.
- It assumes the dataset has already been prepared into the release format.
- If you want to open-source the full public B@150 training set later, ship the
  prepared dataset or a reproducible export pipeline separately.
