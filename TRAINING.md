# Fine-Tuning MAGIC-TTS

## What You Need

1. A prepared dataset directory.
2. Local pretrained assets under `pretrained/`.
3. A downloaded MAGIC-TTS checkpoint under `checkpoints/`.
4. A CUDA-capable PyTorch environment.

The default expected main training dataset location is:

```text
data/b150_public
```

The public release is expected to contain raw audio only. Before fine-tuning,
prepare a local dataset directory with the bundled data-preparation scripts in
this repository, then point `run_finetune.sh` to that prepared directory.

For a lightweight smoke-test split, first download the raw-audio release from
Hugging Face:

```text
https://huggingface.co/datasets/maimai11/b150_official_test_100
```

The default training initialization checkpoint is:

```text
checkpoints/magictts_36k.pt
```

## Dataset Format

The fine-tuning script expects a prepared local dataset generated from the raw
audio release, using the same format consumed by `duration_dataset.py` with
`dataset_type=CustomDatasetPath`.

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

If you use `raw.arrow`, store absolute file paths in each row's `audio_path`
field.

`duration.json` must contain a `duration` array aligned with the dataset rows.
Each row should include:

- `audio_path`
- `text`
- `duration`
- `token_durations`

Optional fields:

- `raw_text`

## Dataset Preparation

For a concrete end-to-end example, first download the
`b150_official_test_100` smoke split from Hugging Face:

```text
https://huggingface.co/datasets/maimai11/b150_official_test_100
```

Assume the dataset is available locally at
`/path/to/b150_official_test_100`, with:

- `selected_samples.exported.jsonl`
- `raw/audio/...`

Then prepare a local training-ready dataset with:

```bash
bash scripts/prepare_finetune_dataset.sh \
  --input-jsonl /path/to/b150_official_test_100/selected_samples.exported.jsonl \
  --audio-root /path/to/b150_official_test_100/raw \
  --output-dir data/smoke_eval100_prepared
```

This produces a prepared dataset under `data/smoke_eval100_prepared`
containing `raw.arrow`, `duration.json`, and `vocab.txt`.

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
  --dataset data/smoke_eval100_prepared \
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

- It assumes the dataset has already been prepared into the release format.
- If you want to open-source the full public B@150 training set later, ship the
  prepared dataset or a reproducible export pipeline separately.
