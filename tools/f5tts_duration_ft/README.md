# Dataset Preparation Tools

This directory bundles the fine-tune dataset preparation scripts that were used
internally before the beta split was created.

For a standard `audio_path + text` JSONL manifest, the recommended flow is:

1. Run `prepare_emilia_1nv_mfa_shards.py` to shard the manifest for MFA.
2. Run `run_mfa_alignment_shard.py` on each shard to generate `*.words.json`
   sidecars.
3. Run `prepare_emilia_1nv_merged_worddur.py` to build `raw.arrow` and
   `duration.json`.

If you want a single entrypoint, use:

```bash
bash scripts/prepare_finetune_dataset.sh \
  --input-jsonl /path/to/manifest.jsonl \
  --text-field text \
  --audio-root /path/to/raw_audio_root \
  --output-dir data/b150_public
```
