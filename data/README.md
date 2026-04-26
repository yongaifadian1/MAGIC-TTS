# Data Placeholder

Download raw-audio releases first, then prepare the local fine-tuning dataset
under this directory with the bundled data-preparation scripts.

Recommended prepared layout:

```text
data/
  b150_public/
    raw/
    duration.json
  smoke_eval100/
    raw/
    duration.json
```

If you publish a lightweight smoke-test split for users, keep the raw-audio
release separate from the main public training split and prepare it locally
with the same scripts before training.

For user smoke testing, download the released 100-sample raw-audio split from:

- https://huggingface.co/datasets/maimai11/b150_official_test_100
