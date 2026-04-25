# Data Placeholder

Place the user-facing prepared fine-tuning dataset under this directory.

Recommended default layout:

```text
data/
  b150_public/
    raw/
    duration.json
  b150_public_eval_smoke_100_pkg/
    raw/
    duration.json
```

This public dataset is expected to be based on the high-confidence B@150 split
with the official 100-sample B@150 evaluation subset removed.

If you publish a lightweight smoke-test split for users, keep it separate from
the main public training split and name it explicitly, for example
`b150_public_eval_smoke_100_pkg`.

The packaged smoke split already included in this repository is intended to be
self-contained and publishable as-is. Its `raw/audio/...` tree contains copied
audio assets, and row-level `audio_path` values are relative to `raw/`.
