# Data Placeholder

Place the user-facing prepared fine-tuning dataset under this directory after
downloading it from the desired release source.

Recommended default layout:

```text
data/
  b150_public/
    raw/
    duration.json
  b150_official_test_100/
    raw/
    duration.json
```

This public dataset is expected to be based on the high-confidence B@150 split
with the official 100-sample B@150 evaluation subset removed.

If you publish a lightweight smoke-test split for users, keep it separate from
the main public training split and name it explicitly, for example
`b150_public_eval_smoke_100_pkg`.

For user smoke testing, download the released 100-sample split from:

- https://huggingface.co/datasets/maimai11/b150_official_test_100
