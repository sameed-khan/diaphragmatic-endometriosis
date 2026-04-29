# `experiments/` — one Pydantic file per experiment

Implements PRD §3 — each `.py` declares a top-level `experiment: ExperimentConfig`. Loaded by `endo.config.load_experiment(path)`. Composition only; no inheritance.

## Files

| File | Purpose |
|---|---|
| `baseline_rtmdet_p2.py` | **Production Week-1 baseline.** ConvNeXt-tiny + 4-level FPN (P2 + P3-P5) + RTMDet head + aux seg head. 60 epochs × 6000 samples/epoch, bf16-mixed, batch_size=8, base_lr=2e-4, weight_decay=0.05, warmup 1 epoch, cosine to min_lr=1e-6, EMA 0.999. Paste augmentation `p_any_paste=0.5`, n_paste_max=7. GRU rescorer enabled at eval. Targets: volume AUROC ≥ 0.80, sens@2FP ≥ 0.70 on 5-fold CV. |
| `smoke.py` | Tiny smoke-test config — `max_epochs=2`, `samples_per_epoch=100`, `batch_size=4`, `bf16-mixed`, `deep_eval_start_epoch=99` (effectively disabled). Used by `scripts/smoke_train.py` to validate plumbing in <5 min. |
| `quickeval.py` | **Eval-pipeline test config** — full fold-0 over 5 epochs × 1000 samples/epoch, `precision="32-true"` (fp32) to dodge the bf16 NaN sensitivity surfaced during validation. Produces a real `best.ckpt` + `deep_eval/epoch{n}_val.npz` so Components 6.5/7/8 can be exercised end-to-end on a real model. NOT for production. |

## Contracts

- **Each file MUST define `experiment: ExperimentConfig`** at module level. The loader raises if missing.
- **`uuid` MUST be uuid4** and unique across experiments — it's part of the run-dir name (`<name>_<short_uuid>/`).
- **Drift guard**: editing an experiment file after a run starts triggers an error from the CLI bootstrap unless `--force-resync` is passed.
- **Precision selection**: bf16-mixed is the production default but has shown intermittent NaN under longer runs. quickeval pins `32-true` as a band-aid; before kicking off baseline_rtmdet_p2 across all 5 folds, profile bf16 stability and consider switching to `16-mixed` (fp16 with grad scaler) — see `agent/complete_spec/IMPLEMENTATION_LOG.md` "Open recommendations".

## Don't

- Don't override config values via CLI flags — copy the file with a new uuid.
- Don't put logic in experiment files. They should be pure config declarations.
- Don't reuse a uuid across files. Both `endo.config.experiment._check_uuid_format` and the run-dir machinery assume uniqueness.
