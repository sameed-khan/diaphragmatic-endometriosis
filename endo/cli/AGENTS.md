# `endo/cli/` — single argparse entrypoint

Implements PRD §4.1 — one process per `(experiment, fold)`, no Hydra, no CLI overrides for config knobs (only orchestration knobs).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `run_experiment.py` | The CLI. Subcommands: `train`, `train_gru`, `eval`, `predict_holdout`, `viz`, `smoke`, `qc_paste`. Builds `runs/<name>_<uuid8>/{experiment.yaml, experiment.py, provenance.json}` on first invocation; subsequent runs validate against `experiment.yaml` (drift guard) and require `--force-resync` to override. Per-fold `_train_one_fold` wires `EmaCallback`, `ModelCheckpoint(monitor=val/slice_auroc)`, `PeriodicDeepEvalCallback`, optional `LearningRateMonitor` (only when `--wandb`), and `pl.Trainer` at the precision/clip/log settings from `experiment.training`. WandB OFF by default (A.9). |

## Contracts

- **`experiments/<name>.py` MUST define a top-level `experiment: ExperimentConfig`** — `endo.config.load_experiment` enforces this.
- **`runs/<name>_<uuid8>/experiment.yaml`** is the single source of truth for "what config did this run use." Editing `experiments/<name>.py` after the first invocation triggers a drift error unless `--force-resync` is passed.
- **`provenance.json.fold_status`** is updated atomically: `pending → running → complete | failed` (I.8.7). Multi-fold runs share one `provenance.json`.
- **Holdout discipline** (A.5): only `predict_holdout` instantiates a DataModule with `allow_holdout=True`. Every other subcommand runs with the default `allow_holdout=False`. Each `predict_holdout` invocation produces a fresh `holdout/run_<ts>_<uuid8>/` subdir with `eval_report.csv` and `invocation.json`.
- **Smoke** delegates to `scripts.smoke_train.run_smoke` so the CLI surface stays small.
- **WandB**: opt-in via `--wandb` on `train`. Logger `project="diaphragmatic-endometriosis"`, group `f"{experiment.name}_{experiment.short_uuid}"`, name `f"fold{fold}"`.

## Invariants enforced

- I.8.6 (drift guard) — `_bootstrap_run_dir` compares the in-memory `ExperimentConfig` to the on-disk YAML and aborts on mismatch unless `--force-resync`.
- I.8.7 (atomic fold-status updates) via `endo.utils.provenance.update_fold_status`.
- I.9.3 (only `predict_holdout` toggles `allow_holdout=True`) — keep the comment in `endo/eval/run_eval.run_holdout_inference` if you refactor.

## Don't

- Don't add config knobs as CLI flags. To change LR / batch size / paste config, copy `experiments/<name>.py` to a new file with a new uuid.
- Don't re-enable `LearningRateMonitor` unconditionally — it errors at trainer init when no logger is registered (`logger=False` is the default).
