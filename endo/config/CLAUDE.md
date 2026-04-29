# `endo/config/` — Pydantic experiment-config tree

Implements PRD §3 — one file per experiment under `experiments/<name>.py`, composed from these sub-configs.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the public config types. |
| `experiment.py` | `ExperimentConfig` — the root composer (`uuid`, `name`, `description`, `tags`, sub-configs, `seed`). `to_yaml` / `from_yaml` round-trip. `diff(other)` returns dotted paths where two configs disagree (used by the CLI drift guard). `short_uuid` (8 hex chars) seeds the run-dir name. |
| `loader.py` | `load_experiment(path)` dynamically imports an `experiments/*.py` file and returns its module-level `experiment: ExperimentConfig`. Raises if the module is missing the symbol or has the wrong type. |
| `paths.py` | `PathsConfig(data_root, cache_root, runs_root, lesion_bank=None)`. |
| `model.py` | `ModelConfig` — backbone, FPN channels/strides, head class count + stacked convs, aux seg channels, etc. |
| `training.py` | `TrainingConfig` — `max_epochs`, `batch_size`, `num_workers`, LR schedule, AMP precision (`bf16-mixed | 16-mixed | 32-true`), `gradient_clip_val`, EMA decay, slice window, target input shape. |
| `sampler.py` | `SamplerConfig` — `epoch_mode` (`fixed_count`), `samples_per_epoch`, pos-frac decay schedule, neg-in-pos-vol share, hard-pool substitution rate + start epoch, deep-eval start epoch + refresh interval. |
| `augmentation.py` | `PasteConfig`, `GeometricConfig`, `IntensityConfig`, and the `AugmentationConfig` composer. |
| `gru.py` | `GRUConfig` — input/hidden dims, bidirectional, dropout, training epochs / lr / weight_decay / aux_loss_weight. |
| `eval.py` | `EvalConfig` — bootstrap N + seed, threshold grids (large vs small), WBF IoU threshold + skip-box threshold, box-size split (mm), inference batch size. |

## Contracts

- **One `experiment: ExperimentConfig`** per `experiments/*.py`. The loader enforces this.
- **`uuid` is uuid4** — the `_check_uuid_format` validator rejects malformed identifiers.
- **`ExperimentConfig.diff` is the basis of the drift guard.** Any new sub-config field must round-trip through `to_yaml` / `from_yaml` cleanly so `diff` doesn't false-positive.
- **`paths.lesion_bank=None` means "use `cache_root/lesion_banks/current.pkl`"** — `endo.augmentation.transform.TrainAugmentation` checks this.

## Invariants

- `experiment.short_uuid` (first 8 chars of the uuid hex with hyphens stripped) determines `runs/<name>_<short_uuid>/`. Don't change the slicing without coordinating with the run-dir bootstrap in `endo/cli/run_experiment.py`.
- Composition only — no inheritance among configs (PRD §3.1.5).

## Don't

- Don't add CLI overrides — to change a value, copy the experiment file with a new uuid.
- Don't store derived state in configs (file mtimes, cache hashes, etc.) — those go in `runs/<exp>/provenance.json`.
