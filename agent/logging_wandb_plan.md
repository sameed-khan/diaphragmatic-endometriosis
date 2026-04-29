# Plan — comprehensive logging + W&B tracking

Status: **APPROVED — ready for implementation on branch `wandb-logging`**.
Scope: turn the existing partial logging (a few `self.log` calls + `[INFO]` lines) into a uniform, configurable observability layer that covers **every stage** of the pipeline (`train` detector → `train_gru` → `eval` → `predict_holdout` → `viz`) with both human-readable file/stdout logs and W&B artifacts/metrics/curves.

This plan does **not** make changes to existing core code; the implementer (a fresh agent on the `wandb-logging` branch) does the edits and runs the end-to-end gate test in §9.

---

## 1. Decisions (locked)

These are the user's resolutions to the 19 clarifying questions; deviation requires explicit re-approval.

### A. Config & control surface

| # | Decision | Notes |
|---|---|---|
| 1 | `LoggingConfig` lives at `endo/config/logging.py` and is wired into `ExperimentConfig` | Drift-exempt fields (see §7 risks) |
| 2 | Different W&B runs per `(fold × stage)` | naming scheme below |
| 3 | Both config and CLI control: `LoggingConfig.wandb.enabled` is the default; `--wandb / --no-wandb` overrides for one-off runs (never modifies experiment.yaml) | **Document this in top-level `CLAUDE.md`** as part of the CLI contract |
| 4 | Default mode `"online"`, configurable to `"offline"`/`"disabled"` | not on HPC anymore |
| 5 | Default log level `INFO` on console + file; `-v` → `DEBUG` | |
| 6 | tqdm stays on stdout; structured `logging` records go to **both** stdout and a per-fold rotating file `<fold_dir>/run.log` | top-level run also gets `<run_dir>/run.log` |

**W&B run naming scheme (locked):**
- `project`: from `LoggingConfig.wandb.project` (default `"diaphragmatic-endometriosis"`).
- `entity`: from `LoggingConfig.wandb.entity` (default `None` → uses API key's default entity).
- `group`: `LoggingConfig.wandb.group` if set, else `f"{experiment_name}_{short_uuid}"`.
- `name`: `LoggingConfig.wandb.run_name` if set, else `f"{experiment_name}/fold{fold}"` for detector and gru stages, `f"{experiment_name}/cv_summary"` for eval, `f"{experiment_name}/holdout"` for holdout.
- `tags`: union of `experiment.tags` (dict values) + `{"fold": str(fold), "stage": <"detector"|"gru"|"eval"|"holdout"|"viz">}` + `LoggingConfig.wandb.tags`.

**`experiment_name` and `run_name` overrideable from config:** `LoggingConfig.wandb.experiment_name` (defaults to `experiment.name`) and `LoggingConfig.wandb.run_name` (defaults to the per-stage scheme above).

### B. What gets logged

| # | Decision |
|---|---|
| 7 | **Validation-prediction image PNGs** are NOT a W&B image-panel; they are **a directory of PNGs uploaded as a W&B artifact**. Off by default. Configurable via `LoggingConfig.viz.log_during_training: bool = False` and `LoggingConfig.viz.log_every_n_epochs: int = N`. Output paths: during training → `<fold_dir>/viz/epoch_{n}/*.png`; post-training (always-on, runs after fit completes) → `<fold_dir>/viz/epoch_post-train/*.png` and includes **all validation patients**. |
| 8 | Gradient/weight histograms: **off**. |
| 9 | System metrics (GPU/CPU/RAM): **on**. |
| 10 | Augmentation samples: log once at epoch 0; **no `--debug-aug` CLI flag** — the toggle is a config-only field `LoggingConfig.aug.log_samples: Literal["never","epoch0","always"] = "epoch0"`. The dedicated **e2e test config** (see §9) sets this to its desired value. |
| 11 | Hard-pool snapshots: log scalar count + score histogram per refresh; do not upload the JSON unless `LoggingConfig.wandb.upload_hard_pool_snapshots = True`. |
| 12 | Lesion-bank stats: logged once at run start as W&B summary, not per-step. |

### C. Artifacts to upload

| # | Decision |
|---|---|
| 13 | Upload `best.ckpt` only as a versioned W&B artifact (alias `best`), gated by `LoggingConfig.wandb.upload_checkpoints: bool`. **Default `True` for production**, but **the e2e-testing config in §9 sets this to `False`.** |
| 14 | Upload eval reports (`eval_report.csv`, `eval_thresholds.json`, FROC PNGs, stratified breakdowns) when wandb is enabled — default `True`. |
| 15 | Visualization upload at end of each fold: a **random sample of 20 TP, 20 FP, and 20 FN slices** rendered through `endo/viz/render.py`, uploaded as a single directory artifact (`viz-fold{N}`), gated by `LoggingConfig.wandb.upload_viz_artifacts: bool = True`. Sampling is reproducible per fold (seeded with `experiment.seed + 1000*fold`). |

### D. Cross-fold rollup

| # | Decision |
|---|---|
| 16 | CV summary is its own W&B run named `<exp>/cv_summary` in the same group. |
| 17 | Holdout is its own W&B run named `<exp>/holdout` in the same group. |

### E. Smoke / quickeval

| # | Decision |
|---|---|
| 18 | Smoke does **not** log to W&B by default. Mechanism: the smoke experiment config sets `LoggingConfig.wandb.enabled = False`. **There is no special CLI handling** — the operator does not have to remember a flag. (Smoke can still be wandb-enabled by editing its config, but the default is off and config-locked.) |
| 19 | Quickeval: same as smoke — default off via its own config. |

---

## 2. `LoggingConfig` schema (locked)

```python
# endo/config/logging.py
from typing import Literal
from pydantic import BaseModel, Field

class WandbConfig(BaseModel):
    enabled: bool = False                  # global gate
    project: str = "diaphragmatic-endometriosis"
    entity: str | None = None              # None → API key's default
    experiment_name: str | None = None     # None → experiment.name
    run_name: str | None = None            # None → per-stage scheme
    group: str | None = None               # None → "{exp}_{short_uuid}"
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "online"
    log_system_metrics: bool = True
    upload_checkpoints: bool = True
    upload_eval_reports: bool = True
    upload_viz_artifacts: bool = True
    upload_hard_pool_snapshots: bool = False

class FileLoggingConfig(BaseModel):
    level_console: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    level_file: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    rotate_max_bytes: int = 50_000_000
    rotate_backups: int = 3

class VizLoggingConfig(BaseModel):
    log_during_training: bool = False
    log_every_n_epochs: int = 0            # 0 → only post-training
    n_train_predictions_logged: int = 8    # cap when log_during_training=True
    sample_tp_per_fold: int = 20
    sample_fp_per_fold: int = 20
    sample_fn_per_fold: int = 20

class AugLoggingConfig(BaseModel):
    log_samples: Literal["never", "epoch0", "always"] = "epoch0"
    n_samples: int = 8

class LoggingConfig(BaseModel):
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbConfig = WandbConfig()
    viz: VizLoggingConfig = VizLoggingConfig()
    aug: AugLoggingConfig = AugLoggingConfig()
    log_every_n_steps: int = 10            # mirrors training.log_every_n_steps if set
```

`ExperimentConfig` gains `logging: LoggingConfig = LoggingConfig()`. The `LoggingConfig` block is **diff-exempt** in `ExperimentConfig.diff(...)` — changing logging settings between resumes does NOT trip the drift guard. This is implemented in step 6 below.

CLI additions to `endo/cli/run_experiment.py`:
- `--wandb / --no-wandb` (mutually exclusive) — overrides `experiment.logging.wandb.enabled`.
- `--wandb-mode {online,offline,disabled}` — overrides `experiment.logging.wandb.mode`.
- `-v` / `-vv` — overrides `experiment.logging.file.level_console`.
- These flags propagate uniformly to all subcommands.

---

## 3. Per-stage metric & artifact catalog

> Implementation reference. Names are final.

### 3.1 `train` (detector — one W&B run per fold, name `<exp>/fold{N}`, tag `stage=detector`)

**Per step (every `LoggingConfig.log_every_n_steps`):**
| key | source |
|---|---|
| `train/loss_total` | already in `endo/lightning_module.py:114` |
| `train/loss_cls`, `loss_bbox`, `loss_aux_seg` | already logged |
| `train/lr` | `LearningRateMonitor` |
| `train/grad_norm` | NEW — `on_after_backward` |
| `train/throughput_samples_per_sec` | NEW — derived from a `StepTimerCallback` |
| `train/skipped_steps_nan` (cumulative) | NEW counter (off the existing NaN guard) |

**Per epoch (training):**
- `train/loss_total_epoch`, `train/seconds_per_epoch`, `train/effective_batch_size` (static).

**Per epoch (validation):**
- `val/loss_total`, `loss_cls`, `loss_bbox`, `loss_aux_seg` (already logged), `val/slice_auroc` (already logged), **NEW** `val/slice_auprc`, `val/secs`.

**Periodic deep-eval (`every deep_eval_refresh_every_epochs` from `deep_eval_start_epoch`):**
- Already logs `deep_eval/val_volume_auroc_coarse`, `deep_eval/val_froc_at_2fp_coarse`. **NEW**: `deep_eval/hard_pool_size`, `deep_eval/val_secs`, `deep_eval/neg_secs`, `deep_eval/n_val_patients_scored`.
- Also: hard-pool score histogram via `wandb.Histogram` per refresh.

**Sampler diagnostics (per epoch end):**
- `sampler/pos_frac_actual`, `sampler/hard_pool_substitution_rate_actual`, `sampler/score_ema_n_tracked`.

**Augmentation diagnostics (per epoch end):**
- `aug/paste_attempts_mean`, `aug/paste_oob_clip_frac_mean`, `aug/elastic_skip_rate`.

**EMA diagnostics:**
- `ema/swap_used_in_val` (boolean per val).

**Once-per-run W&B summary:**
- `params_total_M`, `trainable_M`.
- `n_train_volumes`, `n_val_volumes`, `n_train_slice_index`, `n_val_slice_index`, `pos_neg_volume_ratio_train`, `pos_neg_volume_ratio_val`.
- `lesion_bank_size`, `lesion_size_p50_mm3`, `lesion_size_p95_mm3`.
- GPU name + total memory, CUDA version, PyTorch version.
- Git SHA, hostname, python version (already in `provenance.json` — mirror to W&B summary).

**Artifacts uploaded:**
- `experiment.yaml` + `experiment.py` snapshot (artifact type `config`, alias `current`).
- `provenance.json` (artifact type `provenance`).
- `best.ckpt` (artifact type `model`, alias `best`) — gated by `LoggingConfig.wandb.upload_checkpoints`.
- 20 TP / 20 FP / 20 FN sampled prediction PNGs (artifact `viz-fold{N}`) — gated by `LoggingConfig.wandb.upload_viz_artifacts`.

**Validation-prediction PNG directories (when training-time viz on):**
- During training: `<fold_dir>/viz/epoch_{n}/*.png` written every `LoggingConfig.viz.log_every_n_epochs`, capped at `n_train_predictions_logged`. Optionally uploaded as artifact `viz-fold{N}-epoch{n}`.
- Post-training: `<fold_dir>/viz/epoch_post-train/*.png` for **all validation patients** — always written when wandb enabled. Uploaded as artifact `viz-fold{N}-post-train` (full directory).

### 3.2 `train_gru` (one W&B run per fold, name `<exp>/fold{N}`, tag `stage=gru`)

- `gru/loss_train`, `gru/loss_val`, `gru/val_auroc` per epoch.
- `gru/aux_loss_weight_actual`.
- Once: `gru/sequence_length_distribution` histogram.
- Artifacts: `gru_state.pt` (uploaded; small).

### 3.3 `eval` (one W&B run, name `<exp>/cv_summary`, tag `stage=eval`)

- Per-fold table: `fold`, `large_thr`, `small_thr`, `sens_at_2fp`, `volume_auroc`, `volume_auprc`, `n_lesions_total`, `n_lesions_hit`.
- Pooled CV scalars: `cv/sens_at_2fp_pooled`, `cv/volume_auroc_pooled`, `cv/volume_auprc_pooled`, `cv/best_large_thr_pooled`, `cv/best_small_thr_pooled`.
- Bootstrap CI: `cv/<metric>_lo`, `_hi`, `_n_bootstrap`.
- Curves: FROC (per-fold + pooled), slice & volume ROC, slice & volume PR, threshold heatmap.
- Stratified table by `eval.stratify_keys`.
- Artifacts: `eval_report.csv`, `eval_thresholds.json`, FROC PNGs (artifact type `eval-report`).

### 3.4 `predict_holdout` (one W&B run, name `<exp>/holdout`, tag `stage=holdout`)

- Per-patient predictions table (no labels).
- Aggregate score histogram per fold + ensembled.
- Artifact: predicted boxes parquet.

### 3.5 `viz` (no live metrics)

- Upload a single directory artifact of rendered PNGs, gated by `LoggingConfig.wandb.upload_viz_artifacts`.

### 3.6 `smoke` / `quickeval`

- Default `LoggingConfig.wandb.enabled = False` in their config files; structured stdout/file logs still write.

---

## 4. Stdout/file log shape (representative)

```
2026-04-29 10:00:01 [INFO] endo.cli: ===== run start =====
2026-04-29 10:00:01 [INFO] endo.cli: experiment=baseline-rtmdet-p2 uuid=b3a7f1e9 short=b3a7f1e9
2026-04-29 10:00:01 [INFO] endo.cli: run_dir=runs/baseline-rtmdet-p2_b3a7f1e9 folds=[0,1,2,3,4]
2026-04-29 10:00:01 [INFO] endo.cli: git_sha=… host=… python=…
2026-04-29 10:00:01 [INFO] endo.cli: cuda=True device=0 name=A100-SXM4-80GB total=80.0 GiB
2026-04-29 10:00:01 [INFO] endo.cli: wandb=ON project=diaphragmatic-endometriosis run=baseline-rtmdet-p2/fold0 group=baseline-rtmdet-p2_b3a7f1e9 mode=online
2026-04-29 10:00:01 [INFO] endo.cli: ----- fold 0 -----
2026-04-29 10:00:02 [INFO] endo.data: train_pids=N=98 val_pids=N=24 holdout_excluded=True
2026-04-29 10:00:02 [INFO] endo.data: train_slice_index=12345 val_slice_index=3120 pos/neg train=… val=…
2026-04-29 10:00:03 [INFO] endo.model: LesionDetector params=40.33M trainable=40.33M
2026-04-29 10:00:04 [INFO] endo.augmentation: paste_lesion_bank=… (n=…)
2026-04-29 10:00:05 [INFO] endo.cli: starting fit max_epochs=60 bs=8 precision=bf16-mixed
... tqdm bars ...
2026-04-29 10:11:30 [INFO] endo.lm: epoch=0 train/loss=2.18 grad_norm=4.5 throughput=23.1 samp/s wall=11m
2026-04-29 10:14:31 [INFO] endo.lm: epoch=0 val/loss=1.87 val/slice_auroc=0.703 val/slice_auprc=0.41
2026-04-29 10:14:31 [INFO] endo.ckpt: new best epoch=0 val/slice_auroc=0.703 → ckpts/best.ckpt
2026-04-29 11:30:05 [INFO] endo.deep_eval: epoch=10 val_secs=178.2 neg_secs=421.8 hard_pool_size=812
2026-04-29 11:30:05 [INFO] endo.deep_eval: val_volume_auroc_coarse=0.78 sens@2fp_coarse=0.62
... ...
2026-04-29 19:42:11 [INFO] endo.viz: post-train viz: rendered 24 patient files into fold0/viz/epoch_post-train
2026-04-29 19:42:14 [INFO] endo.cli: fold 0 finished in 9h42m best_val_slice_auroc=0.81 ckpt=…/best.ckpt
2026-04-29 19:42:14 [INFO] endo.wandb: synced — link: https://wandb.ai/<entity>/diaphragmatic-endometriosis/runs/<id>
```

Properties:
- Every line is timestamped + logger-named (`grep`-friendly).
- Lightning's tqdm bars stay on stdout, NOT in the file log.
- Each fold gets `<fold_dir>/run.log` (rotating 50 MB × 3 backups).
- Top-level run gets `<run_dir>/run.log` for cross-fold lines.

---

## 5. W&B dashboard shape

**Per fold (`<exp>/fold{N}`):**
- Charts: all `train/*`, `val/*`, `deep_eval/*`, `sampler/*`, `aug/*`, `ema/*`, `lr` curves vs step and vs epoch.
- Custom panel: small-multiples of the three loss components.
- Custom panel: `val/slice_auroc` and `deep_eval/val_volume_auroc_coarse` overlaid.
- System panel: GPU mem, GPU util, CPU, RAM (auto).
- Summary card: params, dataset sizes, GPU, git SHA, fold, best metric, total wall clock.
- Artifacts tab: `config`, `provenance`, `model:best` (when on), `viz-fold{N}` (with sampled TP/FP/FN PNGs).

**Per group (one experiment):**
- All folds visible side-by-side (group by group → fold).
- A `cv_summary` run with FROC + stratified table.
- Optionally a `holdout` run.

---

## 6. Implementation handoff (the changes the next agent will make)

> **Branch:** `wandb-logging` (cut from `master` after the merge in §10).
> Commit the changes incrementally. Each commit should pass `uv run pytest -q` (existing tests stay green) and add new tests for new code paths.

### 6.1 New files

```
endo/config/logging.py                  # LoggingConfig, WandbConfig, FileLoggingConfig, VizLoggingConfig, AugLoggingConfig
endo/utils/logging_setup.py             # setup_logging(file_cfg, run_dir, fold) → returns the configured root logger
endo/utils/wandb_init.py                # build_wandb_logger(experiment, fold, stage), log_run_summary(...), log_artifact(...)
endo/utils/step_timer.py                # StepTimerCallback (5-line callback) for throughput
endo/utils/aug_counters.py              # AugStatsCallback that drains augmentation pipeline counters at epoch end
experiments/e2e_testing.py              # NEW — end-to-end test config (§9)
tests/utils/test_logging_setup.py       # unit tests for log file rotation, level handling
tests/utils/test_wandb_init.py          # unit tests using wandb's "disabled" mode and a stub
tests/config/test_logging_config.py     # validates serialization + drift-exempt behavior
scripts/preview_logging.py              # OK per directive — emits the §4 log lines without GPU
scripts/preview_wandb_panels.py         # logs synthetic curves to the wandb test project
```

### 6.2 Edits to core code (the implementer makes these — this plan does not)

These are the only places core code is touched, listed in dependency order:

1. **`endo/config/__init__.py`** — re-export `LoggingConfig`.
2. **`endo/config/experiment.py`** — add `logging: LoggingConfig = LoggingConfig()` field; in `ExperimentConfig.diff(...)` exclude the `logging.*` subtree from drift comparison; in `to_yaml/from_yaml` round-trip the new field.
3. **`endo/cli/run_experiment.py`** —
   - Replace `_setup_logging` with `endo.utils.logging_setup.setup_logging(experiment.logging.file, run_dir, fold)`.
   - Replace inline `WandbLogger` ctor with `endo.utils.wandb_init.build_wandb_logger(experiment, fold, stage)` for each subcommand.
   - Add `--wandb`, `--no-wandb`, `--wandb-mode`, `-v`, `-vv` flags via `_add_common`.
   - In `cmd_train`: log run summary (params, dataset sizes, GPU info, git SHA) at fold start; upload `best.ckpt` artifact at fit end (gated).
   - In `cmd_train_gru`: same pattern with `stage="gru"`.
   - In `cmd_eval`: build a `cv_summary` run, log per-fold table + pooled scalars + curves, upload `eval_report.csv` artifact.
   - In `cmd_predict_holdout`: build a `holdout` run, log score histogram, upload predictions parquet.
   - In `cmd_viz`: upload directory artifact at the end.
   - **All five subcommands:** post-training viz step renders `<fold_dir>/viz/epoch_post-train/*.png` for all val patients.
4. **`endo/lightning_module.py`** —
   - Add `on_after_backward` to log `train/grad_norm` (compute via `torch.nn.utils.clip_grad_norm_(... )` returns).
   - Add a `train/skipped_steps_nan` counter incremented inside the existing NaN guard branch (`endo/lightning_module.py:103–110`).
   - Add `val/slice_auprc` next to the existing `val/slice_auroc` computation.
5. **`endo/sampler/periodic_eval.py`** — extend the existing `log_dict` call (line 286) with `deep_eval/hard_pool_size`, `deep_eval/val_secs`, `deep_eval/neg_secs`, `deep_eval/n_val_patients_scored`. Log a `wandb.Histogram` of hard-pool scores when wandb logger is present.
6. **`endo/sampler/weighted.py`** — surface `sampler/pos_frac_actual`, `sampler/hard_pool_substitution_rate_actual` via a small `on_train_epoch_end` hook (a thin Lightning callback owning the sampler reference works equally well to avoid intrusive changes).
7. **`endo/augmentation/transform.py`** — accumulate `paste_attempts`, `oob_clip_frac`, `elastic_skip_rate` counters; expose a `drain_stats()` method called by the new `AugStatsCallback`.
8. **`endo/eval/run_eval.py`** — at end of `run_cv_evaluation`, call a new `endo.utils.wandb_init.log_cv_summary(...)` that takes the per-fold + pooled metrics + curves and posts them.
9. **`endo/gru/train.py`** — keep existing `log.info` line; in addition, when a wandb logger was passed in, call `wandb.log({...})` per epoch.
10. **`endo/viz/run_viz.py`** —
    - Output paths now follow the §1 contract: training-time → `<fold_dir>/viz/epoch_{n}/*.png`; post-training → `<fold_dir>/viz/epoch_post-train/*.png`. Existing callers that rely on the old paths must be migrated (search-and-replace, then run viz tests).
    - Add a `sample_tp_fp_fn(...)` helper that returns 20 each of TP/FP/FN slice indices given val predictions + labels (seed = `experiment.seed + 1000*fold`).
11. **Top-level `CLAUDE.md`** — document the new CLI flags (`--wandb`, `--no-wandb`, `--wandb-mode`, `-v`, `-vv`) and explain the config-vs-CLI override behavior.

### 6.3 Tests to add

- `tests/utils/test_logging_setup.py`: file rotation, level masking, no leaks of the API key.
- `tests/utils/test_wandb_init.py`: building a logger in `mode="disabled"` does not contact the network; tag/group/name composition matches §1; per-stage runs are distinct.
- `tests/config/test_logging_config.py`: yaml round-trip preserves nested logging fields; drift-exempt — changing logging fields between two `ExperimentConfig`s yields empty `diff(...)`.
- `tests/lm/test_grad_norm_logged.py`: a 1-step training stub asserts `train/grad_norm` appears in `trainer.callback_metrics`.
- `tests/eval/test_cv_summary_logged.py`: stub wandb logger captures the right keys.
- `tests/viz/test_viz_paths.py`: `epoch_{n}` and `epoch_post-train` output dirs are created with the expected file count; sampling is deterministic.

### 6.4 Order of work (suggested commits)

1. `feat(config): LoggingConfig + diff-exempt drift handling` (+ tests).
2. `feat(utils): logging_setup + structured per-fold file logs` (+ tests + preview script).
3. `feat(utils): wandb_init wrapper + run name/group/tags scheme` (+ tests + preview script).
4. `feat(cli): --wandb/--no-wandb/--wandb-mode/-v flags + per-stage runs`.
5. `feat(lm): grad_norm + nan_skip_count + val/slice_auprc`.
6. `feat(sampler): periodic_eval extra scalars + hard-pool histogram`.
7. `feat(augmentation): drain_stats counters + AugStatsCallback`.
8. `feat(viz): epoch_{n}/epoch_post-train paths + tp/fp/fn sampler`.
9. `feat(eval): cv_summary + holdout W&B runs + artifact uploads`.
10. `feat(gru): per-epoch W&B logging + checkpoint artifact`.
11. `experiment: e2e_testing config (§9)`.
12. `docs(claude.md): CLI flag contract`.

---

## 7. Risks & open issues

- **Drift guard exemption.** `LoggingConfig` is excluded from `ExperimentConfig.diff(...)`. Implementer: add a unit test that demonstrates two configs differing only in `logging.*` produce an empty diff.
- **W&B offline + checkpoint upload.** Offline mode buffers large artifacts locally; uploading on `wandb sync` later can fail silently if disk fills. The implementer should add a guard that warns when the offline queue exceeds 5 GB.
- **Multi-GPU rank semantics.** All `self.log` calls today implicitly run on rank 0. We use single-GPU; if DDP comes later, `sync_dist=True` is needed for val metrics. Out of scope for this branch.
- **W&B API key handling.** `.env` has `WANDB_API_KEY`; the wrapper must `dotenv.load_dotenv()` once at process start and never log the key value (default in W&B SDK is to redact).
- **tqdm pollution in non-interactive contexts.** When stdout is redirected to a file, tqdm output becomes line-noise. Acceptable trade-off given the per-fold file log captures structured records separately. If that becomes a real annoyance, swap to `RichProgressBar` later.
- **Throughput metric.** Lightning offers no clean per-step time hook. `StepTimerCallback` (5 lines: `on_train_batch_start` records, `on_train_batch_end` diffs, log) is sufficient.

---

## 8. Preview / dry-run scripts

- `scripts/preview_logging.py` — instantiates `LoggingConfig` defaults, calls `endo.utils.logging_setup.setup_logging`, emits a representative sequence of `log.info(...)` lines mimicking the per-fold flow above, and writes the file log. No GPU, no wandb. Use this to eyeball the format before wiring it into `run_experiment.py`.
- `scripts/preview_wandb_panels.py` — uses `WANDB_API_KEY` from `.env` to log synthetic curves to a `diaphragmatic-endometriosis-preview` project (separate from production). Generates fake `train/loss`, `val/slice_auroc`, FROC curve, and a 20-image directory artifact so you can verify the W&B layout looks right before any real training.

---

## 9. End-to-end test (the gate the implementer must pass)

This is the **acceptance test** for the `wandb-logging` branch. It runs after the implementation commits in §6.4 are done and before merging back to `master`.

### 9.1 Test config

**File:** `experiments/e2e_testing.py`

```python
# experiments/e2e_testing.py
"""End-to-end test config for the logging + W&B integration.

Goal: 2 epochs × 1000 samples on fold 0, then holdout eval, all logged to W&B
under experiment "e2e-testing", run "run1" / "run1-holdout". Success criteria
defined in agent/logging_wandb_plan.md §9.3. NOT for production.
"""
from pathlib import Path
from endo.config import (
    AugmentationConfig, EvalConfig, ExperimentConfig, GRUConfig,
    GeometricConfig, IntensityConfig, LoggingConfig, ModelConfig,
    PasteConfig, PathsConfig, SamplerConfig, TrainingConfig,
)
from endo.config.logging import (
    AugLoggingConfig, FileLoggingConfig, VizLoggingConfig, WandbConfig,
)

experiment = ExperimentConfig(
    uuid="00000000-0000-4000-8000-00000000e2e7",
    name="e2e-testing",
    description="2-epoch × 1000-sample end-to-end gate for logging + W&B + holdout.",
    tags={"phase": "e2e-test"},
    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(),
    training=TrainingConfig(
        max_epochs=2,
        batch_size=4,
        num_workers=4,
        base_lr=2e-4,
        warmup_epochs=0,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=1000,
        deep_eval_start_epoch=99,  # disabled — only 2 epochs
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_max=4),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(epochs=1),     # not exercised by this test
    eval=EvalConfig(use_gru=False, bootstrap_n=50),
    logging=LoggingConfig(
        file=FileLoggingConfig(level_console="INFO", level_file="DEBUG"),
        wandb=WandbConfig(
            enabled=True,
            mode="online",
            experiment_name="e2e-testing",
            run_name="run1",                    # detector fold0
            # holdout sub-run will be "run1-holdout" (set by code path)
            upload_checkpoints=False,           # per decision #13
            upload_eval_reports=True,
            upload_viz_artifacts=True,
            upload_hard_pool_snapshots=False,
        ),
        viz=VizLoggingConfig(
            log_during_training=True,
            log_every_n_epochs=1,
            n_train_predictions_logged=4,
            sample_tp_per_fold=20,
            sample_fp_per_fold=20,
            sample_fn_per_fold=20,
        ),
        aug=AugLoggingConfig(log_samples="epoch0", n_samples=4),
    ),
    seed=42,
)
```

Naming detail: the holdout subcommand reads `LoggingConfig.wandb.run_name` and appends `-holdout` if it is set (so run1 → run1-holdout). Default scheme (run_name unset) is unchanged. Document this in the wrapper.

### 9.2 Execution sequence

```bash
# fold 0 detector
uv run -m endo.cli.run_experiment train \
  --experiment experiments/e2e_testing.py --fold 0

# holdout eval (logs to W&B as a separate run, also under e2e-testing group)
uv run -m endo.cli.run_experiment predict_holdout \
  --experiment experiments/e2e_testing.py --ckpts 0
```

(No `--wandb` flag passed — `LoggingConfig.wandb.enabled=True` in the config does the right thing. Verifies the config-driven path.)

### 9.3 Success criteria

The implementer agent **must** verify all of the following before declaring the gate passed; the agent then **stops and asks the human user** to confirm visually in the W&B UI before merging.

**Quantitative (automatable):**
1. `train/loss_total_epoch` for epoch 1 < epoch 0 (monotonic decrease across 2 epochs).
2. `train/loss_cls`, `train/loss_bbox`, `train/loss_aux_seg` each strictly decrease epoch-over-epoch.
3. `val/loss_total` decreases epoch 1 vs epoch 0 (allow ≤ 5 % slack — only 2 epochs).
4. No `train/skipped_steps_nan > 0` (or, if non-zero, document it).
5. `<fold_dir>/run.log` exists and is non-empty; the structured log format from §4 is present.
6. `<fold_dir>/viz/epoch_0/`, `<fold_dir>/viz/epoch_1/`, and `<fold_dir>/viz/epoch_post-train/` exist with at least 1 PNG each.
7. The W&B run `e2e-testing/run1` exists in project `diaphragmatic-endometriosis` and has a non-empty history including the metric keys listed in §3.1.
8. The W&B run `e2e-testing/run1-holdout` exists with the holdout score histogram + predictions table.
9. Both runs share the same `group` (`e2e-testing_<short_uuid>`).
10. `viz-fold0` artifact attached to the detector run with at least 60 PNGs (20 TP + 20 FP + 20 FN).
11. **No `best.ckpt` artifact is uploaded** (per decision #13 + e2e config).
12. Eval report artifact is uploaded.

**Qualitative (the implementer dumps these for the human reviewer):**
- Link to the detector W&B run.
- Link to the holdout W&B run.
- A screenshot or text dump of the W&B "Files" tab listing artifacts.
- `tail -200 <fold_dir>/run.log` printed to stdout for sanity.
- A list of every top-level metric key seen in W&B (the implementer can scrape it via the W&B API).

**Final gate:** the implementer agent **prints the W&B run links + a short summary**, sets a `wandb-logging` PR description with the same checklist, and **stops**. The human user opens the dashboard and confirms (a) loss curves look sane, (b) all metric keys exist, (c) artifacts present. Only after that confirmation should the branch be merged.

If any quantitative check fails: the implementer fixes the issue, re-runs the e2e test, and reports again — does NOT merge to master with a failing gate.

---

## 10. Branch + merge order (operational notes for the implementer)

1. Start from `master` after the codex-audit merges have landed (the previous agent has already done this — verify `git log master --oneline | head` shows the eval-correctness and aug-perf commits).
2. `git checkout -b wandb-logging`.
3. Make the §6 changes in the order in §6.4, one commit per step.
4. Run `uv run pytest -q` after each commit (existing tests stay green).
5. Run the §9 e2e test.
6. On success, push and open a PR against `master` with the §9.3 checklist as the description.
