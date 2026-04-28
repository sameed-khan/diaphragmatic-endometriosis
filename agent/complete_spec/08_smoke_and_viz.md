# Component 8 — Smoke Test + Error-Analysis Visualization

**Status:** Spec locked, ready for implementation.
**Owner files:** `scripts/smoke_train.py`, `scripts/visualize_predictions.py`, `tests/smoke/test_smoke_run.py`
**Date:** 2026-04-27

Two independent artifacts that share no runtime: a smoke training script that catches integration bugs in <5 min, and a visualization script that produces per-slice prediction overlays for human inspection.

---

## 1. Smoke training script — `scripts/smoke_train.py`

### 1.1 Purpose

End-to-end gate that any agent runs **first** after specs are implemented (or after any integration change). Trains 1 fold for 2 epochs on a 5-volume subset and asserts the loss decreases. Surfaces plumbing bugs (data shape mismatches, loss NaN, missing GT, sampler indexing, EMA wiring, callback registration) in seconds, not GPU-hours.

### 1.2 Pipeline

```python
def run_smoke():
    # 1. Pick 5 smallest cohort volumes (by volume.npy file size) — 2 positives + 3 negatives
    smoke_pids = pick_smallest_volumes(cohort='cross-validation', n_pos=2, n_neg=3)

    # 2. Build full DataModule, but override the patient lists to only smoke_pids
    dm = LesionDataModule(
        cache_root=...,
        splits_path=...,
        fold=0,
        batch_size=4,           # smaller for safety
        num_workers=2,
        augment_train=TrainAugmentation(...),   # full aug enabled
        sampler_train=WeightedScheduledSampler(
            cfg=SamplerConfig(epoch_mode='fixed_count', samples_per_epoch=100),
            ...
        ),
        allow_holdout=False,
    )
    dm.train_patient_ids = smoke_pids   # override after setup
    dm.val_patient_ids   = smoke_pids[:2]   # 2 patients for val

    # 3. Build full LightningModule (real model config, not a stub)
    lm = LesionDetectorLM(
        model_cfg=ModelConfig(),
        train_cfg=TrainingConfig(max_epochs=2, warmup_epochs=0),
        score_ema_tracker=ScoreEMATracker(),
    )

    # 4. Build Trainer with lightweight callbacks
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[EmaCallback(decay=0.99)],   # short decay for fast convergence
        logger=False,                           # no W&B for smoke
        log_every_n_steps=1,
        default_root_dir="outputs/smoke",
    )

    # 5. Train and capture step losses
    step_losses = []
    class StepLossCapture(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_):
            step_losses.append(float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs))
    trainer.callbacks.append(StepLossCapture())
    trainer.fit(lm, datamodule=dm)

    # 6. Assertions
    assert len(step_losses) >= 50, f"too few steps: {len(step_losses)}"
    first_10 = np.mean(step_losses[:10])
    last_10  = np.mean(step_losses[-10:])
    assert last_10 < first_10, f"loss did not decrease: first10={first_10:.3f} last10={last_10:.3f}"
    assert all(np.isfinite(step_losses)), "NaN/Inf encountered in training losses"
    assert trainer.callback_metrics.get("val/slice_auroc") is not None, "val never ran"

    # 7. Cleanup
    shutil.rmtree("outputs/smoke", ignore_errors=True)

    print(f"SMOKE PASSED. Loss: {first_10:.3f} → {last_10:.3f}")
```

### 1.3 CLI

```bash
uv run python scripts/smoke_train.py
```

No arguments. Reads cache from the canonical path. Should complete in **< 5 min wall-clock** on the L40S.

### 1.4 Key properties

- **Real model, real data, real cache.** The smoke test exercises the actual code paths that production training uses — no mocks, no stubs.
- **Short epochs.** `samples_per_epoch=100`, `max_epochs=2`, `batch_size=4` → 50 steps total. Enough to see loss decrease; few enough to stay under 5 min.
- **Augmentation enabled.** Catches paste/geom/intensity bugs.
- **No QC signoff dependency.** Smoke test runs before QC review (Component 4 §11.4) by design — that's its point.
- **No W&B logging.** Self-contained, no network dependency.
- **Cleans up after itself.** No persistent state between runs.

### 1.5 Failure modes and what they tell you

| Failure | What it means |
|---|---|
| `len(step_losses) < 50` | Trainer crashed early; check stdout for traceback |
| `last_10 >= first_10` | Model isn't learning — check label/box correctness, loss aggregation, optimizer config |
| NaN/Inf in losses | bf16 + intensity aug interaction, divide-by-zero in Dice loss, or assigner bug |
| `val/slice_auroc is None` | Validation never ran — check val_dataloader setup, patient list overrides |

---

## 2. Visualization script — `scripts/visualize_predictions.py`

### 2.1 Purpose

For a given trained checkpoint, render per-slice prediction overlays for visual error analysis: 20 positive volumes + 20 negative volumes, with green GT boxes, red prediction boxes, and per-event filename tags so issues can be filtered. Output is non-version-controlled (`outputs/`) and optionally logged to W&B.

### 2.2 Per-slice event taxonomy

For each slice within a sampled volume, classify what's present after running inference + per-slice NMS (no WBF — we want raw per-slice predictions for inspection):

| Event | Definition |
|---|---|
| **TP** | A predicted box on this slice has IoU ≥ 0.3 with at least one GT box on this slice |
| **FP** | A predicted box on this slice has IoU < 0.3 with every GT box on this slice (or there are no GT boxes) |
| **FN** | A GT box on this slice has IoU < 0.3 with every predicted box on this slice (or there are no predictions) |

A single slice can produce **multiple** event PNGs — one per event type present. E.g., a slice with a TP detection AND an unrelated FP detection produces both a `_tp_slice<y>.png` AND a `_fp_slice<y>.png`. Both PNGs render the same slice; the filename tag indicates what's *highlighted* in that PNG (TP boxes brightened, others dimmed).

### 2.3 Sampling and filtering

```python
def sample_cases(datamodule, run_seed=42):
    rng = np.random.default_rng(run_seed)
    pos_pids = sorted(datamodule.val_patient_ids_with_label('positive'))
    neg_pids = sorted(datamodule.val_patient_ids_with_label('negative'))
    sampled_pos = rng.choice(pos_pids, size=min(20, len(pos_pids)), replace=False)
    sampled_neg = rng.choice(neg_pids, size=min(20, len(neg_pids)), replace=False)
    return sampled_pos, sampled_neg

# Per-volume slice limit: cap at top-5 most "interesting" slices per volume
# Interestingness = max prediction confidence on that slice (positive vols)
#                 OR max prediction confidence on that slice (negative vols, all FPs)
MAX_SLICES_PER_VOLUME = 5
```

### 2.4 PNG rendering

For each (volume, slice_y, event) tuple to be rendered:

```
- Background: volume[:, slice_y, :] grayscale, contrast-stretched to ROI percentile range
- Overlay GT boxes:    GREEN with 1.5 px width, label "GT"
- Overlay PRED boxes:  RED with 1.0 px width, label "Pred (score=0.XX)"
- Highlight the focused event boxes (per filename tag) with brighter alpha
- Title strip at top: "<patient_id> | slice y=<n> | <event>"
- Save as PNG via matplotlib (dpi=120)
```

### 2.5 Filename convention

```
outputs/<run_name>/<positive|negative>_<patient_id>_<tp|fp|fn>_slice<y>.png
```

Examples:
- `positive_chrome_swallow_bear_tp_slice100.png` — positive volume, slice 100, this PNG highlights a TP detection
- `positive_chrome_swallow_bear_fp_slice128.png` — positive volume, slice 128, this PNG highlights an FP detection on the same volume
- `positive_chrome_swallow_bear_fn_slice95.png` — positive volume, slice 95, this PNG highlights a missed GT
- `negative_glass_puma_glade_fp_slice110.png` — negative volume, slice 110, an FP detection (all detections on negatives are FPs)

### 2.6 Output directory layout

```
outputs/                                           # gitignored — do not version-control
└── <run_name>/                                    # e.g., baseline_fold0_epoch055
    └── viz/
        ├── positive_<pid>_tp_slice<y>.png
        ├── positive_<pid>_fp_slice<y>.png
        ├── positive_<pid>_fn_slice<y>.png
        ├── negative_<pid>_fp_slice<y>.png
        └── manifest.csv             # {filename, patient_id, slice_y, event, max_score, ious, gt_boxes_count, pred_boxes_count}
```

`outputs/` must be added to `.gitignore` if not already.

### 2.7 W&B logging

If a W&B run is active (i.e., `WANDB_RUN_ID` env var set, or `--wandb-run-id` flag passed), each rendered PNG is also logged as a `wandb.Image` to the active run under the key `qc_predictions/<filename_stem>`. The `manifest.csv` is logged as a `wandb.Table` under `qc_predictions/manifest`.

If no W&B run is active, the script writes only to disk and prints the output directory path.

### 2.8 CLI

```bash
# Standalone (no W&B)
uv run python scripts/visualize_predictions.py \
    --run-dir runs/baseline_fold0 \
    --checkpoint best.ckpt \
    --output-dir outputs/baseline_fold0_viz

# With W&B logging (resumes the active run)
uv run python scripts/visualize_predictions.py \
    --run-dir runs/baseline_fold0 \
    --checkpoint best.ckpt \
    --output-dir outputs/baseline_fold0_viz \
    --wandb-run-id <run_id_from_train_py>

# After training: hook into Lightning callbacks to auto-run viz at end of fit
# Add to train.py: pl.callbacks.OnExceptionCheckpoint or on_fit_end
```

### 2.9 Test plan

Tests in `tests/viz/`. Run via `uv run pytest tests/viz/`.

#### 2.9.1 Unit tests (synthetic)

| # | Test | Assertion |
|---|---|---|
| V1 | `test_event_tagging_tp` | Pred IoU=0.5 with GT → tagged tp |
| V2 | `test_event_tagging_fp_no_gt` | No GT, pred score=0.5 → tagged fp |
| V3 | `test_event_tagging_fp_low_iou` | GT exists, pred IoU=0.1 → tagged fp |
| V4 | `test_event_tagging_fn` | GT exists, no pred matches → tagged fn |
| V5 | `test_event_tagging_mixed_slice` | Slice has 1 TP + 1 FP + 1 FN → produces 3 PNGs |
| V6 | `test_filename_format` | Verify structure exactly matches `<positive|negative>_<pid>_<event>_slice<y>.png` |
| V7 | `test_top_k_slice_filter` | Volume with 20 candidate slices → keeps top-5 by max score |
| V8 | `test_manifest_csv_schema` | All required columns present and populated |

#### 2.9.2 Integration test

| # | Test | Assertion |
|---|---|---|
| V9 | `test_visualize_real_run` | Run on real fold-0 best.ckpt; produces ≥ 100 PNGs (40 vols × ≥3 events each on avg); manifest.csv non-empty; all PNGs openable as valid images |

### 2.10 Wall-clock

- Inference on 40 volumes (val fold subset): ~3 min on L40S.
- Rendering 40 vols × ~5 slices × ~3 events ≈ 600 PNGs at 120 dpi: ~2 min.
- **Total ~5 min** per run.

### 2.11 Acceptance checklist

- [ ] `scripts/smoke_train.py` exists; runs to completion; loss decreases; cleans up.
- [ ] `scripts/visualize_predictions.py` exists with the CLI in §2.8.
- [ ] All §2.9.1 unit tests pass.
- [ ] §2.9.2 integration test passes on a real trained run.
- [ ] `outputs/` is in `.gitignore`.
- [ ] W&B logging path tested (script run with active W&B run produces logged images and table).
- [ ] No W&B logging path tested (script run without W&B writes only to disk).
- [ ] Smoke test gate runs as part of CI / pre-train flow (documented in README).

---

## 3. Two scripts, one purpose

Component 8's two scripts answer different questions:

| Script | Question answered | When to run |
|---|---|---|
| `smoke_train.py` | "Is the integration of Components 1–6 actually wired up correctly?" | Before any real training run, after every spec-implementing change |
| `visualize_predictions.py` | "What is this trained model doing right and wrong, qualitatively?" | After every training run, before deciding to retrain or move to evaluation |

Both run independently and have no shared state. Both are CLI-driven, no GUI.

When this file's checklist is green, the entire end-to-end stack (preprocessing → training → eval → visualization) is exercised. The final remaining piece is the cross-component contracts overview, which is the next agent's job per the handoff document.
