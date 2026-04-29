# `endo/` — production training package

Importable as `import endo`. Everything that runs at training, inference, or evaluation time lives here. Cache-construction scripts live in `scripts/`; the importable runtime stays in this package.

## Top-level files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker (empty). |
| `lesion_bank.py` | `LesionBankEntry` dataclass (frozen, PRD §6.4) + `load_bank` / `save_bank` / `current_bank_path`. Donor crops carry `tight_mask`, `tight_intensities`, `tight_shell_mask`, `centroid_offset_in_tight`, `z_extent_voxels`, `intensity_mean`, `intensity_std`, `physical_extent_mm`. Built once by `scripts/build_lesion_bank.py`; consumed by `endo.augmentation.transform.TrainAugmentation`. |
| `lightning_module.py` | `LesionDetectorLM` Lightning wrapper. Owns the `(cls_scores, bbox_preds, aux_seg_logits)` forward, `compute_total_loss` aggregation, slice-AUROC for `val/slice_auroc` (the `ModelCheckpoint` monitor — A.8), warmup→cosine `LambdaLR`, and the score-EMA hook for HNM. NaN guard substitutes a zero-loss tensor with grad-path through `aux_seg_logits.float()` so a single bad mini-batch doesn't poison weights. |
| `ema_callback.py` | `EmaCallback` — fp32 shadow via `timm.utils.ModelEmaV3` (I.8.9). Swaps live ↔ EMA at validation start/end and persists `ema_state_dict` in the Lightning checkpoint (consumed by `endo.gru.feature_cache`, `endo.viz.run_viz`, `endo.eval.run_eval`). |
| `inference_pass.py` | Shared inference primitive: `inference_pass(detector, datamodule, patient_ids, split, batch_size) -> dict[pid, list[SliceScore]]`. Calls `detector.head.predict(cls_scores, bbox_preds, image_size)` (NOT `detector.predict` — see `endo/inference_pass.py` and the cross-component contract fix). `SliceScore` carries `boxes` `(N, 4) float32`, `scores`, and `aux_seg_max`. |

## Subpackages

| Path | Purpose |
|---|---|
| `augmentation/` | Online paste + geometric + intensity + box-rederivation pipeline applied per `Sample` in the training dataloader. |
| `cli/` | `run_experiment` argparse entrypoint (`train`, `eval`, `train_gru`, `predict_holdout`, `viz`, `smoke`, `qc_paste`). |
| `config/` | Pydantic experiment-config tree + `load_experiment(path)` loader. |
| `data/` | `Sample` / `Batch` dataclasses, manifest readers, `LesionDataset`, `LesionDataModule`, custom collate. |
| `eval/` | WBF, FROC, AUROC + bootstrap CIs, threshold search, stratified breakdowns, append-only CSV/JSON writers, CV + holdout orchestrators. |
| `gru/` | Stage-2 BiGRU feature cache, rescorer, training loop. |
| `model/` | Backbone + FPN + aux seg head + vendored RTMDet head + DynamicSoftLabelAssigner + composite `LesionDetector`. |
| `sampler/` | `WeightedScheduledSampler`, `ScoreEMATracker`, `PeriodicDeepEvalCallback`. |
| `utils/` | Generic helpers — seeding, IO, run provenance. |
| `viz/` | Per-slice TP/FP/FN tagging + radiology-coronal PNG renderer + fold-level orchestrator. |

## Cross-package contracts (PRD §6)

- `Sample` (in `endo.data.samples`) is the dataset's per-item output and the augmentation pipeline's input/output. Augmentation MUST preserve the dataclass fields and write back the `(5, Z=384, X=384)` `volume_5ch`, `(384, 384)` `lesion_mask_center`, and `(N, 4)` `boxes`.
- `Batch` (custom collate in `endo.data.collate`) is the LightningModule's `training_step` / `validation_step` input. `boxes` is a `list[Tensor]` of length `B`.
- `LesionDetector.head.predict(cls_scores, bbox_preds, image_size)` is the production NMS-prediction path used by both `LesionDetectorLM._update_score_ema` and `endo.inference_pass`. Don't add a `LesionDetector.predict(cls_scores, bbox_preds, ...)` shim — keep the NMS API on the head.
- The vendored RTMDet head computes CIoU under `torch.amp.autocast(enabled=False)` with an L1 fallback when CIoU goes non-finite. Don't reintroduce a bf16-only CIoU path without restoring the fallback.

## Invariants enforced or tracked here

- I.8.1 (no holdout in train/val), I.8.5 (EMA swap-and-restore), I.8.7 (provenance fold-status), I.8.8 (5-channel `(B, 5, Z=384, X=384)` shape contract), I.8.9 (fp32 EMA shadow).
- `val/slice_auroc` is the ModelCheckpoint monitor (A.8). Don't change without updating both `lightning_module.py` and `cli/run_experiment.py`.
- The training-step NaN guard logs at WARNING level and never updates weights from a non-finite loss; the run continues with the previous step's weights intact.
