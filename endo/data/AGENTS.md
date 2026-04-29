# `endo/data/` — RAM-resident slice dataset + DataModule

Implements Component 3 (`agent/complete_spec/03_dataset_datamodule.md`) and PRD §6.6 (the holdout guard).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `samples.py` | `Sample` dataclass — per-item dataset output. Holds `volume_5ch (5, 384, 384) float32`, `lesion_mask_center (384, 384) uint8`, `boxes (N, 4) float32`, `labels (N,) int64`, plus optional `volume_full_cropped`, `lesion_mask_full_cropped`, `border_band_coords` forwarded to augmentation only at training. `Batch` dataclass — collated training batch, `boxes` is `list[Tensor]` so the RTMDet head can accept variable N per image. |
| `manifest.py` | `read_manifest_jsonl`, `manifest_by_pid` (pid → row dict), `fold_split(rows, fold)` returning `(train_pids, val_pids, holdout_pids)`. |
| `collate.py` | Custom `collate_fn` producing `Batch`. Stacks `volume_5ch` and `lesion_mask_center`, keeps `boxes` / `labels` as Python lists. |
| `dataset.py` | `LesionDataset` — slice-level dataset over an in-RAM cache. `__getitem__(i)` extracts the 5-channel triplet around `slice_y_cached`, applies optional `augment` callable, returns a `Sample`. Per-axis jitter is sampled from `[-jitter_max, +jitter_max]` then **clamped** so the center-slice window stays inside the target frame on edge slices. |
| `datamodule.py` | `LesionDataModule` — Lightning DataModule. Eager-loads every needed patient's `volume.npy`, `lesion_mask.npy`, `border_band.npy` and the global `gt_boxes.parquet`. Builds `slice_index` (per-pid valid `slice_y_cached` range) for train + val. Holdout patients are excluded by default (PRD §6.6) — `setup()` AND `inference_dataloader(patient_ids)` both check against the cohort's holdout pids and raise `HoldoutAccessError` on overlap unless `allow_holdout=True`. `from_experiment(experiment_config, *, fold)` static helper builds the DataModule + `TrainAugmentation` from an `ExperimentConfig`. |

## Contracts

- **Cache layout** (PRD §5.2.2): `cache/v1/preprocessed_manifest.jsonl` lists each patient's `cache_volume_path`, `cache_lesion_mask_path` (None for negatives), `cache_border_band_path` (None for holdout). Volumes are `(408, 174, 408) float16`, masks are `uint8 ∈ {0, 1}`, border bands are `(M, 3) int16` voxel coords.
- **`slice_index`** entries are 4-tuples `(pid, slice_y_cached, is_positive_slice, kind)` where `kind ∈ {"pos_slice", "neg_slice_pos_vol", "neg_slice_neg_vol"}`. The CLI strips this to 3-tuples `(pid, sy, kind)` before passing to `WeightedScheduledSampler`. The `PeriodicDeepEvalCallback` accepts both forms (uses `entry[0]`, `entry[1]` indexing).
- **GT box frame**: `gt_boxes.parquet` rows are in cached `(408, 174, 408)` voxel coords. The dataset translates them into the cropped+padded `(384, 160, 384)` frame at `__getitem__` time and clips boxes that straddle the crop boundary.
- **Validation jitter is centered** — `__getitem__` samples `(jx, jy, jz) = (0, 0, 0)` when `augment is None` (D8 invariant).
- **`allow_holdout`** defaults to `False`. Only `endo.eval.run_eval.run_holdout_inference` legitimately sets it `True`. Re-violate this in only one place — the holdout guard is two-layer (setup + inference_dataloader).

## Invariants checked by tests

D1-D13 from PRD §11.3. D11 / D12 / D13 are the holdout-guard tests.

## Don't

- Don't lazily load patients per-batch — the contract is RAM-resident eager load (cohort fits in 36 GB; PRD §12.2 covers the budget).
- Don't change `slice_index` to 3-tuples without updating the dataset's `__getitem__`, the CLI's sampler-construction shim, AND `endo.sampler.periodic_eval._slice_index_lookup` (which now indexes positionally to support both 3- and 4-tuples).
- Don't bypass the holdout guard. If you need holdout access from a new code path, route through `predict_holdout`.
