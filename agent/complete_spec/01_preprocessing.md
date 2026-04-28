# Component 1 — Preprocessing Pipeline

**Status:** Spec locked, ready for implementation.
**Owner script:** `scripts/preprocess.py`
**Date:** 2026-04-27
**Companion:** Reads §1, §2 of `agent/training_pipeline_decisions_phase1.md`. This document is the implementation contract.

---

## 1. Purpose

Convert the 608 raw NIfTI volumes under `data/raw/` into a uniform-shape, ROI-aware z-scored, fixed-pixel-spacing tensor cache that the dataloader can `mmap` per epoch. Derive per-slice 2D ground-truth bounding boxes from lesion masks **after all spatial transforms have been applied**, so the GT lives in the same coordinate frame as the model input. All augmentation (geometric jitter, lesion copy-paste) happens online — preprocessing produces the *un-augmented baseline* for every volume.

This component runs **once** before training. It is fully deterministic, idempotent, and parallel.

---

## 2. Scope

**In scope:**

- Resample all 608 volumes to a fixed `(0.82 mm, 1.5 mm, 0.82 mm)` voxel grid.
- ROI-aware z-score normalization computed on the resampled, uncropped volume.
- Crop to the post-resample liver-ROI bounding box, then center-pad to a fixed cache shape `(408, 174, 408)` that includes a `±5 mm` jitter margin on every axis.
- Cache `(volume, lesion_mask, liver_mask)` triplets per patient as `.npy` (fp16 / uint8 / uint8).
- Derive per-slice 2D GT boxes from the cached `lesion_mask` and write `gt_boxes.parquet`.
- Precompute the right-hemidiaphragm border-band coordinate list per training/validation volume into `border_bands/<patient_id>.npy` (lesion copy-paste needs this; see Component 4).
- Emit a `preprocessed_manifest.csv` with fold assignment and cache provenance.
- Be idempotent: re-running with no changes produces zero work.
- Be parallel: 16 worker processes by default (cohort completes in ~3 min wall-clock).

**Out of scope (handled by other components):**

- Lesion bank construction → Component 2 (`scripts/build_lesion_bank.py`).
- Online augmentation, jitter, copy-paste → Component 4 (DataModule).
- Model, training, inference → Components 6+.

---

## 3. Inputs

All paths are relative to the repository root unless otherwise stated.

| Input | Path | Used for |
|---|---|---|
| Raw NIfTI volumes | `data/raw/...` (per `manifest.raw_path`) | Source images |
| Lesion masks (positives only) | `data/raw/...` (per `manifest.lesion_mask_path`) | GT |
| Liver masks (binary, TotalSegmentator) | `data/raw/...` (per `manifest.liver_mask_path`) | Copy-paste anchor + provenance |
| Liver ROI (20 mm dilation) | `data/raw/...` (per `manifest.liver_roi_path`) | Norm-stats mask + bbox source |
| Manifest | `data/manifest.csv` | Patient list, paths, scanner, splits |
| Splits | `data/splits.json` | Fold assignment for downstream components |
| Sidecars | `data/sidecars.jsonl` | Variant detection (informational) |

---

## 4. Outputs (the downstream contract)

Other components MUST treat these as the single source of truth. Component 1 is responsible for producing them. No downstream component reads `data/raw/` directly.

```
/scratch/pioneer/users/sak185/diaphragmatic-endometriosis/cache/v1/
├── volumes/
│   └── <patient_id>/
│       ├── volume.npy         # (408, 174, 408) float16
│       └── lesion_mask.npy    # (408, 174, 408) uint8 in {0, 1}      [POSITIVES ONLY]
├── border_bands/
│   └── <patient_id>.npy       # (M, 3) int16 voxel coords (x, y, z) — right-hemidiaphragm 2-mm shell
│                              #   [CV cohort only: positives AND negatives. Holdout skipped.]
├── gt_boxes.parquet           # one row per (patient, slice_y, cc_id); cols: x1, z1, x2, z2, box_max_dim_mm
├── preprocessed_manifest.csv  # see §4.2 for schema
├── code_version.txt           # short SHA of HEAD when cache was written
└── preprocessing.log
```

**`liver_mask` is intentionally NOT in the runtime cache.** It is loaded inside Component 1 from `data/raw/`, used to derive `border_bands/<patient>.npy`, and then discarded. The downstream training pipeline uses `border_bands` (sparse coords) for paste-site selection, not the dense liver mask. This is paste-first ordering: lesion paste happens before geometric augmentation in the dataloader, so the precomputed `border_band` coords remain valid through the paste step. After paste, geometric augs are applied to `(volume, lesion_mask)` only. See Components 3/4 for the runtime detail.

### 4.1 Coordinate conventions

- **Cache axis order:** `axis 0 = X (R-L)`, `axis 1 = Y (slice axis, A-P)`, `axis 2 = Z (I-S)`. Matches the raw NIfTI convention after Orientationd to RAS.
- **GT boxes are 2D**, defined on per-slice `(x, z)` planes. `slice_y` indexes axis 1.
- **Box format:** `(x1, z1, x2, z2)` in cache-cropped+padded voxel coordinates, half-open (`x2 = x1 + width`). Same convention used downstream.
- **Voxel spacing post-cache:** `(0.82 mm, 1.5 mm, 0.82 mm)` — uniform across all 608 volumes.

### 4.2 `preprocessed_manifest.csv` schema

| Column | Type | Description |
|---|---|---|
| `patient_id` | str | Mnemonic ID (matches `manifest.csv`) |
| `cohort` | enum | `cross-validation` \| `holdout` |
| `label` | enum | `positive` \| `negative` |
| `fold` | int \| null | 0–4 for CV, null for holdout |
| `scanner_model` | str | `SIGNA Artist` \| `SIGNA Explorer` |
| `variant` | enum | `A` \| `B` (derived from `series_description`) |
| `cache_volume_path` | str | Relative to cache root |
| `cache_lesion_mask_path` | str \| null | Null for negatives |
| `cache_border_band_path` | str \| null | Null for holdout volumes |
| `roi_bbox_post_resample_x0..z1` | int | Recomputed bbox in cache-cropped+padded coords (the un-padded foreground bbox after centering) |
| `pad_offset_x, pad_offset_y, pad_offset_z` | int | Center-pad offsets used; needed to project boxes |
| `n_lesion_ccs` | int | Count of 3D CCs in cropped lesion mask (sanity check) |
| `roi_norm_p1, roi_norm_p99, roi_norm_mean, roi_norm_std` | float | Normalization stats; logged for QC |
| `lesion_vs_ring_z` | float \| null | Post-cache contrast z-score (regression check vs §1.4 ≥ 0.121 min) |
| `raw_sha256` | str | Source of truth for idempotency |
| `code_version` | str | Git SHA when this row was written |

---

## 5. Pipeline (per-volume function flow)

```
preprocess_one(patient_id) →
    1. load_and_validate(raw, lesion_mask, liver_mask, liver_roi)
    2. resample_all_to_fixed_grid(...)            # (0.82, 1.5, 0.82) mm
    3. compute_roi_norm_stats(volume, liver_roi)  # on resampled, uncropped
    4. apply_clip_and_zscore(volume, stats)
    5. recompute_post_resample_bbox(liver_roi)     # foreground bbox of liver_roi
    6. crop_to_bbox_and_center_pad(volume, lesion_mask, liver_mask, bbox, target_shape=(408, 174, 408))
    7. derive_gt_boxes(lesion_mask_cached)
    8. compute_border_band(liver_mask_cached)
    9. write_cache_files(...)
   10. compute_lesion_vs_ring_z(volume_cached, lesion_mask_cached) [QC]
   11. emit_manifest_row(...)
```

Each step is a pure function with explicit inputs and outputs. No global state.

### 5.1 Step details

**Step 1 — load_and_validate**
- `nibabel.load(raw_path)`; assert `nib.aff2axcodes(affine) == ('R', 'A', 'S')`. If not RAS, run `nib.as_closest_canonical()`. Apply same to all four volumes (raw, lesion_mask, liver_mask, liver_roi).
- Assert shapes match across the four (raw determines truth; masks must match raw in-shape).
- Assert raw shape `(512, N, 512)` with N axis-1.

**Step 2 — resample to fixed grid**
- Target voxel spacing: `(SPACING_X_MM, 1.5, SPACING_Z_MM) mm`, **hardcoded in `scripts/preprocess.py`** as the constant `TARGET_SPACING`.
- The two in-plane values (`SPACING_X_MM`, `SPACING_Z_MM`) are determined **once** by `scripts/analyze_inplane_spacing.py` (see §6.2 below). Default expectation: `(0.82, 0.82)` based on §1.2 cohort median. The analysis script emits the final values; the engineering agent copies them into `preprocess.py` as a one-line edit before the cohort run.
- Use `scipy.ndimage.zoom`:
  - Raw volume: `order=1` (linear).
  - All masks (lesion, liver, liver_roi): `order=0` (nearest neighbour).
- Compute zoom factors per-axis from `nib.header.get_zooms()`. **Do not use `manifest.slice_thickness_mm`** — see §1.2 of the decision doc.

**Step 3 — ROI-aware normalization stats**
- `roi_p1, roi_p99 = np.percentile(volume_resampled[liver_roi_resampled == 1], [1, 99])`
- `roi_mean, roi_std = volume_resampled[liver_roi_resampled == 1].mean(), .std()`
- Compute on the *full resampled, uncropped* volume so masks contain enough voxels for stable percentiles.

**Step 4 — clip + z-score**
- `volume = clip(volume, roi_p1, roi_p99)`
- `volume = (volume - roi_mean) / roi_std`
- Apply to **the entire volume**, not just inside the ROI. Voxels outside the ROI go through the same affine map; this is fine because the model later sees only the bbox-cropped region.

**Step 5 — recompute bbox post-resample**
- Foreground bbox of `liver_roi_resampled` in voxel coords. Used as the crop reference (R2-Q1 answer (a) — single source of truth).
- Use `scipy.ndimage.find_objects(liver_roi_resampled.astype(int))` and take the outer bbox of all foreground voxels regardless of CC count (handles the 18 fragmented liver_roi cases per §1.5).

**Step 6 — crop + center-pad to (408, 174, 408)**
- Crop `volume`, `lesion_mask`, AND `liver_mask` to the bbox from step 5. (`liver_mask` is needed for step 8; not written to cache.)
- Center-pad each axis to target shape; record `pad_offset_{x,y,z}`. Pad value = 0 (volume is z-scored so 0 is the cohort mean).
- **Hard assert:** post-crop foreground extent must fit within `(384, 160, 384)` (the eventual training input). Cohort max is `(334, 147, 321)` post-resample-to-0.82mm — should always pass. If a volume fails (huge liver), abort with a clear error and add to an exclusion file.

**Step 7 — derive 2D GT boxes**
- `cc_labels, n_cc = scipy.ndimage.label(lesion_mask_cached)` (3D 6-connectivity).
- For each CC, iterate `slice_y in [bbox_y0, bbox_y1)`; if `cc_labels[:, slice_y, :] == cc_id` has any True, compute `(x1, z1, x2, z2)` in cached coords.
- Compute `box_max_dim_mm = max((x2-x1)*0.82, (z2-z1)*0.82)`.
- Append rows to in-memory list; write a single global `gt_boxes.parquet` at the end of the cohort run.
- **Sanity check:** total CC count and total box count should match §1.3 figures (197 CCs, ~1,365 boxes). Log discrepancies.

**Step 8 — border band**
- Anisotropic distance transform on `liver_mask_cached`:
  - `outside_1mm = (dist_outside_liver ≤ 1.0 mm) & ~liver_mask`
  - `inside_1mm  = (dist_inside_liver ≤ 1.0 mm) & liver_mask`
  - `border_band = outside_1mm | inside_1mm`
- Right-side restriction: `liver_centroid_x = mean(x where liver_mask)`, then `right_band = border_band & (x_idx > centroid_x)`.
- Convert to coord list: `coords = np.argwhere(right_band).astype(np.int16)` (shape `(M, 3)`).
- Use `scipy.ndimage.distance_transform_edt(... , sampling=(0.82, 1.5, 0.82))` for the anisotropic distances.
- Skip for negative-cohort holdout volumes (only training/val negatives need border bands for paste targets).

**Step 9 — write cache files**
- `np.save` `volume.npy` (fp16) to `volumes/<patient_id>/`.
- For positives only: `np.save` `lesion_mask.npy` (uint8) to `volumes/<patient_id>/`.
- For CV cohort only (positives + negatives, NOT holdout): `np.save` `border_bands/<patient_id>.npy` (int16, shape `(M, 3)`).
- **Do NOT write `liver_mask` to the cache** — it is consumed only inside Component 1 (steps 6 + 8) and discarded.

**Step 10 — QC: lesion-vs-ring contrast z-score**
- Mirrors the QC that §1.4 reports (median 0.810, min 0.121).
- For each lesion CC: `z = (mean_inside_cc - mean_in_3mm_ring) / std_in_3mm_ring`.
- Aggregate per-volume as `min(z over CCs)`. Write to manifest. **Hard fail** if any value is < 0.0 (prior bug regression).

---

## 6. CLI & invocation

```bash
uv run python scripts/preprocess.py \
    --manifest data/manifest.csv \
    --splits data/splits.json \
    --raw-root data/ \
    --cache-root /scratch/pioneer/users/sak185/diaphragmatic-endometriosis/cache/v1 \
    --workers 16 \
    [--patients PATIENT1 PATIENT2 ...]   # optional, for testing on a subset
    [--force]                            # ignore idempotency cache
    [--dry-run]                          # print plan, write nothing
```

`TARGET_SPACING` and `TARGET_SHAPE` are module-level constants in `preprocess.py`, not CLI flags — they are part of the cache-version contract and changing them must invalidate the cache (handled via `code_version` in the cache key per §7).

### 6.2 Companion analysis script — `scripts/analyze_inplane_spacing.py`

Run-once script that determines the in-plane resample spacing.

```bash
uv run python scripts/analyze_inplane_spacing.py \
    --manifest data/manifest.csv \
    --raw-root data/ \
    --output agent/complete_spec/analysis_inplane_spacing.txt
```

Behavior:

1. For each of 608 volumes, read NIfTI header (no voxel data needed) and extract `header.get_zooms()[0]` and `[2]`.
2. Build histograms of `zoom_x` and `zoom_z` at 0.01 mm resolution.
3. Decision rule:
   - If a single bin contains >50% of the cohort, choose that bin's value.
   - Else, choose the cohort median.
4. Print and write to output file:
   - Per-axis chosen spacing (X and Z, in mm).
   - Histogram summary (top 5 bins per axis with counts and percentages).
   - The exact one-liner the engineering agent should paste into `preprocess.py`:
     ```
     TARGET_SPACING = (<chosen_x>, 1.5, <chosen_z>)  # mm; from analyze_inplane_spacing.py YYYY-MM-DD
     ```

This script has no test plan beyond a single smoke test (`uv run` it on the cohort and confirm output file is written). It is run once; its output is checked into the repo as `agent/complete_spec/analysis_inplane_spacing.txt` for provenance.

### 6.1 Implementation skeleton

```python
# scripts/preprocess.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessConfig:
    manifest_path: Path
    splits_path: Path
    raw_root: Path
    cache_root: Path
    target_spacing: tuple[float, float, float] = (0.82, 1.5, 0.82)
    target_shape: tuple[int, int, int] = (408, 174, 408)
    workers: int = 16
    force: bool = False

@dataclass(frozen=True)
class PreprocessResult:
    patient_id: str
    success: bool
    manifest_row: dict | None
    error: str | None

def preprocess_one(patient_id: str, cfg: PreprocessConfig) -> PreprocessResult: ...
def preprocess_cohort(cfg: PreprocessConfig) -> list[PreprocessResult]: ...

# Pure helpers (each is unit-testable):
def load_volume_set(...) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]: ...
def resample_to_grid(volume, source_spacing, target_spacing, mask=False) -> np.ndarray: ...
def roi_normalization_stats(volume, roi) -> dict: ...
def apply_normalization(volume, stats) -> np.ndarray: ...
def post_resample_bbox(roi) -> tuple[slice, slice, slice]: ...
def crop_and_pad(arr, bbox, target_shape, pad_value=0) -> tuple[np.ndarray, tuple[int,int,int]]: ...
def derive_2d_boxes(lesion_mask) -> list[BoxRow]: ...
def compute_border_band(liver_mask, spacing) -> np.ndarray: ...
def lesion_vs_ring_z(volume, lesion_mask) -> float: ...
```

Each helper is a stateless function returning new arrays — never mutates.

---

## 7. Idempotency

- Per patient, compute `cache_key = sha256(raw_sha256 || code_version || target_spacing || target_shape)`.
- If `preprocessed_manifest.csv` already has a row matching `(patient_id, cache_key)` AND all expected files exist, skip. Else process.
- `--force` overrides and re-processes everything.
- `code_version.txt` records `git rev-parse HEAD` of the repo at run time. Changes to preprocessing code force a new cache version.

---

## 8. Parallelism

- `multiprocessing.Pool(processes=16)`; map over `patients`.
- Each worker is independent: opens its own NIfTI, computes, writes its own files.
- Manifest rows are collected via `imap_unordered` and written by the parent process to avoid contention.
- Logging: each worker logs to a per-worker file `preprocessing.<pid>.log`; parent concatenates into `preprocessing.log` at end.
- Memory budget: each worker holds at most ~600 MB (raw + 3 masks at fp32 momentarily during resample). 16 workers × 600 MB = 9.6 GB. CPU node has ample RAM.

---

## 9. Test plan

Tests live in `tests/preprocessing/`. All run via `uv run pytest tests/preprocessing/`.

### 9.1 Unit tests (synthetic data)

Use small synthetic volumes built in-memory:

| Test | Setup | Assertion |
|---|---|---|
| `test_resample_isotropic` | 8×8×8 unit cube, source spacing (1, 1, 1), target (0.5, 0.5, 0.5) | output shape (16, 16, 16); voxel sum scales with volume |
| `test_resample_mask_nn` | Binary mask with single voxel at (4, 4, 4) | After NN resample, output remains binary (no fractional values) |
| `test_norm_stats_inside_roi` | Volume = constant 100 inside ROI, 0 outside | `roi_mean = 100`, `roi_std = 0`, p1 = p99 = 100 |
| `test_clip_zscore_roundtrip` | Random volume + dummy ROI; manually compute expected stats | Output matches |
| `test_post_resample_bbox` | ROI = block at coords (5:15, 3:7, 8:18) | Returns slices `(5:15, 3:7, 8:18)` |
| `test_crop_and_pad_centering` | bbox extent (10, 5, 10), target (20, 11, 20) | Output shape correct; foreground starts at `(5, 3, 5)`; pad offsets recorded |
| `test_crop_and_pad_oversized_bbox` | bbox extent (25, 5, 5), target (20, 11, 20) | Raises with informative error |
| `test_derive_2d_boxes_single_cc` | Lesion mask = 3×2×3 block at (10, 5, 10), spans 2 y-slices | Returns 2 boxes, both `(10, 10, 13, 13)` |
| `test_derive_2d_boxes_disjoint_ccs` | Two non-touching 2×2×2 blocks | Returns 4 boxes (2 per CC × 2 slices), distinct `cc_id` |
| `test_border_band_right_side_only` | Square liver mask centered at (20, 20, 20) extent 10 | Coords have `x > 20` for all rows |
| `test_idempotency_skip` | Process patient once, then call again with same cache_key | Second call returns "skipped" without re-writing |

### 9.2 Integration tests (real data, 2-volume fixture)

Pin fixtures: 1 positive + 1 negative volume, smallest in cohort by raw_path size. Stored as a manifest subset in `tests/preprocessing/fixtures/mini_manifest.csv`. Cached output lives under `tests/preprocessing/.test_cache/`.

| Test | Assertion |
|---|---|
| `test_real_two_volume_e2e` | `preprocess_cohort(cfg, patients=fixtures)` produces all expected files; manifest rows match schema |
| `test_real_volume_shape_correct` | Both volumes cached at `(408, 174, 408)` |
| `test_real_volume_dtype_correct` | `volume.npy` is `float16`; `lesion_mask.npy` (positives only) is `uint8` |
| `test_real_no_liver_mask_in_cache` | `liver_mask.npy` does NOT exist under `volumes/<patient>/` |
| `test_real_norm_zero_centered` | `volume[lesion_mask == 1].mean()` finite and bounded for positives (within ±5σ) |
| `test_real_lesion_recoverable` | `n_lesion_ccs > 0` for the positive volume; matches reference count |
| `test_real_lesion_vs_ring_z_above_floor` | `lesion_vs_ring_z >= 0.121` (regression check vs §1.4 min) |
| `test_real_gt_boxes_inside_cache` | All `(x1, z1, x2, z2)` in `gt_boxes.parquet` are within `[0, 408)`-shaped slices and all `slice_y in [0, 174)` |
| `test_real_border_band_inside_cache` | All coords in `[0, 408) × [0, 174) × [0, 408)` |
| `test_real_border_band_right_only` | All coords have `x > liver_centroid_x` |
| `test_real_border_band_holdout_skipped` | A holdout fixture patient has no `border_bands/<patient>.npy` |

### 9.3 Cohort-level test (single full run)

Invocation: `uv run python scripts/preprocess.py --manifest ... --workers 16` over all 608 volumes.

Acceptance gate before moving to Component 2:

1. **Coverage:** 608/608 succeed OR an exclusion file documents every failure with reason.
2. **Disk:** Total cache size in `[30 GB, 50 GB]`. Point estimate: ~38 GB (35 GB volumes + 3 GB lesion masks + ~30 MB border bands).
3. **Lesion CC count:** Sum of `n_lesion_ccs` across cohort = 197 ± 0 (matches §1.3 exactly — any deviation is a bug).
4. **Box count:** `len(gt_boxes.parquet) ∈ [1300, 1450]` (matches §1.3 ≈ 1,365).
5. **Contrast regression:** `min(lesion_vs_ring_z) >= 0.121` (matches §1.4); `median(lesion_vs_ring_z) >= 0.75` (was 0.810).
6. **Idempotency:** Re-run with no changes completes in < 30 s and writes nothing.
7. **Variant balance:** Per-fold variant-A/variant-B count ratio is within ±20% of cohort proportion (per §1.7 / §8.5).

If any of (1)–(6) fails, do not proceed. Investigate root cause.

---

## 10. Logging

`preprocessing.log` records, per volume:

- Patient ID, source paths, raw_sha256
- Source spacing, target spacing, output shape
- Norm stats (p1, p99, mean, std)
- Bbox extents (pre and post crop)
- Number of lesion CCs and 2D boxes
- Lesion-vs-ring z-score
- Wall-clock time
- Idempotency hit/miss

Cohort summary at end:

- Total runtime, workers
- Distribution of lesion-vs-ring z-scores (quantiles)
- Per-fold patient counts, per-variant counts
- Any exclusions

---

## 11. Failure modes and what to do

| Failure | Detection | Action |
|---|---|---|
| RAS conversion fails | `nib.aff2axcodes` mismatch after `as_closest_canonical` | Abort that volume, log to exclusions, continue cohort |
| Mask shape mismatch | `mask.shape != raw.shape` | Abort that volume, exclusions |
| Bbox larger than target | step 6 hard assert | Abort volume, log; consider widening target_shape (this is a global decision, not per-volume) |
| `lesion_vs_ring_z < 0.121` | step 10 | Hard fail the cohort run; investigate (this would indicate regression vs §1.4) |
| Lesion CC count != 197 | cohort-level check | Hard fail; could indicate mask corruption |
| Disk space exhausted | `np.save` raises | Abort cohort, surface clearly; user must free space (`quotagrp`) |

---

## 12. What this component does NOT decide (forwarded to other components)

- Lesion bank construction (depends on fold; Component 2).
- Online augmentation, bbox jitter, copy-paste (Component 4).
- Whether to mmap or fully load cached arrays at training time (Component 4).
- Conversion of fp16 → bf16 at training time (Component 6 / Lightning).

---

## 13. Estimated wall-clock

- Per-volume: ~5 s on a CPU core (load + 3× resample + crop + pad + save).
- Cohort with 16 workers: ~3–4 min wall-clock.
- Test suite: < 60 s (unit) + ~30 s (integration) on the 2-volume fixture.

---

## 14. Acceptance checklist (Component 1 done)

- [ ] `scripts/preprocess.py` exists with the CLI in §6.
- [ ] All §9.1 unit tests pass.
- [ ] All §9.2 integration tests pass on the 2-volume fixture.
- [ ] Full cohort run satisfies all §9.3 gates.
- [ ] `preprocessed_manifest.csv` schema matches §4.2 exactly.
- [ ] `gt_boxes.parquet` columns match §4.1 exactly.
- [ ] Re-running is a no-op (§7 idempotency).
- [ ] Cache disk in `[30, 50] GB`.

When this checklist is green, Component 2 (lesion bank builder) can begin.
