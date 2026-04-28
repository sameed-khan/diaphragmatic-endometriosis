# Training Pipeline — Phase 1 Decision Document

**Date:** 2026-04-27
**Audience:** The engineering agent who will build the training framework next.
**Scope:** Locked decisions for the Week-1 baseline detector targeting RSNA abstract submission. This document supersedes the modeling synthesis in `agent/eda_synthesis.md` where they conflict, and adds material that synthesis did not commit to (volume-level AUROC pathway, GRU rescorer, lesion copy-paste algorithm, hardware-adapted CV plan).

**Goal:** Train a 2.5D detector that achieves **≥ 0.80 volume-level AUROC** for "at least one diaphragmatic endometriosis lesion in volume" AND **≥ 0.70 sensitivity at 2 FP/volume** on patient-level 5-fold CV. One-week wall-clock budget on a single L40S 46 GB GPU.

---

## 1. Current Data State (post-migration, authoritative)

The `data/` directory is in a known-good state as of 2026-04-27 11:47 after the local-copy migration completed all 7 phases. **The EDA findings document (`eda/FINDINGS.md`) is partially stale** because it was written before the migration; the verification CSVs in `eda/outputs/migration_phase6_*.csv` are authoritative. The post-migration QC overlays at `eda/outputs/post_migration_qc_overlays.png` summarize per-patient alignment visually.

### 1.1 Cohort

- **N = 608 transferred volumes**, 19 GB on disk under `data/raw/`.
- **108 positives + 500 negatives.**
- **Bucket × cohort:**
  | Bucket | Cohort | Count | Folds |
  |---|---|---|---|
  | cross-validation | positive | 86 | fold0:18, fold1:18, fold2:17, fold3:17, fold4:16 |
  | cross-validation | negative | 400 | fold0:82, fold1:81, fold2:79, fold3:79, fold4:79 |
  | holdout | positive | 22 | (locked) |
  | holdout | negative | 100 | (locked) |
- **Holdout is physically separated** under `data/raw/holdout/`. Touch exactly once at the very end after CV is locked.
- **5 patients in `data-local-copy/` have been dropped** (no canonical sequence, lesion not visualizable); 108 is the final positive count.

### 1.2 Hardware metadata

- **All 608 volumes are GE 1.5 T MR**, 3D Dixon LAVA coronal WATER reconstruction.
- **Two scanner models** (post-migration `manifest.scanner_model` is now populated for all 608):
  - `SIGNA Artist`: 369 volumes
  - `SIGNA Explorer`: 239 volumes
- **Two structurally different sequence variants** by `series_description`:
  - **Variant A** (`WATER: COR LAVA DIAF.` and FLEX NAV variants, ~495 vols): both scanners, **1.5 mm reconstructed slice spacing**, 100–160 slices.
  - **Variant B** (`WATER: COR DIAFRAGMA T1 LAVA …`, 113 vols): **Explorer only**, **3.6 mm slice spacing**, 50–60 slices. 11 positives + 102 negatives.
- **In-plane geometry uniform**: all 608 are 512 × N × 512 with axis 1 the through-plane (coronal) axis. In-plane spacing 0.70–0.98 mm (median 0.82 mm).
- **NIfTI zoom_y is the reconstructed slice spacing** (not the acquired DICOM slice thickness — those differ by ~2× because of LAVA-Flex overlap reconstruction). Always use NIfTI `header.get_zooms()` for physical math; do not use `manifest.slice_thickness_mm`.
- **Orientation**: all 608 are RAS, axcodes verified.

### 1.3 Lesion morphology (108 positives)

- **197 connected components** across 108 patients (mean 1.8/patient, max 7).
- **Per-CC max in-plane extent**: P05 5.1 mm · P50 9.8 mm · P95 26.6 mm · max 116 mm. Predominantly 5–30 mm nodules.
- **Per-CC z-extent**: P50 5 slices · P95 13 slices · 7 CCs span only a single slice.
- **2D box count**: 1,365 across 1,090 positive slices (mean 1.25 boxes/slice, max 7).
- **2D box max-dim**: P05 2.3 mm (~3 px on native grid) · P50 7.0 mm · P95 22 mm.
- **Lateralization**: 99% right-sided (vs 87.5% literature). **Horizontal flip will destroy informative signal — do not use it.**
- **Contrast vs background**: median z = 0.64 within ROI (lesion brighter than ring); 100/108 brighter, 8 darker. **Cannot rely on a "find the bright spot" heuristic.**

### 1.4 Mask alignment (post-migration)

- All 108 positives re-staged from `data-local-copy/` (hand-QC'd by user).
- **Lesion-vs-ring contrast z-score**: median **0.810**, P5 0.424, **min 0.121**. Only 1 case below 0.20. (Pre-migration: 6 cases ≤ 0 due to sub-volume picking bug. That bug is fixed.)
- The §13/§14 mask-realignment story in `eda/FINDINGS.md` is resolved. `scripts/realign_masks.py` and `scripts/realign_masks_v2.py` are deprecated.

### 1.5 Liver ROI (TotalSegmentator + 20 mm dilation)

- `liver_masks/` and `liver_rois/` exist for all 608 volumes.
- **20 mm physical-space dilation** of the TotalSeg liver mask (anisotropic, per-volume voxel sizes).
- **Containment**: 106 / 108 positives fully contained in 20 mm liver_roi. Two outliers retain a small number of lesion voxels outside ROI:
  - `glass_puma_glade`: 29 voxels of 251 outside (~12%).
  - `pine_wren_fjord`: 2 voxels of 1,861 outside (~0.1%).
  - Both are annotation-edge artefacts; clip at training time.
- **18/608 liver_masks have 2 connected components** (TotalSeg fragmentation). The dilated `liver_roi` may inherit the fragmentation. This is handled transparently by §1.6 (we crop the outer bbox of all foreground voxels, regardless of CC count) — no per-CC handling is needed in the training pipeline.

### 1.6 Per-volume liver-ROI bounding boxes (newly added)

`data/manifest.csv` now contains 9 new columns (run by `eda/18_liver_roi_bbox.py`):

| Column | Meaning |
|---|---|
| `roi_bbox_x0`, `roi_bbox_x1` | bbox along axis 0 (R-L), inclusive-exclusive voxel indices |
| `roi_bbox_y0`, `roi_bbox_y1` | bbox along axis 1 (slice axis, A-P) |
| `roi_bbox_z0`, `roi_bbox_z1` | bbox along axis 2 (I-S) |
| `roi_bbox_extent_{x,y,z}_mm` | physical extent in mm (using each volume's own zoom) |

Cohort statistics (n=608, all non-empty):

| Axis | P50 vox | P95 vox | max vox | P50 mm | P95 mm | max mm |
|---|---|---|---|---|---|---|
| X (R-L) | 276 | 320 | 371 | 221 | 255 | 274 |
| Y (slices) | 104 | 124 | 147 | 174 | 198 | 220 |
| Z (I-S) | 243 | 287 | 332 | 195 | 226 | 263 |

These are **native (pre-z-resample) voxel counts**. After z-resampling Variant B from 3.6 → 1.5 mm in §2.2, the cohort-wide max y_extent becomes ≈ 220/1.5 = 147 voxels post-resample.

### 1.7 Splits

`data/splits.json` is authoritative, frozen at seed=42. Stratified on `manufacturer_model_name × slice_thickness_bin` for negatives and `manufacturer_model_name` for positives. **Validate post-preprocessing that variant A/B is roughly proportional in each fold**; if it isn't, re-run `scripts/build_splits.py` adding variant to the strata key (do NOT change the seed). Patient-level splits — never leak slices across folds.

### 1.8 Soft negatives

`manifest.soft_negative == True` flags 57 patients reclassified positive→negative during workplan v2.3. Treat as ordinary negatives at training time. They are interesting candidates for negative-side error analysis but do not affect the training spec.

### 1.9 Files and paths

- Raw NIfTIs: `manifest.raw_path` (relative to `data/`).
- Lesion masks (positives only): `manifest.lesion_mask_path` (108 entries).
- Liver masks: `manifest.liver_mask_path`.
- Liver ROIs (20 mm dilation): `manifest.liver_roi_path`.
- BIDS sidecars: `data/sidecars.jsonl`, one line per transferred patient.
- Hashes: `manifest.sha256_raw`, `manifest.liver_mask_sha256` for idempotency checks.

---

## 2. Preprocessing Pipeline (locked)

### 2.1 Order of operations (training and inference are identical)

1. **Read** NIfTI from `raw_path`. Validate RAS axcodes, finite affine, shape `(512, N, 512)`.
2. **Resample only the through-plane axis (axis 1) to 1.5 mm**. Linear interpolation for raw volumes; nearest-neighbour for masks. **Do not resample in-plane** — in-plane spacing variation (0.70–0.98 mm) is small and we want native pixel resolution preserved.
3. **Crop to liver-ROI bounding box** using `manifest.roi_bbox_*` columns. **Do not zero out non-liver voxels inside the bbox** — the model can use peri-liver context (lung, fat planes) that is genuinely informative. The bbox is a crop hint, not a foreground mask.
4. **Pad to a fixed input shape**, axis-by-axis, zero pad, centred. **Target shape: `(384, 160, 384)`** in `(X=axis0, Y=axis1=slices, Z=axis2)`. Justification: covers the cohort max in X and Z with margin, covers post-resample y-max (147) with margin, all dimensions are multiples of 32.
5. **Per-volume intensity normalization**, computed inside the liver_roi mask, applied to the cropped volume:
   - `roi_p1, roi_p99 = percentile(volume[liver_roi == 1], [1, 99])`
   - Clip the cropped volume to `[roi_p1, roi_p99]`.
   - Z-score: `(volume - roi_mean) / roi_std`, where `roi_mean, roi_std` are computed inside the liver_roi mask.
   - This gives ROI-aware normalization that is 2–3× tighter than whole-volume stats and removes the zero-padding bias from µ/σ.
6. **Cache the processed triplet** `(volume_padded, lesion_mask_padded, liver_mask_padded)` to `.npy` per patient. Disk: ~22 GB for the cohort. Cache on `/scratch`. Re-loading + resampling each epoch is more expensive than the disk.
   - Save `lesion_mask_padded` even for negatives (zero array) for uniform code paths.
   - Save `liver_mask_padded` (the binary liver mask, NOT the dilated ROI) — the lesion copy-paste augmentation in §6.3 needs the liver-mask boundary, not the ROI boundary.

### 2.2 Variant B handling

After §2.1 step 2, Variant B's z-extent grows ≈ 2.4× (3.6 mm → 1.5 mm). The upsampling is linear interpolation — there is no physical information at 1.5 mm spacing in Variant B; we are *aligning the grid* so the 5-channel slice triplet covers the same physical depth (4.5 mm × 5/3 = 7.5 mm at 5-channel, see §3.2) for every volume. The network sees Variant B as a slightly smoother through-plane domain. This is the agreed tradeoff — alternative options (drop B, train per-variant heads, resample to 3 mm) are explicitly off the table.

### 2.3 Resolution preserved

In-plane voxel size is preserved at native ~0.82 mm/pixel. A 5 mm micronodule remains ~6 px wide in the input tensor.

---

## 3. Detector Architecture (locked)

### 3.1 Stack

| Component | Choice |
|---|---|
| Detector head | **RTMDet-S** (anchor-free, dense head, SimOTA dynamic label assignment) |
| Backbone | **ConvNeXt-tiny** (28 M params), ImageNet-22k pretrained, sourced via `timm` |
| FPN levels | **strides {4, 8, 16, 32}** (P2–P5). The stride-4 (P2) head is mandatory: P05 box max-dim is ~3 px, anything coarser misses the bottom decile of lesions. Skip stride-64. |
| Auxiliary head | **Segmentation head** (Dice + BCE), supervised on center-slice GT mask, weight 0.3 vs detection loss |
| Activation | RTMDet default (SiLU) |
| Mixed precision | bf16 |
| EMA | decay 0.999 |

### 3.2 Slice-stack input

- **5 adjacent slices, stride 1**, on the resampled grid (1.5 mm/slice).
- Channel order: `[s_{k-2}, s_{k-1}, s_k, s_{k+1}, s_{k+2}]`. Center slice `s_k` is what the loss supervises.
- Physical depth covered: 5 × 1.5 mm = **7.5 mm**. Matches the median per-CC z-span (5 slices at 1.5 mm = 7.5 mm).
- **conv1 surgery for 5-channel input**: replicate pretrained 3-channel conv1 weights to 5 channels and renormalize so expected activation magnitude is preserved (`new_w = pretrained.repeat(1, 2, 1, 1)[:, :5] * (3/5)`). This is a one-time modification at model-init.

### 3.3 Center-slice supervision

For each training step, sample center index `k`, build the 5-channel triplet from the cached resampled volume, and compute losses against the GT extracted from slice `s_k` only. The four context channels flow forward but receive no loss term. This is standard 2.5D supervision.

### 3.4 Input tensor shape

- Per-slice spatial: **384 × 384** (X × Z) after liver-ROI bbox crop and zero-pad to multiple of 32.
- 5-channel slice stack → input tensor shape `(B, 5, 384, 384)`.

### 3.5 Volume-level scoring (the AUROC pathway)

The detector outputs per-slice 2D boxes with confidences. To derive a single volume-level score for AUROC:

`volume_score = max over (slice_index, box) of post_WBF_box_confidence`

Optionally use **top-k mean** (k=5) for stability. In v1, ship max. The GRU rescorer in §11 is the upgrade path that lifts this metric further.

---

## 4. Loss Functions

| Term | Choice | Weight |
|---|---|---|
| Box regression | CIoU | 1.0 (RTMDet default) |
| Classification | Focal loss, **γ=1.5, α=0.25** | 1.0 |
| Auxiliary segmentation | Dice + BCE on center-slice lesion mask | **0.3** |

γ=1.5 (not the YOLO default 2.0) because positives are scarce (86 train pos); γ=2.0 over-suppresses easy positives in this regime.

**Do not use** blob loss, CC-DiceCE, CATMIL, varifocal, or focal-Tversky — they are tuned for primary segmentation, not auxiliary regularization. Plain Dice+BCE is correct here.

---

## 5. Sampling

### 5.1 Slice-level sampling

- **Positive-slice oversampling**: 50% of batch slices come from positive slices (slice has a lesion mask voxel) in epochs 0–10, decaying linearly to 25% by epoch 30.
- **Negative slices** make up the remainder: 50% from off-lesion slices in positive volumes, 50% from negative volumes initially.
- **Hard-negative mining**: starting at epoch 5, after each validation pass, score all negative-volume slices with the current model. Replace 30% of negative-slice draws with the top-FP-prone slices from the previous pass. Refresh the hard-negative pool each epoch.

### 5.2 Volume-level batching

Batch size **8** (5-channel @ 384×384, ConvNeXt-tiny + RTMDet head + aux seg head, bf16 on L40S 46 GB). Should leave 15–20 GB headroom; if OOM, drop to 6. Use gradient accumulation if you need to simulate batch 16 for stability.

### 5.3 Center-slice index sampling

Within a sampled volume, the center index `k` is drawn from valid range `[bbox_y0 + 2, bbox_y1 - 2]` so the 5-channel triplet stays inside the cropped tensor. For oversampled positive slices, `k` is drawn uniformly from the set of slice indices that intersect a lesion mask voxel.

---

## 6. Augmentation

All augmentations applied identically across the 5 channels of a triplet. Geometric transforms applied to lesion mask and liver mask in lockstep. Intensity transforms applied to raw only.

### 6.1 Geometric (every batch)

- Rotation: ±10° in-plane.
- Scale: 0.9–1.1.
- Translation: ±5%.
- Light elastic: σ=2, ~8 control points.
- **No horizontal flip** (load-bearing right-side prior at 99%).
- **No vertical flip** (anatomically wrong).
- **No mosaic, no mixup, no cutmix, no cutout** (incompatible with tiny lesions and the right-side prior).

### 6.2 Intensity (every batch)

- γ correction: γ ∈ [0.8, 1.2].
- Multiplicative bias: 0.9–1.1.
- Gaussian noise: σ=0.01 on z-scored intensity.
- **No multi-window stacking** (MR, not CT).

### 6.3 Lesion copy-paste augmentation (Week-1, in scope)

The single highest-EV augmentation given 86 training positives. Applied with probability **p = 0.5** per batch sample.

#### 6.3.1 Lesion bank (built once at startup)

For each of the 197 lesion CCs in the 86 training positives, store:

- `donor_lesion_3d`: the connected-component foreground mask in donor's grid.
- `donor_lesion_intensities`: the raw intensities at those voxels (post §2.1 normalization).
- `donor_shell_3d`: 1 mm dilation of `donor_lesion_3d` minus `donor_lesion_3d`. Used for boundary feathering only.
- `donor_z_extent`: number of slices the CC spans (1–77, distribution per EDA §4).
- `donor_lesion_intensity_stats`: mean and std of `donor_lesion_intensities`.

#### 6.3.2 Right-hemidiaphragm border band (per target volume)

Computed once per target at load time:

```
liver        = liver_mask (binary, NOT the dilated ROI)
outside_1mm  = binary_dilation(liver, radius_mm=1) AND NOT liver
inside_1mm   = liver AND NOT binary_erosion(liver, radius_mm=1)
border_band  = outside_1mm OR inside_1mm        # 2 mm shell across liver edge

# Right-side restriction: 99% of true lesions are right-sided.
# RAS convention: positive x = anatomical right.
liver_centroid_x = mean(x for x in nonzero(liver))
right_band   = border_band AND (x_voxel_index > liver_centroid_x)
```

The dilation/erosion radii are in physical mm (anisotropic per volume's zooms), implemented via `scipy.ndimage.distance_transform_edt`.

#### 6.3.3 Paste algorithm (per training step, prob 0.5)

```
1.  Pick a target volume from the negative pool (or off-lesion region of a positive).
2.  Pick a random voxel (x*, y*, z*) from `right_band`.
    Reject if local 3-mm-shell std at (x*, y*, z*) is greater than 2× the
    cohort-median local std (avoids pasting onto vessels/high-noise sites).
3.  Pick a donor lesion CC from the lesion-bank (uniform over the 197 CCs).
4.  Translate `donor_lesion_3d` so its centroid lands at (x*, y*, z*). Result:
    `paste_mask` covering some voxels in target's grid, spanning `donor_z_extent`
    slices in z.
5.  Compute target-local intensity stats: mean and std of target volume in a
    3 mm shell AROUND `paste_mask` (target tissue, not the lesion site).
6.  Rescale donor intensities to match target's local brightness:
       donor_normed   = (donor_intensities - donor_mean) / donor_std
       injected       = donor_normed * target_local_std + target_local_mean
7.  Composite into target volume — overwrite ONLY the lesion voxels:
       target[paste_mask] = injected[paste_mask]
    Target's parenchyma, vessels, dome curvature outside `paste_mask` are
    untouched. There is no patch boundary; only the lesion is transplanted.
8.  Soft-blend at the lesion's outer shell only. For voxels v in
    translated `donor_shell_3d`:
       α(v) ∈ (0, 1) decreasing across the 1-mm shell from inside→outside
       target[v] = α(v) · injected[v] + (1 − α(v)) · target[v]
    This kills the hard intensity discontinuity at the lesion boundary
    without transplanting any donor surround tissue.
9.  Update target's labels:
    - 2D bounding boxes: bbox of paste_mask on every slice index it intersects
      (1 to `donor_z_extent` boxes added to the target's box list).
    - Segmentation target: union(target_seg_mask, paste_mask).
```

Through-plane continuity of the synthetic lesion comes from the donor's actual 3D CC shape, not from copying donor surround tissue. The 4 non-center channels of the network input contain target tissue plus, at most, the natural through-plane extent of the donor lesion — which is what real lesions look like.

#### 6.3.4 Implementation notes for the engineering agent

- The lesion bank is built once at startup; per-CC payload is small (typical CC: 2–7 slices × 30 × 30 voxels = ~7 kB raw + mask). Total bank ≤ 5 MB for 197 CCs. Keep in RAM.
- The right-hemidiaphragm `border_band` per target is a small binary mask; cache as a sparse coordinate list per volume in the dataloader.
- `scipy.ndimage.distance_transform_edt` with anisotropic sampling is the right primitive for the 1 mm shells.
- The paste runs entirely on CPU in the dataloader workers.

### 6.4 ROI dilation jitter

To prevent the network from memorizing a fixed liver-edge position as a shortcut, jitter the bbox crop by ±5 mm on each axis at training time (uniformly resample bbox edges). Padding still goes to `(384, 160, 384)`. Disabled at validation/inference.

---

## 7. Training Schedule

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 → cosine to 1e-6 |
| Weight decay | 0.05 |
| Warmup | linear, 1 epoch at 1/10 of base lr |
| Epochs | 60 |
| Batch size | 8 (with grad accum to 16 if loss curve unstable) |
| Mixed precision | bf16 |
| EMA | decay 0.999, reset every fold |
| Early stopping | none in v1 — train full 60 epochs, take best-val-FROC checkpoint |
| Checkpointing | every 5 epochs + best-val-FROC + last |

**Wall-clock estimate on L40S**: ~5 GPU-hours per fold for 5-fold × 1-seed. Total Stage-1 budget ≈ 25 GPU-h.

---

## 8. Cross-Validation and Evaluation

### 8.1 Fold protocol

- **Patient-level 5-fold CV**, frozen seed=42 splits in `data/splits.json`.
- **One seed per fold** (single-seed CV is acceptable for an abstract; cross-fold variance dominates seed variance at 17 pos/fold).
- 122-patient holdout touched **exactly once** at the end after CV is locked. One inference pass with the chosen ensemble.

### 8.2 Metrics (always reported, in this order)

1. **Volume-level AUROC** (at-least-one-lesion). Headline.
2. **Sensitivity at 2 FP/volume** (FROC). Headline.
3. **CPM**: mean sensitivity at {0.125, 0.25, 0.5, 1, 2, 4, 8} FP/volume.
4. **AP@IoU=0.3** (IoU=0.5 is too strict for sub-10 mm lesions).
5. **Patient-level bootstrap 95% CI** on each metric, 1000 resamples, computed per fold.

### 8.3 Stratified breakdowns (also reported)

- FROC stratified by **scanner** (Artist vs Explorer).
- FROC stratified by **variant** (A vs B).
- FROC stratified by **slice-thickness bin** (≤ 2 mm vs > 2 mm).

These are what radiology reviewers will ask for. Fail-fast on any breakdown that shows large disparity between strata.

### 8.4 Operating-point selection

- For volume AUROC: no operating point — report AUROC.
- For per-volume FROC: choose threshold so volume FPR ≤ 2/volume on validation; report sensitivity at that threshold and on the {0.125…8} sweep.

### 8.5 Variant-balance check

Before training: verify that splits.json folds contain Variant A and Variant B in roughly cohort-proportional counts. If not, refold by adding `series_description_variant` to the strata key in `scripts/build_splits.py` and re-running with seed=42 (deterministic).

---

## 9. Inference

### 9.1 Per-slice forward pass

For a target volume:

1. Run §2.1 preprocessing.
2. Slide the center index `k` across all valid slices (stride 1). For each `k`, run the detector forward on the 5-channel triplet.
3. Collect `(slice_k, box_xyxy, score)` tuples across all `k`.

### 9.2 3D Weighted Box Fusion

Treat slice index as z. Run WBF in `(x, y, z)` with:

- xy IoU threshold: 0.3.
- For boxes ≥ 5 mm: require ≥ 2 adjacent slices to agree (boxes that fuse from ≥ 2 slices).
- For boxes < 5 mm: allow single-slice detections with a higher score threshold (grid-search this threshold once per fold on validation).

### 9.3 Volume score derivation

`volume_score = max(post_WBF_confidences)`.

### 9.4 No TTA in v1

Test-time augmentation doubles inference cost and adds noise at our scale. Revisit only if Week-1 sens@2FP < 50%.

---

## 10. Engineering Stack Recommendation

For the engineering agent who builds this next:

| Layer | Recommendation | Rationale |
|---|---|---|
| Trainer | **PyTorch Lightning** | The user's stated preference; mature, handles bf16, EMA, gradient accumulation, DDP if we ever go multi-GPU. |
| Data I/O & augmentation | **MONAI transforms** + custom dataset class | MONAI's `LoadImaged`, `Spacingd` (only along axis 1), `CropForegroundd` (using bbox cols), `NormalizeIntensityd`, and the `RandAffined`/`RandGaussianNoised`/`RandGibbsNoised` family give us most of §6.1 and §6.2 for free. The lesion copy-paste in §6.3 is a custom MapTransform. |
| Backbone | **`timm.create_model('convnext_tiny', pretrained=True, in_chans=5, features_only=True)`** | Direct ImageNet-22k weights via timm; setting `in_chans=5` triggers timm's built-in conv1 replicate (verify it does the renorm — if not, override). |
| Detection head | **MMDetection RTMDet head** ported as a standalone module, OR a hand-rolled RTMDet head (~400 LOC). | MMDetection's full registry is heavy; lifting just `RTMDetHead` + SimOTA assigner + CIoU+focal losses keeps the dependency surface manageable. If MMDetection plumbing eats too much engineering time on day 1–2, switch to **Ultralytics YOLOv11-s** standalone — slightly worse but turn-key. |
| FPN | timm `FeaturePyramidNetwork` or custom 4-level FPN with strides {4,8,16,32}. The P2 head is non-default for most YOLO/RTMDet implementations — must be explicitly added. |
| Aux seg head | A small UNet decoder (4 stages, transposed conv up to stride 4) attached to the backbone features, output 1 channel sigmoid. ~50 LOC. |
| Logging | **Weights & Biases** + **TensorBoard**. Log: train/val losses, per-class FROC, AUROC, scanner-stratified breakdowns, sample slices + GT + predictions every N epochs. |
| Config | **Hydra** or simple OmegaConf. Each fold = one config instance. |
| GPU monitoring | `nvtop` or `gpustat -i 1` during training. |
| Storage | Cache preprocessed `.npy` on `/scratch/pioneer/users/sak185/...`. Do not cache on `/home` (quota-limited; use `quotagrp` to verify per CLAUDE.md). |
| Reproducibility | Set `seed_everything(42)` per fold, log `git rev-parse HEAD` in each run's WandB metadata. |

The core training loop fits in one Lightning module with three losses and the WBF aggregation in `predict_step`. Avoid building a generic detection framework — keep it specialized.

### Things to NOT do

- Do not use MONAI's detection module — it is 3D-RetinaNet-oriented and a poor fit for our 2.5D detector.
- Do not use nnDetection or nnU-Net — they are 3D and self-configuring in ways that will fight our preprocessing decisions.
- Do not implement test-time augmentation in v1.
- Do not introduce DDP / multi-GPU — single L40S is sufficient.

---

## 11. GRU Rescorer (Stage 2, in scope for Week 1)

A small bidirectional GRU is trained AFTER the detector is locked, on frozen-detector features, to lift volume-level AUROC.

### 11.1 Inputs

For each volume in the training/validation set, run the frozen Stage-1 detector once and cache:

- `feat_t`: GAP-pooled feature vector from the last backbone stage (ConvNeXt-tiny: 768-d) for every slice index `t` in the volume.
- The detector's per-slice box list (used at inference for rescoring).

This generates `(num_volumes × num_slices)` feature vectors. Cache to disk.

### 11.2 Architecture

```
Per-volume input sequence: [feat_0, feat_1, ..., feat_{N-1}] each 768-d
   → 1-layer bidirectional GRU, hidden=128 → output [h_0, ..., h_{N-1}] each 256-d
   → linear(256 → 1) + sigmoid
   → per-slice presence probability p_t ∈ [0,1]
   → volume score = max_t p_t  (or top-k mean, k=5)
```

### 11.3 Training

- **Supervision**: volume-level binary label only (`y_volume`). No per-slice label is used directly.
- **Loss**: `BCE(volume_score, y_volume)` plus 0.1 × `BCE(top-k mean of p_t, y_volume)` for stability.
- **Detector and backbone are frozen.** Only GRU weights and the linear head are trained.
- **Optimizer**: Adam, lr 1e-3, weight decay 0.01, 20 epochs.
- **Dropout**: 0.3 on GRU input.
- **Cost**: ~1–2 GPU-h per fold × 5 folds ≈ 5–10 GPU-h total.

### 11.4 Rescoring at inference

For each box `(slice_t, conf=s)` from the detector:

`s' = s × p_t`

Then run §9.2 WBF aggregation using `s'` instead of `s`. Volume score is still `max` over post-WBF box confidences (now rescored).

### 11.5 Reporting

Always report **both** rescored and non-rescored numbers in the abstract. If the lift is < 1 AUROC point, default to non-rescored for simplicity.

---

## 12. Time Budget (One Week)

| Day | Activity | GPU-h |
|---|---|---|
| 1 | Engineering: dataset class, MONAI transform pipeline, conv1-surgery, backbone+FPN+heads wiring. Smoke-test on a single fold. | ~2 |
| 2 | Engineering: lesion copy-paste augmentation (§6.3), hard-negative mining, WBF post-processing, evaluation harness (FROC + AUROC + bootstrap CI). | ~2 |
| 3 | Stage-1 training, fold 0+1 (in parallel-by-time). | ~10 |
| 4 | Stage-1 training, fold 2+3+4. Build Stage-2 GRU feature cache as folds finish. | ~15 |
| 5 | Stage-2 GRU training, all 5 folds. Compute pooled CV metrics. | ~8 |
| 6 | Holdout inference (one shot, full ensemble: detector + GRU rescore). Stratified breakdowns. | ~2 |
| 7 | Abstract draft, figures, writeup. | 0 |

Total: **~40 GPU-h**, well under the 168 GPU-h available on a single L40S, with margin for debugging and one ablation re-run.

If Week-1 baseline lands above the targets early, the highest-EV ablation is **5-fold copy-paste-augmentation off** (negative ablation, ~10 GPU-h) to quantify how much of the lift came from copy-paste — a clean story for the abstract.

---

## 13. Risk Register and Contingencies

| Risk | Trigger | Contingency |
|---|---|---|
| Training is data-limited not architecture-limited | Sens@2FP < 40% across all folds | Skip architecture variation; invest remaining time in label QA + heavier copy-paste (p=0.7 + multi-paste) |
| L40S OOM at batch=8 | First epoch OOM | Drop to bs=6 with grad accum 3; if still OOM, drop input from 384×384 to 320×320 (cohort P95 fits) |
| Variant B drives down stratified FROC | Variant-B sens@2FP > 15 pts below Variant A | Add domain-adversarial loss head (1 day) OR drop Variant B from training and report as out-of-distribution eval |
| RTMDet integration eats >2 days | End of day 2 still no end-to-end run | Switch to Ultralytics YOLOv11-s; same backbone via custom timm-feature-extractor adapter |
| GRU rescorer reduces AUROC | Stage-2 worse than Stage-1 | Default to non-rescored numbers in abstract; document as ablation |
| Holdout sens@2FP much lower than CV | > 10 pts gap | Likely scanner/variant distribution shift in holdout. Report transparently; do not retrain on holdout. |
| ConvNeXt-tiny overfits | val loss diverges from train at epoch ~20 | Increase dropout to 0.2 in head, weight decay to 0.1, early-stop at best-val-FROC |

---

## 14. Out of Scope for Week 1

The following are explicitly NOT part of the Week-1 plan; defer to Week 2+ or after RSNA submission:

- 3D detectors (nnDetection, fully-3D RTMDet, 3D nnU-Net).
- DETR-family detectors (RT-DETR, DINO-DETR, Co-DETR). Bipartite matching is data-hungry; not the right call at 86 positives.
- RAD-DINO, BiomedCLIP, MedSAM-2 backbones. Domain mismatch.
- Self-supervised pretraining on the 500 negatives. Borderline-too-small dataset for SSL to beat ImageNet-22k transfer.
- RadImageNet pretraining ablation. Worth ~5 GPU-h post-baseline; not a Week-1 priority.
- Multi-scale TTA at inference.
- Multi-architecture ensembling (RTMDet + YOLOv11 + DETR). Reserve for if Week-1 baseline already exceeds targets.
- Horizontal-flip augmentation (would destroy the right-side prior).
- Mosaic, mixup, cutmix, cutout (incompatible with tiny lesions or right-side prior).
- 3-channel input ablation (we are committed to 5-channel from the start).

---

## 15. Artifacts and References

### Code artifacts to produce in this phase

- `scripts/preprocess.py` — implements §2.1, writes the `.npy` cache.
- `src/dataset.py` — Lightning DataModule over the cache, implements §5 sampling and §6 augmentation.
- `src/model.py` — backbone + FPN + RTMDet head + aux seg head.
- `src/lesion_bank.py` — implements §6.3.1 and §6.3.3.
- `src/eval.py` — FROC, AUROC, bootstrap CIs, WBF aggregation.
- `src/gru_rescorer.py` — Stage-2 module per §11.
- `configs/baseline_fold{0..4}.yaml` — Hydra configs.
- `train.py` — Lightning entrypoint.

### Data artifacts already available

- `data/manifest.csv` — has 9 new bbox columns, all 608 transferred rows populated (`roi_bbox_*`).
- `data/sidecars.jsonl` — BIDS sidecars with `ManufacturersModelName`, `EchoTime`, `RepetitionTime`.
- `data/splits.json` — frozen seed=42, authoritative.
- `eda/outputs/migration_phase6_*.csv` — post-migration verification (alignment, containment, filesystem).
- `eda/outputs/liver_roi_bbox.csv` — per-volume bbox details.
- `eda/outputs/liver_roi_bbox_summary.txt` — cohort statistics.

### Documents superseded by this one

- `agent/eda_synthesis.md` — the §3a "20 mm dilation insufficient" claim, the open-items punch list, the 320×320 input recommendation, and the 5-fold × 3-seed CV plan are all superseded here. Other sections remain accurate context.

### Documents that remain authoritative context

- `agent/research_2026_modeling.md` — literature review and architecture rationale.
- `agent/research_medical_imaging_approaches.md` — RSNA-competition synthesis and 2.5D background.
- `data/CLAUDE.md` — provenance, layout, conventions.
- `eda/FINDINGS.md` §1, §2, §4–§12, §15 — EDA results that are still accurate post-migration.

---

**End of Phase 1 Decision Document.** Engineering agent: read §1 and §10 first, then §2–§9, then §11. The lesion copy-paste algorithm in §6.3 is non-trivial — budget half a day for it on day 2.
