# EDA → Modeling Synthesis (608-volume cohort)

**Date:** 2026-04-27
**Sources synthesized:**

- `eda/FINDINGS.md` — §1 through §12 EDA results (this round)
- `agent/research_2026_modeling.md` — 2024–2026 detection literature review
- `agent/research_medical_imaging_approaches.md` — prior 2.5D vs 3D background
- `agent/containment_audit.md` — independent audit of §3 containment finding

This document is opinionated and decision-ready. It commits to a Week-1 baseline,
a Week-2 set of ablations, and a punch list of preprocessing decisions.

---

## 1. Headline data characterization

| Property | Value | Implication |
|---|---|---|
| N volumes | 608 (108+ / 500−) | Small. 2.5D, not 3D. |
| Folds | 5×CV (96/97 each) + 122 holdout | 17 pos / fold — small, bootstrap CIs. |
| Scanners | 369 Artist · 239 Explorer | KS p≈10⁻¹⁴⁰ on intensity. **Per-volume normalization is mandatory.** |
| Sequence variants | A: `COR LAVA DIAF.` (478, all scanners) · B: `COR DIAFRAGMA T1 LAVA` (113, Explorer only) | Variant B has 3.6mm spacing, 50–60 slices. **Resample to common z spacing** before stacking. |
| In-plane spacing | 0.70–0.98 mm (median 0.82) | Tight. 0.7–1.0 mm target spacing OK. |
| Through-plane spacing | 1.5–3.6 mm (median 1.7); ratio 2:1 from `manifest.slice_thickness_mm` reflects LAVA-Flex overlap recon | NIfTI zoom = reconstructed spacing, not acquired thickness. Use NIfTI zoom for physical math. |
| Shape | 512 × N × 512, N = 44–156 | Through-plane variability is the headache. |
| Lateralization | **99 % right-sided** (literature: 87.5 %) | **No horizontal flip**, ever. |
| Lesion CCs | 197 across 108 patients (mean 1.8/pt) | Most patients single-CC; some up to 7. |
| Per-CC max extent | P05 5.1 mm · P50 9.8 mm · P95 26.6 mm · max 116 mm | Mostly 5–30 mm nodules. **Stride-4 (P2) head is mandatory** for the bottom decile. |
| Per-CC slices spanned | P50 5 · P95 13; **7 CCs span 1 slice only** | 3-channel stack covers most CCs. Allow single-slice positives at low score threshold. |
| 2D box count | 1,365 across 1,090 positive slices | Mean 1.25 boxes/slice; max 7. |
| 2D box size | P05 max-dim 2.3 mm · P50 7.0 mm · P95 22 mm | Tiny. Anchor / FPN priors must include sub-10 px scales. |
| Slice-level imbalance (CV) | **1 : 53** inside ROI | Pos-slice oversampling required; focal loss insufficient on its own. |
| Liver volume | 676–2 111 cm³ (median 1 094) | Reasonable; 18/608 livers fragmented (2 CCs). |
| Lesion-vs-bg contrast | median z = 0.64 in ROI-bg; 100/108 brighter, 8 darker | **Detector cannot rely on hot-spot heuristic.** |
| Lesion containment in 20 mm liver ROI | 104 / 108 fully contained | Need wider margin or accept clip on 4 patients (audit confirms). |

**Cohort comparability (§6): scanner is balanced (V=0.02). Variant and thickness
differ weakly between cohorts (V≈0.12). All continuous metadata (TE, TR, FOV,
age) statistically indistinguishable. Low shortcut-learning risk overall.**

---

## 2. Preprocessing pipeline (locked)

**Order of operations** for both training and inference:

1. **Read NIfTI, validate.** Confirm 512×N×512, RAS orientation, finite affine.
   (§9 says all 608 pass.)
2. **Resample to a common voxel grid.** Target spacing **0.82 × 1.5 × 0.82 mm**
   (matches the Artist Variant-A median). Linear for raw, nearest for masks.
   Rationale: this aligns variant-A and variant-B onto the same z-grid so a
   k-channel slice stack covers the same physical depth (1.5 mm/slice) for
   every patient. **Variant B (originally 3.6 mm) gets up-sampled** in z by
   ~2.4×; this introduces zero-fill blur but matches what the LAVA-Flex
   overlapping recon does anyway in variant A.
3. **Compute (or load pre-computed) liver ROI.** Use a wider dilation than the
   current 20 mm to fully contain all 108 positives — see §3 decision below.
4. **Crop to ROI bbox + a fixed pad** to a common shape. From §3:
   - At native spacing the P95 ROI bbox is 320 × 124 × 287 voxels.
   - At target spacing 0.82 × 1.5 × 0.82, **target shape = 320 × 144 × 320**
     (round up the slice axis for a clean multiple-of-32 divisor).
   - Centre-pad smaller volumes; for the 1 case where ROI is bigger
     (`spice_viper_vault`, 332 voxels deep), down-resample only that case
     in z by a small factor — or, pragmatically, drop those rare voxels at
     the bottom (least likely to host a lesion). Document either choice.
5. **Per-volume intensity normalization, computed inside the ROI.** Take
   `roi_p1` and `roi_p99`, clip raw to those bounds, then **z-score using
   roi_mean / roi_std** (also computed inside the ROI). Rationale: §7 shows
   ROI-only stats are 2–3× tighter than whole-volume stats and remove the
   full-volume zero-padding bias.
6. **Cache the pre-processed `(volume, roi_mask, lesion_mask)` triplet** to
   `.npy` per patient. Disk pressure: at uint16 + bool×2 ≈ 35 MB / patient ×
   608 ≈ 22 GB. Cheaper than re-loading + resampling each epoch.

### 2a. The dilation-margin decision — REVISED after §14 mask-alignment audit

**Update:** the §3 partial-containment finding (4 patients with lesions outside
the 20 mm ROI) was driven by the **mask-realignment bug** described in §14.
After fixing the 6 misaligned masks (see §14 below), only **2 patients** will
retain any out-of-ROI voxels — both at the annotation edge (29 voxels on
`glass_puma_glade`, 2 voxels on `pine_wren_fjord`).

**Recommendation — committed:** **keep the current 20 mm liver_roi dilation.**
Clip-to-ROI on the 2 minor annotation-edge cases. The user's ITK-SNAP
intuition was correct.

The 30–40 mm recommendation in this document's earlier draft is reversed.

---

## 3. Detector architecture (committed)

### 3a. Input
- **3-channel slice stack**, stride 1, on the resampled grid (1.5 mm/channel
  → 4.5 mm physical depth). Center-slice supervision.
- For **micronodules spanning 1 slice** (7 CCs in the dataset), allow the
  loss to fire only on the center-slice channel.
- Spatial size after pad: **320 × 320** in-plane.
- 5-channel ablation in Week 2 (replicate-and-renorm conv1).

### 3b. Detection head
- **RTMDet-S** (anchor-free, dense head, SimOTA assignment).
- **Strides {4, 8, 16, 32}** — the stride-4 (P2) head is mandatory given the
  P05 box max-dim of 2.3 mm = 3 px.
- Skip stride-64 (head receptive field is bigger than the patch).

### 3c. Backbone
- **ConvNeXt-tiny (28M params)** with ImageNet-22k pretraining.
- Fallback in Week 2: EfficientNetV2-S; SE-ResNeXt50 if both overfit.
- **Skip RAD-DINO / BiomedCLIP** for v1 (CXR domain, ViT first-conv surgery).

### 3d. Auxiliary segmentation head
- **Dice + BCE on the center slice**, weight 0.3 vs detection loss.
- Provides denser supervision at this small sample size — RSNA 2023 1st place
  precedent.

### 3e. Loss
- **Box: CIoU.** SIoU/DIoU not worth the integration risk.
- **Classification: focal γ=1.5, α=0.25.** Lower γ than YOLO default because
  positives are scarce.

### 3f. Augmentation
- Rotation ±10°, scale 0.9–1.1, translation ±5 %.
- Intensity: γ ∈ [0.8, 1.2], multiplicative bias 0.9–1.1, Gaussian noise σ=0.01.
- Light elastic (σ=2, ~8 control points).
- **No horizontal flip** (right-side bias is 99 % — load-bearing).
- **No vertical flip, no mosaic, no mixup** in v1.
- Week 2: **lesion copy-paste augmentation, restricted to the right
  hemidiaphragm region** — this is the single highest-EV intervention given
  86 training positives.

### 3g. Sampling and training schedule
- **Positive-slice oversampling**: 50 % positive slices in epoch 0–10, decaying
  to 25 % by epoch 30. Negatives provide 50 % initially; after epoch 5, 30 % of
  negatives drawn from a **hard-negative pool** (top-k FP slices from previous
  epoch).
- **Optimizer:** AdamW, lr 2e-4 → cosine to 1e-6, weight decay 0.05.
- **Schedule:** 60 epochs, EMA decay 0.999.
- **Mixed precision (bf16)** on A100.
- **Batch size:** 16 slices at 320×320 (≈ 16 GB on A100 with ConvNeXt-tiny);
  scale via gradient accumulation if smaller GPU.

### 3h. Cross-validation
- **Patient-level 5-fold** (frozen seed=42 splits.json), **3 random seeds per
  fold** = 15 runs. ≈ 4–5 GPU-h per run on A100 → 60–75 GPU-h total Week 1.
- **Headline metric: sensitivity at 2 FP/volume.**
- Also report CPM (mean sensitivity at {0.125, 0.25, 0.5, 1, 2, 4, 8}
  FP/volume) and AP@IoU 0.3.
- **Patient-level bootstrap 95 % CIs** on each fold (1000 resamples).
- **Volume-level AUC is unreliable** with 17 positives/fold (CI ≈ ±0.08).
- Holdout (122) touched **once**, after CV is locked.

---

## 4. Inference

1. **Per-slice 2D detection** with 3-slice context.
2. **3D NMS via Weighted Box Fusion** (xy IoU = 0.3) — treat slice index as
   z. Require ≥ 2 adjacent slices for boxes ≥ 5 mm; allow single-slice for
   ≤ 5 mm at a higher score threshold.
3. **Optional Phase-2 sequence rescoring**: small bidirectional GRU on
   GAP-pooled slice embeddings from the 2D backbone. Trained only on volume
   labels. Use σ(GRU) per-slice presence prob to rescore boxes.
4. **No TTA in v1.** Expensive and noisy at this scale; revisit only if Week 1
   sensitivity at 2 FP/vol < 50 %.

---

## 5. Week 1 → Week 2 plan

### Week 1 (single hard-baked baseline)
- The full pipeline above. 75 GPU-hours.
- Ship: **mean ± 95 % CI sensitivity at 2 FP/volume, CPM, AP@0.3** across 5
  folds × 3 seeds. Plus thickness-stratified FROC and scanner-stratified FROC
  for shortcut detection.

### Week 2 (three parallel ablations, one fold-sweep each)
1. **Lesion copy-paste right-hemidiaphragm** — highest EV.
2. **5-channel slice stack** with replicated conv1 — tests through-plane
   context value.
3. **GRU rescoring head** on Week-1 frozen detector — tests sequence
   aggregation lift.

### What I would *not* spend Week 2 on
- 3D detection (nnDetection). Will probably tie or lose at this scale.
- RT-DETR / DINO-DETR. Better baseline first.
- RAD-DINO / BiomedCLIP. Domain mismatch + ViT first-conv surgery is a Week 3
  project.
- SSL pretraining on the 500 negatives — borderline-too-small.
- MedSAM-2 in the loop. Annotation tool, not detector.

---

## 6. Open items / data audit punch list

1. **CRITICAL — fix the mask-realignment bug first.** 6/108 positives have
   misaligned lesion masks (§14). Apply `scripts/realign_masks_v2.py --execute`
   AND for those 6 patients update `manifest.csv.raw_path` to the correct
   sub-volume + re-run TotalSegmentator + dilation on the corrected raw. After
   the fix, re-run §3, §4, §5, §7, §8 EDA scripts; their headline numbers will
   tighten slightly.
2. **Keep the 20 mm liver_roi dilation** (no rebuild needed once §14 is
   resolved).
3. **18 multi-CC livers** flagged in §8 — visual QC; if any fragmented, re-run
   TotalSeg with full-res normal mode and `--ml`.
4. **Annotation-edge 1-pixel boxes** (~10 boxes in §5) — filter at training
   time with `min_box_area_mm² ≥ 4`. Don't propagate into FROC.
5. **Backfill `manifest.scanner_model`** from sidecar — currently empty; every
   downstream script has to re-join sidecar.
6. **Commit splits/sidecars/manifest CSV diff** when the §14 fix lands.

---

## 7. Reality-check operating points

Given §4 morphology (P05 max-extent 5.1 mm = ~6 px after resample) and the
2025 small-object detection surveys, the *prior* on what's achievable:

- **Sensitivity at 2 FP/vol: 50–70 %** is plausible for v1 with this regimen.
- The nnU-Net **Dice ≈ 0.293** result on a similar plaque task is for
  *segmentation*, not detection at 2 FP/vol; the right comparand would be
  something like LIDC nodule detection (sensitivity ~70–80 % at 2 FP/scan)
  but those benchmarks have ~600 training positives, not 86. Halve that
  expectation roughly.
- If Week 1 hits **< 40 % sensitivity at 2 FP/vol**, the bottleneck is
  likely data/labels rather than architecture — invest in lesion copy-paste
  augmentation and a label QA pass.
- If Week 1 hits **> 70 %**, ensemble RTMDet + YOLOv11 + a single DETR
  variant for the holdout submission.

---

## 8. What we now know that we didn't before this EDA

1. **The two GE 1.5T scanners produce intensity on completely different
   scales** (Artist 2.4× brighter median); per-volume percentile normalization
   is non-negotiable.
2. **There are two distinct sequence variants by `series_description`**, and
   variant B is Explorer-only with 3.6 mm spacing — this is the dominant
   source of cohort heterogeneity, not scanner per se.
3. **NIfTI zoom_y is consistently half of the manifest's
   `slice_thickness_mm`** because of LAVA-Flex overlap recon. All physical
   math should use NIfTI zoom (the reconstructed spacing), not manifest
   thickness.
4. **Slice-level pos:neg is 1:53 inside the ROI** — the ROI crop saves voxels
   but not slices. Pos-slice oversampling is mandatory.
5. **99 % of lesions are right-sided** — even more strongly biased than the
   87.5 % literature figure. Horizontal flip would destroy informative signal.
6. **The 20 mm dilated liver ROI fails to fully contain 4/108 positives**;
   30 mm catches 106, 40 mm catches 108 but extends through the diaphragm.
   30 mm + clip-to-ROI on the 2 outliers is the right choice.
7. **`manifest.scanner_model` is empty for all 608 rows** — must use
   sidecars `ManufacturersModelName`. Quick back-fill recommended.
8. **18/608 liver masks (3 %) have 2 CCs** — TotalSeg fragmentation, not
   anatomically correct. Visual QC the 18.
9. **Per-CC slice-spanning has a long tail** — most CCs span 5 slices but the
   max is 77 (a single large plaque). Allow flexible 2.5D context width at
   inference.

---

## 9. References to artifacts

- Per-volume CSVs: `eda/outputs/{volume_geometry,liver_roi_containment_*,lesion_components,lesion_per_patient,slice_per_volume,slice_2d_boxes,intensity_per_volume,liver_mask_qc,fold_balance,soft_negative_spot_check,qc_integrity}.csv`
- Plots: `eda/outputs/*.png` (~30 files including overlays, histograms, heatmaps, QC montages)
- Per-section narrative: `eda/FINDINGS.md`
- Independent audit on §3: `agent/containment_audit.md`
- Literature: `agent/research_2026_modeling.md`,
  `agent/research_medical_imaging_approaches.md`
