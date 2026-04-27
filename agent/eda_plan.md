# Diaphragmatic Endometriosis EDA Plan

**Date:** 2026-04-25
**Status:** Awaiting review

---

## Dataset Summary (from initial exploration)

| Property | Value |
|---|---|
| Total local volumes | 131 |
| Unique patients | 113 |
| Patients with WATER sequence variant | 25 (so 25 volumes are secondary scans) |
| Mask format | 3D Slicer segmentation (NIfTI + CSV metadata) |
| Segments per mask | 2 (all but 1: `ANON0F96D8A953EA` has only Segment_1) |
| Segment meaning | Segment_1 = superior border of liver; Segment_2 = hepatorenal space. Both are endometriosis lesions -- anatomical location distinction only. **Merge to single binary mask for training.** |
| Local negatives | 0 (all negatives on HPC) |
| HPC negatives | ~1000 (not yet converted to NIfTI) |

**Resolved questions:**
- Segment_1 vs Segment_2 = anatomical location of lesion (superior liver border vs hepatorenal space). No quality/type distinction. Merge into binary mask.
- `ANON0F96D8A953EA` has only Segment_1 -- patient simply had lesion(s) in only one location.

---

## EDA Sections

### 1. Volume Geometry & Spacing Consistency

**Goal:** Determine if all volumes share the same voxel spacing and dimensions, or if resampling is needed.

**Tasks:**
- [ ] Load all 131 volumes, extract: shape, voxel spacing (pixdim), affine matrix, orientation (e.g., RAS/LPS)
- [ ] Tabulate and visualize distributions of (x, y, z) spacing
- [ ] Tabulate and visualize distributions of volume dimensions (nx, ny, nz)
- [ ] Identify outliers in spacing or dimensions
- [ ] Check if WATER sequence volumes have different geometry from the primary scans
- [ ] Determine the through-plane (z) slice thickness -- critical for deciding 2.5D slice count
- [ ] **Decision point:** Do we need to resample to a common spacing? What target spacing?

**Rationale:** Anisotropic voxel spacing (e.g., 0.5mm in-plane, 3-5mm through-plane) is common in clinical MRI and directly impacts whether 2.5D or 3D is appropriate. If z-spacing is thick (3-5mm), adjacent slices already cover significant territory and 3-channel 2.5D (center +/- 1) may suffice.

---

### 2. Mask Binarization Verification

**Goal:** Confirm that merging Segment_1 (superior liver border) and Segment_2 (hepatorenal space) into a single binary mask is clean -- no overlaps, no unexpected label values.

**Decision: RESOLVED** -- Merge both segments into a single binary mask (any nonzero voxel = lesion).

**Tasks:**
- [ ] For each mask NIfTI, extract the set of unique voxel values and confirm only {0, 1, 2} are present
- [ ] Verify labels 1 and 2 never overlap spatially (no voxel assigned both labels)
- [ ] Compute voxel counts per label value (1 vs 2) across all masks -- useful to know the anatomical distribution (superior border vs hepatorenal)
- [ ] Count how many masks have only label 1, only label 2, or both
- [ ] Binarize all masks (nonzero -> 1) for downstream analysis

**Rationale:** Quick sanity check before treating all nonzero voxels uniformly. Also gives us a side statistic on the anatomical distribution of lesions across the cohort.

---

### 3. Lesion Size & Morphology Statistics

**Goal:** Characterize lesion dimensions to inform architecture choices (2.5D stack depth, input resolution, anchor sizes if doing detection).

**Tasks:**
- [ ] For each mask, compute connected components per label using `scipy.ndimage.label` or `cc3d`
- [ ] Per connected component, compute:
  - Voxel count
  - Physical volume (mm^3, using voxel spacing)
  - Bounding box dimensions in mm (x, y, z extent)
  - Number of slices spanned in z-axis
  - Centroid location (x, y, z) in physical and voxel coordinates
- [ ] Aggregate statistics: min, max, median, mean, std, percentiles (5th, 25th, 75th, 95th) for each metric above
- [ ] Distribution plots: histogram of lesion volumes, histogram of z-extent (number of slices), histogram of max in-plane diameter
- [ ] Count number of distinct lesions per volume
- [ ] **Critical question:** How many slices do lesions typically span? This directly determines the minimum 2.5D stack depth.
- [ ] **Critical question:** What is the smallest lesion? Can it even be detected at typical downsampled resolutions?

**Rationale:** Literature says diaphragmatic endometriosis lesions can be micronodules (<5mm), nodules, or plaques. With thick MRI slices, a <5mm lesion might span only 1 slice. If so, the center slice in 2.5D is all-or-nothing.

---

### 4. Lesion Location Analysis (Cropping Feasibility)

**Goal:** Map where lesions sit within the volume to determine optimal pre-cropping strategy.

**Tasks:**
- [ ] Plot lesion centroid locations in normalized volume coordinates (fraction of volume extent in each axis)
- [ ] Create a 3D scatter plot or 2D projections of lesion centroids
- [ ] Compute the bounding box that would contain ALL lesion centroids + some margin
- [ ] For each volume, compute the lesion centroid position relative to the volume center
- [ ] Determine if lesions cluster near the liver dome/right hemidiaphragm as expected (literature: 87.5% right-sided)
- [ ] Investigate whether any lesions appear in unexpected locations (left diaphragm, anterior)
- [ ] Estimate how much of the volume could be cropped away without losing any lesion voxels
- [ ] **Decision point:** What crop region (in voxel or physical coords) captures 100% of lesions with margin?

**Rationale:** Pre-cropping to the liver/diaphragm region is universally recommended by competition winners. This dramatically reduces background, improves class balance, and allows higher effective resolution.

---

### 5. Intensity Statistics & Normalization

**Goal:** Characterize intensity distributions to choose normalization strategy.

**Tasks:**
- [ ] For each volume, compute global intensity statistics: min, max, mean, std, and percentiles (0.5th, 1st, 5th, 50th, 95th, 99th, 99.5th)
- [ ] Compute intensity statistics within lesion masks specifically
- [ ] Compute intensity statistics of background (non-lesion) voxels in the diaphragm/liver region
- [ ] Plot intensity histograms for a sample of volumes (overlaying lesion vs background)
- [ ] Compare intensity distributions across patients to assess inter-patient variability
- [ ] Compare intensity distributions between primary and WATER sequence volumes
- [ ] Check for intensity outliers or corrupted volumes
- [ ] **Decision point:** Percentile clipping range and normalization method (min-max, z-score, percentile clip + rescale)

**Rationale:** MRI lacks standardized Hounsfield units like CT. Intensity normalization is critical but method-dependent. The endometriosis lesions are hyperintense on T1W fat-suppressed -- we want normalization that preserves this contrast.

---

### 6. Multi-Sequence Analysis (WATER Variants)

**Goal:** Understand the relationship between primary scans and WATER sequence variants for the 25 patients that have both.

**Tasks:**
- [ ] Compare geometry (spacing, dimensions) between primary and WATER sequences for the same patient
- [ ] Compare intensity distributions between primary and WATER sequences
- [ ] Check if masks for both sequences annotate the same lesions
- [ ] Visualize corresponding slices from both sequences for a few patients
- [ ] Determine if both sequences should be used as separate training examples, or if one is preferred
- [ ] Check if WATER sequences have different contrast characteristics that could confuse the model
- [ ] **Decision point:** Include both sequences as independent training examples? Use as multi-channel input? Exclude WATER?

**Rationale:** Some patients have a primary scan AND a WATER (fat-suppressed) variant. Literature suggests multi-sequence input outperforms single-sequence for endometriosis detection. But mixing heterogeneous sequences without accounting for it can also hurt.

---

### 7. Quality Control Checks

**Goal:** Catch annotation errors, corrupt files, or inconsistencies before training.

**Tasks:**
- [ ] Verify mask and volume dimensions match for all 131 pairs
- [ ] Verify mask and volume affine matrices match (same physical space)
- [ ] Check for empty masks (mask files that contain no nonzero voxels)
- [ ] Check for masks with unexpectedly large annotations (annotator error: labeled half the volume)
- [ ] Check that all masks only contain expected label values (e.g., {0, 1, 2})
- [ ] Visual QC: render mid-slice overlays of mask on volume for all 131 cases (or a random sample of 20-30)
- [ ] Check for duplicate volumes (same patient scanned twice vs. two different sequences)
- [ ] Verify NIfTI headers are well-formed (no NaN in affines, positive voxel sizes)

**Rationale:** At 131 samples, every data point matters. A few bad annotations or corrupt files can meaningfully degrade model performance.

---

### 8. Slice-Level Analysis (2.5D Preparation)

**Goal:** Understand the slice-level characteristics to inform 2.5D training strategy.

**Tasks:**
- [ ] For each volume, identify which slices contain lesion voxels
- [ ] Compute the distribution of "positive slices" per volume
- [ ] Compute the ratio of positive slices to total slices per volume
- [ ] Determine the typical gap between lesion-containing regions (are lesions contiguous or scattered?)
- [ ] For positive slices, compute the 2D area of the lesion in that slice (both voxels and mm^2)
- [ ] Plot the 2D lesion area per slice across the z-axis for a few representative volumes
- [ ] **Critical question:** What fraction of slices per volume are positive? (Determines positive:negative slice ratio)
- [ ] **Critical question:** After liver-region cropping, what fraction of remaining slices are positive?

**Rationale:** For 2.5D training, we'll extract individual slices (or small stacks) as training examples. Understanding the positive:negative slice ratio informs sampling strategy and class weighting.

---

### 9. Patient-Level Summary & Stratification Variables

**Goal:** Collect per-patient metadata for proper train/val splitting.

**Tasks:**
- [ ] Create a patient-level summary table: patient ID, number of volumes, sequence types, total lesion volume, number of lesions, largest lesion size
- [ ] Identify stratification variables: lesion count, total lesion volume, lesion size category (micro <5mm, nodule 5-30mm, plaque >30mm)
- [ ] Check distribution of these variables -- are they balanced enough for stratified splitting?
- [ ] Determine if patients with multiple sequences should be grouped
- [ ] **Decision point:** Stratification strategy for GroupKFold (which variables, how many folds)

---

### 10. Visualizations to Generate

**Goal:** Produce reference visualizations for the paper and for ongoing development.

- [ ] Example volume slices showing typical lesion appearance (3-5 examples)
- [ ] Lesion size distribution histograms
- [ ] Volume spacing/dimension distributions
- [ ] Lesion location heatmap (aggregate across all patients)
- [ ] Intensity distribution comparison (lesion vs background)
- [ ] Mask overlay montage for QC
- [ ] Per-patient lesion summary bar charts

---

### 11. EDA for HPC Negative Dataset (Deferred)

**Goal:** Once negative volumes are available as NIfTI, run a subset of EDA to verify compatibility.

**Tasks (to run on HPC):**
- [ ] Verify spacing/dimension distributions match positive volumes
- [ ] Verify intensity distributions are comparable
- [ ] Check for any scanner/protocol differences between positive and negative cohorts
- [ ] Confirm no labeling errors (spot-check for lesion-like structures in "negatives")
- [ ] **Key concern:** If negatives come from a different scanner or protocol, there is a risk of shortcut learning (model learns scanner artifacts instead of lesions)

---

## Execution Priority

| Priority | Section | Why |
|---|---|---|
| 1 | Volume Geometry (#1) | Determines if resampling is needed; foundational for all downstream analysis |
| 2 | Mask Binarization Verification (#2) | Quick sanity check before all mask-dependent analysis |
| 3 | Lesion Size & Morphology (#3) | Directly informs architecture choices (2.5D depth, resolution) |
| 4 | Lesion Location (#4) | Determines cropping strategy |
| 5 | Quality Control (#7) | Catch problems early -- every sample matters at n=131 |
| 6 | Intensity Statistics (#5) | Informs normalization strategy |
| 7 | Slice-Level Analysis (#8) | Informs 2.5D sampling strategy |
| 8 | Multi-Sequence (#6) | Secondary concern |
| 9 | Patient Summary (#9) | Needed before training, not urgent for EDA |
| 10 | Visualizations (#10) | Generated alongside above |
| 11 | HPC Negatives (#11) | Deferred |

---

## Dependencies & Tools Needed

```
nibabel        - NIfTI loading
numpy          - array operations
scipy          - connected components, morphology
cc3d           - fast 3D connected components (optional, faster than scipy)
pandas         - tabular summaries
matplotlib     - plotting
seaborn        - statistical plots
```

## Notes

- The research document (`agent/research_medical_imaging_approaches.md`) contains detailed findings from RSNA Kaggle competitions and endometriosis ML papers that informed this plan.
- Key concern from research: **Dice = 0.293** was achieved by nnU-Net on a similar plaque segmentation task. Expectations should be calibrated accordingly.
- **RESOLVED:** Segment_1 = superior liver border lesion, Segment_2 = hepatorenal space lesion. Both are endometriosis -- merge to binary mask for training.
