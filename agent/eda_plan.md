# Diaphragmatic Endometriosis EDA Plan (v2 — 608-volume cohort)

**Date:** 2026-04-27
**Status:** Drafted, awaiting first-pass execution
**Supersedes:** v1 (131-volume, positive-only) — completed and archived in git history

---

## 1. Dataset Summary (post-Phase 1 migration)

| Property | Value |
|---|---|
| Total transferred volumes | 608 |
| Cohort split | 108 positive · 500 negative |
| Bucket split | 122 holdout · 486 cross-validation (5 folds, seed=42) |
| Manufacturer / field | GE / 1.5 T (uniform) |
| Sequence | 3D Dixon LAVA coronal `WATER` (`DERIVED\PRIMARY\DIXON\WATER\MAGNITUDE`) |
| In-plane shape | 512 × 512 (uniform) |
| Through-plane (axis 1) slice count | 52–160 (varies; ~10 distinct values, mode 120) |
| Slice thickness | 2.20–6.94 mm (Artist median 3.0; Explorer median 3.6) |
| Scanner models | SIGNA Artist (369) · SIGNA Explorer (239) |
| Series descriptions | 6 variants, dominated by `WATER: COR LAVA DIAF.` (478) |
| BodyPart codes | 7 variants (ABDOMEN, PELVIS, UTERUS, LIVER, ABDOMENPELVIS, …) |
| PatientSex | F=607, M=1 (single male in positives) |
| Lesion masks | 108 (positives only) — destructively binarized in place to {0,1}; see `scripts/binarize_lesion_masks.py` and `eda/outputs/binarize_lesion_masks_report.csv` |
| Liver masks | 608 (TotalSegmentator `total_mr`, `task=liver`); voxel counts 338 k–1.94 M |
| Liver ROIs | 608 (20 mm anisotropic dilation of liver masks; per-NIfTI zooms) |
| Soft-negatives | 5 of 608 transferred (52 others in phase2_unsupervised, out of scope) |

**Sources of truth (do not glob):**
- `data/manifest.csv` filtered to `transferred_to_home == True` → 608 rows
- `data/sidecars.jsonl` → full BIDS sidecars
- `data/splits.json` → frozen fold assignments

## 2. Policy decisions (locked for this round)

- **Holdout in EDA.** Holdout (122) is included in all aggregates and modeling decisions because the 608-cohort is treated as the *target deployment population* for this single-center clinical use case. Holdout is excluded only from training and hyperparameter tuning.
- **Modeling target.** 2.5D detection: 3-channel slice-stack input → 2D bounding boxes on the center slice. Boxes derived from connected components of binarized lesion masks per slice.
- **Mask semantics.** Binary {0,1}. No per-anatomic-location classes. Lesions = any nonzero voxel. (Destructive binarization handled separately.)
- **Soft-negatives.** Treated as hard negatives — no special downstream handling beyond a brief sanity check.
- **Phase 2 SSL pool.** Strictly out of scope for this EDA.
- **Compute.** 20 CPUs, parallel via `ProcessPoolExecutor(max_workers=16)`.

---

## 3. Sections

### Section 1 — Volume Geometry & Acquisition Heterogeneity

**Goal:** Confirm in-plane uniformity, characterize through-plane variability, identify outliers, decide whether resampling is needed.

**Tasks (608 volumes, header-only reads, parallel):**
- Per volume: shape, voxel zooms, affine, orientation codes (`nib.aff2axcodes`), determinant sign
- Identify the through-plane axis per volume (axis whose zoom matches `SliceThickness`); confirm it is consistently axis 1 in this cohort
- Cross-validate header zooms vs `manifest.slice_thickness_mm` and `sidecar.SpacingBetweenSlices`; flag mismatches
- Tabulate distributions stratified by `cohort × bucket × scanner_model × series_description`
- Flag outliers: thickness > 5 mm, slice count < 60, atypical orientation
- Recompute FOV (mm) per axis and check FOV consistency

**Decision points:**
- Single uniform target spacing (e.g., 0.7 × 3.0 × 0.7 mm)? Or train at native and only force a fixed input crop?
- Through-plane resampling for the variable-N axis to a common slice count, or pad/crop in fixed window?

---

### Section 2 — Mask Binarization Verification

**Goal:** Confirm the destructive binarization left the 108 lesion masks clean and lossless w.r.t. nonzero footprint.

**Tasks:**
- For each lesion_mask: confirm `np.unique == {0,1}`, dtype `uint8`, shape and affine still match raw volume
- Cross-check `total_nonzero_after == total_nonzero_before` from the binarization report (lossless on union of label 1 and label 2)
- No empty masks (every positive must have ≥ 1 lesion voxel)

---

### Section 3 — Liver-ROI Containment Analysis (new in v2)

**Goal:** Decide whether the 20 mm-dilated liver ROI is a viable hard crop. If lesions ever fall outside, dilation must widen.

**Tasks (108 positive triples: raw + lesion_mask + liver_roi, parallel):**
- For each positive: fraction of lesion voxels inside `liver_mask`, inside `liver_roi`, outside both
- Identify cases with any lesion voxels outside `liver_roi` → crop failures (worth investigating individually)
- For each contained lesion: smallest bbox inside ROI containing all lesion voxels with N-mm margin (N ∈ {0, 5, 10, 20}); summarise resulting per-volume crop size
- Per-volume `liver_roi_voxels / total_voxels` ratio (cropping savings) and the resulting in-plane / through-plane patch shape
- Render 5 worst-containment cases (mid-coronal slice with lesion + ROI overlay)

**Decision points:**
- Adopt liver_roi crop in pipeline? If <100 % containment, increase dilation (re-run `dilate_segmentations.py` with larger margin) or fall back to a different anatomic prior
- Effective post-crop class-balance and per-volume input size

---

### Section 4 — Lesion Size & Morphology

**Goal:** Characterize lesion 3D shape distribution to inform 2.5D stack depth, input resolution, and 2D bbox anchor priors.

**Tasks (108 binarized masks, parallel):**
- 3D connected components (26-connectivity) per mask
- Per CC: voxel count, physical volume mm³, bbox extent (x,y,z) in voxels and mm, slices spanned along through-plane, centroid (voxel + mm)
- Aggregate distributions: P5, P25, P50, P75, P95 for volume, max-extent, slices-spanned
- Number of distinct lesions per patient
- Categorize: micronodule (<5 mm max-extent), nodule (5–30 mm), plaque (>30 mm)
- Smallest 10 lesions: bbox at native resolution and at typical detector input sizes (256, 384, 512)
- Stratify size distribution by scanner_model and slice_thickness_bin (does thicker slicing under-detect small lesions?)

**Decision points:**
- Minimum 2.5D stack depth (informed by `slices_spanned` distribution)
- Native-resolution vs downsample input
- Anchor box scale priors

---

### Section 5 — Slice-Level Analysis & 2D Bounding Box Statistics (new in v2)

**Goal:** Extract per-slice GT boxes; characterize their distribution to inform detector head design and slice-sampling strategy.

**Tasks:**
- **Positives (108):** for each positive volume, identify positive slices along through-plane axis; within each positive slice, 2D connected components → axis-aligned bboxes (`x,y,w,h` in voxels and mm)
- Per-slice: number of boxes, per-box area, aspect ratio
- Per-volume: total positive slices, fraction of total slices, gap structure (contiguous or scattered), per-slice box count
- **Crop comparison:** repeat all of the above inside the liver_roi crop (Section 3); record positive-slice fraction with and without crop
- **Negatives (500):** count slices contributed (no positives) — combined with positives to compute the global positive:negative slice ratio at training time
- Smallest 2D box (pixels, mm²) → minimum resolution constraint
- Aspect-ratio distribution → anchor design

**Decision points:**
- 2.5D context window (3, 5, 7 slices)
- Center-slice positive sampling vs any-slice-with-overlap sampling
- Positive-slice oversampling ratio at the data loader
- Anchor scales / aspect ratios

---

### Section 6 — Cohort Comparability & Shortcut-Learning Risk (new in v2)

**Goal:** Surface any acquisition-metadata difference between positives and negatives that the model could exploit instead of learning lesion morphology.

**Tasks:**
- For each of {`scanner_model`, `slice_thickness`, `series_description`, `body_part`, `patient_age`, `n_slices`, FOV_z, `pulse_sequence_name`, `repetition_time`, `echo_time`}: compute distribution per cohort
- Statistical tests: KS / chi-squared as appropriate; report p-values and effect sizes (Cliff's δ for ordinal)
- Cross-tabulate (`scanner_model × series_description × thickness_bin`): are positive and negative both populated in every stratum?
- Flag any "positive-only" or "negative-only" stratum

**Decision points:**
- Hard imbalance → consider rebalancing the negative pool (sample-weight at training, or excluding outlier strata)
- Soft imbalance → instrument inference-time monitoring per stratum

---

### Section 7 — Intensity Statistics & Normalization

**Goal:** Choose normalization strategy. Detect cohort-level intensity drift.

**Tasks (608 volumes, parallel):**
- Per volume: P0.5, P1, P5, P25, P50, P75, P95, P99, P99.5; mean, std
- Per volume *inside liver_roi*: same percentiles (better proxy for the in-distribution intensity range the model sees)
- **Lesion intensity** (positives, inside binary mask): same percentiles + lesion-vs-non-lesion-bg contrast
- Stratify intensity stats by `cohort × scanner_model × thickness_bin`
- Statistical test: are lesion-bg contrast and per-volume P99 different between cohorts? Between scanners?
- Visualize: overlaid intensity histograms per scanner, per cohort

**Decision points:**
- Normalization method: percentile-clip → z-score (recommended for MR), percentile-clip → min-max, or z-score
- Clip bounds (P1/P99 vs P0.5/P99.5)
- Per-volume vs per-scanner-fitted normalization
- Compute over full volume vs liver_roi only

---

### Section 8 — Liver Mask Quality Control (new in v2)

**Goal:** Catch any TotalSegmentator failures before downstream pipeline trusts liver_masks/liver_rois.

**Tasks:**
- Distribution of `liver_voxel_count` and physical liver volume (cm³) across 608 (5.7× range observed)
- Identify outliers: bottom 10 and top 10 by physical volume
- Per-volume: 3D connected components in liver_mask — flag any with >1 component (TotalSeg should produce a single connected liver)
- Render mid-coronal-slice mask overlay PNG for the 20 size-outliers + 10 random middle-of-distribution cases → visual QC
- Cross-check: for positive volumes, does the lesion centroid sit superior-to (above) the liver dome as expected anatomically?

**Decision points:**
- Manual exclusion list, or re-run TotalSegmentator with different settings, or accept

---

### Section 9 — Volume / Mask / Affine Quality Control

**Goal:** Final integrity sweep before any training script runs.

**Tasks (608 triples for raw+liver, 108 for raw+lesion):**
- Affine match: raw vs lesion_mask; raw vs liver_mask; raw vs liver_roi
- Spacing match
- Shape match
- NaN / Inf in affines, non-positive zooms, malformed headers
- Empty lesion masks (none expected) — and any lesion mask with `lesion_voxel_count > 1 % * total_voxels` (annotation error)
- Shape sanity: every volume `512 × N × 512`
- Visual QC montage:
  - 30 random positives (mid-positive-slice with lesion+liver overlay)
  - 10 random negatives (mid-coronal with liver overlay)
  - The 9 thick-slice cases (carry-over from v1's thick_slice QC, but mapped to mnemonic IDs from manifest filter `slice_thickness_mm > 4.5`)

---

### Section 10 — Fold Balance & Stratification Audit (new in v2)

**Goal:** Verify the seed=42 splits.json folds are balanced for our metrics.

**Tasks:**
- Per fold (5 + holdout): cohort counts, scanner_model breakdown, slice-thickness distribution, age distribution, lesion size category counts (positives), total positive-slice count, mean lesion volume
- Identify any fold extreme (e.g., a fold with disproportionately many thick-slice cases)
- Single-row-per-fold summary table + small-multiples bar charts

**Decision points:**
- Accept frozen splits as-is, or document deviations to monitor in CV reports

---

### Section 11 — Soft-Negative Spot Check (new in v2, small)

**Goal:** Quick verification the 5 transferred soft_negatives are unremarkable as negatives.

**Tasks:**
- Tabulate the 5 mnemonic_ids' `cohort=negative` row from manifest, plus liver_voxel_count, intensity stats, scanner_model
- Compare against the negative-cohort distribution; flag any that look anomalous

---

### Section 12 — Reference Visualizations

Generated alongside the above scripts:
- 3 example positive slices with lesion + liver overlay
- 3 example negative slices with liver overlay
- Lesion-volume histogram (overall + per scanner)
- Spacing/dimension joint scatter
- Lesion-centroid heatmap (coronal/axial/sagittal projections, normalized to volume)
- Lesion-vs-bg intensity histogram overlay
- Per-fold balance bar chart
- Worst liver-ROI containment cases (Section 3 deliverable)

---

## 4. Execution Priority

| # | Section | Why |
|---:|---|---|
| 1 | Volume Geometry (§1) | Foundational; used by everything |
| 2 | Mask Binarization Verification (§2) | Quick gate after destructive binarization |
| 3 | Liver-ROI Containment (§3) | Gates whether liver_roi crop strategy is viable |
| 4 | Lesion Size & Morphology (§4) | Architecture choices |
| 5 | Slice-Level + 2D Bbox (§5) | Direct input to 2.5D detector design |
| 6 | Cohort Comparability (§6) | Shortcut-learning risk surfaced early |
| 7 | Intensity Stats (§7) | Normalization decision |
| 8 | Liver Mask QC (§8) | Catch any TotalSeg failures |
| 9 | QC sweep (§9) | Final integrity check |
| 10 | Fold Balance (§10) | Verify splits before training |
| 11 | Soft-Negative Spot Check (§11) | 1-shot sanity |
| 12 | Visualizations (§12) | Generated alongside above |

---

## 5. Implementation Notes

**Path discovery.** Always read `manifest.csv` (`raw_path`, `lesion_mask_path`, `liver_mask_path`, `liver_roi_path`). Never glob the directory tree (per `data/CLAUDE.md`).

**Parallelism.** `concurrent.futures.ProcessPoolExecutor(max_workers=16)`. Header-only and stat operations are CPU-bound; full-volume reads are IO-bound — for the latter, monitor whether 16 workers is too many for the disk and tune down if IO-saturated.

**Memory & tool-call rule.** `eda/CLAUDE.md` rule "no Read tool call > 5 MB" applies to direct image reads via the Read tool, not to scripts. Scripts may read the full ~19 GB dataset over their run.

**Outputs.** All CSVs and PNGs go to `eda/outputs/` (gitignored). After every script, append findings to `eda/FINDINGS.md`.

**Reproducibility.** Each script logs its inputs (manifest filter), the polars DataFrame schema of its output CSV, and the runtime. Aim for idempotent re-runs.

---

## 6. Migration of v1 scripts

| v1 script | Action |
|---|---|
| `01_volume_geometry.py` | Rewrite for manifest-driven discovery, parallel, cohort×scanner stratification |
| `02_mask_binarization.py` | Rewrite as a *post*-binarization sanity check (binarization itself moved to `scripts/binarize_lesion_masks.py`) |
| `03_lesion_size_morphology.py` | Rewrite manifest-driven, parallel; binarized inputs only |
| `04_lesion_location.py` | Subsumed by `03_liver_roi_containment.py` (new) |
| `05_quality_control.py` | Rewrite, extend to liver_mask + liver_roi checks |
| `06_intensity_statistics.py` | Rewrite; add cohort comparability stratification, in-ROI stats |
| `07_slice_level_analysis.py` | Rewrite; add 2D bbox extraction, post-crop slice ratio |
| `08_multi_sequence_analysis.py` | **Delete** — obsolete (canonical sequence selection upstream removed primary/WATER pairing) |
| `09_thick_slice_qc.py` | Adapt patient list (mnemonic IDs from `manifest.slice_thickness_mm > 4.5`) and fold into §9 visual QC |

**New scripts (numbered by execution priority):**
- `01_volume_geometry.py`
- `02_mask_binarization_verification.py`
- `03_liver_roi_containment.py` ★
- `04_lesion_size_morphology.py`
- `05_slice_level_2d_bboxes.py` ★
- `06_cohort_comparability.py` ★
- `07_intensity_statistics.py`
- `08_liver_mask_qc.py` ★
- `09_quality_control.py`
- `10_fold_balance.py` ★
- `11_soft_negative_spot_check.py` ★

★ = new question that the 608-cohort enabled.

---

## 7. Out-of-scope (for this round)

- Phase 2 unsupervised pool (~4,476 volumes) characterization
- Per-anatomic-location modeling (Segment_1 vs Segment_2 distinction is binarized away)
- Multi-sequence input (only WATER canonical sequence exists)
- Resampling/normalization implementation (this EDA round only *decides* the strategy; implementation is downstream)

## 8. Notes carried over from v1

- nnU-Net Dice ≈ 0.293 was achieved on a similar plaque-segmentation task. Calibrate expectations accordingly.
- Literature: 87.5 % of diaphragmatic endometriosis lesions are right-sided. Section 3 will check whether the 20 mm liver_roi captures the geometric footprint of these lesions.
- Lesions can be micronodules (<5 mm), nodules (5–30 mm), or plaques (>30 mm). With 3.0–3.6 mm typical slice thickness, a 5 mm micronodule may span only 1–2 slices — minimum 2.5D stack depth is bounded below by this fact.
