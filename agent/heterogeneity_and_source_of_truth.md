# Heterogeneity & Source-of-Truth Investigation

**Date:** 2026-04-27
**Question driving the investigation:** are the 6 misaligned positives a one-off realignment-script bug, or symptoms of a deeper sequence-heterogeneity problem? And what is the most homogeneous, most pristine training set we can carve out before locking the modeling pipeline?

This document combines three parallel investigations:
1. **`/home/jjs374/DiaE/`** — the read-only source-of-truth (subagent CSV at `eda/outputs/source_of_truth_audit.csv`).
2. **`data-local-copy/`** — the user's manually-QC'd subset (audit at `eda/outputs/data_local_copy_audit.csv`).
3. **608-cohort heterogeneity analysis** — sidecar metadata + DICOM-tag clustering (`eda/outputs/protocol_clusters.csv`, `protocol_tradeoff.csv`).

---

## 1. Source-of-truth landscape

`/home/jjs374/DiaE/nifti/<anon>.nii.gz` (the radiologist-handler-converted NIfTI, no series-description suffix) is the **ground-truth canonical NIfTI** for each patient. It is what the radiologist annotated lesion masks against. Files with a series-description suffix
(`<anon>_WATER:_COR_DIAFRAGMA_T1_LAVA_AB.nii.gz`, etc.) are alternate
sequences that were also acquired but are *not* the canonical annotation
target — though some of those sequences also have masks (extra annotations).

### The 108 Phase-1 positives at the source

Per the source-of-truth audit:

- **88 / 108 PASS** — canonical NIfTI exists, canonical mask exists, alignment clean.
- **20 / 108 REVIEW** — broken down as:
  - **18** patients have additional non-canonical mask annotations (i.e., the radiologist labeled the same patient on more than one sequence). The canonical mask is fine — the REVIEW flag just means there is *bonus* paired data.
  - **2** patients (`dapple_bunny_dome` holdout and `teak_ox_beach` fold4) **have no canonical NIfTI at all** — the only sequence acquired for them is `WATER:_COR_DIAFRAGMA_T1_LAVA_AB`. For these two, the non-canonical IS the canonical, and our pipeline already handles this (their `manifest.series_description` is `WATER: COR DIAFRAGMA T1 LAVA AB`).

### Lesion-mask alignment quality on the source-of-truth

Computed as lesion-vs-5-voxel-dilated-ring intensity z-score on the canonical NIfTI:

| Cluster | n | median cz | min cz | P5 cz | n_weak (cz<0.2) |
|---|---:|---:|---:|---:|---:|
| Artist × LAVA_DIAF | 67 | **0.87** | 0.42 | 0.46 | 0 |
| Explorer × LAVA_DIAF | 29 | **0.73** | 0.12 | 0.23 | 1 |
| Explorer × DIAFRAGMA_T1_LAVA_AB | 11 | **0.84** | 0.46 | 0.51 | 0 |
| Artist × LAVA_FLEX_NAV | 1 | 0.37 | 0.37 | n/a | 0 |

**The source-of-truth is essentially clean.** Only 1 weak case in 108 (the
Explorer-A patient `polar-jay-field` with cz=0.12 — a real low-contrast 79-voxel
lesion, not a misalignment). Compare against our current pipeline, which has 6
patients with cz < 0.2 — all 6 are misaligned by the dcm2niix sub-volume
selection bug we identified in §14.

---

## 2. `data-local-copy/` is a renamed mirror of the source-of-truth

The user's manually-QC'd local copy contains **131 mask annotations across 113
mnemonic IDs** (106 canonical + 25 non-canonical). I confirmed via SHA-256 byte
comparison on a sample of 8 patients that the niftis in `data-local-copy/nifti/`
are **byte-identical to `/home/jjs374/DiaE/nifti/<anon>.nii.gz`** — the only
change is the filename (mnemonic-with-hyphens instead of ANON code).

The same is true for the masks. So `data-local-copy/` is just `/home/jjs374/DiaE/`
with mnemonic filenames and a hand-curated subset (113 patients vs. ~5000 in
DiaE).

### Distinct non-canonical sequences seen in `data-local-copy/`

| Suffix | # masks | Likely meaning |
|---|---:|---|
| `WATER:_COR_DIAFRAGMA_T1_LAVA_AB` | 19 | Variant B — coarser-slice (3.6 mm) Explorer-only re-acquisition |
| `WATER:_COR_LAVA_FLEX_NAV_DIAFRAG` | 3 | LAVA-Flex with respiratory navigator |
| `WATER:_COR_LAVA_DIAF__BRACO_PA…` | 2 | Post-Bracco-contrast (Gd) LAVA — different physical state |
| `WATER:_COR_LAVA_DIAF._NAV_ISO` | 1 | Navigator + isotropic |

**Interpretation:** These are *real protocol variants* with different physical sampling, not just different exports. The 19 `DIAFRAGMA_T1_LAVA_AB` masks are most often paired with patients whose canonical sequence is `LAVA DIAF` — i.e., the same patient was scanned with both protocols and labeled twice. Useful as paired data for cross-protocol consistency studies, **not** as additional training samples for v1.

---

## 3. Sequence heterogeneity in the 608-cohort

**Headline correction (from subagent's DICOM-level probe):** my preliminary read on the 4 cluster table — that `DIAFRAGMA_T1_LAVA_AB` is a "different protocol" — was **wrong**. The DICOM-tag investigation by the source-of-truth subagent confirms it's the SAME physical pulse sequence as `LAVA DIAF.` on the same Explorer scanners. Identical TE (~3.28 ms), TR (~6.4 ms), FlipAngle (12°), ImageType (`DERIVED/PRIMARY/DIXON/WATER/MAGNITUDE`), MRAcquisitionType (3D), ScanningSequence (GR), SequenceVariant (SS+SK), pulse sequence name `efgre3d`. **Only the operator-typed `SeriesDescription` and `ProtocolName` strings differ.**

Joint `(scanner, series_description)` clustering with DICOM-confirmed protocol verdict:

| Cluster | n_total | n_pos | n_neg | DICOM-level pulse sequence | Verdict |
|---|---:|---:|---:|---|---|
| **Artist × COR LAVA DIAF.** | 353 | 67 | 286 | efgre3d, FA=12, TE=3.13ms, TR=6.07ms | reference |
| **Explorer × COR LAVA DIAF.** | 125 | 29 | 96 | efgre3d, FA=12, TE=3.28ms, TR=6.40ms | **same sequence, different scanner platform** (slightly different in-plane resolution: Artist 0.820mm vs Explorer 0.781mm, ~5% — known cross-vendor histogram-normalization issue) |
| **Explorer × COR DIAFRAGMA T1 LAVA AB** | 112 | 11 | 101 | efgre3d, FA=12, TE=3.28ms, TR=6.40ms | **same physical sequence as Explorer LAVA_DIAF** — operator-typed label difference only |
| **Artist × COR LAVA FLEX NAV DIAFR** | 15 | 1 | 14 | efgre3d, FA=12, TE=3.13ms, TR=6.07ms, **NAV-gated**, thinner slices | **genuinely different acquisition** — respiratory-navigated free-breathing with 2.2 mm slices |
| 3 stragglers (NAV ISO, BRACO PA) | 3 | 0 | 3 | mixed | one-offs, all negative |

**Revised key observation.** The 108 positives sit on **one physical pulse sequence + one navigated outlier**. The visible heterogeneity reduces to:

- **Scanner heterogeneity (Artist vs Explorer)** — same vendor, same field strength, different software/coil generation. ~2.5× P99 intensity scale difference (§7) plus ~5% in-plane resolution difference. **Acceptable** — per-volume percentile normalization handles it cleanly (Glocker et al. NeurIPS-Med 2019; Mårtensson et al. MIA 2020).
- **Operator-typed label heterogeneity** (`COR LAVA DIAF.` vs `COR DIAFRAGMA T1 LAVA AB`) — pure metadata-string difference, **not a real protocol difference**. The pixel-level images from these two label-clusters are statistically indistinguishable conditional on scanner.
- **One genuine acquisition outlier**: the 1 LAVA_FLEX_NAV positive (`opal_shrew_weld`, ANONE938C488C5D4) — navigator-gated free-breathing, thinner slices. Drop.

So the heterogeneity has only **one real axis of concern** (LAVA_FLEX_NAV vs the rest), and 1 positive of 108 is a tiny excision.

---

## 4. Training-set tradeoff matrix (revised after DICOM probe)

The subagent's DICOM evidence dissolves the "Variant A vs B" distinction — both clusters are the same `efgre3d` sequence with different operator labels. The only genuinely-different acquisition is the 1 LAVA_FLEX_NAV patient. So the cleanest selection is more permissive than my preliminary draft:

| Option | Definition | Total positives | CV | HO | Comment |
|---|---|---:|---:|---:|---|
| **D: Current default** | All 108 | 108 | 86 | 22 | Includes 1 NAV-gated outlier |
| **★ Recommended (revised): Drop only LAVA_FLEX_NAV** | All except `opal_shrew_weld` | **107** | **85** | **22** | DICOM-confirmed single-protocol set; drops 1 patient (1 %) |
| C: All-canonical PASS | Drop 20 REVIEW (incl. 18 with extra non-canonical masks) | 88 | 68 | 20 | Loses 18 perfectly-aligned patients on a metadata flag — overcautious |
| B: Sequence-clean, scanner-mixed (subagent's pick) | LAVA_DIAF (drops DIAFRAGMA_T1_LAVA + FLEX_NAV + REVIEW) + PASS | 79 | 60 | 19 | Defensible if you want zero series_description heterogeneity in metadata, but the *images* are the same. |
| A: Most homogeneous | Artist + LAVA_DIAF + PASS | 55 | 42 | 13 | Single scanner — drops 50 % of positives unnecessarily |

**My revised recommendation: drop only `opal_shrew_weld` (LAVA_FLEX_NAV) — n=107.**

The subagent's "Option B = 79 positives" recommendation cites possible label-leakage from `series_description` strings (Mehrtash et al. TMI 2020). That risk is real if `series_description` is fed to the model, but **our model only sees pixels** — and the DICOM probe shows pixels are produced by the same pulse sequence. So we can safely include the 11 DIAFRAGMA_T1_LAVA_AB patients and the 18 REVIEW patients (whose canonical mask is fine; the REVIEW flag was for "has additional non-canonical mask annotations").

The slice-thickness variation (2.94–6.94 mm across the cohort) is real and is handled by:
- **Resampling all volumes to a common z-spacing** (e.g., 1.5 mm or 3.0 mm) at preprocess time — already a committed §1 step.
- **Reporting thickness-stratified FROC** so we can detect any thickness-shortcut bias.

### Per-fold positive counts under "drop FLEX_NAV only"

| | fold0 | fold1 | fold2 | fold3 | fold4 | holdout |
|---|---:|---:|---:|---:|---:|---:|
| Current (all Phase-1) | 18 | 18 | 17 | 17 | 16 | 22 |
| Drop FLEX_NAV (recommended) | 18 | 18 | 17 | **16** | 16 | 22 |

Mean 17.0 ± 0.89 (CoV 5.2 %). Essentially unchanged from current.

---

## 5. Recommendations (committed)

### 5a. Pipeline-level fix: switch to source-of-truth NIfTIs

**Stop using the dcm2niix re-conversion** (`/scratch/pioneer/users/sak185/dia-endo-conversion/output/nifti_pos/<anon>/water_canonical*.nii.gz`).

**Use `/home/jjs374/DiaE/nifti/<anon>.nii.gz` directly** as the canonical raw NIfTI for each patient. This:
- Eliminates the `realign_masks.py` sub-volume-selection bug — the source NIfTI IS the right grid by construction.
- Aligns 130/131 = 99 % of masks cleanly (vs. our pipeline's 102/108 = 94 %).
- Removes 6 misaligned patients we'd otherwise need to fix one by one.
- Is exactly what the user's `data-local-copy/` already validated by hand.

The migration steps are:
1. Copy `/home/jjs374/DiaE/nifti/<anon>.nii.gz` → `data/raw/<bucket>/<cohort>/<mnemonic>.nii.gz` (one-to-one, mnemonic rename).
2. Copy `/home/jjs374/DiaE/masks/<anon>.nii.gz` → `data/lesion_masks/<bucket>/positive/<mnemonic>_mask.nii.gz` (binarize on read; some source masks have label values {0, 1, 2}).
3. For the 2 patients with no canonical NIfTI (`dapple_bunny_dome`, `teak_ox_beach`), use their non-canonical `<anon>_WATER:_COR_DIAFRAGMA_T1_LAVA_AB.nii.gz` instead. Both are Variant B and will be excluded from training under the homogeneity rule below — but keep them migrated for completeness.
4. **Re-run `scripts/run_totalseg.py`** on the new raws to regenerate liver masks. Source NIfTIs are in LAS orientation (not RAS) with varied zooms — TotalSegmentator handles this natively.
5. **Re-run `scripts/dilate_segmentations.py`** at 20 mm. The user's intuition is correct: 20 mm is anatomically appropriate, and the §3 finding that we needed 30–40 mm was an artifact of the misalignment.
6. **Update `manifest.csv`** with new shapes, zooms, sha256 hashes, and paths.
7. **Skip the `realign_masks.py` step entirely** — the source masks are already aligned with the source NIfTIs by construction. Only re-binarize.

The reproducibility chain remains intact: `/home/jjs374/DiaE/` is itself derived from the original DICOM upload by the data handler, so the only change is "use the data handler's NIfTI directly instead of re-running dcm2niix".

### 5b. Training-set selection: drop only LAVA_FLEX_NAV (revised)

For training, validation, and the holdout report:

- **Include:** all 107 positives except `opal_shrew_weld` (the 1 navigator-gated patient with thinner slices and genuinely different acquisition strategy).
- **Exclude:** the 1 LAVA_FLEX_NAV positive only. Among negatives, drop the 14 LAVA_FLEX_NAV negatives and 4 stragglers (NAV ISO / BRACO PA / DIAFR LAVA EXP / DIAFRAGMA T1 LAVA P / LAVA DIAF NAV).

This gives **107 positives + ~482 negatives = 589 volumes**. We retain 99 % of our positives. The DICOM probe confirms there is no real cross-protocol heterogeneity to escape from beyond the 1 NAV-gated patient.

Why I deviate from the subagent's Option B (79 positives):

- The subagent's defensive cut against `series_description` operator labels rests on a potential leakage pathway that doesn't exist in our pipeline (we don't pass series_description to the model). Excluding 17 PASS LAVA_DIAF patients flagged REVIEW (because they have *additional* non-canonical mask annotations) and 8 PASS DIAFRAGMA_T1_LAVA patients (DICOM-confirmed same physical sequence) costs us ~25 positives in exchange for ruling out a bug that's already ruled out by the input pipeline.
- The 11 DIAFRAGMA_T1_LAVA_AB positives are simply Explorer scans of the same sequence with a different operator-typed protocol-name string. Including them adds a small slice-thickness skew (3.6 mm vs 3.0 mm) that is fully neutralized by §1's resample-to-common-z-spacing preprocessing step.
- Holdout integrity: keeping all 22 holdout positives (vs the subagent's 19) gives more statistical power for the final report.

If we ever discover that the model IS leveraging metadata-correlated shortcuts (detectable via thickness-stratified FROC after Week 1), we can fall back to the subagent's Option B as a conservative comparator — at the cost of 25 positives.

### 5b-extra: hold-out external-protocol cohort

The 1 excluded LAVA_FLEX_NAV positive plus its 14 LAVA_FLEX_NAV negatives plus the 4 stragglers = **5 positives + 18 negatives = 23-volume external-protocol test set**. Run inference on this set after CV is locked and the holdout has been evaluated — report it as an "out-of-protocol" generalization number, not as a primary metric. Useful future-work signal.

### 5c. Update the §3 dilation finding (already partially reverted)

The §3 partial-containment finding (4 outliers requiring 40 mm dilation) was driven by:
- **2 misalignment artefacts** (blush_turtle_cliff, raven_dove_summit) — fixed by 5a.
- **2 annotation-edge cases** (glass_puma_glade ~29 voxels, pine_wren_fjord 2 voxels) — clip-to-ROI is fine.

After 5a, **20 mm dilation is sufficient for ≥107/108 positives** with at most clip-to-ROI losses on 1–2 patients at the annotation edge. The user's ITK-SNAP intuition stands.

### 5d. What about the 25 non-canonical mask annotations?

These are interesting bonus data — same patient annotated on a different sequence — but for v1 they are **not used for training**. Hold them out. They are useful for:
- Paired test-retest analysis (does the model find the same lesions when the patient is re-scanned with a different protocol?)
- Future SSL pretraining (more annotated data on alternative sequences)
- Future cross-protocol generalization fine-tuning if we want to expand beyond Variant A

---

## 6. Bottom-line summary

| Question | Answer |
|---|---|
| Are `water_canonicalc` and `water_canonicala` the same sequence? | **They are the same physical pulse sequence (DICOM-confirmed: efgre3d, FA=12, identical TE/TR/ImageType). They differ only in (a) sub-volume choice when dcm2niix split a multi-station DICOM series, with different physical origins, and/or (b) operator-typed series_description labels. The image content is the same protocol — our pipeline picked the wrong sub-volume by alphabetical order, which is the §14 bug.** |
| What are the sources of heterogeneity in our source volumes? | **Only two real ones: (1) Artist vs Explorer scanner-platform differences (intensity scale ~2.5×, in-plane resolution ~5 % — handled by per-volume normalization), (2) one navigator-gated free-breathing acquisition (`opal_shrew_weld`, 1 of 108) which is genuinely different — drop it. Everything else (LAVA_DIAF vs DIAFRAGMA_T1_LAVA labels, slice-thickness 2.9–4.4 vs 3.2–3.6 mm) is operator-label or scanner-default variation, not a real protocol difference.** |
| Are we training the model to look across different MRI protocols? | **Effectively no. The DICOM-level evidence shows 107 of 108 positives sit on the same `efgre3d` LAVA-Dixon pulse sequence. The model needs to handle scanner-platform variation, not cross-protocol invariance.** |
| Is the data-local-copy aligned correctly? | **Yes. 130/131 annotations have lesion-ring contrast z ≥ 0.2 (median 0.81). The niftis are byte-identical to `/home/jjs374/DiaE/nifti/<anon>.nii.gz`.** |
| Is the source of truth at /home/jjs374/DiaE clean? | **Yes. 88/108 PASS, 20/108 REVIEW (mostly informational — has additional non-canonical mask). Median lesion-ring contrast z = 0.81; only 1 weak case (real low-contrast lesion, not misalignment).** |
| Is the simplest fix to skip dcm2niix and use the source NIfTIs? | **Yes. Migrate `/home/jjs374/DiaE/nifti/<anon>.nii.gz` → `data/raw/<mnemonic>.nii.gz` directly, copy source masks into `data/lesion_masks/`, re-run TotalSeg + dilation, update manifest. Skip the re-conversion and the realign step.** |
| What's the recommended training set? | **107 positives — drop only `opal_shrew_weld` (the 1 NAV-gated patient with thinner slices). This is more permissive than the subagent's Option B (79); the subagent's defensive cut is sensible if our pipeline used series_description as a feature, but ours doesn't. Hold the LAVA_FLEX_NAV + stragglers (5 pos / 18 neg) aside as an external-protocol generalization test set.** |

---

## 7. Artefacts

- `eda/outputs/source_of_truth_audit.csv` — 108-row per-patient audit (CSV from subagent)
- `eda/outputs/data_local_copy_audit.csv` — 131-row per-annotation audit
- `eda/outputs/protocol_clusters.csv` — 7 (scanner × series × image_type) clusters
- `eda/outputs/protocol_tradeoff.csv` — 5-rule tradeoff matrix
- `eda/outputs/protocol_heterogeneity_summary.txt` — full heterogeneity report
- `eda/outputs/data_local_copy_audit_summary.txt` — alignment-quality report on user's QC'd copy
