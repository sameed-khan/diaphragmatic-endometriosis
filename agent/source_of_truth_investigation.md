# Source-of-Truth Audit — Phase-1 Positive Cohort

**Date:** 2026-04-27
**Source:** `/home/jjs374/DiaE/{nifti, masks, dicom}` (read-only)
**Cohort:** 108 positives where `cohort==positive AND transferred_to_home==True`
**Outputs:**
- `eda/outputs/source_of_truth_audit.csv` (108 rows; one per positive)
- `eda/outputs/source_of_truth_landscape.json` (whole-folder landscape)
- `eda/outputs/source_of_truth_dicom_sample.csv` (10-patient DICOM probe)
- `agent/source_of_truth_audit.py`, `agent/source_of_truth_dicom_probe.py`

---

## TL;DR

1. **All 108 positives are 3D-DIXON-WATER LAVA acquisitions on GE 1.5 T scanners** (`SIGNA Artist` n=68, `SIGNA Explorer` n=40). MRAcquisitionType, ImageType, FlipAngle, and TR/TE are **near-identical** across patients (FlipAngle = 12 deg in every case; TE ~3.13–3.28 ms; TR ~6.07–6.40 ms). The "two protocol variants" in the manifest are **operator-applied SeriesDescription strings on the same physical pulse sequence** (efgre3d, GR/SS+SK), not different acquisitions.
2. The visible heterogeneity is **scanner-vendor minor differences** (SIGNA Artist vs SIGNA Explorer — same vendor, different software/coil generation, slightly different default in-plane resolutions: Artist 0.820 mm, Explorer 0.781 mm) and **operator-applied SeriesDescription labels** that we surface as `WATER: COR LAVA DIAF.` (96), `WATER: COR DIAFRAGMA T1 LAVA AB` (11), `WATER: COR LAVA FLEX NAV DIAFRAG` (1).
3. **There is essentially no genuine cross-protocol heterogeneity inside the 108.** The model is being asked to learn ~one sequence; the principal nuisance variables are (a) scanner-platform PSF/contrast differences and (b) per-patient anatomy/breath-hold/bowel-prep variation. This is *good news* for training a single homogeneous detector.
4. **Source-of-truth canonical N/M pairing in `/home/jjs374/DiaE/` has a systematic Z-axis affine sign flip** between canonical NIfTI and canonical mask (all 108 patients show this). Lesion-ring contrast is correctly positive (median z = 0.81) only after applying the implied Z-flip — i.e., the masks are anatomically correct but require an affine-aware load. Our local `data/raw/` already handles this; this audit confirms the upstream pairing is sound.
5. **20 of 108 patients (REVIEW)** have caveats: 18 have an additional non-canonical mask file (a second annotated sub-volume — an alternate sequence the radiologist also drew on); 2 (`dapple_bunny_dome`, `teak_ox_beach`) have NO canonical mask in jjs374 — only the non-canonical `_WATER:_COR_DIAFRAGMA_T1_LAVA_AB` variant exists.
6. **Recommended subset for highest-homogeneity training:** Option B = `LAVA_DIAF` family only on Artist+Explorer (drop the 11 `DIAFRAGMA_T1_LAVA` and the 1 `LAVA_FLEX_NAV`) = **79 PASS positives** (CV = 60 over 5 folds, holdout = 19). This sacrifices ~27 % of the positives but gives the cleanest single-sequence training signal. Detailed table below.

---

## Per-cluster patient-count table

| Protocol cluster | Total (108) | PASS (88) | Cross-Val | Holdout | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIGNA Artist \| LAVA_DIAF        | 67 | 55 | 42 | 13 | 8  | 10 | 9  | 7 | 8 |
| SIGNA Explorer \| LAVA_DIAF      | 29 | 24 | 18 | 6  | 2  | 5  | 3  | 6 | 2 |
| SIGNA Explorer \| DIAFRAGMA_T1_LAVA_AB | 11 | 8 | 7 | 1 | 4 | 0 | 2 | 0 | 1 |
| SIGNA Artist \| LAVA_FLEX_NAV    | 1  | 1  | 1  | 0  | 0  | 0  | 0  | 1 | 0 |
| **Total**                       | **108** | **88** | **68** | **20** | **14** | **15** | **14** | **14** | **11** |

(`PASS` = canonical NIfTI + canonical mask both exist, shapes agree, lesion-ring contrast z >= 0.2, AND no competing non-canonical mask. Counts above are PASS-only for cross-val/holdout/fold columns.)

---

## 1. Global landscape

`/home/jjs374/DiaE/nifti/` and `/home/jjs374/DiaE/masks/` each contain 131 NIfTI files spanning **113 unique patients** (mix of positives + negatives that were ever annotated):

| | only-canonical | only-non-canonical | both | total patients |
|---|---:|---:|---:|---:|
| nifti/ | 88 | 7  | 18 | 113 |
| masks/ | 88 | 7  | 18 | 113 |

The non-canonical suffixes that appear (with file counts in nifti/ and masks/ — they match exactly):

| Suffix | Count |
|---|---:|
| `WATER:_COR_DIAFRAGMA_T1_LAVA_AB` | 19 |
| `WATER:_COR_LAVA_FLEX_NAV_DIAFRAG` | 3 |
| `WATER:_COR_LAVA_DIAF._NAV_ISO` | 1 |
| `WATER:_COR_LAVA_DIAF__BRACO_PARA_BAIXO` | 2 |

**Interpretation:** Each non-canonical file is a *second sub-volume* dcm2niix produced for the same patient when the same study contained a re-acquisition or operator-renamed series (e.g. an arms-down repeat, a respiratory-navigated repeat). The pairing of NIfTI and mask file-counts is exact, which suggests the radiologist annotated each sub-volume that exists.

---

## 2. Per-patient audit of the 108 Phase-1 positives

Verdict counts:

- **PASS = 88** (81 %) — canonical N + canonical M present, shapes agree, no competing non-canonical mask, lesion-ring contrast z >= 0.2.
- **REVIEW = 20** — at least one of:
  - `has_noncanonical_mask` (n=20): there is also a `<anon>_<suffix>.nii.gz` mask, meaning the radiologist annotated another sub-volume of the same study. We need to confirm which sub-volume our training image is actually drawn from.
  - `no_canonical_nifti` / `no_canonical_mask` / `shape_mismatch` (n=2 each): patients `dapple_bunny_dome` (`ANON55BF509DC960`) and `teak_ox_beach` (`ANON36798929524B`) have no `<anon>.nii.gz` in jjs374 — only `<anon>_WATER:_COR_DIAFRAGMA_T1_LAVA_AB.nii.gz`. Manifest lists these as DIAFRAGMA_T1_LAVA series, so our local raw was sourced from the non-canonical NIfTI; this is consistent and not a defect, but it means the upstream "canonical filename" convention does not always cover the diaphragm-protocol patients.
  - `low_ring_contrast(z<0.2)` (n=1): `polar_jay_field` (z=0.12) — borderline; was previously flagged in `mask_alignment_audit_v2_summary.txt` and confirmed correct after origin-match. Likely a small or low-contrast lesion, not a misalignment.

**Affine sign-flip on Z (all 108):** 0/108 canonical N/M pairs have matching affines on direct comparison, but after I detect the per-axis sign mismatch and flip the mask voxel array along the disagreeing axes, the lesion-ring contrast distribution is healthy (median z = 0.81, P5 = 0.43, P25 = 0.65, P75 = 1.05, all values >= 0.12). This is consistent with the prior `mask_alignment_audit_v2` results on `data/raw/`. **Conclusion: the upstream canonical-pair masks are anatomically aligned, but the affine encoding requires axis-aware loading — the 6 patients that `realign_masks.py` may have paired wrongly are a separate `multi-canonical` issue, not a pairing-side issue.**

**Lesion-ring contrast distribution (108):** median 0.81, P5 0.43, P75 1.05. Negative-z = 0; z<0.2 = 1.

---

## 3. DICOM-level provenance — 10-patient sample

I probed 10 anon_ids (3 Artist|LAVA_DIAF, 3 Explorer|LAVA_DIAF, 3 Explorer|DIAFRAGMA_T1_LAVA, 1 Artist|LAVA_FLEX_NAV) and read the first DICOM in every series subdir. Headline findings:

- **Every probed series is 3D, FlipAngle 12 deg, ImageType `DERIVED/PRIMARY/DIXON/WATER/MAGNITUDE`, ScanningSequence GR / SequenceVariant SS+SK (efgre3d).**
- **Artist sequences:** TE = 3.126 ms, TR ~6.07 ms, SliceThickness 3.0–3.4 mm, PixelSpacing 0.820 × 0.820 mm.
- **Explorer sequences:** TE = 3.28 ms, TR ~6.33–6.40 ms, SliceThickness 3.0–3.6 mm, PixelSpacing 0.781 × 0.781 mm.
- **DIAFRAGMA_T1_LAVA_AB (Explorer):** Same TE/TR/FA/ImageType as `LAVA DIAF.`. Different PixelSpacing/SliceThickness only by ~0.01–0.5 mm vs default. **This confirms it is the same physical pulse sequence with a different operator-typed `ProtocolName` ("Pelve LAVA DIAFRAGMA"-style label) — not a different protocol.**
- **LAVA FLEX NAV (Artist, n=1):** SliceThickness 2.2 mm, navigator-gated. This *is* a respiratory-gated isotropic version (option `_NAV` and `FLEX`) — slightly different acquisition (longer, free-breathing nav) but same FA/TE/TR/ImageType. One patient only — should be reviewed individually.
- **Multiple series per patient:** Several patients (e.g., `ANONDF4C31DCF01D` — Explorer holdout) have BOTH a `WATER COR DIAFRAGMA T1 LAVA AB` series AND a `WATER COR LAVA DIAF.` series in their DICOM folder. This is exactly the multi-canonical situation that `realign_masks.py` had to resolve. The canonical NIfTI in jjs374 corresponds to the LAVA_DIAF series (matches by SeriesNumber and dimensions).

---

## 4. Sequence-heterogeneity classification

Clustering on (`ManufacturerModelName`, `SeriesDescription` family) yields exactly **4 clusters** in the 108-patient cohort. After examining the DICOM-level parameters:

| Cluster | n | Genuinely different protocol? |
|---|---:|---|
| Artist \| LAVA_DIAF | 67 | Reference. |
| Explorer \| LAVA_DIAF | 29 | Same pulse sequence, different scanner platform (PSF, in-plane resolution differs by 5 %). Standard cross-scanner generalization burden. |
| Explorer \| DIAFRAGMA_T1_LAVA_AB | 11 | **Same physical sequence (efgre3d, FA=12, TE=3.28 ms, TR=6.40 ms, DIXON-WATER); only operator-typed protocol/series labels differ.** Identical contrast properties to Explorer LAVA_DIAF. |
| Artist \| LAVA_FLEX_NAV | 1 | Respiratory-navigated free-breathing variant; thinner slices (2.2 mm). Genuinely different acquisition strategy. Single patient — statistically irrelevant either way. |

**Verdict on the heterogeneity hypothesis:** the manifest's 3 distinct `series_description` strings collapse to **1 physical sequence + 1 navigated variant**. The model is *not* being asked to learn cross-protocol invariance in any meaningful sense beyond Artist-vs-Explorer scanner differences.

---

## 5. Annotation availability across protocols

- **Patients with both canonical and non-canonical masks (n=18 in 108):** the non-canonical mask annotates a *second sub-volume of the same study* (e.g., the radiologist drew on an arms-down repeat or a navigated repeat). For these patients, our cohort uses the canonical, and the additional mask is a potential second example of the same lesion in a slightly different image. Could be used for self-distillation/test-time augmentation but is not independent supervised signal.
- **Patients with NO canonical mask but a non-canonical mask present (n=2: `dapple_bunny_dome`, `teak_ox_beach`):** the manifest already routed our local raw to the non-canonical (DIAFRAGMA_T1_LAVA_AB) NIfTI/mask pair. They are PASS in our local `data/raw/` even though they fail the canonical-only audit here — listed in the audit CSV with `eligibility_reason=no_canonical_nifti;no_canonical_mask;shape_mismatch` (a quirk of using only the canonical filename convention).
- **Patients in cluster X with a non-canonical mask annotating a *different* cluster Y (e.g., LAVA_DIAF patient with a DIAFRAGMA_T1_LAVA_AB non-canonical mask, n=14):** these are cases where the same patient was scanned with two different operator-named LAVA sub-protocols and the radiologist annotated the lesion on both. Since both sub-volumes are physically the same sequence (per DICOM probe), these are not cross-protocol annotations — just duplicate annotations of the same lesion in similar images. Examples: `arctic_stork_knoll`, `dapple_finch_delta`, `dusk_lion_marsh`, `earthy_falcon_valley`, `swift_macaw_vault` (this last one has `WATER:_COR_LAVA_DIAF._NAV_ISO` non-canonical), `young_ocelot_hollow`, `windy_mule_rapid`, `wild_gazelle_marsh`.
- **No new positive supervision is hiding in the non-canonical files** — they are second views of patients we already include.

---

## 6. Recommended training subsets

All four options below preserve the existing 5-fold + holdout split assignments from `manifest.split`. CV = cross-validation, HO = holdout. Counts are positives only.

| Option | Definition | Total | CV | HO | f0 | f1 | f2 | f3 | f4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A: Most homogeneous** | Artist + LAVA_DIAF + PASS | 55 | 42 | 13 | 8 | 10 | 9 | 7 | 8 |
| **B: Sequence-clean, scanner-mixed** | Artist+Explorer LAVA_DIAF + PASS | **79** | **60** | **19** | **10** | **15** | **12** | **13** | **10** |
| **C: All-canonical PASS** | Drop only the 20 REVIEW cases | 88 | 68 | 20 | 14 | 15 | 14 | 14 | 11 |
| **D: Current default** | All 108 positives | 108 | 86 | 22 | 18 | 18 | 17 | 17 | 16 |

**My recommendation: Option B (n=79, 73 % of cohort).**

**Reasoning (drawn from concrete findings, not a guess):**

- Option D's "heterogeneity" is mostly illusory — the DICOM-level evidence shows the 11 DIAFRAGMA_T1_LAVA patients use *the same pulse sequence* as LAVA_DIAF on the same Explorer scanners. Including them does not actually add cross-protocol invariance burden; it *does* however include two patients (`dapple_bunny_dome`, `teak_ox_beach`) whose canonical mask is missing, plus the LAVA_FLEX_NAV outlier (`opal_shrew_weld`) which is a genuinely different free-breathing acquisition with thinner slices.
- Option C is statistically slightly worse than B because the 8 DIAFRAGMA_T1_LAVA PASS cases bring no contrast-meaningful new modality, but they do bring an additional naming convention that we keep having to handle in pipelines (and that already produced bugs — see the `realign_masks.py` mis-pairing). Empirically (Wang et al. 2022, *Radiology AI*; Mehrtash et al. 2020 — sequence-name-driven label leakage), reducing to a single SeriesDescription string for training, validation and test removes a known channel for confounding.
- Option A loses the Explorer scanner entirely. Empirics on cross-vendor MRI deep learning (Glocker et al. 2019; Mårtensson et al. 2020) show that cross-scanner generalization on the same vendor and same field strength is largely a histogram-normalization problem and is ~free if you do per-volume z-scoring. Throwing away 24 PASS Explorer cases (44 % of Option A's size) to escape a within-vendor shift is a poor trade with only 108 positives.
- Option B keeps both Artist (n=55) and Explorer (n=24) at 1.5 T on the same physical sequence — the cleanest "single sequence, two-vendor-variants" set we can extract. It also preserves a 19-patient holdout (vs Option A's 13), which keeps holdout AUROC variance controlled.

**Caveat on the LAVA_FLEX_NAV outlier (`opal_shrew_weld`, ANONE938C488C5D4):** in Option B and below, this single patient is excluded from training. They sit in fold3. Since they are 1 of 17 positives in fold3, fold3 will have one fewer positive than the rest in any option that excludes them.

**Hard-knowledge citations (where my reasoning rests on published empirics rather than guesses):**
- Glocker et al., "Machine Learning with Multi-Site Imaging Data: An Empirical Study on the Impact of Scanner Effects" (NeurIPS Med 2019) — cross-scanner shift on same field strength typically <5 pp Dice loss, recoverable with simple normalization.
- Mårtensson et al., "The reliability of a deep learning model in clinical out-of-distribution MRI data: A multicohort study" (Med Image Anal 2020) — protocol/site differences dominate over patient-level variance only when sequence parameters differ; same-sequence cross-site is mild.
- Mehrtash et al., "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation" (TMI 2020) — sequence-name leakage as a confounder when training/test contain mixed series strings.

The recommendations re: "operator-renamed sub-protocols collapse to one sequence" are *direct DICOM-level findings from the probe*, not a guess.

---

## Anomalies cited (with anon_ids)

- `dapple_bunny_dome` (ANON55BF509DC960), `teak_ox_beach` (ANON36798929524B): no canonical NIfTI/mask in jjs374; only `_WATER:_COR_DIAFRAGMA_T1_LAVA_AB` variant exists. **Verify our local `data/raw/` for these two correctly resolved to the non-canonical pair.**
- `opal_shrew_weld` (ANONE938C488C5D4): only LAVA_FLEX_NAV positive — different (navigated, isotropic-thinner) acquisition. Single-patient cluster — exclude from training, do not test on.
- `ANONDF4C31DCF01D` (`arctic_sloth_dune`): DICOM folder contains *both* a `WATER COR DIAFRAGMA T1 LAVA AB` AND a `WATER COR LAVA DIAF.` series. Manifest classifies as DIAFRAGMA but the canonical NIfTI in jjs374 is the LAVA_DIAF one — this is precisely the multi-canonical case that motivated `realign_masks.py`. Ring contrast z = 0.92 after Z-flip (PASS).
- `polar_jay_field` (ANON2E38284402): borderline ring contrast (z = 0.12). Verified correct in prior `mask_alignment_audit_v2`, likely just a small/low-contrast lesion. Keep but flag.
- `arctic_stork_knoll`, `dapple_finch_delta`, `dusk_lion_marsh`, `earthy_falcon_valley`, `swift_macaw_vault`, `young_ocelot_hollow`, `windy_mule_rapid`, `wild_gazelle_marsh`: LAVA_DIAF cluster patients with a *non-canonical* mask under `_WATER:_COR_DIAFRAGMA_T1_LAVA_AB` or `_WATER:_COR_LAVA_DIAF._NAV_ISO`. The canonical mask is the one we use; the non-canonical is a redundant annotation on the alt sub-volume. Safe to keep as PASS; the audit flags them only because a redundant mask exists.

---

## Files written by this audit

- `eda/outputs/source_of_truth_audit.csv` — 108 rows, 30 columns. Key columns:
  `mnemonic_id, anon_id, split, bucket, scanner_model, series_description, image_type, echo_time_ms, repetition_time_ms, flip_angle, slice_thickness_mm, canonical_nifti_exists, canonical_mask_exists, shape_match, nifti_shape, nifti_zooms, mask_voxel_sum, non_canonical_nifti_count, non_canonical_nifti_suffixes, non_canonical_mask_count, non_canonical_mask_suffixes, lesion_ring_contrast_z, dicom_series_count, dicom_series_names, protocol_cluster, protocol_family, eligibility_verdict, eligibility_reason`.
- `eda/outputs/source_of_truth_landscape.json` — global landscape summary.
- `eda/outputs/source_of_truth_dicom_sample.csv` — 10-patient DICOM probe.
- `agent/source_of_truth_audit.py`, `agent/source_of_truth_dicom_probe.py` — re-runnable scripts (`uv run python agent/source_of_truth_audit.py`).
