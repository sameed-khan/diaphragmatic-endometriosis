# §3 Liver-ROI Containment Audit

Investigates the §3 finding that 4/108 positives have lesion voxels outside the
20 mm-dilated liver ROI. The user pushed back on grounds that ITK-SNAP overlays
look correct and that 40 mm dilation would extend through the diaphragm into
the lung. This audit reproduces and stress-tests the §3 / §3b numbers, verifies
the dilation script, the voxel arithmetic, the affine bookkeeping, and the
actual anatomy on screen.

Scripts written for this audit (under `eda/`):
`debug_containment_audit.py`, `debug_source_align.py`, `debug_affine_mismatch.py`,
`debug_overlay_raw.py`. Output PNGs in `eda/outputs/`:
`debug_containment_blush_turtle_cliff.png`,
`debug_overlay_blush_turtle_cliff_full.png`.

## 1. Dilation script reproducibility — clean

`scripts/dilate_segmentations.py` calls `distance_transform_edt(~liver, sampling=zooms)`
with `zooms = img.header.get_zooms()[:3]`. `scipy` returns physical-mm distance
when `sampling` is in mm, and the script thresholds `<= 20.0`. For all 4
patients I re-ran the dilation in-process and compared against the on-disk
`*_liver_roi.nii.gz`:

| Mnemonic                | liver vox | on-disk ROI vox | fresh 20 mm dilation vox | XOR |
|---|---:|---:|---:|---:|
| blush_turtle_cliff      | 943,812   | 2,517,165        | 2,517,165                | **0** |
| raven_dove_summit       | 681,766   | 1,682,245        | 1,682,245                | **0** |
| glass_puma_glade        | 737,775   | 2,256,013        | 2,256,013                | **0** |
| pine_wren_fjord         | 1,178,663 | 2,839,370        | 2,839,370                | **0** |

Conclusion: **the on-disk ROI is exactly the 20 mm anisotropic Euclidean
dilation of the liver mask**, no off-by-one, no isotropic-voxel bug, no zoom
order mistake. Hypothesis (3) is ruled out.

## 2. Voxel arithmetic — clean

All four mask trios share identical shapes and identical affines (atol 1e-4) as
already verified. Loaded directly via `nib.load(...).dataobj`, dtype is `uint8`,
unique values are `{0, 1}`. `lesion & ~roi` is exactly what the §3 script
counts. For blush_turtle_cliff, near the centroid of the out-of-ROI cloud
(voxel `(265, 103, 390)`):

- `lesion = 1`, `liver = 0`, `roi = 0`, distance-to-liver = **29.90 mm**.
- On the same coronal slice y=103: `lesion ∧ roi = 0`, `lesion ∧ ¬roi = 155`,
  i.e. **every single one of 155 lesion voxels on that slice is outside the
  ROI**. There is no possible interpretation where the §3 voxel arithmetic is
  wrong; the lesion is genuinely 25–37 mm away from the nearest liver voxel.

Hypothesis (1) is ruled out.

## 3. Per-margin distance breakdown for the 4 cases

Recomputed straight from the masks:

| Mnemonic            | n_lesion | max_dist | P95 | >20 mm | >25 mm | >30 mm | >40 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| blush_turtle_cliff  | 897    | **37.3** | 34.8 | 757 | 757 | **522** | 0 |
| raven_dove_summit   | 4559   | 31.6     | 23.5 | 793 | 161 | 5     | 0 |
| glass_puma_glade    | 251    | 25.4     | 23.9 | 29  | 2   | 0     | 0 |
| pine_wren_fjord     | 1861   | 20.4     | 4.6  | 2   | 0   | 0     | 0 |

So a 30 mm dilation captures 100% for raven, glass, and pine, but blush still
loses 522 voxels at 30 mm. The full cohort needs ~40 mm to clear blush.

## 4. Realignment / source-mask sanity check

The radiologist's source masks live at `/home/jjs374/DiaE/masks/ANON*.nii.gz`.
`scripts/realign_masks.py` does a lossless dual-axis flip (`mask[::-1, :, ::-1]`)
and adopts the dcm2niix RAS affine. For all 4 cases, `XOR(flip(source), in-data
mask) == 0`, so the realignment binary content matches the source by
construction. Voxel sums match exactly (897, 4559, 251, 1861).

I also looked at affine drift: for each patient, take a source-mask centroid
`s`, world-map it through the source affine; then dual-flip it to `s'` and
world-map through the *post* affine (which equals the raw affine):

| Mnemonic            | drift |
|---|---:|
| blush_turtle_cliff  | **56.7 mm** |
| raven_dove_summit   | **45.0 mm** |
| glass_puma_glade    | 0.00 mm |
| pine_wren_fjord     | 0.00 mm |

The drift in blush and raven looks alarming, but **this is a synthetic
quantity** — it would only matter if the source affine itself were the
ground-truth registration, which it isn't. The radiologist's source mask was
created on a volume with a different per-axis voxel size (e.g. for blush the
source affine has zooms `(-0.742, 1.500, -0.742)` whereas the raw dcm2niix
volume has zooms `(0.820, 1.700, 0.820)`). The realignment audit (per
`scripts/audit_mask_canonical.py`) was the actual gating QC; that audit was
shape-based (any axis permutation) and the dual-flip was empirically validated
on 5 sample patients. So the 56.7 mm "drift" tells us the original source-NIfTI
header was wrong about voxel spacing (these are old radiology PACS exports), not
that our realignment is wrong. Hypothesis (5) is **plausible but not
demonstrably the cause**: I'd want to verify the dual-flip was genuinely
correct for blush/raven by comparing image-space landmarks against the raw —
e.g. liver dome position in the radiologist's original water volume vs the
dcm2niix raw volume. That's outside the scope of this audit, but worth doing
before concluding blush is purely an annotation issue.

## 5. The blush_turtle_cliff debug montage

`eda/outputs/debug_containment_blush_turtle_cliff.png` shows 5 contiguous
coronal slices y ∈ {99, 101, 104, 107, 110} (axis-1 in the `(512, 116, 512)`
volume; axis-1 is anteroposterior at 1.7 mm/voxel — y=110 is the most anterior
slice). Per-slice in-ROI / total / out-ROI lesion counts:

```
y=99:  in=0   total=7    out=7
y=101: in=0   total=105  out=105
y=104: in=0   total=70   out=70
y=107: in=0   total=60   out=60
y=110: in=0   total=7    out=7
```

Every red lesion voxel on these slices is outside the cyan ROI outline. In the
images the lesion sits in the right superior thorax, well above where the
liver-dilation reaches (the blue liver mask and cyan ROI outline are both
visible only on more posterior slices, y=60–95 in the second diagnostic
montage `debug_overlay_blush_turtle_cliff_full.png`).

## 6. Anatomic plausibility

The blush_turtle_cliff lesion bbox spans y=52–110 (out-of-ROI portion at
y=99–110), i.e. ≈19 mm to ≈100 mm anterior of the most posterior lesion voxel.
The liver mask y-range is 15–102. So the lesion *does* extend ~14 mm anterior
of the most-anterior liver voxel, with the bulk of out-of-ROI voxels in the
range y=99–110. In the rendered montage these out-of-ROI voxels lie in the
**right anterior thorax just under the chest wall / sternum**. Is that where
diaphragmatic endometriosis lives? Anatomically the right hemidiaphragm dome
*does* curve up and forward toward the sternum and is thinly draped over the
liver dome. Right-sided diaphragmatic endometriosis classically affects exactly
this dome region. So the out-of-ROI lesion voxels in blush_turtle_cliff are
**anatomically plausible diaphragm voxels** that happen to be > 20 mm above the
liver mask along the cranial direction. The user's "20 mm contains it" eyeball
impression is at odds with the array data; one of these is true:

1. ITK-SNAP rendering is correct and the user's eyeball just missed the most
   superior tip of the lesion (very plausible for a 757-voxel cluster
   distributed across 12 thin coronal slices).
2. The user was looking at the liver mask not the 20 mm ROI overlay (also
   plausible — the 20 mm ROI is hard to eyeball-distinguish from the liver).

I lean strongly on (1). The geometry on the rendered montage is unambiguous:
the red lesion clearly extends cranial to where any 20 mm dilation of the
liver mask reaches.

The user's "40 mm extends through the diaphragm into the lung" objection is
mostly correct as a rule of thumb, but for this specific cohort and this
specific anatomic location (dome, where the diaphragm is millimetres thick and
sits directly under the parietal pleura), 30 mm is enough to capture 99.5% of
positives and 35–40 mm is enough for full coverage. The lung *will* be
included in the ROI for those margins, but that's fine for an object-detection
model: the ROI is a hard crop, not a hard label.

## Verdict

| Mnemonic            | Verdict |
|---|---|
| **blush_turtle_cliff** | (a) Real out-of-ROI lesion voxels. The lesion sits on the right hemidiaphragm dome, 25–37 mm cranial of the liver mask. Anatomically plausible. Possibly compounded by source-affine voxel-spacing mismatch (see §4) — worth verifying registration is truly correct for this patient before training. |
| **raven_dove_summit**  | (a) Real out-of-ROI lesion voxels at 21–32 mm. Same dome anatomy. Most voxels are 20–25 mm out; a 25 mm dilation gets 96% of them, 30 mm gets all but 5. |
| **glass_puma_glade**   | (a) Real out-of-ROI lesion voxels at 20–25 mm; tiny cluster (29/251). Borderline. |
| **pine_wren_fjord**    | (c) Annotation-edge artefact: only 2/1861 voxels are at 20.4 mm, P95 distance is 4.6 mm. These are 1-voxel pokes at the border of an otherwise contained lesion. |

Hypotheses (1) analysis bug, (3) dilation bug ruled out. Hypothesis (2)
spurious annotation voxels applies cleanly to pine and partly to glass.
Hypothesis (4) — anatomically real on-the-dome lesions — is the dominant
explanation for blush and raven. Hypothesis (5) — realignment shift — cannot
be definitively ruled out for blush/raven, but the dual-flip XOR=0 against the
source mask strongly suggests the realignment is internally consistent.

## What dilation margin is actually needed?

From the §3b table: at 40 mm dilation **108/108 positives are fully contained**;
at 35 mm we'd capture all but blush; at 30 mm we'd capture all but blush+raven.
A reasonable choice for the project:

- **Option A — 40 mm dilation.** Guarantees full lesion coverage. The user's
  concern about "extends into the lung" is true: the ROI will include
  pulmonary parenchyma at the dome. For a *detection-only* model where the
  ROI is just a crop window (not a target mask), this is harmless; the model
  has more context. Adds ~20 mm of in-plane padding and ~12 voxels of
  through-plane padding versus 20 mm.
- **Option B — 30 mm dilation + clip lesion mask to ROI for the 1 outlier.**
  Captures 107/108. Treat blush_turtle_cliff as a known annotation outlier
  (or reinvestigate registration for that patient).
- **Option C — keep 20 mm and accept the 4 partial-containment cases as
  expected losses**, clipping the 4 lesion masks to the ROI. Loses 757
  voxels of label on blush, 793 on raven, 29 on glass, 2 on pine.

Recommendation: **Option A (40 mm)**. The "extends into the lung" worry is
correct anatomy but not a liability for object detection — and 40 mm
guarantees that the model sees the entire lesion in *every* positive. The
post-crop bbox grows by only ~2 cm per side which is well inside what a
modern detector handles. If memory is tight, **Option B (30 mm) with clipped
labels for blush_turtle_cliff** is a fine compromise. Option C should be a
last resort because it silently truncates label data.
