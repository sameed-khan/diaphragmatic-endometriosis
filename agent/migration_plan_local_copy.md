# Migration plan — `data-local-copy/` → `data/` rebuild for Phase-1 positives

**Date:** 2026-04-27
**Scope:** Replace the 108 misaligned positive volumes + masks under `data/` with their hand-QC'd correctly-aligned versions from `data-local-copy/`. Re-run downstream artefacts (liver_masks, liver_rois). Update `manifest.csv` + `sidecars.jsonl`. Verify, then delete `data-local-copy/`.
**Out of scope:** the 500 negatives are unchanged; `splits.json` is untouched (the frozen seed=42 splits already cover all 108 positive anon_ids by mnemonic, which we keep stable).

---

## Pre-flight audit (already executed, ✅ all clear)

| Check | Result |
|---|---|
| `data-local-copy` mnemonics (with underscore swap) collide with any `data/` negative | **0 collisions** |
| All 108 positive `anon_id`s present in `data-local-copy/patient_id_mapping.csv` | **108/108 found** |
| Mnemonic discrepancies (after hyphen ↔ underscore swap) | **0** |
| Canonical nii+msk available in `data-local-copy/` | **106/108** |
| Patients requiring explicit non-canonical routing | **2** (`dapple_bunny_dome`, `teak_ox_beach` — both map to `_WATER:_COR_DIAFRAGMA_T1_LAVA_AB`) |
| 5 extras (`granite_*`, `ivory_tern_sage`, `steady_*`, `wheat_shrew_road`) | **DROP** per user (lesions not visualizable) |

**Mnemonic strategy:** keep the existing underscored mnemonics from `data/`. Resolve `data-local-copy` files via the `anon_id` foreign key and rename hyphens → underscores at copy time. This keeps `splits.json`, `patient_id_mapping.csv`, and all path columns in `manifest.csv` stable.

---

## Migration phases

The migration is split into 7 phases. Phases are idempotent where reasonable; Phase 5 onwards depends on success of earlier phases.

### Phase 0 — Snapshot the current state (read-only)

- Compute and persist a snapshot of `manifest.csv`, `sidecars.jsonl`, `splits.json`, and `patient_id_mapping.csv` into `agent/migration_2026_04_27_snapshots/` (timestamped, gitignored). Lets us diff or roll back.
- Compute SHA-256 for every current `data/raw/<bucket>/positive/<m>.nii.gz` and `data/lesion_masks/<bucket>/positive/<m>_mask.nii.gz`. Save to `agent/migration_2026_04_27_snapshots/before_sha.csv`. The full manifest already has these for negatives; this snapshot lets us prove what changed.

**Output:** `agent/migration_2026_04_27_snapshots/` with frozen pre-migration state.

### Phase 1 — Backup current positive directories

Move (don't delete) the existing positive directories:

- `data/raw/<bucket>/positive/` → `data/_pre_migration_backup/raw/<bucket>/positive/`
- `data/lesion_masks/<bucket>/positive/` → `data/_pre_migration_backup/lesion_masks/<bucket>/positive/`
- `data/liver_masks/<bucket>/positive/` → `data/_pre_migration_backup/liver_masks/<bucket>/positive/`
- `data/liver_rois/<bucket>/positive/` → `data/_pre_migration_backup/liver_rois/<bucket>/positive/`

Negatives are untouched.

**Why move not delete:** if Phase 6 verification fails, we restore from backup and roll back manifest changes.

**Output:** `data/_pre_migration_backup/` (gitignored).

### Phase 2 — Stage `data-local-copy/` files into `data/raw/` and `data/lesion_masks/`

For each of the 108 positives, looked up via `anon_id`:

```
src_nii  = data-local-copy/nifti/<hyphen-mnemonic>{._suffix}.nii.gz
src_msk  = data-local-copy/masks/<hyphen-mnemonic>{._suffix}.nii.gz
dst_nii  = data/raw/<bucket>/positive/<underscore-mnemonic>.nii.gz
dst_msk  = data/lesion_masks/<bucket>/positive/<underscore-mnemonic>_mask.nii.gz
```

For 106 patients: use the canonical (`_suffix` = empty).
For 2 patients (`dapple_bunny_dome`, `teak_ox_beach`): use `_WATER:_COR_DIAFRAGMA_T1_LAVA_AB`.

**File operations:**
- `cp` the raw NIfTI byte-for-byte (no transformation — preserves byte-identity to source-of-truth).
- For the mask: load, **binarize to {0, 1}** (some sources use {0, 1, 2}), recast to uint8, save with the source mask's affine (which is identical to the raw's affine in data-local-copy by construction).
- Emit a per-patient row of (anon_id, src_nii, src_msk, dst_nii, dst_msk, raw_sha256, mask_sha256, raw_shape, raw_zooms, axcodes) to `eda/outputs/migration_phase2_report.csv`.

**Skip the 5 extras** (`granite_elk_quartz`, `granite_marten_valley`, `ivory_tern_sage`, `steady_gorilla_crest`, `wheat_shrew_road`): the user has visually QC'd these and cannot see the lesion on the T1-FS canonical, so the masks may not correspond to actual diaphragmatic lesions. Drop entirely. They were never in our 108-positive cohort anyway (`transferred_to_home=false`), so no manifest-level removal is needed beyond ensuring they don't leak in.

**Sanity check at end of Phase 2:**
- 108 raw NIfTIs and 108 masks now live under `data/raw/<bucket>/positive/` and `data/lesion_masks/<bucket>/positive/`.
- Every dst_nii's SHA-256 matches data-local-copy's SHA-256 byte-for-byte.

### Phase 3 — Re-run TotalSegmentator on the 108 new raws

Use existing `scripts/run_totalseg.py` driver, scoped to the 108 positive mnemonics:

```bash
uv run -m scripts.run_totalseg \
    --data-root data \
    --filter-cohort positive \
    --execute
```

(or pass an explicit `--mnemonic-list` if the existing CLI accepts that — verify before running)

This regenerates:
- `data/liver_masks/<bucket>/positive/<m>_liver_mask.nii.gz` (binary uint8)

**Compute resources:** ~108 patients × ~30-60 s/patient on GPU = ~1-2 hours wallclock. The TotalSegmentator MR `task=total_mr roi_subset=liver` flow is what was used originally — no changes.

**Sanity check at end of Phase 3:**
- 108 new liver_mask files exist; for each, shape/affine match the new raw, all values in {0,1}, voxel count > 100k (sanity floor for an adult liver).

### Phase 4 — Re-dilate the 108 new liver masks at 20 mm

```bash
uv run -m scripts.dilate_segmentations \
    --input-dir data/liver_masks \
    --output-dir data/liver_rois \
    --margin-mm 20 \
    --filter-cohort positive \
    --execute
```

This regenerates `data/liver_rois/<bucket>/positive/<m>_liver_roi.nii.gz`.

**Sanity check at end of Phase 4:**
- 108 new liver_roi files exist; for each, shape/affine match the new raw, ROI voxel count > liver voxel count, ROI voxel fraction in [0.05, 0.30].

### Phase 5 — Update metadata files

#### 5a. `manifest.csv`

For each of 108 positive rows, recompute and update:

| Column | Source |
|---|---|
| `sha256_raw` | hash of new dst_nii |
| `shape` | new NIfTI shape |
| `n_slices_actual` | shape[1] (through-plane axis) |
| `slice_thickness_mm` | re-extract from sidecar (DICOM SliceThickness) — patient-level so should not change, but recompute for consistency |
| `pixel_spacing_x_mm`, `pixel_spacing_y_mm` | new NIfTI zooms[0], zooms[2] |
| `liver_mask_sha256` | hash of new liver_mask file |
| `liver_voxel_count` | sum(liver_mask) |
| `migration_timestamp` | `datetime.now().isoformat()` |
| `liver_roi_margin_mm` | 20 (unchanged) |
| `transferred_to_home` | True (unchanged) |
| `*_path` columns | unchanged — file paths preserved |

**Backfill `scanner_model`:** the §6 finding showed the column is empty for all 608 rows in the current manifest. Take the opportunity to back-fill it from the sidecar `ManufacturersModelName` for all 608 rows (positive + negative). One-line polars join.

Save to `data/manifest.csv` (overwriting). Diff vs the Phase-0 snapshot must show exactly:
- 108 positive rows changed in the listed columns
- 608 rows changed in `scanner_model`
- 0 other rows changed

#### 5b. `sidecars.jsonl`

The sidecars contain DICOM-level patient metadata (Modality, MagneticFieldStrength, Manufacturer, ManufacturersModelName, EchoTime, RepetitionTime, ImageType, etc.) plus a `raw_path` field.

The DICOM metadata is patient-level and unchanged by the conversion choice. The `raw_path` is also unchanged (we kept the same destination filenames). **So the sidecars need no edits.**

Run a verification: every `mnemonic_id` in the new manifest exists in sidecars; every sidecar `raw_path` resolves on disk after migration. If any sidecar's `mnemonic_id` is missing from the new manifest → flag and abort.

#### 5c. `splits.json`

The frozen seed=42 splits.json `assignments` map is keyed by `anon_id`. Since:
- We kept all 108 positive anon_ids identical
- We didn't add the 5 dropped extras (they were never in splits anyway because `transferred_to_home=false`)
- We didn't change any mnemonic
- We didn't add/remove any negative

…**no edits needed.** Verification: every anon_id in the new manifest's positive rows exists in `splits["assignments"]`.

#### 5d. `data/patient_id_mapping.csv`

This is `anon_id ↔ mnemonic_id`. Since we kept all 108 mnemonic IDs unchanged, and added zero patients, **no edits needed.** Verification: file unchanged via SHA-256 compared to Phase-0 snapshot.

#### 5e. `data-local-copy/patient_id_mapping.csv`

Will be deleted with the rest of `data-local-copy/` in Phase 7.

### Phase 6 — Verification suite (gate to Phase 7)

A single script (`scripts/verify_migration.py`) that runs all checks and exits non-zero on any failure. Phases 7 cannot run unless this exits 0.

#### 6a. Filesystem integrity (must all be True)

For each of 108 positive mnemonics:
- `data/raw/<bucket>/positive/<m>.nii.gz` exists; SHA-256 matches Phase 2 record
- `data/lesion_masks/<bucket>/positive/<m>_mask.nii.gz` exists; values ⊆ {0,1}; uint8; non-empty
- `data/liver_masks/<bucket>/positive/<m>_liver_mask.nii.gz` exists; values ⊆ {0,1}; uint8; voxel count ∈ [100k, 5M]
- `data/liver_rois/<bucket>/positive/<m>_liver_roi.nii.gz` exists; values ⊆ {0,1}; uint8; ROI ⊇ liver mask
- All 4 files share the same shape, zooms, and affine (atol 1e-4 on affine)

For all 500 negatives:
- raw + liver_mask + liver_roi all still exist and SHA-256 matches Phase-0 snapshot (i.e., negatives were not touched)

#### 6b. Manifest consistency

- `manifest.csv` row count: 5,060 (unchanged)
- `manifest.csv` filtered to `transferred_to_home=True`: exactly 608 rows (108 positive + 500 negative)
- For every row, the path columns resolve on disk
- `scanner_model` populated for all 608 transferred rows; values ⊆ {`SIGNA Artist`, `SIGNA Explorer`}
- `liver_voxel_count` populated for all 608

#### 6c. Splits consistency

- All 108 positive `anon_id`s appear in `splits["assignments"]`
- The fold-distribution counts match `splits.phase1_targets` exactly:
  - holdout_pos=22, holdout_neg=100, cv_pos=86, cv_neg=400 — unchanged

#### 6d. Mask-alignment quality (the critical check)

Re-run `eda/14_mask_alignment_audit_v2.py` (or an equivalent inline) to compute per-positive lesion-vs-ring intensity z-score on the NEW `data/raw/`. Expected results:
- median contrast_z ≥ 0.75 (was 0.80 in pipeline; data-local-copy showed 0.81)
- P5 contrast_z ≥ 0.40 (was 0.12 in pipeline; data-local-copy showed 0.42)
- Number of patients with contrast_z < 0.2 ≤ 1 (was 6 in pipeline; data-local-copy showed 1, the genuinely-low-contrast `polar-jay-field`)

**This is the headline gate** — if these thresholds aren't hit, something went wrong in Phase 2 or 3.

#### 6e. Lesion-containment in 20 mm liver_roi (Section 3 re-run)

Re-run `eda/03_liver_roi_containment.py` on the rebuilt cohort. Expected:
- ≥ 106/108 positives fully contained (was 104/108, but 2 of the 4 outliers were misalignment artefacts that should now be fixed)
- The remaining 1-2 partial cases (likely `glass_puma_glade`, `pine_wren_fjord`) lose ≤ 30 voxels each — annotation edge artefacts.

#### 6f. Slice geometry (Section 1 re-run)

Re-run `eda/01_volume_geometry.py`. Expected:
- All 608 volumes still have 512×N×512 shape
- All RAS axcodes (data-local-copy uses LAS, but our migration preserves the source affine — verify what axcodes the migrated files have)
- **Decision point:** if migrated raws are LAS (per data-local-copy), do we leave them or canonicalize to RAS at preprocessing time? Either is fine; document the choice.

#### 6g. End-to-end re-run of EDA §3, §4, §5, §7

These will produce slightly different numbers from the original report (better alignment + tightened distributions). Save outputs to `eda/outputs/post_migration_*.csv` so we can diff against the originals and confirm only the expected directional shifts.

#### 6h. Negative-cohort untouched (regression check)

Re-compute SHA-256 for all 500 negative `raw_path` files. All must match Phase-0 snapshot.

### Phase 7 — Cleanup

**Only if Phase 6 exits 0:**

1. Delete `data/_pre_migration_backup/` (the Phase 1 backup is no longer needed).
2. Delete `data-local-copy/` (the source-of-truth has been migrated).
3. Update the snapshot in `agent/migration_2026_04_27_snapshots/` with a `migration_completed.txt` marker including the verification report.
4. Remove `eda/outputs/realign_masks_v2_report.csv` if it exists (the v2 realign script is superseded — we're not using it).
5. Mark `scripts/realign_masks.py` and `scripts/realign_masks_v2.py` as deprecated in their docstrings (or just delete them — they're no longer in the data flow).

**If Phase 6 fails:**

1. Restore `data/raw/<bucket>/positive/`, `data/lesion_masks/<bucket>/positive/`, `data/liver_masks/<bucket>/positive/`, `data/liver_rois/<bucket>/positive/` from `data/_pre_migration_backup/`.
2. Restore `manifest.csv` from `agent/migration_2026_04_27_snapshots/`.
3. Leave `data-local-copy/` in place.
4. Investigate the failure, fix, re-run from Phase 1.

---

## Implementation: a single driver script

I'll write `scripts/migrate_local_copy_to_data.py` with a phased CLI:

```bash
# Dry-run by default — prints what would happen, writes Phase 0 snapshot, no destructive ops
uv run -m scripts.migrate_local_copy_to_data --phase 0
uv run -m scripts.migrate_local_copy_to_data --phase 1   # backup
uv run -m scripts.migrate_local_copy_to_data --phase 2   # copy raws + binarize masks
uv run -m scripts.migrate_local_copy_to_data --phase 3   # totalseg (calls run_totalseg internally)
uv run -m scripts.migrate_local_copy_to_data --phase 4   # dilation
uv run -m scripts.migrate_local_copy_to_data --phase 5   # update manifest, scanner_model backfill
uv run -m scripts.migrate_local_copy_to_data --phase 6   # verification suite
uv run -m scripts.migrate_local_copy_to_data --phase 7   # cleanup, only after phase 6 passes
```

Each phase prints a pass/fail summary and writes a phase-report CSV under `eda/outputs/migration_phase{N}_report.csv` for audit.

A convenience flag `--phase all --execute` runs phases 0 → 6 in sequence (stopping on first failure), and prompts before running phase 7.

---

## Risk & rollback summary

| Risk | Mitigation |
|---|---|
| Phase 2 copies wrong file | Phase 2 emits per-patient SHA-256 record; Phase 6a verifies SHA-256 matches data-local-copy. |
| TotalSeg fails on a few patients | Phase 3 fails fast on any patient with empty liver_mask; can be re-run patient-scoped. |
| Manifest update introduces inconsistency | Phase 5 always writes a NEW manifest.csv; Phase 0 snapshot of the OLD is kept. |
| Migration breaks negatives | Phase 6h SHA-256 check on all 500 negatives against Phase 0 snapshot. |
| splits.json drift | Phase 6c verifies all 108 anon_ids still map to the same fold/holdout assignment. |
| Decision regret on 5 extras | The 5 extras can be added later with a follow-up patch (rerun TotalSeg + dilation on just those 5, append to manifest, append to splits.json with stratified rebalance). Not blocking. |
| LAS vs RAS orientation in migrated raws | The data-local-copy niftis are LAS; original `data/raw/` was RAS. Several downstream EDA scripts and §1 finding rely on RAS. **Decision required:** (a) keep LAS, document, update the through-plane-axis identification logic to read from header rather than assume axis-1; or (b) at copy time, apply `nib.as_closest_canonical()` to convert to RAS — produces a byte-different file from data-local-copy but matches the prior pipeline convention. **Recommendation: (b)** — it preserves all the EDA/preprocessing assumptions that we already validated. |

---

## Expected post-migration deltas

| Metric | Pre-migration | Post-migration (expected) |
|---|---:|---:|
| Mask-aligned positives (cz ≥ 0.2) | 102/108 | **107/108** |
| Median lesion-ring contrast z | 0.80 | **~0.81** |
| Lesion fully contained in 20 mm liver_roi | 104/108 | **≥ 106/108** |
| Total positive lesion voxels | 78,673 | will tighten by ~1–3 % |
| §7 lesion-vs-bg z-score median | 0.64 | **~0.75** (the 6 misaligned were dragging this down) |
| `manifest.scanner_model` populated | 0/608 | **608/608** |
| Other §6, §10, §11 findings | (unchanged) | (unchanged) |

---

## After migration: model-training implications

- **Cohort:** 108 positives, of which 1 (`opal_shrew_weld` — LAVA_FLEX_NAV) should still be excluded from training per the heterogeneity finding (genuinely different acquisition). Effective training cohort = 107 positives.
- **The §3 dilation finding (use 20 mm)** is restored to its original recommendation. No 30/40 mm rebuild needed.
- **All other modeling commitments** in `agent/eda_synthesis.md` and `agent/heterogeneity_and_source_of_truth.md` stand unchanged: 2.5D RTMDet-S + ConvNeXt-tiny, stride-4 P2 head, per-volume percentile normalization, no horizontal flip, etc.
- **`opal_shrew_weld` + 4 stragglers** become the held-out external-protocol generalization test set, evaluated once at the end as a non-gated number.

---

## Open question for user before execution

The orientation question (LAS in data-local-copy vs RAS in current data/) is the only design choice still open:

**Q:** Should we apply `nib.as_closest_canonical()` at copy time to canonicalize the migrated raws/masks to RAS (matching the existing pipeline convention), or keep them LAS and update downstream code to be orientation-agnostic?

**Default if you don't reply:** RAS canonicalization at copy time. It's a single function call per file, preserves byte-identity-to-content (just relabels orientation), and avoids touching every EDA script. The byte identity to data-local-copy is lost in the process, but the *physical* identity of the volume is preserved.
