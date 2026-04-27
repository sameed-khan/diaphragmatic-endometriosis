# Phase 1 Data Migration Plan — /scratch → /home/data with mnemonic renaming

**Author:** Claude (planning)
**Date:** 2026-04-26
**Status:** PLAN (revised after user review) — ready for executor agent post-context-rotation.
**Predecessor docs:** [`convert-plan-v2.md`](./convert-plan-v2.md), [`handoff-2026-04-26-post-phase1.md`](./handoff-2026-04-26-post-phase1.md)

---

## 1. Goal

Copy the **Phase 1 development cohort** (608 patients, canonical-sequence-only) from /scratch to a new persistent working tree at `/home/sak185/dia-endo-conversion/data/`, renaming each patient from its `ANON…` ID to a deterministic 3-word mnemonic (`arctic_snow_tiger` style, **underscore-separated**) on the fly. The new tree must:

1. **Physically separate the holdout set** from the CV pool (so the locked test set is hard to accidentally include in training), and reflect **cohort** (`positive` vs `negative`) directly in the path, so a glob like `data/raw/cross-validation/positive/*.nii.gz` is meaningful. **Per-fold sub-directories are NOT used** — the fold assignment for each CV patient lives in `splits.json` and the `split` column of `manifest.csv`, both of which the training code references at runtime.
2. Be ergonomic for **visual inspection** — flat NIfTI files openable in FSLeyes / ITK-SNAP / 3D Slicer with no extractor.
3. Accept future **derivative layers** (liver masks, cropped volumes, p1–p99 normalized, etc.) as parallel top-level directories with the same internal `<bucket>/<cohort>/` structure, so a path-substitution rule (`raw/X.nii.gz` ↔ `cropped_raw/X.nii.gz`) is the only mental model needed.
4. **Separate ground-truth masks from imaging volumes** — lesion masks live in their own parallel tree (`lesion_masks/`), not co-located with volumes.
5. Preserve **bidirectional traceability** — `data/manifest.csv` tracks **all 5,120 patients** in the project (Phase 1 + Phase 2 + pilot + remask23), with a `transferred_to_home` boolean column distinguishing what's actually on /home vs what still lives only on /scratch.

**Out of scope for this migration (NOT transferred to /home):**
- Phase 2 negatives (4,476 patients, SSL pretraining pool — stays on /scratch; will get `transferred_to_home=False` rows in the manifest)
- Alternate water series (`water_alt_*`), FAT, in-phase, out-phase volumes
- dcm2niix sub-volume splits (`water_canonicala/b`, `water_canonical_e1/e2`)

---

## 2. Source-of-truth and scope filter

**Source manifest:** `/scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest.csv` (8,409 rows, refreshed 2026-04-26).

### 2.1 What to migrate (Phase 1 dev cohort)

```python
df = pl.read_csv(SOURCE_MANIFEST)
df = df.filter(pl.col("role") == "canonical")              # canonical sequence only
df = df.filter(pl.col("volume_index") == 0)                # drop dcm2niix sub-volume splits (a/b/_e1/_e2)
phase1_splits = ["holdout", "fold0", "fold1", "fold2", "fold3", "fold4"]
to_transfer = df.filter(pl.col("split").is_in(phase1_splits))

# 20 patients have duplicate rows (converted in pilot AND phase1 — same file content,
# idempotent overwrite). Dedup by phase preference: phase1 > remask23 > pilot.
to_transfer = to_transfer.sort([
    pl.col("phase").map_elements(
        lambda s: {"phase1": 0, "remask23": 1, "pilot": 2}.get(s, 3),
        return_dtype=pl.Int32,
    )
])
to_transfer = to_transfer.unique(subset=["patient_id"], keep="first")
# Assert resulting count == 608. Abort migration if not.
```

### 2.2 What to track but NOT migrate

After deriving `to_transfer` (608 rows), **also** derive a parallel set for the rest of the project — these patients get rows in `data/manifest.csv` with `transferred_to_home=False`:

```python
not_transferred = df.filter(pl.col("role") == "canonical")
not_transferred = not_transferred.filter(pl.col("volume_index") == 0)
not_transferred = not_transferred.filter(~pl.col("split").is_in(phase1_splits))
# Includes split=phase2_unsupervised AND split is null (the 5 pre-splits pilot leftovers).
not_transferred = not_transferred.sort([...phase preference...])
not_transferred = not_transferred.unique(subset=["patient_id"], keep="first")
# Expect ~4,512 patients (4,476 phase2 + 31 pre-splits pilot/remask23 + 5 split-null).
```

Combined: `to_transfer` (608) + `not_transferred` (~4,512) = **5,120 unique patients**, one canonical row each. This is the row count of `data/manifest.csv`.

### 2.3 Per-bucket breakdown (for sanity checks during migration)

The migration uses **two physical buckets** (`holdout` and `cross-validation`), but per-fold counts are preserved for downstream verification — the manifest's `split` column carries the fold assignment.

| Bucket / split | positive | negative (incl. soft-neg) | total |
|---|---:|---:|---:|
| **holdout** (physically separate dir) | 22 | 100 | **122** |
| `cross-validation` total (sum of folds, in one dir) | **86** | **400** | **486** |
| ↳ fold0 (annotation only) | 18 | 82 | 100 |
| ↳ fold1 (annotation only) | 18 | 81 | 99 |
| ↳ fold2 (annotation only) | 17 | 79 | 96 |
| ↳ fold3 (annotation only) | 17 | 79 | 96 |
| ↳ fold4 (annotation only) | 16 | 79 | 95 |
| **transferred_to_home=True** | **108** | **500** | **608** |
| phase2_unsupervised + null | — | ~4,512 | **~4,512** |
| **manifest total** | | | **~5,120** |

Soft-negatives (5 in Phase 1) live under `negative/`; the `soft_negative` column in the manifest is the boolean flag, never used for path routing.

---

## 3. Mnemonic naming

### 3.1 Algorithm — reuse the existing generator

**File:** `scripts/generate_patient_names.py` — already implemented. Hashes ANON ID via SHA-256, indexes into `scripts/wordlists.json` (116 adj × 120 animals × 116 nouns = **1.61 M unique combinations**; ~5,120 patients = 0.32% utilization → no realistic collision risk; the existing collision-resolution loop covers the edge case anyway).

**Separator: underscore.** Names look like `arctic_snow_tiger`, `azure_fox_meadow`, `bright_owl_canyon`. The existing script uses hyphen — change the format string accordingly:

```python
return f"{adjectives[adj_idx]}_{animals[animal_idx]}_{nouns[noun_idx]}"
```

This is a one-character change at line 60 of `generate_patient_names.py`.

### 3.2 Scope — generate names for all 5,120 patients

Source the patient list from `manifest.csv["patient_id"].unique()` (covers all 5,120 patients converted across phase1+phase2+pilot+remask23). Reasons:
- Future-proofs Phase 2 migration. Same SHA-256 + same wordlists ⇒ same names.
- The mapping is a project-wide invariant. Patients shouldn't get renamed twice.
- Cost is trivial (<1 sec for generation, <100 KB for the CSV).

The existing script reads patient IDs from a flat `nifti/*.nii.gz` glob — this won't work for our setup. **Update it to take `--manifest` and `--output` CLI args.** See §6.1.

### 3.3 Output: `data/patient_id_mapping.csv`

Two columns: `anon_id`, `mnemonic_id`. Sorted by `anon_id` for deterministic diff. **Treat this file as immutable once written** — regenerating with a different wordlist or hashing scheme would shift names and break every downstream reference. The script's `argparse` should refuse to overwrite an existing mapping unless `--force` is passed.

---

## 4. Filesystem hierarchy

### 4.1 Top-level layout

```
/home/sak185/dia-endo-conversion/data/
├── raw/                                  # original full-resolution canonical NIfTIs (volumes + JSON only)
│   ├── holdout/                          # 122 patients — locked test set, physically separated
│   │   ├── positive/
│   │   │   ├── arctic_snow_tiger.nii.gz
│   │   │   ├── arctic_snow_tiger.json    # BIDS sidecar
│   │   │   └── ...
│   │   └── negative/
│   │       ├── azure_fox_meadow.nii.gz
│   │       ├── azure_fox_meadow.json
│   │       └── ...
│   └── cross-validation/                 # 486 patients across all 5 folds (fold ID via splits.json)
│       ├── positive/
│       │   └── ...
│       └── negative/
│           └── ...
├── lesion_masks/                         # GT radiologist masks (positives only); parallel structure to raw/
│   ├── holdout/positive/
│   │   └── arctic_snow_tiger.nii.gz
│   └── cross-validation/positive/
│       └── ...
├── liver_masks/                          # ADDED LATER (TotalSegmentator output, mirrored structure)
│   ├── holdout/<cohort>/<mnemonic>.nii.gz
│   └── cross-validation/<cohort>/<mnemonic>.nii.gz
├── cropped_raw/                          # ADDED LATER (liver-bbox cropped canonical volumes)
│   ├── holdout/<cohort>/<mnemonic>.{nii.gz,json}      # JSON includes crop_bbox metadata
│   └── cross-validation/<cohort>/<mnemonic>.{nii.gz,json}
├── cropped_lesion_masks/                 # ADDED LATER (GT masks cropped to same bbox)
│   ├── holdout/positive/<mnemonic>.nii.gz
│   └── cross-validation/positive/<mnemonic>.nii.gz
├── normalized_p1p99/                     # ADDED LATER, only if precomputed (vs runtime DataLoader transform)
│   ├── holdout/<cohort>/<mnemonic>.nii.gz
│   └── cross-validation/<cohort>/<mnemonic>.nii.gz
├── predictions/                          # ADDED DURING TRAINING
│   └── <run_id>/<bucket>/<cohort>/<mnemonic>.nii.gz   # bucket = holdout | cross-validation
├── manifest.csv                          # 5,120 rows; project-wide tracker
├── patient_id_mapping.csv                # immutable two-column mapping
├── splits.json                           # copied verbatim from /scratch/.../workplan/splits.json
└── README.md                             # auto-generated by migration script
```

**Where the fold assignment lives** (since it's NOT in the directory tree):
- `data/splits.json` — frozen JSON; `assignments[patient_id] -> "fold0" | "fold1" | ... | "fold4" | "holdout"`
- `data/manifest.csv` — `split` column carries the same value per row

The training DataLoader resolves `cross-validation/` patients into folds at runtime by joining on `splits.json` (or by reading the `split` column from the manifest). The directory tree just gives you a clean glob for "all CV-pool inputs" vs "all holdout inputs".

### 4.2 Per-patient files (after this migration)

For one Phase 1 CV-pool patient e.g. `arctic_snow_tiger` (assigned to fold0 in `splits.json`):

| Path | Always present? |
|---|---|
| `raw/cross-validation/positive/arctic_snow_tiger.nii.gz` | ✅ |
| `raw/cross-validation/positive/arctic_snow_tiger.json` | ✅ |
| `lesion_masks/cross-validation/positive/arctic_snow_tiger.nii.gz` | ✅ for positives, absent for negatives |

For a holdout positive e.g. `bright_owl_canyon`:

| Path | Always present? |
|---|---|
| `raw/holdout/positive/bright_owl_canyon.nii.gz` | ✅ |
| `raw/holdout/positive/bright_owl_canyon.json` | ✅ |
| `lesion_masks/holdout/positive/bright_owl_canyon.nii.gz` | ✅ |

For a negative (either bucket): just `.nii.gz` and `.json` under `raw/`. The corresponding `lesion_masks/<bucket>/negative/` directory **doesn't exist** (negatives have no GT mask).

### 4.3 Why this hierarchy

- **Two physical buckets only — `holdout/` and `cross-validation/`** — instead of six (holdout + 5 folds). The locked test set is the only split that benefits from physical separation (so you can never accidentally `find data/raw/cross-validation -name "*.nii.gz"` and pull in test data). Folds within the CV pool are an analysis-time construct; their assignment lives in `splits.json` + `manifest.csv["split"]`, both of which the training code reads anyway.
- **Cohort sub-dir** lets you `ls raw/cross-validation/positive/` to count CV positives without parsing the manifest. Useful for sanity checks and EDA.
- **Flat per-patient** (no `<mnemonic>/` subdir): every patient has at most 2 files in `raw/` (volume + JSON). A subdir per patient would create 608 mostly-trivial directories. Flat is cheaper to glob and easier to `rsync`.
- **Masks in their own tree** (separate from `raw/`): clear separation between "imaging input" and "ground-truth label". Easier to reason about — you never accidentally train on a tree that includes label files. Easier to diff/sync the imaging tree without copying labels. Path substitution still trivial: `raw/holdout/positive/X.nii.gz` ↔ `lesion_masks/holdout/positive/X.nii.gz`.
- **Derivatives as parallel top-level dirs** mirroring `raw/`'s internal structure: a derivative is a 1-to-1 transform on `raw/`. Mirroring the path means `raw/cross-validation/positive/X.nii.gz` ↔ `cropped_raw/cross-validation/positive/X.nii.gz` is a path-substitution rule, no manifest re-resolution.

### 4.4 What does NOT need a directory

- **2.5D read-in**: runtime sampling strategy (concatenate 3 adjacent slices as channels). DataLoader pattern, no disk format change.
- **Augmentations** (flips, intensity jitter, etc.): runtime only.
- **Inference-time transforms**: runtime only.
- **p1–p99 normalization**: borderline. Cheap as a runtime transform; precomputing it doubles disk size (fp32 vs int16). **Default plan: implement as runtime DataLoader transform first.** Only precompute to `normalized_p1p99/` if visualization needs the normalized volumes.

---

## 5. `data/manifest.csv` schema

The single source-of-truth lookup table for the migrated tree AND the project-wide patient inventory. **5,120 rows** at migration time (one per project patient), augmented by additional columns as derivative pipelines run (rows never get added; column count grows).

### 5.1 Columns at migration time

| Column | Source | Example | Notes |
|---|---|---|---|
| `mnemonic_id` | from `patient_id_mapping.csv` | `arctic_snow_tiger` | unique across all 5,120 patients |
| `anon_id` | source manifest `patient_id` | `ANON01042AC6BED6` | round-trip key |
| `split` | source manifest `split` | `fold0` / `holdout` / `phase2_unsupervised` / null | |
| `cohort` | derived: `pos`/`neg` → `positive`/`negative` | `positive` | |
| `soft_negative` | source manifest | `False` | boolean flag |
| **`transferred_to_home`** | derived: `split in phase1_splits` | `True` for 608 rows, `False` for ~4,512 | **the migration's headline column** |
| `raw_path` | constructed (transferred only); empty otherwise | `raw/cross-validation/positive/arctic_snow_tiger.nii.gz` | relative to `data/` |
| `raw_json_path` | constructed (transferred only) | `raw/cross-validation/positive/arctic_snow_tiger.json` | |
| `lesion_mask_path` | constructed (transferred positives only) | `lesion_masks/cross-validation/positive/arctic_snow_tiger.nii.gz` | |
| `bucket` | derived: `"holdout"` if `split=="holdout"` else `"cross-validation"` | `cross-validation` | the physical-tree directory; `split` carries the fold ID |
| `original_filename` | source manifest `output_filename` | `water_canonical.nii.gz` | for trace-back |
| `original_subdir` | source manifest `output_subdir` | `nifti_pos/ANON01042AC6BED6` | |
| `source_series_path` | source manifest | `/scratch/.../input/positive/ANON…/WATER:_…` | DICOM source dir |
| `image_type` | source manifest (BIDS sidecar) | `DERIVED\\PRIMARY\\DIXON\\WATER` | |
| `series_description` | source manifest | `WATER: COR LAVA DIAF` | |
| `scanner_model` | source manifest | `SIGNA Artist` | |
| `magnetic_field_strength` | source manifest | `1.5` | |
| `slice_thickness_mm` | source manifest | `6.0` | |
| `pixel_spacing_x_mm`, `pixel_spacing_y_mm` | source manifest | `1.5625` | |
| `shape` | source manifest | `512x80x512` | |
| `n_slices_actual` | source manifest | `80` | |
| `n_volumes_from_series` | source manifest | `1` (or `2`/`3` for multi-canonical patients — flag) | |
| `had_multi_canonical` | derived: `n_volumes_from_series > 1` | `False` (448 patients) / `True` (160 patients) | |
| `phase` | source manifest | `phase1` / `phase2` / `pilot` / `remask23` | SLURM-job-level provenance |
| `sha256_raw` | source manifest `sha256` | `…` | |
| `migration_timestamp` | timestamp of this migration (transferred only) | `2026-04-26T15:30:00` | empty for non-transferred |

For `transferred_to_home=False` rows: all the path columns (`raw_path`, `raw_json_path`, `lesion_mask_path`, `migration_timestamp`) are empty strings. All the other columns (BIDS metadata, sha256, etc.) are populated — the manifest is a complete project-wide registry, with paths populated only for what's been physically migrated.

### 5.2 Columns added by future derivative steps

Each derivative pipeline augments the manifest by adding columns and updating only the rows for migrated patients. None of these are written by THIS migration; documented here so the schema is forward-compatible.

| Pipeline | Columns added |
|---|---|
| TotalSegmentator | `liver_mask_path`, `liver_mask_sha256` |
| Crop | `crop_bbox_x0`, `crop_bbox_x1`, `crop_bbox_y0`, `crop_bbox_y1`, `crop_bbox_z0`, `crop_bbox_z1`, `cropped_raw_path`, `cropped_lesion_mask_path`, `cropped_shape`, `cropped_sha256` |
| p1–p99 normalize | `normalized_p1p99_path`, `p1_value`, `p99_value`, `normalized_sha256` |

Each step does `pl.read_csv → with_columns → write_csv`. Manifest is a single tabular file; column count grows monotonically.

---

## 6. Scripts to write or update

### 6.1 `scripts/generate_patient_names.py` (UPDATE existing)

Currently reads patient IDs from `nifti/*.nii.gz`. Two updates needed:

**(a)** Parameterize input/output paths via argparse:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Source manifest.csv (e.g., /scratch/.../output/manifest.csv)")
    ap.add_argument("--wordlists", type=Path,
                    default=Path(__file__).parent / "wordlists.json")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write patient_id_mapping.csv")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing mapping (DANGEROUS — shifts all names)")
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"{args.output} already exists. Refusing to overwrite without --force "
            f"(regenerating would shift names and break downstream references)."
        )

    adjectives, animals, nouns = load_wordlists(args.wordlists)
    df = pl.read_csv(args.manifest, infer_schema_length=10000)
    patient_ids = sorted(df["patient_id"].unique().to_list())
    # ... existing collision-resolution loop ...
    pl.DataFrame(mapping).sort("anon_id").write_csv(args.output)
```

**(b)** Change the separator from `-` to `_` in `generate_name`:

```python
return f"{adjectives[adj_idx]}_{animals[animal_idx]}_{nouns[noun_idx]}"
```

The hashing and collision-resolution logic is correct — keep it.

### 6.2 `scripts/migrate_to_home.py` (NEW)

Main migration script. **Idempotent, dry-run by default.**

```python
"""Copy + rename Phase 1 dev cohort from /scratch to /home/dia-endo-conversion/data.

CLI:
    python scripts/migrate_to_home.py \\
        --source-manifest /scratch/.../output/manifest.csv \\
        --output-root /scratch/.../output \\
        --mapping data/patient_id_mapping.csv \\
        --splits-json /scratch/.../workplan/splits.json \\
        --target-root /home/sak185/dia-endo-conversion/data \\
        [--execute]    # default: dry-run

Pseudocode:

1. Load: source_manifest, mapping, splits_json. Validate every column required is present.

2. Apply scope filters (§2):
    a. canonicals = manifest.filter(role=="canonical" AND volume_index==0)
    b. dedup canonicals by patient_id with phase preference (phase1 > remask23 > pilot)
    c. assert canonicals.height == 5120 (one per project patient)
    d. mark transferred_to_home = split in {holdout, fold0..4}
    e. assert sum(transferred_to_home) == 608

3. Build per-row migration plan (only for transferred rows):
    for each row r where transferred_to_home:
        mnemonic = mapping[r.patient_id]
        cohort_dir = "positive" if r.cohort == "pos" else "negative"
        bucket = "holdout" if r.split == "holdout" else "cross-validation"
        raw_dir = target_root / "raw" / bucket / cohort_dir
        mask_dir = target_root / "lesion_masks" / bucket / "positive"

        src_nii  = output_root / r.output_subdir / r.output_filename   # water_canonical.nii.gz
        src_json = src_nii.with_suffix("").with_suffix(".json")
        dst_nii  = raw_dir / f"{mnemonic}.nii.gz"
        dst_json = raw_dir / f"{mnemonic}.json"

        if r.cohort == "pos":
            src_mask = output_root / "masks_pos" / r.patient_id / "mask_canonical.nii.gz"
            dst_mask = mask_dir / f"{mnemonic}.nii.gz"
        else:
            src_mask, dst_mask = None, None

        plan.append((src_nii, dst_nii, src_json, dst_json, src_mask, dst_mask, r))

    # Note: the `bucket` derivation collapses fold0..fold4 into "cross-validation".
    # The fold ID is preserved in the manifest's `split` column and in splits.json.

4. Pre-flight checks (abort and report if any fail):
    - Every src_nii exists.
    - Every src_json exists.
    - For every positive: src_mask exists.
    - All dst paths are unique within the plan (no two patients map to the same target).
    - mapping covers every patient_id in the plan.
    - mapping covers every patient_id in the full canonicals set (for non-transferred rows).
    - Estimated target size (sum of source file sizes) <= remaining /home quota * 0.5
      (i.e., we leave at least half the quota free).
    - target_root either doesn't exist or is empty (or has only this script's previous artifacts).

5. Print dry-run summary:
    - 608 patient migrations, by split x cohort
    - 1,324 total file copies (608 NIfTI + 608 JSON + 108 lesion masks)
    - Estimated size: ~28 GB
    - Estimated wall: ~3-5 min sequential cp over NFS

6. If --execute:
    a. mkdir -p every unique dst directory.
    b. shutil.copy2 each (preserves mtime + permissions).
    c. After every NIfTI copy: re-compute sha256 of dst, compare to source manifest's sha256.
       Abort and report on first mismatch (corruption indicator).
    d. Track elapsed time; print progress every 50 patients.

7. Build target_root/manifest.csv (5,120 rows, schema per §5.1):
    - For transferred rows: populate raw_path, raw_json_path, lesion_mask_path
      (relative to data/), migration_timestamp.
    - For non-transferred rows: those columns are empty strings.
    - All other columns populated for every row.

8. Copy target_root/splits.json verbatim from --splits-json.

9. Copy target_root/patient_id_mapping.csv from --mapping (verbatim — it lives logically
   alongside the manifest, even if it was generated separately in Phase A).

10. Generate target_root/README.md (auto-generated, see §6.3).

11. Final report:
    - N patients migrated (must == 608), by split x cohort
    - N lesion masks copied (must == 108)
    - Total bytes copied
    - sha256 mismatches (must == 0)
    - Manifest row count (must == 5120)
    - Wall time
    - Any pre-flight warnings (note quota usage, etc.)
"""
```

### 6.3 README.md auto-generation (inside `migrate_to_home.py`)

The migration script writes a one-page README.md to `data/README.md`. Template (filled in at migration time):

```markdown
# dia-endo-conversion data tree

**Generated:** {migration_timestamp}
**Source:** /scratch/pioneer/users/sak185/dia-endo-conversion/output/
**Migration script:** scripts/migrate_to_home.py
**Naming:** mnemonic IDs from scripts/generate_patient_names.py + scripts/wordlists.json

## Layout

```
raw/<bucket>/<cohort>/<mnemonic>.{nii.gz,json}
    where <bucket> in {holdout, cross-validation}
          <cohort> in {positive, negative}
lesion_masks/<bucket>/positive/<mnemonic>.nii.gz       # GT masks, positives only
liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz        # ADDED LATER (TotalSegmentator)
cropped_raw/<bucket>/<cohort>/<mnemonic>.{nii.gz,json} # ADDED LATER
cropped_lesion_masks/<bucket>/positive/<mnemonic>.nii.gz # ADDED LATER
normalized_p1p99/<bucket>/<cohort>/<mnemonic>.nii.gz   # ADDED LATER (if precomputed)
predictions/<run_id>/<bucket>/<cohort>/<mnemonic>.nii.gz # ADDED DURING TRAINING
manifest.csv                # project-wide; transferred_to_home column gates what's actually here;
                            # `split` column carries fold ID for CV-pool patients
patient_id_mapping.csv      # ANON ↔ mnemonic; immutable
splits.json                 # frozen splits (seed=42); authoritative source for fold assignment
```

**Two physical buckets, five logical folds:** `cross-validation/` contains all 486 CV-pool patients in one tree. The fold assignment (fold0..fold4) is in `manifest.csv["split"]` and `splits.json["assignments"][<patient_id>]`. The training DataLoader reads splits.json (or the manifest) at runtime to determine which patients to use for which fold.

## Migration counts (this migration only)

| Bucket            | positive | negative | total |
|-------------------|---------:|---------:|------:|
| holdout           | 22       | 100      | 122   |
| cross-validation  | 86       | 400      | 486   |
| **total**         | **108**  | **500**  | **608** |

Per-fold breakdown of the cross-validation bucket (from `manifest.csv`, not the directory tree):

| split | positive | negative | total |
|-------|---------:|---------:|------:|
{filled_in_fold_table}

Lesion masks copied: 108 (one per positive patient).

## Project totals (in manifest.csv)

- Total patients tracked: 5,120
- Transferred to /home (this run): 608
- Phase 2 (still on /scratch only): ~4,512

Filter `transferred_to_home == True` in `manifest.csv` to scope to this directory.

## Verification

```bash
find raw -name "*.nii.gz" | wc -l                  # → 608
find lesion_masks -name "*.nii.gz" | wc -l         # → 108
find raw -name "*.json" | wc -l                    # → 608
ls raw/holdout/positive/ | wc -l                   # → 22 (.nii.gz)
ls raw/cross-validation/positive/ | wc -l          # → 86 (.nii.gz)
```

## Re-running

The migration is idempotent — re-running with the same inputs produces no changes (script
detects existing files via sha256 match). To force re-migration of a single patient,
delete the target files and re-run.

## Provenance

See `agent/migration-plan.md` for the design rationale, decisions, and the full execution
checklist.
```

### 6.4 No update to `scripts/rename_files.py`

That script does in-place rename in flat dirs. **Don't reuse it.** Leave as-is for any future flat-tree rename use case.

---

## 7. Execution checklist (for the executor agent)

Run in order. Each phase is independent; pause and report between if anything looks off.

### Phase A — Generate names

```bash
cd /home/sak185/dia-endo-conversion
source .venv/bin/activate
mkdir -p data

python scripts/generate_patient_names.py \
    --manifest /scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest.csv \
    --wordlists scripts/wordlists.json \
    --output data/patient_id_mapping.csv
```

**Verify:**
- `wc -l data/patient_id_mapping.csv` → **5121** (5,120 patients + 1 header).
- `cut -d, -f2 data/patient_id_mapping.csv | tail -n +2 | sort -u | wc -l` → **5120** (all mnemonics unique).
- `awk -F, 'NR>1 {print $2}' data/patient_id_mapping.csv | head -3` shows underscore-separated names like `arctic_snow_tiger`.
- Treat the file as immutable from this point on.

### Phase B — Dry-run migration

```bash
python scripts/migrate_to_home.py \
    --source-manifest /scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest.csv \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --mapping data/patient_id_mapping.csv \
    --splits-json /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/splits.json \
    --target-root /home/sak185/dia-endo-conversion/data
    # no --execute → dry-run
```

**Verify the dry-run report:**
- 608 patient migrations planned.
- 1,324 file copies (608 NIfTI + 608 JSON + 108 lesion masks).
- 0 missing source files, 0 duplicate target paths, 0 mapping-misses.
- Per-bucket counts match §2.3:
  - `holdout`: 122 patients (22 pos + 100 neg)
  - `cross-validation`: 486 patients (86 pos + 400 neg)
- Estimated target size: ~28 GB.

### Phase C — Execute migration

```bash
python scripts/migrate_to_home.py \
    [same args as Phase B] \
    --execute
```

Wall time estimate: ~3–5 min sequential cp over NFS for 28 GB.

### Phase D — Post-execution verification

```bash
# Counts
find data/raw -name "*.nii.gz" | wc -l            # expect 608
find data/raw -name "*.json"   | wc -l            # expect 608
find data/lesion_masks -name "*.nii.gz" | wc -l   # expect 108

# Per-bucket sanity (just two buckets now, not six)
for b in holdout cross-validation; do
    n_pos=$(ls data/raw/$b/positive/*.nii.gz 2>/dev/null | wc -l)
    n_neg=$(ls data/raw/$b/negative/*.nii.gz 2>/dev/null | wc -l)
    n_msk=$(ls data/lesion_masks/$b/positive/*.nii.gz 2>/dev/null | wc -l)
    echo "$b: pos=$n_pos, neg=$n_neg, masks=$n_msk (expect masks==pos)"
done
# Expected:
#   holdout:           pos=22, neg=100, masks=22
#   cross-validation:  pos=86, neg=400, masks=86

# Per-fold sanity (via manifest, NOT directory tree)
python -c "
import polars as pl
m = pl.read_csv('data/manifest.csv', infer_schema_length=10000)
assert m.height == 5120, f'Expected 5120 rows, got {m.height}'
assert m.filter(pl.col('transferred_to_home')).height == 608, 'Transferred count wrong'
assert m.filter(pl.col('transferred_to_home') & (pl.col('cohort')=='positive')).height == 108
print(f'manifest.csv: {m.height} rows, {m.filter(pl.col(\"transferred_to_home\")).height} transferred')
print()
print('By bucket (matches directory tree):')
print(m.filter(pl.col('transferred_to_home')).group_by(['bucket','cohort']).agg(pl.len()).sort(['bucket','cohort']))
print()
print('By split / fold (per the manifest, not the tree):')
print(m.filter(pl.col('transferred_to_home')).group_by(['split','cohort']).agg(pl.len()).sort(['split','cohort']))
"

# sha256 round-trip (paranoia check; the migration script already verified all):
RANDOM_FILE=$(find data/raw/cross-validation -name "*.nii.gz" | shuf -n 1)
sha256sum "$RANDOM_FILE"
# Compare to the corresponding sha256_raw column in data/manifest.csv

# Visual spot-check (run from a node with FSLeyes / ITK-SNAP):
SOME_POS=$(ls data/raw/cross-validation/positive/*.nii.gz | head -1)
SOME_MASK=$(echo "$SOME_POS" | sed 's|/raw/|/lesion_masks/|')
fsleyes "$SOME_POS" "$SOME_MASK" -cm red -a 60
```

### Phase E — Report back to user

One short note covering: counts (must match §2.3), sha256 mismatches (must be 0), total bytes, wall time, any pre-flight warnings, and the manifest row counts (5120 / 608).

---

## 8. Decisions captured in this plan (for the executor — do NOT reopen without explicit user input)

1. **Mnemonic separator: underscore** (`arctic_snow_tiger`). One-character change in `generate_patient_names.py`.
2. **`cohort` directory naming: `positive` / `negative`** (full words, not `pos`/`neg`).
3. **Mnemonic IDs come from the existing `generate_patient_names.py` algorithm** (SHA-256 hash + wordlists indexing + collision retry). Don't redesign.
4. **Generate mnemonics for all 5,120 patients, not just Phase 1.** Future-proofs Phase 2.
5. **`patient_id_mapping.csv` is immutable once written.** Script refuses to overwrite without `--force`.
6. **One canonical file per patient.** Pick `volume_index == 0`. Multi-canonical patients (160 of 608 in Phase 1) drop their `_e1`/`a`/`b` sub-volumes; flagged in manifest via `had_multi_canonical=True`.
7. **20 patients with duplicate manifest rows** (pilot+phase1) deduped by phase preference: phase1 > remask23 > pilot. File content is byte-identical (idempotent overwrite during conversion); deduplication picks a deterministic source.
8. **Soft-negatives go under `negative/`** with the `soft_negative` column flagged. No `soft_negative/` dir.
9. **Lesion masks are NOT co-located with volumes.** They live in a parallel tree `lesion_masks/<bucket>/positive/<mnemonic>.nii.gz`. Path-substitution rule: replace `raw/` with `lesion_masks/`.
10. **`data/manifest.csv` tracks all 5,120 patients** with a `transferred_to_home` boolean column. Filter on that column to scope to what's actually on /home.
11. **`splits.json` lives at top-level in /home/data** (not in a `metadata/` subdir).
12. **README.md is auto-generated** by the migration script. Hand-edit later if needed.
13. **Dry-run is the default; `--execute` is required to copy.** Don't change this.
14. **Two physical buckets only: `holdout/` and `cross-validation/`.** No per-fold sub-directories. The five folds (fold0..fold4) live as values in `manifest.csv["split"]` and in `splits.json` — the training DataLoader resolves them at runtime. The `bucket` column in the manifest carries the directory-tree value (`holdout` or `cross-validation`) for trivial path construction.
15. **Directory naming uses hyphen for `cross-validation`** (matches the user's wording). Mnemonic patient names use underscores (e.g., `arctic_snow_tiger`). The two conventions don't conflict because the directory name and the patient name occupy different path components.

---

## 9. Things to NOT do (anti-patterns for the executor)

- ❌ Don't move files; **always copy.** /scratch is the source of truth and gets backed up to SSD; /home is the working tree.
- ❌ Don't `mv` or rename anything in /scratch. All renaming happens at copy time in /home.
- ❌ Don't write into /home/data without `--execute`. The script must fail loudly if `--execute` is set without first having a passing dry-run.
- ❌ Don't generate new mnemonic names if `data/patient_id_mapping.csv` already exists. The generator script refuses to overwrite without `--force`. (Idempotency.)
- ❌ Don't create per-patient subdirectories (`data/raw/cross-validation/positive/arctic_snow_tiger/water_canonical.nii.gz`). Keep flat per-cohort: `data/raw/cross-validation/positive/arctic_snow_tiger.nii.gz`.
- ❌ Don't create per-fold sub-directories under `cross-validation/`. The fold ID is in `manifest.csv` and `splits.json`; baking it into the path adds churn (six dirs instead of two) without helping the training code, which already reads the JSON.
- ❌ Don't co-locate masks with volumes. Lesion masks go in the parallel `lesion_masks/` tree.
- ❌ Don't put a `_lesion` suffix on mask files. They're already in their own tree, so the bare mnemonic name suffices: `lesion_masks/cross-validation/positive/arctic_snow_tiger.nii.gz`.
- ❌ Don't drop columns from the source manifest when writing `data/manifest.csv`. Augment, don't replace — `anon_id`, `original_filename`, `source_series_path`, `phase`, `sha256_raw` all need to round-trip.
- ❌ Don't change the `splits.json` content in /home. Copy verbatim.
- ❌ Don't migrate Phase 2 patients. They get rows in the manifest with `transferred_to_home=False` and empty path columns; their files stay in /scratch.
- ❌ Don't migrate alt/fat/inphase/outphase volumes. Canonical only.
- ❌ Don't follow symlinks during copy if /scratch contains any (use `shutil.copy2` not `cp -L`).
- ❌ Don't create `lesion_masks/<bucket>/negative/` directories. Negatives don't have lesion masks.

---

## 10. Future steps (out of scope for THIS plan, sketched for context)

The migration plan above gets us to:
- `data/raw/<bucket>/<cohort>/<mnemonic>.{nii.gz,json}` (608 patients × 2 files; `bucket` ∈ {`holdout`, `cross-validation`})
- `data/lesion_masks/<bucket>/positive/<mnemonic>.nii.gz` (108 positives)
- `data/manifest.csv` (5,120 rows)

Subsequent steps (each layered on the same hierarchy pattern — same two `<bucket>/` dirs, same `<cohort>/` sub-dirs, same mnemonic filenames):

1. **`scripts/run_totalseg.py`** — runs TotalSegmentator (`task=total_mr`, `roi_subset=["liver"]`) on every file in `data/raw/`, writes the liver segmentation to `data/liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz`. Augments manifest with `liver_mask_path`, `liver_mask_sha256`.
2. **`scripts/apply_liver_crop.py`** — reads `data/raw/` + `data/liver_masks/`, computes bbox + 20 mm margin, writes:
   - cropped volume → `data/cropped_raw/<bucket>/<cohort>/<mnemonic>.{nii.gz,json}` (JSON includes `crop_bbox`)
   - cropped lesion mask (positives only) → `data/cropped_lesion_masks/<bucket>/positive/<mnemonic>.nii.gz`
   - augments manifest with `crop_bbox_*`, `cropped_raw_path`, `cropped_lesion_mask_path`, etc.
3. **`scripts/apply_p1p99.py`** *(only if precomputing)* — reads `data/cropped_raw/`, applies p1–p99 normalization, writes to `data/normalized_p1p99/`. Default plan: implement as runtime DataLoader transform first.
4. **`scripts/sanity_check_dataset.py`** — runs visual + numeric QC on the final tree (orientation, intensity ranges, mask voxel counts, shape consistency within scanner-thickness strata).

Each step follows the same pattern as the migration: idempotent, dry-run default, sha256 verification, manifest augmentation. The hierarchy from this plan is the load-bearing structure they all build on.

---

## 11. Pointers

- **Source manifest:** `/scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest.csv`
- **Splits:** `/scratch/pioneer/users/sak185/dia-endo-conversion/workplan/splits.json`
- **Lesion masks (already realigned):** `/scratch/pioneer/users/sak185/dia-endo-conversion/output/masks_pos/<ANONID>/mask_canonical.nii.gz`
- **Mnemonic generator:** `scripts/generate_patient_names.py` (UPDATE per §6.1)
- **Wordlists:** `scripts/wordlists.json` (1.6M unique combinations, immutable)
- **Plan predecessor:** [`handoff-2026-04-26-post-phase1.md`](./handoff-2026-04-26-post-phase1.md) §7 mentions HDF5; THIS plan supersedes that with a flat-NIfTI design per the user's preference for visual inspection.

---

**End of plan.** Ready for executor.
