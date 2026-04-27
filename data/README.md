# dia-endo-conversion data tree

**Generated:** 2026-04-26T15:21:36
**Source:** /scratch/pioneer/users/sak185/dia-endo-conversion/output/
**Migration script:** scripts/migrate_to_home.py
**Naming:** mnemonic IDs from scripts/generate_patient_names.py + scripts/wordlists.json

## Layout

```
raw/<bucket>/<cohort>/<mnemonic>.nii.gz                # volumes only — sidecars consolidated
    where <bucket> in {holdout, cross-validation}
          <cohort> in {positive, negative}
lesion_masks/<bucket>/positive/<mnemonic>.nii.gz       # GT masks, positives only
liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz        # ADDED LATER (TotalSegmentator)
cropped_raw/<bucket>/<cohort>/<mnemonic>.nii.gz        # ADDED LATER
cropped_lesion_masks/<bucket>/positive/<mnemonic>.nii.gz # ADDED LATER
normalized_p1p99/<bucket>/<cohort>/<mnemonic>.nii.gz   # ADDED LATER (if precomputed)
predictions/<run_id>/<bucket>/<cohort>/<mnemonic>.nii.gz # ADDED DURING TRAINING
sidecars.jsonl             # one record per transferred patient — full BIDS sidecar + provenance
manifest.csv               # project-wide; transferred_to_home gates physical presence
patient_id_mapping.csv     # ANON ↔ mnemonic; immutable
splits.json                # frozen splits (seed=42); authoritative fold assignment
```

### sidecars.jsonl

One JSON object per line, 608 lines (one per transferred patient). Each record:

```json
{
  "mnemonic_id":  "amber_bear_quartz",
  "anon_id":      "ANOND1293F75ED74",
  "bucket":       "cross-validation",  // or "holdout"
  "split":        "fold0",              // fold0..fold4 or "holdout"
  "cohort":       "negative",           // or "positive"
  "raw_path":     "raw/cross-validation/negative/amber_bear_quartz.nii.gz",
  "sidecar":      { ... full dcm2niix BIDS sidecar (Modality, ImageType, etc.) ... }
}
```

Stream it with `polars.read_ndjson("data/sidecars.jsonl")` or `jq -c '.' data/sidecars.jsonl`.

**Two physical buckets, five logical folds:** `cross-validation/` contains all 486 CV-pool
patients in one tree. The fold assignment (fold0..fold4) is in `manifest.csv["split"]` and
`splits.json["assignments"]`. The training DataLoader reads splits.json (or the manifest)
at runtime to determine which patients to use for which fold.

## Migration counts (this migration only)

| Bucket            | positive | negative | total |
|-------------------|---------:|---------:|------:|
| holdout           | 22       | 100      | 122   |
| cross-validation  | 86       | 400      | 486   |
| **total**         | **108**  | **500**  | **608** |

Per-fold breakdown of the cross-validation bucket (from `manifest.csv`, not the directory tree):

| split | cohort | count |
|-------|--------|------:|
| fold0 | negative | 82 |
| fold0 | positive | 18 |
| fold1 | negative | 81 |
| fold1 | positive | 18 |
| fold2 | negative | 79 |
| fold2 | positive | 17 |
| fold3 | negative | 79 |
| fold3 | positive | 17 |
| fold4 | negative | 79 |
| fold4 | positive | 16 |
| holdout | negative | 100 |
| holdout | positive | 22 |

Lesion masks copied: 108 (one per positive patient).

## Project totals (in manifest.csv)

- Total patients tracked:    5060
- Transferred to /home:      608
- Not transferred (Phase 2 + leftovers): 4452

Filter `transferred_to_home == True` in `manifest.csv` to scope to this directory.

## Verification

```bash
find raw -name "*.nii.gz" | wc -l              # → 608
find lesion_masks -name "*.nii.gz" | wc -l     # → 108
find raw -name "*.json" | wc -l                # → 0  (consolidated into sidecars.jsonl)
wc -l sidecars.jsonl                           # → 608
ls raw/holdout/positive/ | wc -l               # → 22 (.nii.gz)
ls raw/cross-validation/positive/ | wc -l      # → 86 (.nii.gz)
```

## Migration stats

- Bytes copied: 18.76 GB
- Wall time:    428.9 s

## Re-running

The migration is idempotent — re-running with the same inputs produces no changes
(script detects existing files via sha256 match for NIfTIs, existence for JSONs and masks).
To force re-migration of a single patient, delete the target files and re-run.

## Provenance

See `agent/migration-plan.md` for the design rationale, decisions, and the full
execution checklist.
