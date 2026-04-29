# `scripts/` — cache-construction and dev workflow scripts

Scripts that produce or migrate authoritative artifacts. None of these are imported by the runtime package (`endo/`) — they all run as `uv run python scripts/<name>.py`. Many predate the current PRD and exist only as historical pipeline stages; the active production scripts are tagged below.

## Production-active scripts

| File | Purpose |
|---|---|
| `preprocess.py` | **Component 1.** Reads `data/manifest.jsonl` + `data/cohort.json`, resamples each volume to `(0.82, 1.5, 0.82) mm`, ROI-z-scores against the liver, crop+pad to `(408, 174, 408)`, derives 2D GT boxes, computes border bands, writes `cache/v1/{volumes,border_bands,gt_boxes.parquet,preprocessed_manifest.jsonl}`. `--probe-connectivity` runs the one-time CC count probe (6-conn vs 26-conn) at NATIVE resolution and writes `cache/v1/runtime/connectivity_lock.json`. Idempotent on `(raw_sha256, code_version, target_spacing, target_shape)`. |
| `analyze_inplane_spacing.py` | One-time analysis script. Reads each volume's NIfTI header, picks the cohort's median in-plane spacing, writes `agent/complete_spec/analysis_inplane_spacing.txt` with the recommended `TARGET_SPACING` constant. The result was already pasted into `preprocess.py` — re-run only if the cohort changes. |
| `build_lesion_bank.py` | **Component 2.** Reads `cache/v1/preprocessed_manifest.jsonl`, filters to `cohort=='cross-validation' AND label=='positive'` (86 donors), reads the locked connectivity from `runtime/connectivity_lock.json`, multiprocesses over donors to extract `LesionBankEntry` records, writes `lesion_bank_<git_sha8>.pkl`, the atomic `current.pkl` symlink, and `bank_provenance.json`. |
| `smoke_train.py` | **Component 8.** 5-min integration gate. Picks 5 smallest CV volumes (2 pos + 3 neg ensuring fold-0 has at least one positive AND at least one positive lives in another fold), writes `data/.smoke_manifest.jsonl`, builds the real DataModule + LightningModule, captures step losses, asserts SM1-SM4 (≥20 steps, last10 < first10, finite, val/slice_auroc logged). The CLI's `smoke` subcommand delegates here. |
| `build_unified_manifest.py` | Phase 0a one-shot migration that produced `data/manifest.jsonl` + `data/cohort.json` from the legacy multi-file format. Idempotent — running it now is a no-op given `data/_legacy/` is populated. |
| `build_splits.py` | Phase 0a frozen 5-fold split builder (stratified). Already executed; `data/cohort.json` carries the result. |

## Phase-0 / migration / one-time scripts

| File | Purpose |
|---|---|
| `consolidate.py`, `consolidate_sidecars.py` | Walked the upstream DICOM tree to a clean structure. Done. |
| `prescan.py`, `convert_one_patient.py`, `build_workplan.py` | Per-patient DICOM → NIfTI conversion driver. Done. |
| `dilate_segmentations.py`, `binarize_lesion_masks.py`, `realign_masks.py`, `realign_masks_v2.py` | Mask-canonical alignment + 20 mm liver-ROI dilation. Done. |
| `audit_mask_canonical.py`, `qc.py`, `preflight_check.py` | QC + audits run during the migration. Done. |
| `select_pilot.py`, `monitor.py` | SLURM-era pilot pickers + live monitor. Not used on the Lambda Labs A10. |
| `run_totalseg.py` | TotalSegmentator liver-mask driver. Done. |
| `migrate_local_copy_to_data.py`, `migrate_to_home.py`, `rename_files.py`, `generate_patient_names.py`, `build_remask_package.py` | Migration / cohort-renaming utilities. Frozen; consult `data/_legacy/` for inputs. |
| `_common.py`, `wordlists.json` | Helpers shared across the migration scripts. |

## Contracts

- **Cache contract** (PRD §5.2): `preprocess.py` is the sole producer of `cache/v1/`. Anything else that writes there violates the cache versioning. The cache is keyed on `(preprocess code SHA, target spacing, target shape, raw_sha256)`.
- **Bank contract**: `build_lesion_bank.py` is the sole producer of `cache/v1/lesion_banks/`. The atomic `current.pkl` symlink is what `endo.augmentation.transform.TrainAugmentation` loads by default.
- **Manifest contract**: `build_unified_manifest.py` enforces I.1.1-I.1.10 on write; treat `data/manifest.jsonl` as immutable post-Phase-0.

## Invariants

- `preprocess.py` cohort run produces I.7.1-I.7.10 (cache shapes, dtypes, CC count, contrast floor, border-band coverage, idempotency, disk budget).
- `build_lesion_bank.py` produces I.4.1-I.4.4 (86 donors, no holdout leak, ~157 CCs, connectivity matches the lock file).

## Don't

- Don't bypass `preprocess.py` to write to `cache/v1/` directly — the cache version provenance assumes this script is the only producer.
- Don't run the migration scripts on the current data tree. They'd no-op at best, but they're not part of the steady-state pipeline.
- Don't add `import endo` to a script that's part of the cache-construction path. Cache scripts must be runnable BEFORE the runtime is fully wired (Phase 0d → Phase 1 ordering).
