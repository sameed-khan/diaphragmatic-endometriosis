# Diaphragmatic Endometriosis Dataset — DICOM → NIfTI Conversion Plan (v2)

**Owner:** sak185 (CWRU pioneer cluster)
**Source data:** `/home/jjs374/DiaE/`
**Project root:** `/home/sak185/dia-endo-conversion/`
**Working scratch:** `/scratch/pioneer/users/sak185/dia-endo-conversion/` *(persistent user scratch on pioneer; `/scratch/<user>/` is permission-denied. Per-job scratch is `$PFSDIR=/scratch/pioneer/jobs/job.<JOBID>.pioneer`. 14-day untouched-file purge.)*
**Final canonical output:** `/home/sak185/dia-endo-conversion/data/`
**Tool:** `dcm2niix` v1.0.20250505 (in `~/.local/bin`, on `$PATH`)

> **For implementing agents:** This plan is self-contained and can be executed end-to-end with no prior context. Every script, command, and decision is specified. There is **one human-review gate** between Stage 4 and Stage 5: the user runs the SLURM `sbatch` command themselves after reviewing the workplan. Do not submit SLURM jobs autonomously.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Investigation findings](#2-investigation-findings)
3. [Architectural decisions](#3-architectural-decisions)
4. [Project setup (uv)](#4-project-setup-uv)
5. [Directory & filename conventions](#5-directory--filename-conventions)
6. [Pipeline overview](#6-pipeline-overview)
7. [Stage details](#7-stage-details)
8. [Script specifications](#8-script-specifications)
9. [SLURM scripts](#9-slurm-scripts)
10. [Progress monitoring](#10-progress-monitoring)
11. [Manifest schemas](#11-manifest-schemas)
12. [Failure handling](#12-failure-handling)
13. [Runtime & storage](#13-runtime--storage)
14. [Pre-flight checklist](#14-pre-flight-checklist)
15. [Human review gates](#15-human-review-gates)
16. [Open questions](#16-open-questions)
17. [Appendix — ML pipeline context](#17-appendix--ml-pipeline-context)

---

## 1. TL;DR

Convert **~4,950 negative + 165 positive** patient folders of GE 1.5T LAVA Dixon coronal MRI to NIfTI (5 positives are hard-excluded — see §2.4). Write a `uv`-managed Python project at `/home/sak185/dia-endo-conversion/`. **Consolidate** `/home/jjs374/DiaE` (the canonical source) into a flat `/scratch/.../input/{positive,negative}/<ANONID>/<series>/` layout — collapsing the original five `dicom_neg<N>` batches into a single `negative/` tree, picking the per-ANONID canonical from the batch with the most series, dropping the four pos-overlap IDs from negatives, and excluding the redundant `dicom/Dicom upload/` subdir. Run a SLURM array (~100 jobs × ~50 patients each), have each task write to node-local `/tmp` and `rsync` back to `/scratch`. Convert positives in a second small SLURM array, then apply a deterministic dual-axis mask flip (`mask[::-1,:,::-1]` + new affine) to realign the existing masks to the fresh re-converted volumes. Final output lives at `/home/sak185/dia-endo-conversion/data/`. Pilot 10–20 patients first; user runs `sbatch` after reviewing the workplan; a `monitor.py` script gives live ETA against `squeue` + completed manifest rows.

> **v2.1 layout change (2026-04-25):** The original `data.tgz` tarball was structurally inconsistent (`dicom_neg1` had an extra `dicom_anon/` nesting level, `dicom_neg5` was absent). The pipeline now sources from `/home/jjs374/DiaE` directly via `scripts/consolidate.py`, which produces the flat `positive/`+`negative/` layout on /scratch in a single step. The downstream prescan/build_workplan no longer need cross-batch dedup logic (handled at consolidation time).

> **v2.2 mask alignment + exclusions (2026-04-26):** `realign_masks.py` rewritten to shape-match each existing mask against ALL freshly-converted WATER outputs for the patient (canonical + sub-volumes + alts) and save as `mask_<basename>.nii.gz`. `build_workplan.py` now emits one `alignment_audit.csv` row per existing mask file (no more single-mask-per-patient assumption). Five positives hard-excluded from the study via `EXCLUDED_PIDS` in `scripts/_common.py` because their lesions are not visible on the canonical sequence (see §2.4).

> **v2.3 phased rollout + CV splits (2026-04-26):** Conversion is split into Phase 1 (108 positives + 500 stratified-random negatives, ~15–30 min on SLURM) and Phase 2 (the remaining ~4,400 negatives, run on demand for SSL pretraining or supervised scaling). Splits are pre-computed and frozen via `scripts/build_splits.py` with `seed=42`: 22 positives + 100 negatives held out for the final test set; the rest 5-fold cross-validated. Stratification keys: `manufacturer_model_name` × slice-thickness binned (`≤4mm` vs `>4mm`) **on the canonical sequence only**. See §3a, §6, and §11 for details.

End-to-end wall time: ~5–6 hours for negatives, ~30 min for positives + mask fix.

---

## 2. Investigation findings

### 2.1 Source data inventory

Counted from `/home/jjs374/DiaE` (canonical source; ANON-named dirs only):

| Folder (source) | ANON dirs | Notes |
|---|---:|---|
| `dicom_neg1` | 1,001 | |
| `dicom_neg2` | 1,238 | |
| `dicom_neg3` | 1,052 | |
| `dicom_neg4` | 1,584 | |
| `dicom_neg5` |   239 | also contains a metadata file `_mapeamento_ids.csv` (240th entry; not a patient) |
| **Negatives total** | **5,114** | |
| Unique negative patient IDs after dedup | **4,950** | 160 cross-batch duplicates dropped at consolidation; 4 dropped because they're also positives |
| `dicom` (positives) | 170 | the prior plan said 171 — the 171st was the `Dicom upload/` subfolder |
| `dicom/Dicom upload/` | 17 | exact byte-level duplicates of 17 patients in `dicom/`; **excluded** at consolidation |
| Existing `nifti/` (prior conversion of positives) | 131 .nii.gz | unchanged |
| Existing `masks/` (radiologist annotations) | 131 .nii.gz + 131 .csv | unchanged |
| **Post-consolidation net to convert** | **4,950 negatives + 170 positives** | |

### 2.2 Hierarchy

**Source** (`/home/jjs374/DiaE`) is two levels deep under each batch, **not** standard DICOM patient/study/series:

```
dicom_neg<N>/<ANONID>/<SERIES_DESCRIPTION>/<sopinstanceUID>.dcm
dicom/<ANONID>/<SERIES_DESCRIPTION>/<sopinstanceUID>.dcm
```

**Post-consolidation layout** (`/scratch/.../input/`) flattens the batches:

```
positive/<ANONID>/<SERIES_DESCRIPTION>/<sopinstanceUID>.dcm
negative/<ANONID>/<SERIES_DESCRIPTION>/<sopinstanceUID>.dcm
nifti/   <ANONID>[_<series>].nii.gz       (prior radiologist-conv NIfTIs)
masks/   <ANONID>[_<series>].{nii.gz,csv} (radiologist masks)
```

There is no study-level folder. ~30% of patients have multiple series subfolders (max observed: 5). Series folder names contain spaces (e.g., `WATER COR LAVA DIAF.`); always quote in shell.

### 2.3 Sequence characteristics

- **Scanner:** GE Medical Systems, 1.5T. Two models: SIGNA Artist and SIGNA Explorer.
- **Pulse sequence:** `efgre3d` (LAVA 3D SPGR Dixon).
- **Canonical sequence:** Dixon WATER reconstructions. DICOM `ImageType` (0008,0008) 4th token == `WATER`; full string `DERIVED\PRIMARY\DIXON\WATER`. SeriesDescription always begins with `WATER:`. **100% (108/108 unambiguously mapped) of segmented series in the existing dataset are WATER.**
- **FAT, IN_PHASE, OUT_PHASE** present sporadically; FAT in some patients, in/out-phase in only 4.
- **Slice thickness varies systematically by scanner:**
  - SIGNA Artist: ~6.0 mm typical (range 2.2–7.13 mm)
  - SIGNA Explorer: ~3.5 mm typical (range 3.5–6.4 mm)
- **In-plane voxel spacing:** captured per-volume from DICOM `PixelSpacing` (0028,0030).

### 2.4 Edge cases

Most are now resolved at consolidation time (`scripts/consolidate.py`); prescan/build_workplan see only the cleaned tree.

| Issue | Count | Action |
|---|---|---|
| Patients hard-excluded from the study (lesion not visible on canonical sequence) | 5 | Listed in `scripts/_common.py::EXCLUDED_PIDS`; dropped at workplan build with reason `excluded_no_visible_lesion_on_canonical`. ANON25C6C345BBDA, ANON474B6A632EC1, ANONB37185FC9DAF, ANONC0DC7E3FB015, ANONC4A3AEBA378D. Re-introduce by removing from the set once the issue is resolved. |
| Empty patient folders (0 .dcm files inside series subfolder) | 85 (36 in neg4, 49 in neg5) | copied through; prescan flags `read_status=empty_folder`; skipped at workplan build |
| `.DS_Store`, `._*` files scattered | many | excluded by `rsync --exclude` at consolidation |
| `dicom/Dicom upload/` subdir | 17 patients (all duplicates of patients in `dicom/`) | excluded at consolidation (`Dicom upload` is in the consolidate.py exclude list) |
| `dicom_neg5/_mapeamento_ids.csv` | 1 file | excluded at consolidation (not an ANON dir) |
| Patient IDs duplicated across `dicom_neg*` | 160 (verified at consolidation) | resolved at consolidation: keep batch with most series; alphabetical batch tiebreak. Audit trail in `consolidation.csv`. |
| Patient IDs in both `dicom/` and `dicom_neg*` | 4 (`ANON4EF24D0EDFA5`, `ANON5DCB62C77550`, `ANON7317255BC6B3`, `ANONA9B87788C42B`) | excluded from `negative/` at consolidation; positives win. |
| `dicom_neg1/dicom_anon/` extra nesting (in the abandoned `data.tgz`) | n/a | not an issue: pipeline now reads from `/home/jjs374/DiaE` directly, which has no such nesting. |
| WATER series with unusual slice count (e.g., 564 in `ANON148BD54809B2`) | rare | let dcm2niix split if it detects geometry inconsistency; numeric suffixes (`_e1`, `_e2`) preserved |
| FAT-only patients (no WATER) | 2 known in neg1 | manifest row with `role=fat`; no `_canonical`; downstream training filters them |
| Series with no WATER/FAT prefix in folder name | a few | classify by DICOM `ImageType`, not folder name |
| Symlinks, non-ASCII filenames | none found | n/a |

### 2.5 HPC environment

- **Scheduler:** SLURM. Max wall time 13d 8h. Partitions: `batch` (CPU, nodes `compt230-289`, `compt291-399`), `gpu`. Single node = 40 cores / 256 GB RAM.
- **Filesystems:**
  - `/home` — NFSv4 on `vstorvip.lb.cwru.edu`, ~5 endpoints, 14 TB free.
  - `/scratch` — NFSv3 on same backend, ~16 endpoints (better aggregate throughput), 23 TB free, **14-day retention**.
  - `/tmp` — node-local XFS on `/dev/sda`, 868 GB free per node.
- **`dcm2niix`:** v1.0.20250505 at `~/.local/bin/dcm2niix`, on `$PATH`. Invoke as `dcm2niix`.
- **Python:** system is 3.6.8 (too old). Use `module load Python/3.11.3-GCCcore-12.3.0` and `uv` for venv management.
- **`uv`:** user installs to `~/.local/bin/uv` (already planned).

### 2.6 Affine / orientation / mask alignment (CRITICAL)

A focused investigation compared the original DICOMs, the prior-conversion NIfTIs in `/home/jjs374/DiaE/nifti/`, and the radiologist's masks in `/home/jjs374/DiaE/masks/`. **The masks are NOT in the same coordinate frame as the prior NIfTIs.** Findings:

| Comparison | Shapes | Zooms | Affine | Axcodes |
|---|---|---|---|---|
| Old NIfTI ↔ Mask | match | match | **flip on axis-2 + Z translation shift** | (L,A,S) vs (L,A,**I**) |
| Old NIfTI ↔ Fresh dcm2niix v1.0.20250505 | match | match | **flip on axis-0 + X translation shift** | (L,A,S) vs (**R**,A,S) |
| Mask ↔ Fresh dcm2niix | match | match | **flip on BOTH axis-0 and axis-2** | (L,A,I) vs (R,A,S) |
| DICOM `ImagePositionPatient` ↔ Mask origin | — | — | **identical** | — |
| qform/sform codes | old NIfTI: qform=0, sform=2 | mask: qform=1, sform=1 | fresh: qform=1, sform=1 | — |

**Interpretation.** The radiologist's annotation tool saved masks anchored to the raw DICOM physical origin (matches `ImagePositionPatient` exactly), in LAS-with-S↓ orientation. The old dcm2niix wrote LAS-with-S↑ NIfTIs, with a non-standard `qform=0`. The fresh dcm2niix v1.0.20250505 writes standard RAS NIfTIs with `qform=1, sform=1`. Net: re-converting positives with the new pipeline produces NIfTIs that **do not align voxel-for-voxel with existing masks** unless the masks are flipped on both axis-0 and axis-2 and re-headered with the fresh affine.

**The fix is lossless** (pure index reversal, no interpolation):

```python
fixed_data = mask_data[::-1, :, ::-1]
fixed_img  = nib.Nifti1Image(fixed_data, affine=fresh.affine, header=fresh.header)
```

This was verified deterministic across 5 sample patients (3 plain-style + 2 extended-style filenames, including one oblique-affine patient). Implementation: `scripts/realign_masks.py` (§8.4).

---

## 3. Architectural decisions

| # | Decision |
|---|---|
| Cohort scope | Convert all 5,115 negatives + all 171 positives. |
| Canonical sequence selection | DICOM `ImageType` 4th token == `WATER`; SeriesDescription starts with `WATER:` (belt-and-suspenders). |
| Multi-WATER per patient | Keep all WATER series. Mark exactly one as `_canonical` per patient via tiebreak rule (§5.3). Flag alternates as `water_alt_NN`. |
| FAT series | Convert if present; name `fat_NN.nii.gz`. Reserved for future second-channel training. |
| Other Dixon (in/out phase) | Convert if encountered (rare); name `inphase_NN`, `outphase_NN`. |
| Re-convert positives | **Yes.** Apply mask realignment (axis-0 + axis-2 flip) so existing masks align with fresh RAS NIfTIs. |
| Resampling at conversion time | None. Convert at native voxel resolution. Harmonization happens in training preprocessing. |
| BIDS JSON sidecars | `-b y`, `-ba n` (DICOMs already anonymized; preserve scanner/protocol metadata). |
| Compression | `-z y` (.nii.gz). |
| Output organization | Nested per-patient: `data/<cohort>/[<batch>/]<ANONID>/<role>.{nii.gz,json}`. |
| Failure policy | dcm2niix failures → `failed.csv`, no retry, no abort. Excluded patients → `skipped.csv`. |
| Working location | `/scratch/pioneer/users/sak185/dia-endo-conversion/` (input + intermediate). Final → `/home/sak185/dia-endo-conversion/data/`. |
| Pilot | Stage 0 runs on a hand-picked subset of ~12 patients; user reviews before launching full arrays. |
| Human review gate | Between Stage 3 (workplan generated) and Stage 4 (SLURM submission). User runs `sbatch` themselves. |

---

## 4. Project setup (uv)

`uv` is the user's preferred Python project manager. Already installed at `~/.local/bin/uv`.

### 4.1 Initialize the project

```bash
cd /home/sak185/
mkdir -p dia-endo-conversion
cd dia-endo-conversion
uv init --python 3.11
```

This creates `pyproject.toml`, `.python-version`, and an empty `.venv/`. The `.python-version` should be `3.11` (matches the cluster's `Python/3.11.3-GCCcore-12.3.0` module).

### 4.2 Add dependencies

```bash
uv add pydicom nibabel numpy polars tqdm
```

Resulting `pyproject.toml` should contain:
```toml
[project]
name = "dia-endo-conversion"
version = "0.1.0"
description = "DICOM → NIfTI conversion pipeline for diaphragmatic endometriosis dataset"
requires-python = ">=3.11"
dependencies = [
    "pydicom>=3.0",
    "nibabel>=5.4",
    "numpy>=1.26",
    "polars>=1.0",
    "tqdm>=4.66",
]
```

### 4.3 Verify

```bash
uv run python -c "import pydicom, nibabel, polars, tqdm; print('OK')"
```

Should print `OK`. From now on, run all Python scripts via `uv run python scripts/<name>.py …` (or activate `.venv/bin/activate` once and call `python` directly).

### 4.4 Project layout (after setup)

```
/home/sak185/dia-endo-conversion/
├── .venv/                                    # uv-managed venv (~150 MB)
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md                                 # quickstart for collaborators
├── convert-plan-v2.md                        # this document (symlink or copy from /home/sak185/)
├── agent/                                    # human-facing planning + handoff docs
│   ├── convert-plan-v2.md                    # THIS document
│   └── handoff-2026-04-26.md                 # session handoff snapshot (created at context-rotation points)
├── scripts/
│   ├── _common.py                            # shared constants (POSITIVE_OVERLAP_IDS, EXCLUDED_PIDS, helpers)
│   ├── consolidate.py                        # Stage 0a: rsync /home/jjs374/DiaE → /scratch with dedup
│   ├── prescan.py                            # Stage 1: read DICOM headers
│   ├── build_workplan.py                     # Stage 2: classify + filter + plan
│   ├── select_pilot.py                       # Stage 3: auto-pick pilot patients
│   ├── convert_one_patient.py                # called inside SLURM array tasks
│   ├── build_splits.py                       # Stage 3a: holdout + 5-fold CV splits, write splits.json (TO IMPLEMENT)
│   ├── realign_masks.py                      # Stage 4b/5b: shape-match + dual-axis flip masks
│   ├── qc.py                                 # Stage 6: post-conversion QC + manifest finalize
│   ├── monitor.py                            # progress tracker (run separately during SLURM stages)
│   ├── audit_mask_canonical.py               # one-off: which patients have masks on alt vs canonical
│   └── build_remask_package.py               # one-off: stage manual-remask packages for offline re-segmentation
├── slurm/
│   ├── convert_phase1.slurm                  # Stage 4: ~600-patient subset (TO IMPLEMENT — currently named convert_neg.slurm + convert_pos.slurm)
│   └── convert_phase2.slurm                  # Stage 5: ~4,400 remaining negatives (TO IMPLEMENT)
├── data/                                     # CANONICAL OUTPUT (rsync'd from /scratch at end)
│   ├── nifti_neg/<ANONID>/water_canonical.{nii.gz,json}
│   ├── nifti_pos/<ANONID>/water_canonical.{nii.gz,json}, water_alt_NN.{nii.gz,json}, ...
│   ├── masks_pos/<ANONID>/mask_<basename>.nii.gz   # one per existing radiologist mask, basename matches the matched volume
│   ├── manifest.csv                          # all volumes (neg + pos), one row each, with `split` column
│   ├── splits.json                           # frozen patient-level splits (seed + assignments)
│   ├── skipped.csv                           # excluded patients with reasons
│   ├── failed.csv                            # dcm2niix failures
│   ├── pre_scan_index.csv                    # pre-conversion DICOM header tally
│   ├── alignment_audit.csv                   # per-positive: mask-fresh-NIfTI alignment check
│   └── README.md                             # frozen run summary + this plan
└── logs/                                     # symlink to /scratch/pioneer/users/sak185/dia-endo-conversion/logs/
```

---

## 5. Directory & filename conventions

### 5.1 Output tree

```
/home/sak185/dia-endo-conversion/data/
├── nifti_neg/<batch>/<ANONID>/<role>.{nii.gz,json}
├── nifti_pos/<ANONID>/<role>.{nii.gz,json}
└── masks_pos/<ANONID>/mask_canonical.nii.gz
```

### 5.2 Filename schema

| Role | Filename | Notes |
|---|---|---|
| Canonical WATER | `water_canonical.nii.gz` + `water_canonical.json` | Always present unless patient has no WATER. |
| Additional WATER | `water_alt_01.nii.gz`, `water_alt_02.nii.gz`, … | Sorted deterministically by SeriesDescription. |
| FAT | `fat_01.nii.gz`, `fat_02.nii.gz`, … | If present. |
| IN_PHASE | `inphase_01.nii.gz`, ... | Rare. |
| OUT_PHASE | `outphase_01.nii.gz`, ... | Rare. |
| dcm2niix geometry-split sub-volumes | filenames preserve dcm2niix's `_e1`, `_e2` suffix verbatim, e.g., `water_alt_01_e1.nii.gz` | Documented per row in manifest. |
| Mask (positives only) | `mask_canonical.nii.gz` | Realigned to `water_canonical.nii.gz` of the same patient. |

The patient ID (`ANONID`) is in the path, not in the leaf filename. This supports clean glob patterns:
- `data/nifti_neg/*/*/water_canonical.nii.gz`
- `data/nifti_pos/*/water_canonical.nii.gz`
- `data/masks_pos/*/mask_canonical.nii.gz`

### 5.3 Canonical-selection algorithm (multi-WATER tiebreak)

```python
# After Stage 1 builds pre_scan_index.csv:
seriesdesc_freq = (
    pre_scan_index
    .query("image_type_token == 'WATER'")
    .series_description.value_counts()
    .to_dict()
)

def select_canonical(patient_water_rows):
    """Returns the series row to mark as canonical for one patient."""
    return sorted(
        patient_water_rows,
        key=lambda r: (
            -seriesdesc_freq.get(r.series_description, 0),  # most-common across dataset first
            -r.n_dcm_files,                                  # then most slices
            r.series_description,                            # then alphabetical (deterministic)
        )
    )[0]
```

### 5.4 Positive cohort: identifying which series was segmented

For each of the 171 positive patients, we must determine which series the existing mask corresponds to (so we can mark the same series as `_canonical` and apply the realigned mask only to that volume). Three cases:

1. **Existing nifti filename is extended-style** (e.g., `ANON01042AC6BED6_WATER:_COR_DIAFRAGMA_T1_LAVA_AB.nii.gz`): the series description is encoded in the filename — match it back to the source series folder (replace `_` with ` `, strip `WATER:` prefix). 25 patients.
2. **Existing nifti filename is plain ANONID-only AND the patient's `dicom/<ANONID>/` has only one series subfolder**: that's the segmented series. 83 patients (per the prior pre-scan).
3. **Existing nifti filename is plain ANONID-only AND the patient has multiple series subfolders** (23 ambiguous patients): match by **slice count** — the existing nifti's z-dim must equal the .dcm count of exactly one series. If there's still a tie or no match, log to `alignment_audit.csv` with `ambiguous=True` and skip the mask realignment for that patient (the patient still gets converted; only the mask is dropped).

This mapping is computed once in `scripts/build_workplan.py` and recorded in `workplan.csv` with a `mask_source_path` column for positive rows.

---

## 6. Pipeline overview

| Stage | What happens | Where it runs | Wall time |
|---|---|---|---|
| 0 | One-time setup: project init, install deps, create scratch dirs | head node | ~5 min |
| 0a | **Consolidate**: rsync `/home/jjs374/DiaE` → `/scratch/.../input/{positive,negative,nifti,masks}` with dedup | head node (parallel rsync) | ~30–60 min |
| 1 | Pre-scan: read DICOM headers from every series, write `pre_scan_index.csv` | head node (parallel) | ~10 min |
| 2 | Build workplan: classify + filter + select canonicals; write `workplan.csv`, `skipped.csv` | head node | <1 min |
| 3 | **Pilot run**: convert ~13 auto-picked patients sequentially; user inspects outputs | head node | ~30 min |
| 3a | **Build splits + Phase 1 subset**: stratified holdout + 5-fold CV; write `splits.json` and `subset_phase1.txt` (~600 patients) | head node | <1 min |
| 4a | **Phase 1 SLURM array**: convert subset_phase1 (~108 pos + 500 neg) — *gated by user `sbatch`* | compute nodes | ~15–30 min |
| 4b | **Phase 1 mask realign + QC** | head node | ~10 min |
| **— Initial baseline-training milestone here —** | | | |
| 5 | **Phase 2 SLURM array**: convert remaining ~4,400 negatives — *gated; runs whenever Phase 2 is needed (SSL or scaling supervised set)* | compute nodes | ~3–4 hours |
| 6 | Post-conversion QC + manifest finalization (re-run after Phase 2) | head node | ~10 min |
| 7 | Copy `/scratch/.../output/` → `/home/sak185/dia-endo-conversion/data/` | head node | ~15 min |

---

## 7. Stage details

### Stage 0 — One-time setup

```bash
# 0.1 Project init (see §4)
cd /home/sak185/ && mkdir -p dia-endo-conversion && cd dia-endo-conversion
uv init --python 3.11
uv add pydicom nibabel numpy polars tqdm

# 0.2 Working directories on /scratch
SCRATCH=/scratch/pioneer/users/sak185/dia-endo-conversion
mkdir -p "$SCRATCH"/{input,output,logs,workplan}
ln -sfn "$SCRATCH/logs" /home/sak185/dia-endo-conversion/logs

# 0.3 Verify dcm2niix on PATH
which dcm2niix && dcm2niix --version | head -1
# expect: /home/sak185/.local/bin/dcm2niix
#         Chris Rorden's dcm2niiX version v1.0.20250505 ...
```

### Stage 0a — Consolidate source data

Replaces the v2.0 `tar -xzf data.tgz` step. `data.tgz` was structurally inconsistent
(neg1 nesting; missing neg5). Source from `/home/jjs374/DiaE` instead.

```bash
cd /home/sak185/dia-endo-conversion
uv run python scripts/consolidate.py \
    --input-root  /home/jjs374/DiaE \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/input \
    --workers 16
```

Produces:
- `/scratch/.../input/positive/<ANONID>/<series>/...` (170 patients)
- `/scratch/.../input/negative/<ANONID>/<series>/...` (~4,950 patients post-dedup)
- `/scratch/.../input/nifti/`  (131 .nii.gz, prior radiologist conversions)
- `/scratch/.../input/masks/`  (131 .nii.gz + 131 .csv)
- `/scratch/.../input/consolidation.csv` — full audit trail (which batch each
  ANONID came from, which alternates were dropped, rsync exit per patient)

Excludes `.DS_Store`, `._*`, `dicom/Dicom upload/`, `dicom_neg5/_mapeamento_ids.csv`.
Dedup rules: pos-overlap-wins (4 IDs); for cross-batch neg duplicates, keep the
batch with the most series subfolders (alphabetical batch tiebreak).

Wall time: ~30–60 min depending on NFS contention.

Sanity check after consolidation:
```bash
ls /scratch/pioneer/users/sak185/dia-endo-conversion/input/positive | wc -l   # expect 170
ls /scratch/pioneer/users/sak185/dia-endo-conversion/input/negative | wc -l   # expect ~4950
ls /scratch/pioneer/users/sak185/dia-endo-conversion/input/nifti    | wc -l   # expect 131
ls /scratch/pioneer/users/sak185/dia-endo-conversion/input/masks    | wc -l   # expect 262
```

### Stage 1 — Pre-scan

Build `pre_scan_index.csv` — one row per (patient, series, cohort).

```bash
cd /home/sak185/dia-endo-conversion
uv run python scripts/prescan.py \
    --input-root /scratch/pioneer/users/sak185/dia-endo-conversion/input \
    --output     /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/pre_scan_index.csv \
    --workers 32
```

Walks `/scratch/.../input/positive/<ANONID>/<series>` and `/scratch/.../input/negative/<ANONID>/<series>`, reads ONE `.dcm` per series for headers, parallelized across CPU cores.

Expected output: ~24,000 rows (one per series across ~5,120 consolidated patients).

### Stage 2 — Build workplan

Apply exclusion rules, classify by ImageType, select canonicals, identify mask-source mapping for positives.

```bash
uv run python scripts/build_workplan.py \
    --pre-scan-index /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/pre_scan_index.csv \
    --existing-nifti /scratch/pioneer/users/sak185/dia-endo-conversion/input/nifti \
    --existing-masks /scratch/pioneer/users/sak185/dia-endo-conversion/input/masks \
    --output-dir     /scratch/pioneer/users/sak185/dia-endo-conversion/workplan
```

Produces in `workplan/`:
- `workplan.csv` — one row per series-to-convert (negatives + positives)
- `skipped.csv` — one row per excluded patient/series
- `alignment_audit.csv` — one row per positive, with `mask_source_path` and `mask_mappable` flag

**Exclusion order** (cross-batch dedup + pos-overlap already done at consolidation; see `consolidation.csv`):
1. Empty source folder (no .dcm)
2. Series with non-Dixon ImageType: skip
3. For each remaining patient: WATER series → `canonical` (one) + `alt_NN` (rest); FAT → `fat_NN`; IN/OUT phase → respective.

### Stage 3 — Pilot run

Convert a hand-picked subset and visually verify before launching the full arrays.

```bash
# Pick 12 diverse patients from workplan.csv:
#   5 single-WATER negatives
#   3 multi-WATER negatives
#   2 FAT-bearing patients
#   2 positives (one with extended-style nifti, one with plain) — for mask-realignment validation
# Save the patient ID list as workplan/pilot_patients.txt

uv run python scripts/convert_one_patient.py \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --patient-list /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/pilot_patients.txt \
    --workers 4

# For positive pilots, also run mask realignment:
uv run python scripts/realign_masks.py \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --masks-root /scratch/pioneer/users/sak185/dia-endo-conversion/input/masks \
    --alignment-audit /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/alignment_audit.csv \
    --patient-list /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/pilot_patients.txt
```

**Pilot QC checklist (user does this manually):**
- [ ] Open 3 random `water_canonical.nii.gz` in FSLeyes/ITK-SNAP/MRIcroGL. Verify orientation looks coronal, intensity reasonable, slice count plausible.
- [ ] Open 1 positive `water_canonical.nii.gz` + the realigned `mask_canonical.nii.gz` as overlay. Verify the mask voxels overlap the lesion in the volume.
- [ ] Inspect 3 BIDS JSON sidecars: confirm `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`, `SliceThickness`, `ImageType` populated.
- [ ] Inspect manifest rows (run a subset `qc.py`) — canonical/alt classification matches expectation.

If anything looks wrong: fix the pipeline and re-run the pilot before proceeding.

### Stage 3a — Build splits + Phase 1 subset

**Phased rollout strategy** (added v2.3, 2026-04-26). The full conversion of ~5,120 patients would take ~3–4h, but the initial baseline model only needs a small working subset. We split the conversion into two phases so model iteration can begin in parallel with the bulk negative conversion.

**Pool design** (post-exclusions, 165 active positives, 4,950 negatives):

| Pool | Patients | Purpose |
|---|---:|---|
| **Held-out test** (LOCKED, never touched during dev) | 22 pos + 100 neg | Final paper evaluation only |
| **5-fold CV pool** | 86 pos + 400 neg | Training + validation; 5 folds, ~17 pos + ~80 neg per fold |
| **Phase 2 negatives** (added later) | +4,400 neg | MAE SSL pretraining (and optional augmentation of supervised set) |

**Stratification keys** (applied independently to positives and negatives within each cohort):
1. `manufacturer_model_name` — SIGNA Artist vs SIGNA Explorer (~50/50 mix; real distribution shift per §17).
2. **Slice thickness binned coarsely** on the **canonical sequence only**: `≤4mm` vs `>4mm`.

This produces 4 strata per cohort (Artist-thin, Artist-thick, Explorer-thin, Explorer-thick). Stratification ensures every fold sees both scanner models and both thickness regimes, so per-fold performance isn't dominated by which scanner happens to be over-represented.

**Determinism**: a single seed (`SUBSET_SEED = 42`) governs all stratified random sampling. Re-running with the same seed produces identical splits. The seed and per-patient assignments are serialized in `splits.json` and never re-rolled.

```bash
uv run python scripts/build_splits.py \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --pre-scan-index /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/pre_scan_index.csv \
    --alignment-audit /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/alignment_audit.csv \
    --output-dir /scratch/pioneer/users/sak185/dia-endo-conversion/workplan \
    --holdout-pos 22 --holdout-neg 100 \
    --phase1-neg 500 \
    --n-folds 5 --seed 42
```

Produces:
- `splits.json` — per-patient assignment to one of: `holdout` | `fold0` | `fold1` | `fold2` | `fold3` | `fold4` | `phase2_unsupervised`. Includes the seed and stratification rule for reproducibility.
- `subset_phase1.txt` — patient IDs in scope for Phase 1 (108 with masks + 100 holdout neg + 400 CV neg = ~608).
- `subset_phase2.txt` — remaining ~4,400 negatives (assigned `phase2_unsupervised`).
- `splits_summary.csv` — per-stratum row counts for QC.

### Stage 4 — Phase 1 SLURM array (~600 patients)

**HUMAN GATE.** The user runs:

```bash
cd /home/sak185/dia-endo-conversion
sbatch slurm/convert_phase1.slurm
```

`convert_phase1.slurm` is small (array=1-10%5, ~60 patients/task, ~15–30 min wall) and processes the union of Phase 1 positives + Phase 1 negatives in `subset_phase1.txt`. Per-task logic identical to the original `convert_neg.slurm` (write to node-local `$PFSDIR`, rsync results to /scratch).

### Stage 4b — Phase 1 mask realign + QC

```bash
uv run python scripts/realign_masks.py \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --masks-root /scratch/pioneer/users/sak185/dia-endo-conversion/input/masks \
    --alignment-audit /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/alignment_audit.csv \
    --patient-list /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/subset_phase1.txt
```

For each existing radiologist mask, the script:
1. Loads the mask. Reads its shape.
2. Searches `output/nifti_pos/<ANONID>/` for a freshly-converted WATER volume (canonical, canonicala/b sub-volumes, or alt_NN) whose shape matches.
3. Applies the lossless `mask_data[::-1, :, ::-1]` dual-axis flip; adopts the matched volume's affine + header (with `scl_slope=1`, `scl_inter=0`, dtype=uint8 to avoid float-precision drift).
4. Saves to `output/masks_pos/<ANONID>/mask_<basename>.nii.gz` where `<basename>` is e.g. `canonical`, `canonicala`, `alt_01`.
5. Logs to `alignment_audit_results.csv` with `qc_status ∈ {ok, fail, skip}`. Multi-label masks (`{0,1,2}` etc.) are passed through with `label_values` flagged in `qc_notes` (not failed).

Then run `qc.py` to assemble the Phase 1 manifest. **Now you can train the baseline model on Phase 1 data.**

### Stage 5 — Phase 2 negatives SLURM array (run later)

When you need the rest of the negatives — for SSL pretraining or to scale the supervised set — run:

```bash
sbatch slurm/convert_phase2.slurm
```

`convert_phase2.slurm` partitions `subset_phase2.txt` (~4,400 negatives) into ~90 array tasks throttled to 30 concurrent. Wall time: ~3–4 hours.

After Phase 2 completes, re-run `qc.py` to refresh the manifest with the new rows.

### Stage 6 — Post-conversion QC + manifest finalization

```bash
uv run python scripts/qc.py \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --workplan    /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --skipped     /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/skipped.csv \
    --alignment-audit /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/alignment_audit.csv
```

Concatenates `manifest_part_*.csv`, parses every BIDS sidecar, computes per-volume metadata (scanner model, voxel spacings, slice counts), runs sanity checks, writes:
- `output/manifest.csv`
- `output/qc_flags.csv`
- `output/summary.txt`

### Stage 7 — Copy to canonical location

```bash
DEST=/home/sak185/dia-endo-conversion/data
mkdir -p "$DEST"
rsync -av --info=progress2 \
    /scratch/pioneer/users/sak185/dia-endo-conversion/output/ \
    "$DEST"/
cp /home/sak185/convert-plan-v2.md "$DEST"/README.md
echo "Conversion complete: $(date)" >> "$DEST"/README.md
```

After verifying `$DEST` looks complete, optionally clean up:
```bash
# Optional: free /scratch DICOMs (but keep output until /home is verified backed up)
rm -rf /scratch/pioneer/users/sak185/dia-endo-conversion/input/{positive,negative}
```

---

## 8. Script specifications

All scripts live in `/home/sak185/dia-endo-conversion/scripts/`. They share a common helper module `scripts/_common.py` with constants and small helpers.

### 8.0 `scripts/_common.py`

```python
"""Constants and helpers shared across scripts."""
from pathlib import Path

# Patients to exclude entirely from the negatives cohort
# (they appear in both dicom/ as positives and dicom_neg* as negatives)
POSITIVE_OVERLAP_IDS = {
    "ANON4EF24D0EDFA5",
    "ANON5DCB62C77550",
    "ANON7317255BC6B3",
    "ANONA9B87788C42B",
}

NEG_BATCHES = ("dicom_neg1", "dicom_neg2", "dicom_neg3", "dicom_neg4", "dicom_neg5")
POS_BATCH = "dicom"

# Mapping ImageType 4th token -> role prefix
IMAGETYPE_TO_ROLE = {
    "WATER":     "water",
    "FAT":       "fat",
    "IN_PHASE":  "inphase",
    "OUT_PHASE": "outphase",
}

# QC thresholds
MIN_SLICES_QC_FLAG = 30


def first_dcm_in(series_dir: Path) -> Path | None:
    """Return the first .dcm file in a series folder, or None if empty."""
    for p in sorted(series_dir.iterdir()):
        if p.is_file() and (p.suffix.lower() == ".dcm" or "." not in p.name):
            return p
    return None


def cohort_of(batch: str) -> str:
    return "pos" if batch == POS_BATCH else "neg"
```

### 8.1 `scripts/prescan.py`

```python
"""Stage 1: scan every series subfolder, read one DICOM, write pre_scan_index.csv."""
import argparse, csv, multiprocessing as mp
from pathlib import Path
import pydicom
from tqdm import tqdm
from _common import NEG_BATCHES, POS_BATCH, first_dcm_in

CSV_FIELDS = [
    "cohort", "batch", "patient_id", "series_path", "series_folder_name",
    "n_dcm_files", "image_type_full", "image_type_token",
    "series_description", "manufacturer", "manufacturer_model_name",
    "magnetic_field_strength", "slice_thickness_mm",
    "spacing_between_slices_mm", "pixel_spacing_x_mm", "pixel_spacing_y_mm",
    "rows", "cols", "sequence_name", "scanning_sequence",
    "read_status", "read_error",
]


def list_series(input_root: Path):
    """Yield (cohort, batch, patient_id, series_dir) tuples."""
    for batch in (POS_BATCH,) + NEG_BATCHES:
        batch_dir = input_root / batch
        if not batch_dir.is_dir():
            continue
        cohort = "pos" if batch == POS_BATCH else "neg"
        for patient in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            for series in sorted(s for s in patient.iterdir() if s.is_dir()):
                yield cohort, batch, patient.name, series


def scan_one(args):
    cohort, batch, pid, series = args
    n_dcm = sum(1 for _ in series.iterdir() if _.is_file())
    row = {f: "" for f in CSV_FIELDS}
    row.update(cohort=cohort, batch=batch, patient_id=pid,
               series_path=str(series), series_folder_name=series.name,
               n_dcm_files=n_dcm)
    sample = first_dcm_in(series)
    if sample is None:
        row["read_status"] = "empty_folder"
        return row
    try:
        ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        it = list(ds.get("ImageType", []))
        row["image_type_full"] = "\\".join(it)
        row["image_type_token"] = it[3] if len(it) >= 4 else ""
        row["series_description"] = str(ds.get("SeriesDescription", ""))
        row["manufacturer"] = str(ds.get("Manufacturer", ""))
        row["manufacturer_model_name"] = str(ds.get("ManufacturerModelName", ""))
        row["magnetic_field_strength"] = str(ds.get("MagneticFieldStrength", ""))
        row["slice_thickness_mm"] = str(ds.get("SliceThickness", ""))
        row["spacing_between_slices_mm"] = str(ds.get("SpacingBetweenSlices", ""))
        ps = ds.get("PixelSpacing", [None, None])
        row["pixel_spacing_x_mm"] = str(ps[0]) if ps[0] is not None else ""
        row["pixel_spacing_y_mm"] = str(ps[1]) if ps[1] is not None else ""
        row["rows"] = str(ds.get("Rows", ""))
        row["cols"] = str(ds.get("Columns", ""))
        row["sequence_name"] = str(ds.get("SequenceName", ""))
        row["scanning_sequence"] = str(ds.get("ScanningSequence", ""))
        row["read_status"] = "ok"
    except Exception as e:
        row["read_status"] = "error"
        row["read_error"] = repr(e)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    series_list = list(list_series(args.input_root))
    print(f"Scanning {len(series_list)} series across {args.workers} workers...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool, args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in tqdm(pool.imap_unordered(scan_one, series_list, chunksize=10),
                        total=len(series_list)):
            w.writerow(row)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
```

### 8.2 `scripts/build_workplan.py`

```python
"""Stage 2: classify, dedupe, filter, select canonicals, build workplan.csv."""
import argparse, re
from pathlib import Path
from collections import defaultdict
import polars as pl
from _common import POSITIVE_OVERLAP_IDS


def select_canonical(rows, freq):
    """Pick the canonical WATER row for one patient using the documented tiebreak.
    rows: list of dicts."""
    return sorted(rows, key=lambda r: (
        -freq.get(r["series_description"], 0),
        -int(r["n_dcm_files"] or 0),
        r["series_description"],
    ))[0]


def write_csv_or_empty(rows, path, schema_columns):
    """Polars raises on empty-list construction without a schema; handle that."""
    if rows:
        pl.DataFrame(rows).write_csv(path)
    else:
        pl.DataFrame(schema={c: pl.String for c in schema_columns}).write_csv(path)


def common_cols(r, cohort, batch, out_subdir):
    return {
        "cohort": cohort, "batch": batch, "patient_id": r["patient_id"],
        "source_series_path": r["series_path"],
        "source_series_description": r["series_description"],
        "image_type_full": r["image_type_full"],
        "image_type_token": r["image_type_token"],
        "n_dcm_files_in_source": r["n_dcm_files"],
        "output_subdir": out_subdir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-scan-index", type=Path, required=True)
    ap.add_argument("--existing-nifti", type=Path, required=True)
    ap.add_argument("--existing-masks", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    # infer_schema_length=10000 so polars doesn't mistype sparse columns
    df = pl.read_csv(args.pre_scan_index, infer_schema_length=10000)

    skipped, workplan, alignment = [], [], []

    # === Step 1: split into pos/neg cohorts ===
    pos = df.filter(pl.col("cohort") == "pos")
    neg = df.filter(pl.col("cohort") == "neg")

    # === Step 2: drop empty folders ===
    for r in df.filter(pl.col("read_status") == "empty_folder").iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "batch": r["batch"],
                        "source_path": r["series_path"],
                        "reason": "empty_folder", "detail": ""})
    pos = pos.filter(pl.col("read_status") == "ok")
    neg = neg.filter(pl.col("read_status") == "ok")

    # === Step 3: drop the 4 positive-overlap IDs from neg cohort ===
    overlap_list = list(POSITIVE_OVERLAP_IDS)
    for r in neg.filter(pl.col("patient_id").is_in(overlap_list)).iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "batch": r["batch"],
                        "source_path": r["series_path"],
                        "reason": "duplicate_with_positive",
                        "detail": "positive cohort wins"})
    neg = neg.filter(~pl.col("patient_id").is_in(overlap_list))

    # === Step 4: dedupe inter-neg-batch duplicates ===
    # For each patient, keep the batch with the most series subfolders.
    counts = (
        neg.group_by(["patient_id", "batch"])
           .agg(pl.len().alias("n_series"))
    )
    # Per-patient: pick the batch with max n_series (alphabetical batch name as tiebreak)
    keep = (
        counts.sort(["patient_id", "n_series", "batch"], descending=[False, True, False])
              .group_by("patient_id", maintain_order=True)
              .agg(pl.col("batch").first().alias("keep_batch"))
    )
    neg_with_keep = neg.join(keep, on="patient_id", how="left")
    drop_rows = neg_with_keep.filter(pl.col("batch") != pl.col("keep_batch"))
    for r in drop_rows.iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "batch": r["batch"],
                        "source_path": r["series_path"],
                        "reason": "duplicate_in_neg",
                        "detail": f"kept {r['keep_batch']}"})
    neg = (neg_with_keep.filter(pl.col("batch") == pl.col("keep_batch"))
                        .drop("keep_batch"))

    # === Step 5: drop series with non-Dixon ImageType ===
    is_dixon = pl.col("image_type_full").fill_null("").str.contains("DIXON")
    for cohort_df in (pos, neg):
        for r in cohort_df.filter(~is_dixon).iter_rows(named=True):
            skipped.append({"patient_id": r["patient_id"], "batch": r["batch"],
                            "source_path": r["series_path"],
                            "reason": "non_dixon",
                            "detail": r["image_type_full"]})
    pos = pos.filter(is_dixon)
    neg = neg.filter(is_dixon)

    # === Step 6: SeriesDescription frequency tally (WATER only, across both cohorts) ===
    water_all = (
        pl.concat([pos, neg], how="diagonal_relaxed")
          .filter(pl.col("image_type_token") == "WATER")
    )
    freq_df = water_all["series_description"].value_counts()
    # Polars value_counts returns columns: <original_name>, "count"
    sd_freq = dict(zip(freq_df["series_description"].to_list(),
                       freq_df["count"].to_list()))

    # === Step 7: build workplan rows ===
    def emit_patient(rows, cohort):
        """rows: list of dicts (one patient's series rows)."""
        pid = rows[0]["patient_id"]
        batch = rows[0]["batch"]
        out_subdir = (f"nifti_neg/{batch.replace('dicom_', '')}/{pid}"
                      if cohort == "neg" else f"nifti_pos/{pid}")
        token_groups = defaultdict(list)
        for r in rows:
            token_groups[r["image_type_token"]].append(r)
        water_rows = token_groups.get("WATER", [])
        if water_rows:
            canonical = select_canonical(water_rows, sd_freq)
            alts = [r for r in water_rows
                    if r["series_path"] != canonical["series_path"]]
            alts.sort(key=lambda r: r["series_description"])
            workplan.append({**common_cols(canonical, cohort, batch, out_subdir),
                             "role": "canonical",
                             "output_basename": "water_canonical"})
            for i, r in enumerate(alts, start=1):
                workplan.append({**common_cols(r, cohort, batch, out_subdir),
                                 "role": "alt",
                                 "output_basename": f"water_alt_{i:02d}"})
        for i, r in enumerate(sorted(token_groups.get("FAT", []),
                                      key=lambda x: x["series_description"]), start=1):
            workplan.append({**common_cols(r, cohort, batch, out_subdir),
                             "role": "fat",
                             "output_basename": f"fat_{i:02d}"})
        for token, prefix in (("IN_PHASE", "inphase"), ("OUT_PHASE", "outphase")):
            for i, r in enumerate(sorted(token_groups.get(token, []),
                                          key=lambda x: x["series_description"]), start=1):
                workplan.append({**common_cols(r, cohort, batch, out_subdir),
                                 "role": prefix,
                                 "output_basename": f"{prefix}_{i:02d}"})

    for cohort_df, cohort_name in ((pos, "pos"), (neg, "neg")):
        by_patient = defaultdict(list)
        for r in cohort_df.iter_rows(named=True):
            by_patient[r["patient_id"]].append(r)
        for pid in sorted(by_patient.keys()):
            emit_patient(by_patient[pid], cohort_name)

    # === Step 8: positives — identify which series the existing mask corresponds to ===
    pos_water_canonical = [w for w in workplan
                           if w["cohort"] == "pos" and w["role"] == "canonical"]
    nifti_files = list(args.existing_nifti.glob("*.nii.gz"))
    nifti_by_pid = defaultdict(list)
    for f in nifti_files:
        m = re.match(r"^(ANON[A-F0-9]+)(?:_(.*))?\.nii\.gz$", f.name)
        if m:
            nifti_by_pid[m.group(1)].append((f, m.group(2)))

    for w in pos_water_canonical:
        pid = w["patient_id"]
        candidates = nifti_by_pid.get(pid, [])
        mask_source, mappable, reason = "", False, ""
        if not candidates:
            reason = "no_existing_nifti_or_mask"
        elif len(candidates) == 1 and candidates[0][1] is None:
            mask_source = str(args.existing_masks / candidates[0][0].name)
            mappable = Path(mask_source).exists()
            reason = "" if mappable else "mask_file_not_found"
        else:
            wsd_norm = w["source_series_description"].replace(" ", "_")
            matched = [c for c in candidates if c[1] == wsd_norm]
            if matched:
                mask_source = str(args.existing_masks / matched[0][0].name)
                mappable = Path(mask_source).exists()
                reason = "" if mappable else "mask_file_not_found"
            else:
                # Fall back to slice-count match — deferred to realign_masks.py (has nibabel)
                reason = "ambiguous_will_match_by_slice_count_at_realign_time"
                mappable = True  # tentative
        alignment.append({
            "patient_id": pid,
            "canonical_series_path": w["source_series_path"],
            "mask_source_path": mask_source,
            "mask_mappable": mappable,
            "reason": reason,
        })

    # === Step 9: write outputs ===
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_or_empty(workplan, args.output_dir / "workplan.csv",
        ["cohort", "batch", "patient_id", "source_series_path",
         "source_series_description", "image_type_full", "image_type_token",
         "n_dcm_files_in_source", "output_subdir", "role", "output_basename"])
    write_csv_or_empty(skipped, args.output_dir / "skipped.csv",
        ["patient_id", "batch", "source_path", "reason", "detail"])
    write_csv_or_empty(alignment, args.output_dir / "alignment_audit.csv",
        ["patient_id", "canonical_series_path", "mask_source_path",
         "mask_mappable", "reason"])

    n_neg = sum(1 for w in workplan if w["cohort"] == "neg")
    n_pos = sum(1 for w in workplan if w["cohort"] == "pos")
    print(f"workplan: {len(workplan)} rows ({n_neg} neg, {n_pos} pos)")
    print(f"skipped:  {len(skipped)} rows")
    print(f"alignment: {len(alignment)} positives")


if __name__ == "__main__":
    main()
```

### 8.3 `scripts/convert_one_patient.py`

Called both by SLURM array tasks (per chunk) and by the pilot run (per `--patient-list`). Reads workplan rows for the assigned patients, copies each series folder to node-local `/tmp`, runs `dcm2niix`, writes outputs to `<output_root>/<output_subdir>/<output_basename>.nii.gz`, then `rsync`s to the shared output, appends to a per-task manifest CSV.

Key semantics:
- One `dcm2niix` invocation per (patient, series).
- Failures: capture stderr, append to `failed.csv` (per-task), exit 0 (don't fail the SLURM task).
- Idempotent: overwrites existing outputs.

```python
"""Convert a chunk of patients from workplan.csv. Called from SLURM or pilot."""
import argparse, csv, os, subprocess, shutil, sys, tempfile
from pathlib import Path
import polars as pl

MANIFEST_PART_FIELDS = [
    "cohort", "batch", "patient_id", "source_series_path", "role",
    "output_subdir", "output_basename", "exit_code",
    "produced_files", "stderr_excerpt",
]


def convert_one(row, output_root: Path, work: Path) -> dict:
    src = Path(row["source_series_path"])
    out_dir = output_root / row["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy to node-local /tmp first
    local_src = work / "in" / row["patient_id"] / src.name
    local_src.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, local_src / f.name)
    local_out = work / "out" / row["output_subdir"]
    local_out.mkdir(parents=True, exist_ok=True)
    # Run dcm2niix
    cmd = ["dcm2niix", "-z", "y", "-b", "y", "-ba", "n",
           "-f", row["output_basename"], "-o", str(local_out), str(local_src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    produced = sorted(local_out.glob(f"{row['output_basename']}*.nii.gz"))
    # rsync results to shared
    for f in produced:
        shutil.copy2(f, out_dir / f.name)
        json_sib = f.with_suffix("").with_suffix(".json")
        if json_sib.exists():
            shutil.copy2(json_sib, out_dir / json_sib.name)
    # cleanup local
    shutil.rmtree(local_src, ignore_errors=True)
    shutil.rmtree(local_out, ignore_errors=True)
    return {
        **{k: row[k] for k in ["cohort", "batch", "patient_id",
                               "source_series_path", "role", "output_subdir",
                               "output_basename"]},
        "exit_code": proc.returncode,
        "produced_files": ";".join(f.name for f in produced),
        "stderr_excerpt": proc.stderr[-500:] if proc.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--manifest-part", type=Path, required=True,
                    help="Where to write per-task manifest CSV")
    ap.add_argument("--patient-list", type=Path,
                    help="Optional file with one ANONID per line; only convert these")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--task-count", type=int, default=1,
                    help="Total array tasks; this task processes rows where "
                         "(row_idx %% task_count) == (task_id-1)")
    args = ap.parse_args()

    df = pl.read_csv(args.workplan, infer_schema_length=10000)
    if args.patient_list and args.patient_list.exists():
        keep = set(args.patient_list.read_text().split())
        df = df.filter(pl.col("patient_id").is_in(list(keep)))
    elif args.task_count > 1:
        # Round-robin partition across array tasks
        df = df.with_row_index("_idx").filter(
            (pl.col("_idx") % args.task_count) == (args.task_id - 1)
        ).drop("_idx")

    work = Path(tempfile.mkdtemp(prefix=f"dcm2niix_{os.getpid()}_", dir="/tmp"))
    args.manifest_part.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_part.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_PART_FIELDS)
        w.writeheader()
        for row in df.iter_rows(named=True):
            try:
                rec = convert_one(row, args.output_root, work)
            except Exception as e:
                rec = {**{k: row.get(k, "") for k in MANIFEST_PART_FIELDS},
                       "exit_code": -1, "produced_files": "",
                       "stderr_excerpt": repr(e)[:500]}
            w.writerow(rec)
            f.flush()
    shutil.rmtree(work, ignore_errors=True)
    print(f"Task {args.task_id}/{args.task_count} done; manifest: {args.manifest_part}")


if __name__ == "__main__":
    main()
```

### 8.4 `scripts/realign_masks.py`

```python
"""Realign existing radiologist masks to fresh dcm2niix RAS NIfTIs.

Lossless dual-axis flip: mask[::-1, :, ::-1] + adopt fresh affine.
Verified against 5 sample patients — see plan §2.6.
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import polars as pl


def realign_one(fresh_path: Path, mask_path: Path, out_path: Path) -> dict:
    fresh = nib.load(fresh_path)
    mask = nib.load(mask_path)
    notes = []
    if mask.shape != fresh.shape:
        return {"shape_match": False, "qc_status": "fail",
                "qc_notes": f"shape {mask.shape} vs fresh {fresh.shape}"}
    # Apply lossless dual-axis flip (axis-0 and axis-2)
    fixed = mask.get_fdata(dtype=np.float32)[::-1, :, ::-1]
    out = nib.Nifti1Image(fixed, affine=fresh.affine, header=fresh.header.copy())
    out.header.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, out_path)
    # Reload and QC
    reloaded = nib.load(out_path)
    affine_diff = float(np.abs(reloaded.affine - fresh.affine).max())
    axc_fresh = nib.aff2axcodes(fresh.affine)
    axc_mask = nib.aff2axcodes(reloaded.affine)
    vals = np.unique(reloaded.get_fdata())
    is_binary = set(vals.tolist()).issubset({0.0, 1.0})
    voxel_count = int(reloaded.get_fdata().sum())
    status = "ok"
    if axc_fresh != axc_mask: status = "fail"; notes.append("axcodes_mismatch")
    if affine_diff > 1e-3:    status = "fail"; notes.append(f"affine_diff={affine_diff:.6f}")
    if not is_binary:         status = "fail"; notes.append(f"non_binary:{vals[:5]}")
    if voxel_count == 0:      status = "fail"; notes.append("empty_mask")
    return {"shape_match": True,
            "axcodes_fresh": str(axc_fresh),
            "axcodes_mask_fixed": str(axc_mask),
            "affine_max_diff": affine_diff,
            "is_binary": is_binary,
            "mask_voxel_count": voxel_count,
            "qc_status": status,
            "qc_notes": ";".join(notes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--masks-root", type=Path, required=True)
    ap.add_argument("--alignment-audit", type=Path, required=True)
    ap.add_argument("--patient-list", type=Path)
    args = ap.parse_args()

    audit = pl.read_csv(args.alignment_audit, infer_schema_length=10000)
    plan = pl.read_csv(args.workplan, infer_schema_length=10000)

    pos_canonicals = plan.filter(
        (pl.col("cohort") == "pos") & (pl.col("role") == "canonical"))

    fresh_paths = {
        r["patient_id"]: args.output_root / r["output_subdir"] / f"{r['output_basename']}.nii.gz"
        for r in pos_canonicals.iter_rows(named=True)
    }
    out_subdir_by_pid = {r["patient_id"]: r["output_subdir"]
                         for r in pos_canonicals.iter_rows(named=True)}

    if args.patient_list and args.patient_list.exists():
        keep = set(args.patient_list.read_text().split())
        audit = audit.filter(pl.col("patient_id").is_in(list(keep)))

    qc_rows = []
    for row in audit.iter_rows(named=True):
        pid = row["patient_id"]
        fresh = fresh_paths.get(pid)
        out_subdir = out_subdir_by_pid.get(pid, "")
        out_path = (args.output_root /
                    out_subdir.replace("nifti_pos", "masks_pos") /
                    "mask_canonical.nii.gz")
        rec = {"patient_id": pid,
               "fresh_path": str(fresh) if fresh else "",
               "mask_source_path": row["mask_source_path"],
               "mask_out_path": str(out_path)}
        if not fresh or not Path(fresh).exists():
            rec.update(qc_status="skip", qc_notes="fresh_canonical_not_found")
        elif (not row["mask_source_path"]
              or not Path(row["mask_source_path"]).exists()):
            if row["reason"] == "ambiguous_will_match_by_slice_count_at_realign_time":
                fresh_z = nib.load(fresh).shape[2]
                matched = False
                for mf in args.masks_root.glob(f"{pid}*.nii.gz"):
                    if nib.load(mf).shape[2] == fresh_z:
                        rec.update(realign_one(Path(fresh), mf, out_path),
                                   mask_source_path=str(mf))
                        matched = True
                        break
                if not matched:
                    rec.update(qc_status="skip", qc_notes="no_mask_z_dim_match")
            else:
                rec.update(qc_status="skip", qc_notes="mask_source_missing")
        else:
            rec.update(realign_one(Path(fresh), Path(row["mask_source_path"]),
                                    out_path))
        qc_rows.append(rec)

    out_csv = args.output_root / "alignment_audit_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df = pl.DataFrame(qc_rows)
    result_df.write_csv(out_csv)
    vc = result_df["qc_status"].value_counts()
    summary = dict(zip(vc["qc_status"].to_list(), vc["count"].to_list()))
    print(f"Realignment summary: {summary}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
```

### 8.5 `scripts/qc.py`

```python
"""Stage 6: concatenate per-task manifests, harvest BIDS sidecars, sanity-check."""
import argparse, glob, hashlib, json
from pathlib import Path
import nibabel as nib
import polars as pl
from _common import MIN_SLICES_QC_FLAG


def sha256_of(path: Path, blocksize=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b: break
            h.update(b)
    return h.hexdigest()


def harvest(nifti_path: Path) -> dict:
    img = nib.load(nifti_path)
    json_path = nifti_path.with_suffix("").with_suffix(".json")
    side = json.loads(json_path.read_text()) if json_path.exists() else {}
    pix = side.get("PixelSpacing", [None, None])
    return {
        "n_slices_actual": int(img.shape[2]),
        "shape": "x".join(str(s) for s in img.shape),
        "scanner_model": side.get("ManufacturerModelName", ""),
        "magnetic_field_strength": side.get("MagneticFieldStrength", ""),
        "slice_thickness_mm": side.get("SliceThickness", ""),
        "spacing_between_slices_mm": side.get("SpacingBetweenSlices", ""),
        "pixel_spacing_x_mm": pix[0] if pix else "",
        "pixel_spacing_y_mm": pix[1] if pix else "",
        "image_type": "\\".join(side.get("ImageType", [])),
        "series_description": side.get("SeriesDescription", ""),
        "conversion_software_version": side.get("ConversionSoftwareVersion", ""),
        "sha256": sha256_of(nifti_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--skipped", type=Path, required=True)
    ap.add_argument("--alignment-audit", type=Path)
    args = ap.parse_args()

    parts = sorted(glob.glob(str(args.output_root / "manifest_part_*.csv")))
    if not parts:
        raise SystemExit("No manifest_part_*.csv found in output root")
    df = pl.concat(
        [pl.read_csv(p, infer_schema_length=10000) for p in parts],
        how="diagonal_relaxed",
    )

    # Expand "produced_files" to one row per actual .nii.gz output
    rows = []
    for r in df.iter_rows(named=True):
        files = [f for f in str(r["produced_files"] or "").split(";") if f]
        base = {k: r[k] for k in r}
        if not files:
            rows.append({**base, "output_filename": "",
                         "volume_index": 0, "n_volumes_from_series": 0,
                         "conversion_status": ("failed" if r["exit_code"] != 0
                                               else "no_output")})
            continue
        for i, fname in enumerate(files):
            row = {**base, "output_filename": fname,
                   "volume_index": i,
                   "n_volumes_from_series": len(files),
                   "conversion_status": ("ok" if r["exit_code"] == 0 else "failed")}
            nifti_path = args.output_root / r["output_subdir"] / fname
            if nifti_path.exists():
                row.update(harvest(nifti_path))
                if row.get("n_slices_actual", 0) < MIN_SLICES_QC_FLAG:
                    row["conversion_status"] = "qc_flag:low_slice_count"
            rows.append(row)

    manifest = pl.DataFrame(rows)
    manifest.write_csv(args.output_root / "manifest.csv")
    qc_flags = manifest.filter(
        pl.col("conversion_status").str.starts_with("qc_flag")
        | (pl.col("conversion_status") != "ok"))
    qc_flags.write_csv(args.output_root / "qc_flags.csv")

    n_ok = manifest.filter(pl.col("conversion_status") == "ok").height
    n_failed = manifest.filter(pl.col("conversion_status") == "failed").height
    n_qc = manifest.filter(
        pl.col("conversion_status").str.starts_with("qc_flag")).height
    n_skipped = pl.read_csv(args.skipped)["patient_id"].n_unique()

    summary = (
        f"Manifest rows:    {manifest.height}\n"
        f"OK:               {n_ok}\n"
        f"Failed:           {n_failed}\n"
        f"QC-flagged:       {n_qc}\n"
        f"Skipped patients: {n_skipped}\n"
    )
    (args.output_root / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
```

### 8.6 `scripts/monitor.py`

See §10.

---

## 9. SLURM scripts

### 9.1 `slurm/convert_neg.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dcm2niix_neg
#SBATCH --array=1-100%30
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --output=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/neg_%A_%a.out
#SBATCH --error=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/neg_%A_%a.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0

PROJECT=/home/sak185/dia-endo-conversion
SCRATCH=/scratch/pioneer/users/sak185/dia-endo-conversion

cd "$PROJECT"

# Use uv-managed venv
source .venv/bin/activate

WORKPLAN=$SCRATCH/workplan/workplan.csv
OUT_ROOT=$SCRATCH/output
MANIFEST_PART=$OUT_ROOT/manifest_part_neg_${SLURM_ARRAY_TASK_ID}.csv

# Filter workplan to negatives only, then this task takes its share
# (round-robin partition over 100 tasks)
TMP_PLAN=$(mktemp /tmp/workplan_neg_${SLURM_ARRAY_TASK_ID}.XXXX.csv)
head -1 "$WORKPLAN" > "$TMP_PLAN"
awk -F, -v task=$SLURM_ARRAY_TASK_ID -v ntasks=100 '
  NR > 1 && $1 == "neg" {
    if ((neg_idx % ntasks) == (task - 1)) print
    neg_idx++
  }' "$WORKPLAN" >> "$TMP_PLAN"

python scripts/convert_one_patient.py \
    --workplan "$TMP_PLAN" \
    --output-root "$OUT_ROOT" \
    --manifest-part "$MANIFEST_PART" \
    --task-id $SLURM_ARRAY_TASK_ID \
    --task-count 100

rm -f "$TMP_PLAN"
echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
```

### 9.2 `slurm/convert_pos.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=dcm2niix_pos
#SBATCH --array=1-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --output=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/pos_%A_%a.out
#SBATCH --error=/scratch/pioneer/users/sak185/dia-endo-conversion/logs/pos_%A_%a.err

set -euo pipefail
module load Python/3.11.3-GCCcore-12.3.0

PROJECT=/home/sak185/dia-endo-conversion
SCRATCH=/scratch/pioneer/users/sak185/dia-endo-conversion
cd "$PROJECT"
source .venv/bin/activate

WORKPLAN=$SCRATCH/workplan/workplan.csv
OUT_ROOT=$SCRATCH/output
MANIFEST_PART=$OUT_ROOT/manifest_part_pos_${SLURM_ARRAY_TASK_ID}.csv

TMP_PLAN=$(mktemp /tmp/workplan_pos_${SLURM_ARRAY_TASK_ID}.XXXX.csv)
head -1 "$WORKPLAN" > "$TMP_PLAN"
awk -F, -v task=$SLURM_ARRAY_TASK_ID -v ntasks=4 '
  NR > 1 && $1 == "pos" {
    if ((pos_idx % ntasks) == (task - 1)) print
    pos_idx++
  }' "$WORKPLAN" >> "$TMP_PLAN"

python scripts/convert_one_patient.py \
    --workplan "$TMP_PLAN" \
    --output-root "$OUT_ROOT" \
    --manifest-part "$MANIFEST_PART" \
    --task-id $SLURM_ARRAY_TASK_ID \
    --task-count 4

rm -f "$TMP_PLAN"
```

---

## 10. Progress monitoring

`scripts/monitor.py` polls SLURM via `squeue` and counts completed manifest rows to give a live ETA.

```python
"""Live progress monitor for the SLURM array. Run from the head node."""
import argparse, glob, subprocess, time
from pathlib import Path
import polars as pl


def squeue_state(job_id: str) -> dict:
    """Returns counts of array tasks in each state."""
    out = subprocess.run(
        ["squeue", "-j", job_id, "--noheader", "-r", "-o", "%T"],
        capture_output=True, text=True)
    states = out.stdout.split()
    return {s: states.count(s) for s in set(states)}


def manifest_progress(output_root: Path, expected_total: int) -> tuple[int, int]:
    parts = glob.glob(str(output_root / "manifest_part_*.csv"))
    if not parts: return 0, 0
    done = sum(sum(1 for _ in open(p)) - 1 for p in parts)  # subtract header
    failed = 0
    for p in parts:
        try:
            df = pl.read_csv(p, infer_schema_length=10000)
            if "exit_code" in df.columns:
                failed += df.filter(pl.col("exit_code") != 0).height
        except Exception:
            pass
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True, help="SLURM job ID (the %A part)")
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--cohort", choices=["neg", "pos", "all"], default="neg")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    plan = pl.read_csv(args.workplan, infer_schema_length=10000)
    if args.cohort != "all":
        plan = plan.filter(pl.col("cohort") == args.cohort)
    expected = plan.height

    history = []  # list of (timestamp, done_count)
    print(f"Monitoring SLURM job {args.job_id}; expected {expected} conversions")
    print(f"{'time':<19} {'done':>7} {'failed':>7} {'rate/min':>9} {'ETA':>10} {'queue states':<40}")
    while True:
        states = squeue_state(args.job_id)
        done, failed = manifest_progress(args.output_root, expected)
        now = time.time()
        history.append((now, done))
        history = [(t, d) for t, d in history if now - t <= 600]  # last 10 min
        if len(history) >= 2:
            dt = history[-1][0] - history[0][0]
            dd = history[-1][1] - history[0][1]
            rate = (dd / dt * 60) if dt > 0 else 0
            eta_sec = ((expected - done) / (rate / 60)) if rate > 0 else float("inf")
            eta = (f"{eta_sec/3600:5.1f}h" if eta_sec < float("inf") and eta_sec > 3600
                   else f"{eta_sec/60:6.1f}m" if eta_sec < float("inf")
                   else "    --")
        else:
            rate, eta = 0, "  --"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sstr = " ".join(f"{k}={v}" for k, v in sorted(states.items()))
        print(f"{ts}  {done:>6}/{expected:<6}  {failed:>5}  {rate:>7.1f}  {eta:>9}  {sstr}")
        if not states:
            print("Job no longer in queue. Final progress reported above.")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
```

Usage during Stage 4:

```bash
# In a separate terminal:
JOBID=$(squeue -u $USER -h -o "%A" -n dcm2niix_neg | head -1)
uv run python scripts/monitor.py \
    --job-id $JOBID \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --cohort neg
```

Sample output:

```
time                done    failed  rate/min        ETA  queue states
2026-04-25 14:30:01    0/4882      0      0.0       --   PENDING=70 RUNNING=30
2026-04-25 14:30:31  152/4882      0    304.0    15.5m   PENDING=68 RUNNING=30 COMPLETED=2
2026-04-25 14:31:01  311/4882      0    317.4    14.4m   PENDING=66 RUNNING=30 COMPLETED=4
...
```

Quick one-shot status (no live loop):

```bash
squeue -u $USER -t RUNNING,PENDING -o "%i %j %T %M %L"
ls /scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest_part_*.csv | wc -l
```

---

## 11. Manifest schemas

### `manifest.csv` (final)

| Column | Source | Notes |
|---|---|---|
| `cohort` | workplan | `neg` or `pos` |
| `patient_id` | workplan | `ANONID` |
| `split` | splits.json | `holdout` \| `fold0`..`fold4` \| `phase2_unsupervised` — assigned by `build_splits.py`, joined into manifest at QC time |
| `source_series_path` | workplan | absolute path to source series folder |
| `source_series_description` | workplan | from SeriesDescription |
| `image_type_full` | sidecar | `\`-joined ImageType string |
| `image_type_token` | workplan | 4th token (`WATER`, `FAT`, …) |
| `role` | workplan | `canonical` \| `alt` \| `fat` \| `inphase` \| `outphase` |
| `output_subdir` | workplan | `nifti_neg/<ANONID>` or `nifti_pos/<ANONID>` |
| `output_basename` | workplan | e.g., `water_canonical` |
| `output_filename` | qc | e.g., `water_canonical.nii.gz` (handles dcm2niix `_e1` splits) |
| `n_volumes_from_series` | qc | usually 1; >1 if dcm2niix split on geometry |
| `volume_index` | qc | 0 for primary; 1+ for split sub-volumes |
| `n_dcm_files_in_source` | workplan | DICOM file count in source series |
| `n_slices_actual` | qc (nibabel) | z-dim of converted volume |
| `shape` | qc | e.g., `512x120x512` |
| `scanner_model` | sidecar | `SIGNA Artist` \| `SIGNA Explorer` \| other |
| `magnetic_field_strength` | sidecar | always 1.5 in this dataset |
| `slice_thickness_mm` | sidecar | per-volume |
| `spacing_between_slices_mm` | sidecar | through-plane sampling |
| `pixel_spacing_x_mm` | sidecar | in-plane (PixelSpacing[0]) |
| `pixel_spacing_y_mm` | sidecar | in-plane (PixelSpacing[1]) |
| `conversion_software_version` | sidecar | dcm2niix version stamp |
| `conversion_status` | qc | `ok` \| `failed` \| `qc_flag:<flag>` |
| `stderr_excerpt` | per-task | last 500 chars of dcm2niix stderr if failed |
| `sha256` | qc | SHA-256 of the .nii.gz |

### `splits.json` (v2.3, frozen artifact)

Top-level keys: `seed` (int), `n_folds` (5), `stratification_keys` (list), `assignments` (dict `patient_id → split_label`), `summary` (per-stratum counts).

### `skipped.csv`

`patient_id, cohort, source_path, reason, detail`

`reason` ∈ {`empty_folder`, `non_dixon`, `excluded_no_visible_lesion_on_canonical`}.
(Cross-batch dedup and pos/neg overlap are now resolved at consolidation, not workplan build — see `consolidation.csv`.)

### `failed.csv`

(Generated post-hoc by qc.py from rows in `manifest.csv` with `conversion_status=failed`):
`patient_id, cohort, source_series_path, exit_code, stderr_excerpt, slurm_job_id`

### `alignment_audit_results.csv` (positives only — v2.2 schema)

`patient_id, mask_filename, filename_suffix, mask_source_path, target_volume, mask_out_path, shape_match, axcodes_target, axcodes_mask_fixed, affine_max_diff, label_values, n_unique_labels, fg_voxel_count, qc_status, qc_notes`

One row per existing mask file (not per patient). `target_volume` is which freshly-converted volume the mask was shape-matched to; `mask_out_path` is the realigned output.

---

## 12. Failure handling

| Condition | Action |
|---|---|
| Empty source folder | `skipped.csv` (reason `empty_folder`) |
| 4-overlap positive ID in neg cohort | `skipped.csv` (reason `duplicate_with_positive`) |
| Inter-neg duplicate | `skipped.csv` (reason `duplicate_in_neg`, kept = batch with most series) |
| Non-Dixon ImageType | `skipped.csv` (reason `non_dixon`) |
| Patient with no WATER series | manifest row with `role=fat` (or whatever exists), no canonical for them |
| dcm2niix non-zero exit | row in manifest with `conversion_status=failed` and stderr captured. **Continue** to next patient. No retry, no abort. |
| dcm2niix exit 0 but 0 outputs | `conversion_status=no_output` |
| Output z-dim < 30 | `qc_flag:low_slice_count`; row kept |
| Mask realignment QC fail (axcodes mismatch, non-binary, empty) | `alignment_audit_results.csv` `qc_status=fail` with notes; mask file still written but flagged |
| Mask source missing for ambiguous positive | `qc_status=skip`; positive volume still in dataset, just no mask |
| Re-running on partially-completed output | dcm2niix overwrites; manifest is regenerated from scratch in Stage 6 |

---

## 13. Runtime & storage

### Wall time

| Stage | Wall time | Resource |
|---|---|---|
| 0 — extract `data.tgz` | ~30 min | head node, single-threaded `tar -xz` |
| 1 — pre-scan | ~10 min | head node, 32 cores |
| 2 — build workplan | <1 min | head node, single Python process |
| 3 — pilot (12 patients) | ~30 min | head node, sequential |
| 4 — neg SLURM array (~4,882 patients, 100 tasks × 50, throttled to 30 concurrent) | ~3–4 hours | ~50 CPU-hours total |
| 5a — pos SLURM array (171 patients, 4 tasks concurrent) | ~30 min | ~3 CPU-hours |
| 5b — mask realignment | ~5 min | head node, sequential |
| 6 — QC + manifest | ~10 min | head node |
| 7 — `rsync` /scratch → /home | ~15 min | one rsync |
| **Total** | **~5–6 hours end-to-end** | |

### Storage

| Location | Peak usage |
|---|---|
| `/scratch/pioneer/users/sak185/dia-endo-conversion/input/{positive,negative,nifti,masks}` | ~135 GB (consolidated DICOMs + nifti + masks) |
| `/scratch/pioneer/users/sak185/dia-endo-conversion/output/` | ~50 GB (NIfTI + sidecars + manifests + logs) |
| `/home/sak185/dia-endo-conversion/data/` (final) | ~50 GB |
| `/home/sak185/dia-endo-conversion/.venv/` | ~150 MB |
| `/tmp` per node during a SLURM task | ~1–2 GB transient |

`/scratch` peak: ~185 GB. Comfortably under 23 TB free.

---

## 14. Pre-flight checklist

Before launching Stage 4 SLURM array:

- [ ] `uv` is installed at `~/.local/bin/uv`
- [ ] Project `/home/sak185/dia-endo-conversion/` exists with `pyproject.toml`, `uv.lock`, `.venv/`
- [ ] `uv run python -c "import pydicom, nibabel, polars, tqdm"` exits 0
- [ ] `dcm2niix --version` reports v1.0.20250505
- [ ] `scripts/consolidate.py` completed (see `consolidation.csv` for the audit trail)
- [ ] `ls /scratch/.../input/positive | wc -l` returns 170
- [ ] `ls /scratch/.../input/negative | wc -l` returns ~4,950
- [ ] `ls /scratch/.../input/nifti    | wc -l` returns 131
- [ ] `ls /scratch/.../input/masks    | wc -l` returns 262 (.nii.gz + .csv)
- [ ] `pre_scan_index.csv` exists with ~24,000 rows
- [ ] `workplan.csv` exists; row counts: ~5,500–6,000 series total (canonical + alts + fat)
- [ ] `skipped.csv` exists; row count ≥ 85 (empties); cross-batch dedup is in `consolidation.csv`, not `skipped.csv`
- [ ] `alignment_audit.csv` exists with 170 positive rows
- [ ] **Pilot run completed** (Stage 3) and visually QC'd
- [ ] `/scratch` free space > 200 GB
- [ ] `sbatch --test-only slurm/convert_neg.slurm` passes
- [ ] User has reviewed `workplan.csv` and `skipped.csv` and is satisfied

---

## 15. Human review gates

Five points where a human (the user) must intervene:

1. **After Stage 3 (pilot)** ✅ *passed 2026-04-26*: visually inspect ~3 random `water_canonical.nii.gz` files in a NIfTI viewer; verify orientation looks coronal, intensity reasonable. Inspect 1 positive's `mask_canonical.nii.gz` overlaid on its `water_canonical.nii.gz` — mask voxels should be over the visible lesion.
2. **After Stage 3a (splits)**: review `splits_summary.csv` — confirm per-stratum counts are sensible; confirm seed in `splits.json` is the one you want locked. The splits are FROZEN once Phase 1 conversion runs.
3. **Before Stage 4 (Phase 1 SLURM submission)**: review `subset_phase1.txt` row counts (~600). User runs `sbatch slurm/convert_phase1.slurm`.
4. **After Stage 4b (Phase 1 realign)**: review `alignment_audit_results.csv` for the Phase 1 patient list — `qc_status=ok` for all mask-bearing volumes (multi-label `qc_notes` are warnings, not failures). Now you can begin baseline training.
5. **Before Stage 5 (Phase 2 SLURM submission)**: optional — only run Phase 2 when you actually need the bulk negatives (for SSL pretraining or scaling the supervised set). User runs `sbatch slurm/convert_phase2.slurm`.

Implementing agents must NOT submit `sbatch` jobs autonomously. They prepare everything and stop.

---

## 16. Open questions

1. **HDF5 dataset format** — out of scope here; build a separate `nifti_to_hdf5.py` after Stage 7 completes.
2. **dcm2niix multi-volume behavior on outlier patients** — for series like `ANON148BD54809B2` (564 slices), dcm2niix may emit `water_canonical_e1.nii.gz`, `water_canonical_e2.nii.gz`. Manually inspect one such case in the pilot. If splitting is undesirable, add `-m y` to the dcm2niix command (forces merge).
3. **Outlier slice thickness** (2.2 mm and 7.13 mm cases) — flagged in manifest, not excluded. Decide at training time whether to drop or augment.
4. **Re-running on new data** — pipeline is idempotent: re-run from Stage 1 with refreshed `pre_scan_index.csv`. Note: SeriesDescription frequency tally may shift, so the `_canonical` selection for an existing patient could change in edge cases. Document the run timestamp in `data/README.md`.
5. **Mask alignment regression detection** — the QC test in `realign_masks.py` catches affine/axcode/binary/empty issues. If new dcm2niix versions ever change orientation conventions again, the QC will catch it.

---

## 17. Appendix — ML pipeline context

Captured here for design rationale; not part of the conversion implementation.

- **Recommended training stack:** MONAI + PyTorch Lightning. Most common pairing in published medical imaging work.
- **2.5D approach:** No native 2.5D in MONAI — set `in_channels = N_slices` on a 2D U-Net and stack adjacent slices manually as a transform.
- **Detection approach:** Segmentation + connected-components → bounding boxes is the established pattern in endometriosis MRI papers. Pure detection (RetinaNet 3D, nnDetection) needs more labeled data than 131 cases. Skip nnDetection (3D-only, PyTorch 1.x).
- **SSL pretraining:** Strongly recommended. CVPR 2025 evidence: MAE-pretrained ResEnc U-Net beat fully-tuned nnU-Net by ~3 Dice on 11 datasets. Pretrain on the ~5,000 negative WATER volumes with masked autoencoder (60–90% mask ratio), then fine-tune encoder + decoder on the 131 positives.
- **Why we kept all WATER series, not just one per patient:** more pretraining volumes. The `_canonical` flag lets you deduplicate at training time if the alternates turn out to be near-duplicates of the same anatomy.
- **Why the manifest captures scanner model + slice thickness + pixel spacing:** Artist (~6 mm) vs Explorer (~3.5 mm) is a real distribution shift. Stratify train/val/test by scanner model. Resample to common voxel spacing (e.g., 1.5×1.5×3 mm) in the training preprocessing.
- **No prior published work on diaphragmatic endometriosis MRI deep learning** — this is a blank space. The WATER-only Dixon coronal protocol is unique to this dataset.
- **Dataset format for training:** HDF5 conversion as a separate downstream step. MONAI `PersistentDataset` is a low-friction alternative if you want to stay in-stack.
