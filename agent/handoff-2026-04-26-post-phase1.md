# Session Handoff — 2026-04-26 (post Phase-1, Phase-2 in-flight, pre-preprocessing)

This is a snapshot at the milestone **after Phase 1 conversion + mask realignment + Phase 1 manifest are complete, while Phase 2 conversion is running on the cluster, and immediately before the TotalSegmentator-based preprocessing pipeline is implemented.**

The earlier snapshot (`handoff-2026-04-26.md`, ~03:39 today) covers everything up through pilot QC. Read it for project background and decision history through 2026-04-25; this document covers everything after.

---

## 1. Project at a glance

**Goal:** Convert ~5,120 GE 1.5T LAVA Dixon coronal MRI patient folders (diaphragmatic endometriosis dataset) from DICOM → NIfTI on the CWRU pioneer cluster, then preprocess into a training-ready HDF5 format.

**Authoritative plan:** [`convert-plan-v2.md`](./convert-plan-v2.md). The preprocessing pipeline (TotalSegmentator crop → normalize → binarize → thickness harmonize → HDF5) is **not yet in the plan** — it's the next thing to design.

**Owner:** sak185 on CWRU pioneer SLURM HPC.

**Project root:** `/home/sak185/dia-endo-conversion/`

**Working scratch:** `/scratch/pioneer/users/sak185/dia-endo-conversion/`

---

## 2. Where we are in the plan

| Stage | Status | Notes |
|---|---|---|
| 0 — project setup | ✅ done | |
| 0a — consolidate `/home/jjs374/DiaE` → /scratch | ✅ done | |
| 1 — prescan | ✅ done | |
| 2 — build_workplan | ✅ done (v2.3 schema with `soft_negative` column) | |
| 3 — pilot conversion + QC | ✅ done | |
| 3a — build splits | ✅ done | seed=42, 5-fold CV, 22+100 holdout, Phase 1 = 608 patients |
| 4a — Phase 1 SLURM array | ✅ **done** | job 3269215 ran 5:02–5:35; 755/755 series, 0 failed |
| 4b — Phase 1 mask realign | ✅ **done** | 126/126 positives realigned (qc_status=ok) |
| 4c — Phase 1 QC / manifest.csv | ✅ **done** | 1,079 rows in `manifest.csv`, all OK |
| **5 — Phase 2 SLURM array** | 🟡 **IN FLIGHT** | job 3269236 submitted ~07:00; 45 tasks × ~125 series, %40 concurrent; ~1,067 conversions done in first 21 min; ETA total ~50–60 min wall |
| 6 — Phase 2 QC re-run | pending | re-run `qc.py` after Phase 2 finishes |
| 7 — copy /scratch → /home/.../data/ | pending | |
| **8 — preprocessing pipeline (NEW, not in plan v2 yet)** | ⏳ **NEXT** | TotalSegmentator liver crop + p1–p99 normalize + binarize labels + slice-thickness harmonize → HDF5 |

---

## 3. What's on disk right now

### Source data (consolidated, RO)

```
/scratch/pioneer/users/sak185/dia-endo-conversion/input/
├── positive/<ANONID>/<series>/<*.dcm>          170 patient dirs (re-rsynced 04:55 today after a still-unexplained 03:35 wipe)
├── negative/<ANONID>/<series>/<*.dcm>        4,950 patient dirs
├── nifti/<ANONID>[_<series>].nii.gz            131 prior-conversion nifti files
├── masks/<ANONID>[_<series>].{nii.gz,csv}      262 mask + label files
└── consolidation.csv                         5,122 audit rows
```

### Workplan artifacts

```
/scratch/pioneer/users/sak185/dia-endo-conversion/workplan/
├── pre_scan_index.csv               6,464 rows
├── workplan.csv                     6,409 rows  ← now has `soft_negative` column
├── skipped.csv                      55 rows
├── alignment_audit.csv              183 rows / 165 unique pos / 126 with mask file
├── splits.json                      ← seed=42, 608 phase1 + 4,476 phase2 + 57 soft_negative_pids
├── splits_summary.csv               per-stratum counts
├── subset_phase1.txt                608 ANONIDs (Phase 1 patient list)
├── subset_phase2.txt              4,476 ANONIDs (Phase 2 patient list)
└── pilot_patients.txt
```

### Phase 1 output (DONE)

```
/scratch/pioneer/users/sak185/dia-endo-conversion/output/
├── nifti_neg/<ANONID>/water_canonical*.{nii.gz,json}    509 patients, 810 .nii.gz
├── nifti_pos/<ANONID>/water_canonical*.{nii.gz,json}    114 patients, 207 .nii.gz
├── masks_pos/<ANONID>/mask_<basename>.nii.gz            126 patients (Phase 1 cohort), all qc_status=ok
├── manifest_part_phase1_{1..10}.csv                     755 conversions
├── manifest_part_phase2_{1..10+}.csv                    🟡 GROWING — Phase 2 in flight
├── manifest_part_pilot.csv                              23 pilot rows
├── manifest_part_remask23.csv                           47 remask-investigation rows
├── alignment_audit_results.csv                          126 rows, all OK
├── manifest.csv                                         1,079 rows  ← Phase 1 snapshot, will refresh after Phase 2
├── qc_flags.csv                                         empty (no flags)
└── summary.txt                                          OK: 1,079, Failed: 0
```

`manifest.csv` has a `split` column (joined from `splits.json`) and a `soft_negative` boolean. The 5 patients with `split=null` and 10 with `split=phase2_unsupervised` are pre-splits pilot/remask leftovers — downstream training should filter to `split.is_in(['holdout', 'fold0', ..., 'fold4'])`.

### Phase 2 progress (live)

```
$ squeue -u $USER | grep phase2
# ~40 tasks RUNNING, others PENDING (AssocGrpCpuLimit throttling — expected)
# manifest_part_phase2_*.csv files growing live in output/
```

---

## 4. Code state — what's new since the morning handoff

| File | Change |
|---|---|
| `scripts/_common.py` | unchanged |
| `scripts/build_workplan.py` | **modified**: emits `soft_negative` column; reclassifies the 57 mask-less ex-positives from `pos` → `neg` cohort (output_subdir `nifti_neg/`) so they join the negative pool but stay trackable |
| `scripts/build_splits.py` | **NEW**: stratified holdout + 5-fold CV, seed=42; outputs `splits.json`, `subset_phase{1,2}.txt`, `splits_summary.csv` |
| `scripts/convert_one_patient.py` | **modified twice today**: (a) `--patient-list` and `--task-count` now compose (filter first, then partition); (b) **resume logic** — reads existing `manifest_part_*.csv` on startup, skips already-completed `(output_subdir, output_basename)` pairs, appends rather than truncates. Failed rows (`exit_code != 0`) are NOT skipped — they retry on the next submit |
| `scripts/preflight_check.py` | **NEW**: validates source data presence + DICOM counts + splits.json coverage before SLURM submission. Exits non-zero on missing/empty source dirs (fatal); warns on dcm-count mismatches (non-fatal). Wired into both `convert_phase1.slurm` and `convert_phase2.slurm` as the first step |
| `scripts/realign_masks.py` | unchanged from v2.2 (already shape-matches against all fresh WATER outputs; multi-label values preserved with `label_values` in qc_notes) |
| `scripts/qc.py` | **modified**: accepts `--splits-json` and joins `split` + `soft_negative` columns into `manifest.csv` |
| `slurm/convert_phase1.slurm` | **NEW**: `--array=1-10%20`, time **10h** (was 1h originally — too tight; user bumped after Phase 1 ran tight at 12–17 min/task). 10 array tasks completed cleanly in one shot |
| `slurm/convert_phase2.slurm` | **NEW**: `--array=1-45%40` (45 tasks × ~125 series, max 40 concurrent — under the 48 MaxSubmit cap), time **24h** |

`MEMORY.md` index and per-topic memory files under `~/.claude/projects/-home-sak185-dia-endo-conversion/memory/` were not modified today. They were correct and current as of the morning handoff.

---

## 5. Decision log (this session, post-morning)

- **Soft-negative reclassification.** The 57 ex-positives whose lesions don't appear on the canonical sequence (no canonical mask file) were moved out of the `pos` cohort into `neg` (with `soft_negative=True` in workplan + manifest). They go through the same conversion pipeline as true negatives, share output_subdir conventions (`nifti_neg/<ANONID>/`), and are eligible for inclusion in the holdout/CV pools — but downstream training can filter on `soft_negative` if needed. User rationale: *"these aren't even positives so they should be moved out of that designation."* Patient list in `splits.json["soft_negative_pids"]` (57 IDs).
- **5-fold CV split design (seed=42).** 22 positives + 100 negatives held out; 86 positives + 400 negatives in CV pool, distributed round-robin across 5 folds. Stratification on (manufacturer_model_name × thickness_bin) for negatives; (manufacturer_model_name only — thickness collapsed) for the small positive pool. Tiny-stratum allocation is "best-effort" largest-remainder. Two independent rng streams (`rng_pos = default_rng(42)`, `rng_neg = default_rng(43)`). 29 FAT-only patients (no canonical) directly assigned `phase2_unsupervised`.
- **Phase 1 ran twice.** First submission (job 3269196, 04:35) was cancelled by user when it began returning `FileNotFoundError` on every positive — investigation showed `/scratch/.../input/positive/` had been wiped at 03:35 (cause still unknown, possibly cluster maintenance). Re-rsynced positives from `/home/jjs374/DiaE/dicom`, added `preflight_check.py` as a defense, resubmitted as job 3269215 at 05:02 — completed cleanly 5:35.
- **Resume logic added.** Originally `convert_one_patient.py` opened `manifest_part_*.csv` in `"w"` mode, truncating any prior progress on restart. After the user pointed out a 1h timeout would have lost state, added prior-manifest indexing + append mode + skip-already-done. Verified by dry-run on the existing phase1_1 manifest: 76/76 skipped, 0 new, manifest unchanged.
- **Phase 2 concurrency.** Bumped `convert_phase2.slurm` from `%30` to `%40` simultaneous after `sinfo` showed ~93 idle batch nodes and the workload is per-task serial I/O. Stayed under the 48 MaxSubmit cap to leave headroom for any other jobs.
- **Phase 1 manifest is a complete usable snapshot.** Even though Phase 2 is in flight, Phase 1's `manifest.csv` (1,079 rows, 0 failed, 0 QC-flagged) is enough to start downstream model design / preprocessing work. Re-running `qc.py` after Phase 2 finishes will fold the new ~5,654 rows in.

---

## 6. What's running right now (state at handoff write time)

```
$ squeue -u $USER
JOBID         NAME              STAT   TIME   NODES
3269039       interactive       RUN    7+ h   compt337   ← user's CPU shell
3269236_[1]   dcm2niix_phase2   PEND   0:00              ← throttled (AssocGrpCpuLimit)
3269236_2..   dcm2niix_phase2   RUN    21+ min × ~40 nodes
```

Phase 2 manifest_parts at handoff:
```
manifest_part_phase2_1.csv:  126 rows (running)
manifest_part_phase2_2.csv:  118
... (10 manifest parts already, ~1,067 conversions logged)
```

**Don't kill this job.** Resume logic means a restart would be safe, but 21 min in there's nothing to gain.

---

## 7. Immediate next step — preprocessing pipeline

User just decided (this session) on the post-conversion preprocessing pipeline:

1. **TotalSegmentator liver crop** with 3D margin (per-volume bbox of liver segmentation, expand by N voxels in each direction, crop to that bbox).
2. **p1–p99 normalize** intensity over the cropped volume.
3. **Binarize labels** (mask: anything non-zero → 1; the multi-label `{0,1,2}` patients flagged in `alignment_audit_results.csv` collapse to `{0,1}`).
4. **Slice-thickness harmonize** — resample to a target z-spacing (TBD; Artist ~6mm, Explorer ~3.5mm, want to converge).
5. **Package to HDF5** — one HDF5 per split (`train.h5`, `val.h5`, `holdout.h5`) with one group per patient.

**Recommendation given to user:** make the NIfTI → HDF5 jump *after step 1 (the crop)*, since TotalSegmentator is the only expensive irreversible step worth caching. Steps 2–4 stay as on-the-fly DataLoader transforms (cheap, hyperparameter-dependent — especially target slice thickness). User accepted; specific HDF5 schema not yet drafted.

**Right now the user is trying to get an interactive GPU node** to develop the TotalSegmentator wrapper script.

### TotalSegmentator allocation recommendation (this session)

```bash
salloc --partition=gpu --constraint=gpul40s --gres=gpu:1 \
       --cpus-per-task=8 --mem=32G --time=8:00:00
```

Best idle-capacity targets at the time of writing: **gput067** (L40S, 40 idle CPUs), **gput064** (L40S, 24 idle CPUs), **gput073** (H100, 44 idle CPUs — fall-back for max speed). RTX 4090 (`gput075`, 44 idle CPUs) also works.

**TotalSegmentator inference specs** (from agent research, citing wasserth/TotalSegmentator README):
- VRAM: ~7 GB for `total_mr` task at 1.5 mm; ~3 GB with `--fast` (3 mm).
- RAM: 16–32 GB system, 4–8 CPUs.
- PyTorch ≥2.0; nnUNetv2 ≥2.3.1; Python ≥3.10. **First-run weight download** to `~/.totalsegmentator/` (set `TOTALSEG_HOME_DIR` to override). Pre-fetch via `totalseg_download_weights -t total_mr`.
- For liver crop: `python_api.totalsegmentator(input, output, task="total_mr", roi_subset=["liver"])`. Liver class index in label space = 5. Derive bbox from `np.where(mask==5)`, expand by margin, crop source.
- Per-volume runtime on RTX 3090: ~10–60 s for `total_mr`, ~5–10 s with `--fast`. L40S should be ~30–40% faster. **608 Phase 1 volumes ≈ 1–4 hours total** depending on mode.
- Known install gotcha: `pip install SimpleITK==2.0.2` if you hit "ITK only supports orthonormal direction cosines."

---

## 8. Open questions (for the next session)

- **HDF5 schema**: one big file with `split` attribute per group, or one file per split? User's preference is "doesn't have to fiddle around with dozens of files" — strongly suggests file-per-split. Per-patient: `/<ANONID>/{volume, mask?, voxel_spacing, model, slice_thickness, original_shape, crop_bbox, ...}` is the obvious shape but not yet specified.
- **Target slice thickness for harmonization**: median (4.5 mm)? Artist's 6 mm? Explorer's 3.5 mm? Or keep native + resample at DataLoader time? User leaning toward the latter per recommendation but not finalized.
- **TotalSegmentator margin**: how many mm beyond liver bbox? 10–20 mm is conventional but task-dependent.
- **Multi-label mask handling**: 13 of 126 Phase 1 masks have values ∉ {0,1}. Plan v2.2 says "preserve as-is and flag" — at preprocessing time, binarize all to {0,1} per user's step 3? Confirm.
- **Phase 2 patients in preprocessing**: Phase 2 negatives are SSL pretraining material. Run them through the same crop pipeline? Probably yes — masks irrelevant, but the cropped intensity volume is reusable.

---

## 9. Things to NOT do

- ❌ Don't `scancel` job 3269236 (Phase 2). It's running cleanly.
- ❌ Don't re-run `consolidate.py` or any of stages 0–4. Everything is settled and on disk.
- ❌ Don't change the splits seed (42). The whole point of the freeze is reproducibility.
- ❌ Don't binarize masks at conversion time. Multi-label values are preserved on disk; binarization happens in the preprocessing pipeline (step 3 above), not before.
- ❌ Don't autosubmit any `sbatch` jobs. The user runs all SLURM submissions.
- ❌ Don't write to `/scratch/<user>/` directly; use `/scratch/pioneer/users/<user>/`.
- ❌ Don't pre-bake target slice thickness into the HDF5 unless the user has confirmed a target — keep it as a load-time transform until then.

---

## 10. Useful commands for resuming

```bash
# Activate env
cd /home/sak185/dia-endo-conversion
source .venv/bin/activate

# Phase 2 progress
squeue -u $USER
ls /scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest_part_phase2_*.csv | wc -l

# After Phase 2 finishes, re-run QC
python scripts/qc.py \
    --output-root /scratch/pioneer/users/sak185/dia-endo-conversion/output \
    --workplan /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/workplan.csv \
    --skipped /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/skipped.csv \
    --alignment-audit /scratch/pioneer/users/sak185/dia-endo-conversion/output/alignment_audit_results.csv \
    --splits-json /scratch/pioneer/users/sak185/dia-endo-conversion/workplan/splits.json

# Phase 1 manifest peek
python -c "
import polars as pl
m = pl.read_csv('/scratch/pioneer/users/sak185/dia-endo-conversion/output/manifest.csv', infer_schema_length=10000)
print(m.group_by('split').agg(pl.col('patient_id').n_unique().alias('n_patients')).sort('split'))
"

# Get an L40S GPU node
salloc --partition=gpu --constraint=gpul40s --gres=gpu:1 \
       --cpus-per-task=8 --mem=32G --time=8:00:00

# Pre-stage TotalSegmentator weights (do once, on a node with internet)
totalseg_download_weights -t total_mr
```

---

## 11. Pointer to memory + earlier handoff

- Earlier handoff (pilot QC era): [`handoff-2026-04-26.md`](./handoff-2026-04-26.md) — the morning snapshot. Most of its decision history is still valid; section 7 (its TODO list) is now all done.
- Authoritative plan: [`convert-plan-v2.md`](./convert-plan-v2.md). Plan still ends at stage 7 (rsync to /home/.../data); the new preprocessing stage isn't documented yet — that's the first thing to spec in the next session.
- Memory files at `~/.claude/projects/-home-sak185-dia-endo-conversion/memory/` are still current.
