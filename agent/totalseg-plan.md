# TotalSegmentator + 20mm Liver-ROI Pipeline — End-to-End Plan

**Author:** Claude (planning)
**Date:** 2026-04-26
**Status:** PLAN — ready for executor agent post-context-rotation.
**Predecessor:** [`migration-plan.md`](./migration-plan.md) §10 sketch of `run_totalseg.py`.

---

## 1. Goal

Generate, for every Phase 1 patient already on `/home/dia-endo-conversion/data/raw/` (608 patients), a **liver region-of-interest mask** that downstream training will multiply against the source volume to gate non-liver voxels. The pipeline runs in two stages:

1. **TotalSegmentator** (`task=total_mr`, `roi_subset=["liver"]`, full-res normal mode) → binary liver mask in `data/liver_masks/`.
2. **Distance-transform dilation** by 20 mm physical (anisotropic-aware) → binary liver-ROI mask in `data/liver_rois/`.

The end state of this plan:
- `data/liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz` — 608 binary uint8 NIfTIs (raw TotalSegmentator output).
- `data/liver_rois/<bucket>/<cohort>/<mnemonic>.nii.gz` — 608 binary uint8 NIfTIs (raw mask dilated by 20 mm, used as training-time ROI).
- `data/manifest.csv` — augmented with 5 new columns (`liver_mask_path`, `liver_mask_sha256`, `liver_roi_path`, `liver_voxel_count`, `liver_roi_margin_mm`).
- `data/_totalseg_test/` — deleted.
- `.gitignore` extended; the work since `init` committed in 6 logical commits.

**Use-case downstream:** `volume * (roi_mask > 0)` gates the input tensor to liver-vicinity voxels. Tensors are kept the same shape via padding (no cropping at this stage).

---

## 2. Inputs and dependencies

### 2.1 Source files (read-only)
- `/home/sak185/dia-endo-conversion/data/raw/<bucket>/<cohort>/<mnemonic>.nii.gz` — 608 source volumes
- `/home/sak185/dia-endo-conversion/data/manifest.csv` — manifest with `transferred_to_home`, `bucket`, `cohort`, `mnemonic_id`, `raw_path`

### 2.2 Tooling — confirmed working as of 2026-04-26
- **TotalSegmentator** installed via `uvx`. The wheel set is currently `torch==2.11.0+cu128` (matches the L40S driver 12.8 — see decision §3.7).
  - Verify with `~/.local/share/uv/tools/totalsegmentator/bin/python -c "import torch; print(torch.cuda.is_available())"` → must print `True`. If `False`, reinstall per §6.0 before continuing.
- **GPU**: NVIDIA L40S, 46 GB VRAM. Driver 570.158.01 / CUDA 12.8.
- **Python deps for the pipeline scripts**: `polars`, `nibabel`, `numpy`, `scipy`, `tqdm` (already in the project's `.venv` per `pyproject.toml` / `uv.lock`).

### 2.3 Pre-existing partial work to reuse
- `data/_totalseg_test/segmentations_normal/` already contains valid normal-mode liver segmentations for **10 Phase 1 patients** (5 pos + 5 neg). These were produced with the same `uvx TotalSegmentator -ta total_mr -rs liver -ml` invocation we'll use in production. **Reuse them**: copy/move to the production location during Phase B prep so they count toward the 608. Saves ~12 min.

The 10 reusable patients (from `data/_totalseg_test/selected_patients.csv`):
```
opal_shrew_weld         pos  fold3
arctic_sloth_dune       pos  holdout
hushed_crow_glen        pos  fold2
wild_gazelle_marsh      pos  fold3
steep_walrus_creek      pos  holdout
bold_cheetah_hedge      neg  fold1
arctic_ferret_grove     neg  holdout
arctic_rabbit_knoll     neg  fold4
dusty_hare_pass         neg  fold2
dusty_raccoon_basin     neg  fold1
```

---

## 3. Decisions captured (locked in — do NOT reopen without explicit user input)

1. **TotalSegmentator mode: full-res normal** (NOT `--fast`). Smoother output; ~6 % runtime overhead is acceptable for one-time pre-processing.
2. **Single ROI: `--roi_subset liver`** (with `--ml` to get a single-file output). Locator+crop+full-res-on-crop pattern.
3. **Output format: binary uint8 NIfTI for both raw and ROI masks.** Both `data/liver_masks/<...>.nii.gz` and `data/liver_rois/<...>.nii.gz` are `{0, 1}`-valued. TotalSegmentator's multi-label output (liver class label) is binarized post-inference.
4. **Dilation: distance-transform-based, physical-space anisotropic, 20 mm radius.** Uses `scipy.ndimage.distance_transform_edt` with `sampling=voxel_sizes` from each volume's header. NOT a uniform voxel-radius dilation — the per-axis voxel count is computed from each volume's true zooms.
5. **Dilation margin is configurable via `--margin-mm` CLI arg, default 20.**
6. **No bbox coordinates stored.** Downstream training applies the binary ROI as a multiplicative gate on the source volume; cropping is handled later (or never — pad to fixed shape instead).
7. **Parallelism: 6-way Python `multiprocessing.Pool`** for TotalSegmentator (each worker shells out to `uvx TotalSegmentator …`) and same for dilation (CPU-only `distance_transform_edt`). All TotalSegmentator workers share the L40S; the test phase confirms VRAM headroom.
8. **Idempotent.** Both scripts skip patients whose target file already exists with non-trivial content. `--force` flag re-runs; `--retry-failed` re-runs only patients in `failures.csv`.
9. **VRAM gate on the test phase.** If peak GPU memory during the 12-case test exceeds **41 GB** (90% of 46 GB), the script halts and asks the user to decide (drop to 4-way? 3-way?). It does NOT auto-scale.
10. **Failure handling: skip-with-log, never abort.** Each TotalSegmentator failure (non-zero exit) and each empty-mask result is logged to `data/_pipeline/failures.csv`. Each suspiciously-small mask (< 1000 voxels) is logged to `data/_pipeline/qc_warnings.csv` but the patient is still processed normally.
11. **Test phase outputs are PRODUCTION outputs.** The 12-case test writes directly to `data/liver_masks/`; the production phase processes the remaining 596 patients via idempotent skip.
12. **Reuse the 10 existing test segmentations from `data/_totalseg_test/`.** Move/copy them to production location at the start of Phase B.
13. **`.gitignore` excludes large data**, tracks small project metadata. See §10.1.
14. **Commits: 6 logical groupings.** See §10.2.

---

## 4. Filesystem layout after this pipeline

```
data/
├── raw/                                    # unchanged
├── lesion_masks/                           # unchanged
├── sidecars.jsonl                          # unchanged
├── manifest.csv                            # 5 columns added (see §5)
├── splits.json                             # unchanged
├── patient_id_mapping.csv                  # unchanged
├── README.md                               # updated to mention liver_masks/ and liver_rois/
├── liver_masks/                            # NEW — raw TotalSegmentator binary outputs
│   ├── holdout/
│   │   ├── positive/<mnemonic>.nii.gz       (22 files)
│   │   └── negative/<mnemonic>.nii.gz       (100 files)
│   └── cross-validation/
│       ├── positive/<mnemonic>.nii.gz       (86 files)
│       └── negative/<mnemonic>.nii.gz       (400 files)
├── liver_rois/                             # NEW — 20mm dilated ROIs (binary)
│   └── (mirrors liver_masks/ structure exactly, 608 files)
└── _pipeline/                              # NEW — pipeline metadata, NOT for downstream consumption
    ├── failures.csv                         # patients that failed TotalSegmentator (initially empty)
    ├── qc_warnings.csv                      # liver masks < 1000 voxels (initially empty)
    ├── timing_test.csv                      # per-patient wall time, 12-case test
    ├── timing_full.csv                      # per-patient wall time, full run
    ├── vram_log_test.csv                    # 2-second nvidia-smi samples during test
    └── pipeline_run.log                     # human-readable log, appended each run
```

**Why the `_pipeline/` underscore prefix:** sorts before `lesion_masks/`/`liver_*/` and signals "internal artifact, not downstream input". `.gitignore`d.

---

## 5. Manifest schema additions

After this pipeline, `data/manifest.csv` gains **5 new columns**, populated only for `transferred_to_home==True` rows. All other rows have empty values.

| Column | Type | Example | Notes |
|---|---|---|---|
| `liver_mask_path` | str (relative) | `liver_masks/cross-validation/positive/arctic_snow_tiger.nii.gz` | Empty if TotalSegmentator failed for this patient. |
| `liver_mask_sha256` | str | hex sha256 of the binary mask file | Empty on failure. |
| `liver_roi_path` | str (relative) | `liver_rois/cross-validation/positive/arctic_snow_tiger.nii.gz` | Empty if dilation step skipped (e.g., raw mask missing). |
| `liver_voxel_count` | int | `1234567` | Foreground voxel count of the **raw** liver mask. Useful for QC. Empty on failure. |
| `liver_roi_margin_mm` | int | `20` | The margin used by the dilation step that produced this ROI. Lets us track if any patient was processed with a different margin. |

Column order: append to the right of existing columns. No reordering.

---

## 6. Scripts to write or rewrite

### 6.0 (optional) Re-verify TotalSegmentator GPU setup

```bash
~/.local/share/uv/tools/totalsegmentator/bin/python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible'; print('OK')"
```

If this fails, reinstall TotalSegmentator with the cu128 wheel set:
```bash
uv tool install --reinstall --python 3.12 TotalSegmentator \
    --index https://download.pytorch.org/whl/cu128 \
    --index https://pypi.org/simple \
    --index-strategy unsafe-best-match
```

### 6.1 `scripts/run_totalseg.py` (NEW)

**Purpose:** Orchestrate parallel TotalSegmentator inference over the 608 Phase 1 patients. Idempotent, dry-run by default, configurable parallelism, retry-on-failure.

**CLI:**
```
python scripts/run_totalseg.py \
    --data-root /home/sak185/dia-endo-conversion/data \
    --workers 6 \
    [--limit 12]            # process only first N patients (for the test phase)
    [--patients <csv>]      # explicit subset of mnemonic_ids (one per line, or single column CSV)
    [--retry-failed]        # only process patients listed in _pipeline/failures.csv
    [--force]               # re-run patients whose output already exists
    [--vram-monitor]        # spawn an nvidia-smi sampler that writes vram_log.csv
    [--execute]             # default: dry-run
```

**Behavior:**

```
1. Load manifest.csv, filter transferred_to_home==True. Should have 608 rows.

2. Build the work queue:
   - If --patients: filter to those mnemonics.
   - Elif --retry-failed: load _pipeline/failures.csv, take its mnemonics.
   - Else: full 608.
   - If not --force: drop patients whose target liver_mask file exists AND is non-trivial
     (file size > 1 KB, OR sha256 matches a previously-recorded one — file size is enough
     for our purposes).
   - If --limit N: keep first N (sorted by mnemonic_id for determinism).
   - Emit dry-run summary: N patients to process, target output paths, estimated wall time
     (N × 80s / workers).

3. Pre-flight:
   - Verify TotalSegmentator's GPU is visible: subprocess check (§6.0). Abort if False.
   - Verify all source NIfTIs exist.
   - Verify all target dirs are writable (mkdir -p).
   - If --vram-monitor: spawn `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
     -l 2 > _pipeline/vram_log_test.csv` as a background process. Save its PID; SIGTERM at end.
   - If not --execute: stop here with the dry-run summary.

4. Execute (multiprocessing.Pool of N workers):
   - For each patient in the queue, the worker invokes:
       uvx TotalSegmentator \
           -i <data-root>/raw/<bucket>/<cohort>/<mnemonic>.nii.gz \
           -o <data-root>/liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz \
           -ta total_mr -rs liver -ml --quiet
   - Capture exit code + stderr + wall time.
   - On exit code == 0:
       - Open the output NIfTI, count foreground voxels (data > 0).
       - If voxel_count == 0:
           - Append to failures.csv: {mnemonic_id, anon_id, reason='empty_mask', stderr=''}
           - Delete the empty output file.
       - Elif voxel_count < 1000:
           - Append to qc_warnings.csv: {mnemonic_id, voxel_count}
           - Binarize the file in place (cast > 0 to uint8) and KEEP IT.
       - Else:
           - Binarize the file in place (cast > 0 to uint8). Done.
       - Append to timing CSV: {mnemonic_id, cohort, thickness_mm, wall_seconds, voxel_count, exit_code=0}
   - On exit code != 0:
       - Append to failures.csv: {mnemonic_id, anon_id, reason='exit_code_<N>', stderr=<last 500 chars>}
       - Append to timing CSV: {..., exit_code=<N>, voxel_count=0}

5. Post-processing:
   - If --vram-monitor: SIGTERM the sampler; compute peak VRAM; print.
   - Print summary: N attempted, N succeeded, N failed, N qc_warnings, total wall, mean wall.
   - If --limit was set AND failures count == 0 AND peak VRAM < 41 GB:
       Print: "TEST PHASE OK — safe to proceed with --workers 6 on full set."
   - If --limit was set AND peak VRAM >= 41 GB:
       Print: "TEST PHASE WARNING — peak VRAM <X> GB. Consider reducing --workers."
   - Exit code: 0 if all succeeded, 1 if any failed (so caller can branch).

6. Update manifest (only when no --limit, i.e., full run completed):
   - For each transferred patient:
       - If liver_masks/<...>.nii.gz exists: populate liver_mask_path, liver_mask_sha256,
         liver_voxel_count.
       - Else: leave empty.
   - Write manifest.csv (atomic — write to .tmp then mv).
```

**Notes on the worker:**

- `multiprocessing.Pool` with method `spawn` (avoid fork issues on shared CUDA state — though uvx subprocesses are independent, spawn is safer).
- Each worker function takes a `(mnemonic, src_path, dst_path)` tuple, calls `subprocess.run([...uvx command...], capture_output=True, timeout=600)`, returns a result dict.
- `tqdm` progress bar with `imap_unordered`.

**Helper functions to factor:**

```python
def binarize_inplace(nifti_path: Path) -> int:
    """Load NIfTI, cast (data > 0) to uint8, save back, return foreground voxel count."""
    img = nib.load(nifti_path)
    data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.header.set_data_dtype(np.uint8)
    out.header.set_slope_inter(1.0, 0.0)
    nib.save(out, nifti_path)
    return int(data.sum())

def sha256_file(path: Path) -> str:  # (copy from scripts/migrate_to_home.py)
    ...
```

### 6.2 `scripts/dilate_segmentations.py` (REWRITE)

**Purpose:** Apply 20 mm distance-transform dilation to each binary liver mask in `data/liver_masks/`, write the binary dilated ROI to `data/liver_rois/` mirroring the source structure. Per-volume voxel sizes from each NIfTI's header.

**Replaces the existing one-off draft.** The current `scripts/dilate_segmentations.py` (75 lines, hardcoded paths, single voxel-size from first file, multi-radius output, label-preserving) is a research prototype. The production version below is configurable, per-file voxel-size aware, single-radius, binary output, idempotent.

**CLI:**
```
python scripts/dilate_segmentations.py \
    --input-dir /home/sak185/dia-endo-conversion/data/liver_masks \
    --output-dir /home/sak185/dia-endo-conversion/data/liver_rois \
    --margin-mm 20 \
    --workers 6 \
    [--force]
    [--execute]             # default: dry-run
```

**Behavior:**

```
1. Recursively list all *.nii.gz in --input-dir. Should be 608 (after Phase B-D complete).

2. Build work queue: list of (src, dst) where dst mirrors src structure under --output-dir.
   If not --force: drop pairs where dst exists and size > 1 KB.

3. Dry-run summary: N pairs, output dirs to be created, --margin-mm value.

4. If --execute:
   - mkdir -p every unique dst parent.
   - multiprocessing.Pool(--workers) over the queue. Each worker:
       a. img = nib.load(src)
       b. data = np.asarray(img.dataobj) > 0     # binary input
       c. voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])   # PER-FILE
       d. dist = distance_transform_edt(~data, sampling=voxel_sizes)
       e. dilated = (dist <= margin_mm).astype(np.uint8)
       f. out = nib.Nifti1Image(dilated, img.affine, img.header)
       g. out.header.set_data_dtype(np.uint8); out.header.set_slope_inter(1.0, 0.0)
       h. nib.save(out, dst)
       i. Return (mnemonic, wall_seconds, voxel_count_in, voxel_count_out)

5. Print summary: N processed, mean wall, mean voxel_count growth ratio (out/in).

6. Update manifest:
   - For each transferred patient with a dst NIfTI present:
       - liver_roi_path = relative path
       - liver_roi_margin_mm = --margin-mm
   - Write manifest.csv atomically.
```

**Note on `nib.Nifti1Image` behavior:** when you pass an existing header, nibabel will preserve the affine but overwrite dtype/scl_slope/scl_inter as specified. The explicit `set_data_dtype` + `set_slope_inter` calls prevent any drift from float-promotion or scl-rescaling on save/reload.

**Note on edge cases:**
- If the input mask is all-zero (shouldn't happen — `run_totalseg.py` deletes empty outputs), `distance_transform_edt(~mask)` returns 0 everywhere and `dilated` becomes a full-volume mask of 1s. Worker should detect input voxel_count == 0 and skip with a log line ("input mask empty, skipping").

### 6.3 No update to `scripts/realign_masks.py`, `scripts/migrate_to_home.py`, etc.

This pipeline is additive. Don't touch existing scripts.

---

## 7. Execution checklist (for the executor agent)

Each phase is independent. Pause and report if anything is unexpected.

### Phase A — Verify environment

```bash
cd /home/sak185/dia-endo-conversion
source .venv/bin/activate

# A1. Confirm uvx + GPU
~/.local/share/uv/tools/totalsegmentator/bin/python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU OK:', torch.cuda.get_device_name(0))"

# A2. Confirm 608 transferred patients in manifest
python -c "import polars as pl; m=pl.read_csv('data/manifest.csv', infer_schema_length=10000); n=m.filter(pl.col('transferred_to_home')).height; assert n==608, n; print('manifest OK:', n, 'transferred')"

# A3. Confirm 608 source NIfTIs exist
test $(find data/raw -name '*.nii.gz' | wc -l) -eq 608 && echo "raw OK"

# A4. Confirm both scripts exist
test -f scripts/run_totalseg.py && test -f scripts/dilate_segmentations.py && echo "scripts OK"

# A5. Create _pipeline dir
mkdir -p data/_pipeline
```

### Phase B — Reuse the 10 existing test segmentations

```bash
# B1. For each of the 10 mnemonics in _totalseg_test/segmentations_normal/, copy the file to
#     data/liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz, binarizing on the way.
python <<'PYEOF'
from pathlib import Path
import polars as pl
import nibabel as nib
import numpy as np

m = pl.read_csv('data/manifest.csv', infer_schema_length=10000)
src_dir = Path('data/_totalseg_test/segmentations_normal')
n = 0
for src in sorted(src_dir.glob('*.nii.gz')):
    mnem = src.stem.replace('.nii', '')
    row = m.filter(pl.col('mnemonic_id') == mnem).to_dicts()
    if not row:
        print(f'WARN: {mnem} not in manifest'); continue
    r = row[0]
    bucket, cohort = r['bucket'], r['cohort']
    dst = Path(f'data/liver_masks/{bucket}/{cohort}/{mnem}.nii.gz')
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(src)
    data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.header.set_data_dtype(np.uint8); out.header.set_slope_inter(1.0, 0.0)
    nib.save(out, dst)
    n += 1
    print(f'  copied: {mnem} -> {dst}  (voxels: {int(data.sum())})')
print(f'\nDone. Pre-populated {n} liver masks.')
PYEOF

# B2. Verify
test $(find data/liver_masks -name '*.nii.gz' | wc -l) -eq 10 && echo "B OK"
```

### Phase C — VRAM-gated 12-case test

```bash
# C1. Pick 12 fresh patients with varied thickness AND large volumes (stress VRAM).
#     Strategy: pick 12 patients NOT already in liver_masks/, sort by middle-shape dim
#     descending, and from the top-30 by size, pick 12 spread by thickness.
python <<'PYEOF'
import polars as pl, json
m = pl.read_csv('data/manifest.csv', infer_schema_length=10000)
t = m.filter(pl.col('transferred_to_home'))
already = set(t['mnemonic_id']) - set(p.stem.replace('.nii','') for p in __import__('pathlib').Path('data/liver_masks').rglob('*.nii.gz'))
# Compute slice count from shape "AxBxC" middle dim
def slc(s): p=s.split('x'); return int(p[1])
remaining = t.filter(pl.col('mnemonic_id').is_in(list(already))).with_columns(
    sc=pl.col('shape').map_elements(slc, return_dtype=pl.Int64),
    thk=pl.col('slice_thickness_mm').cast(pl.Float64),
)
# Top 60 by slice count, then pick 12 spread across thickness quintiles
top = remaining.sort('sc', descending=True).head(60)
picks = []
for q in [0.05, 0.2, 0.4, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95, 0.5, 0.3, 0.7]:
    qval = top['thk'].quantile(q)
    cand = top.with_columns(d=(pl.col('thk') - qval).abs()).filter(~pl.col('mnemonic_id').is_in([p['mnemonic_id'] for p in picks])).sort('d').head(1)
    if cand.height: picks.append(cand.to_dicts()[0])
sel_path = 'data/_pipeline/test12_patients.csv'
pl.DataFrame(picks).select(['mnemonic_id','cohort','split','slice_thickness_mm','shape']).write_csv(sel_path)
print(f'Wrote {sel_path} ({len(picks)} patients)')
PYEOF

# C2. Dry-run on the test 12
python scripts/run_totalseg.py \
    --data-root /home/sak185/dia-endo-conversion/data \
    --patients data/_pipeline/test12_patients.csv \
    --workers 6
# Verify the dry-run summary lists 12 patients, no missing source files.

# C3. Execute the test (with VRAM monitor)
python scripts/run_totalseg.py \
    --data-root /home/sak185/dia-endo-conversion/data \
    --patients data/_pipeline/test12_patients.csv \
    --workers 6 \
    --vram-monitor \
    --execute 2>&1 | tee -a data/_pipeline/pipeline_run.log

# C4. Inspect VRAM log + decide
python -c "
import polars as pl
v = pl.read_csv('data/_pipeline/vram_log_test.csv', has_header=False, new_columns=['mb'])
peak = v['mb'].max()
print(f'Peak VRAM during test: {peak} MB ({peak/1024:.1f} GB)')
assert peak < 41 * 1024, f'PEAK EXCEEDS 41 GB — STOP, ask user'
print('OK to proceed with --workers 6')
"
```

**HALT POINT:** If C4 prints "PEAK EXCEEDS 41 GB", do NOT continue. Report to user and ask whether to drop to 4 workers (`--workers 4`) for the full run.

### Phase D — Full TotalSegmentator run on remaining 596 patients

```bash
# D1. Confirm count
test $(find data/liver_masks -name '*.nii.gz' | wc -l) -eq 22 && echo "10+12=22 already done"

# D2. Dry-run (idempotent skip handles the 22 already done)
python scripts/run_totalseg.py \
    --data-root /home/sak185/dia-endo-conversion/data \
    --workers 6
# Should report 586 to process (608 - 22).

# D3. Execute
python scripts/run_totalseg.py \
    --data-root /home/sak185/dia-endo-conversion/data \
    --workers 6 \
    --execute 2>&1 | tee -a data/_pipeline/pipeline_run.log
# Estimated wall time: 586 × 80s / 6 workers ≈ 130 min ≈ 2.2 h

# D4. Verify
test $(find data/liver_masks -name '*.nii.gz' | wc -l) -eq 608 && echo "608 OK"
# If less than 608, check failures.csv
python -c "
import polars as pl
n = sum(1 for _ in __import__('pathlib').Path('data/_pipeline/failures.csv').open()) - 1 if __import__('pathlib').Path('data/_pipeline/failures.csv').exists() else 0
print(f'failures: {n}')
"
```

**HALT POINT:** If failures > 0, run `python scripts/run_totalseg.py --retry-failed --workers 6 --execute` once. If failures persist after retry, report to user.

### Phase E — Dilation

```bash
# E1. Dry-run
python scripts/dilate_segmentations.py \
    --input-dir /home/sak185/dia-endo-conversion/data/liver_masks \
    --output-dir /home/sak185/dia-endo-conversion/data/liver_rois \
    --margin-mm 20 \
    --workers 6
# Should report 608 pairs (or however many liver_masks exist).

# E2. Execute
python scripts/dilate_segmentations.py \
    --input-dir /home/sak185/dia-endo-conversion/data/liver_masks \
    --output-dir /home/sak185/dia-endo-conversion/data/liver_rois \
    --margin-mm 20 \
    --workers 6 \
    --execute 2>&1 | tee -a data/_pipeline/pipeline_run.log
# Estimated wall time: 608 × ~5s / 6 workers ≈ 8 min (CPU-bound EDT)

# E3. Verify
test $(find data/liver_rois -name '*.nii.gz' | wc -l) -eq 608 && echo "608 ROIs OK"
```

### Phase F — Manifest update + verification

The manifest is updated *inside* `run_totalseg.py` and `dilate_segmentations.py` already. This phase is just verification.

```bash
python <<'PYEOF'
import polars as pl
m = pl.read_csv('data/manifest.csv', infer_schema_length=10000)
t = m.filter(pl.col('transferred_to_home'))
print(f'transferred: {t.height}')
print(f'with liver_mask_path: {t.filter(pl.col("liver_mask_path") != "").height}')
print(f'with liver_roi_path:  {t.filter(pl.col("liver_roi_path") != "").height}')
print(f'liver_roi_margin_mm distribution: {t["liver_roi_margin_mm"].value_counts()}')
print(f'liver_voxel_count: min={t["liver_voxel_count"].min()} max={t["liver_voxel_count"].max()} median={t["liver_voxel_count"].median()}')
# Sanity: ROI voxel count > raw voxel count for every patient (dilation grows the mask)
# (Quick spot check on 5 patients)
import nibabel as nib, numpy as np
from pathlib import Path
for r in t.head(5).to_dicts():
    raw = nib.load(f'data/{r["liver_mask_path"]}'); roi = nib.load(f'data/{r["liver_roi_path"]}')
    rv = int((np.asarray(raw.dataobj)>0).sum()); ov = int((np.asarray(roi.dataobj)>0).sum())
    assert ov >= rv, f'ROI smaller than raw for {r["mnemonic_id"]}'
    print(f'  {r["mnemonic_id"]}: raw={rv} roi={ov} ratio={ov/rv:.2f}')
print('OK')
PYEOF
```

### Phase G — Cleanup

```bash
# G1. Delete the test directory
rm -rf data/_totalseg_test

# G2. Update data/README.md to mention liver_masks/ and liver_rois/.
#     (Add two lines under "Layout" and one paragraph explaining the ROI use case.)
#     The plan §10.3 below has the exact diff to apply.
```

### Phase H — `.gitignore` + commits

See §10.

---

## 8. VRAM monitoring details (Phase C only)

The `--vram-monitor` flag in `run_totalseg.py` does:

```python
vram_proc = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-l", "2"],
    stdout=open(data_root / "_pipeline" / "vram_log_test.csv", "w"),
)
try:
    # ... main work ...
finally:
    vram_proc.terminate()
    vram_proc.wait(timeout=5)
```

After the run, parse `vram_log_test.csv` (one integer MB per line, every 2 seconds) and compute the max. Report peak in MB and as a fraction of the 46 GB total. Threshold for "OK to scale": **< 41 GB (90%).**

Why 90% and not 100%: a 6th worker spinning up at the wrong moment could spike past the watermark; we want a small buffer so the OS doesn't OOM-kill the process.

---

## 9. Failure handling — full taxonomy

| Scenario | Detection | Logged to | File outcome | Pipeline continues? |
|---|---|---|---|---|
| TotalSegmentator exit != 0 | subprocess returncode | `failures.csv` (reason=`exit_code_<N>`, stderr tail) | not created | yes |
| TotalSegmentator exit 0 + empty mask | `voxel_count == 0` after binarize | `failures.csv` (reason=`empty_mask`) | deleted | yes |
| TotalSegmentator exit 0 + tiny mask (< 1000 voxels) | post-binarize check | `qc_warnings.csv` | KEPT, binarized normally | yes |
| Dilation worker exception | try/except around `dilate_one` | `failures.csv` (reason=`dilation_error`, exception repr) | not created | yes |
| Source NIfTI missing | pre-flight | abort with assert error | n/a | no |
| Manifest row missing for a mnemonic | pre-flight | abort with assert error | n/a | no |

**`failures.csv` schema:** `mnemonic_id, anon_id, phase, reason, stderr_excerpt, timestamp`
**`qc_warnings.csv` schema:** `mnemonic_id, voxel_count, timestamp`

Re-running with `--retry-failed`:
- Loads `failures.csv`, takes its mnemonic_ids, ignores the rest of the queue.
- On success, removes that row from `failures.csv` (rewrites the file without it).
- On failure, updates the row in `failures.csv` with the new timestamp.

---

## 10. Cleanup, .gitignore, commits

### 10.1 `.gitignore` additions

The current `.gitignore` is:
```
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

.venv
```

Append:
```
# Large derivative data — never commit
data/raw/
data/lesion_masks/
data/liver_masks/
data/liver_rois/
data/cropped_raw/
data/cropped_lesion_masks/
data/normalized_p1p99/
data/predictions/
data/_pipeline/
data/_totalseg_test/

# Slurm logs (already symlinked to /scratch but be explicit)
logs/
slurm-*.out

# Editor / OS
.DS_Store
*.swp
```

**Tracked (small project metadata):**
- `data/manifest.csv` (~2 MB after this pipeline; sub-MB compressed)
- `data/sidecars.jsonl` (1.2 MB)
- `data/splits.json` (240 KB)
- `data/patient_id_mapping.csv` (175 KB)
- `data/README.md`

### 10.2 Commit organization

Six logical commits in order. Each commit's body should explain *why*, not *what* (the diff already shows the what). Use HEREDOC for clean multi-line messages.

**Commit 1 — Mnemonic naming overhaul**
- Files: `scripts/generate_patient_names.py`
- Subject: `Add CLI args + underscore separator to patient name generator`
- Body: Why: needed parameterized input/output paths and underscore-separated names per migration plan §3. Refuses overwrite without --force to keep the mapping immutable.

**Commit 2 — Phase 1 migration to /home**
- Files: `scripts/migrate_to_home.py`, `data/patient_id_mapping.csv`, `data/splits.json`, `data/manifest.csv`, `data/README.md`
- Subject: `Add Phase 1 migration script + 5,060-row project manifest`
- Body: Why: separates the 608 Phase-1 dev cohort onto /home with mnemonic naming and a `transferred_to_home` boolean tracking the rest of the project on /scratch. Implements the cohort-aware sub-volume selection rule (positives match mask file, negatives pick max-slice DIXON-WATER row).

**Commit 3 — Sidecar consolidation**
- Files: `scripts/consolidate_sidecars.py`, `data/sidecars.jsonl`, `data/manifest.csv` (drop `raw_json_path` col), `data/README.md`
- Subject: `Consolidate per-patient BIDS sidecars into single JSONL`
- Body: Why: 608 tiny .json files alongside .nii.gz files added clutter; one JSONL with provenance fields (mnemonic_id, anon_id, bucket, split, cohort, raw_path) is easier to stream and keeps `data/raw/` strictly volume-only.

**Commit 4 — TotalSegmentator + dilation pipeline scripts**
- Files: `scripts/run_totalseg.py` (new), `scripts/dilate_segmentations.py` (rewrite)
- Subject: `Add liver-segmentation pipeline (TotalSegmentator + 20mm dilation)`
- Body: Why: needed liver-vicinity ROI to gate input volumes during training. Two-stage: TotalSegmentator (`task=total_mr`, `roi_subset=liver`, full-res normal) then physical-space distance-transform dilation. Both scripts are 6-way-parallel idempotent, with `--retry-failed` and dry-run-by-default. Confirmed 6 workers fit within the L40S 46 GB.

**Commit 5 — Liver mask & ROI artifacts**
- Files: `data/manifest.csv` (add 5 new columns: liver_mask_path, liver_mask_sha256, liver_roi_path, liver_voxel_count, liver_roi_margin_mm), `data/README.md` (mention liver_masks/ and liver_rois/)
- NOTE: `data/liver_masks/` and `data/liver_rois/` are gitignored, so this commit only carries manifest + README updates that *describe* what's now on disk.
- Subject: `Track liver masks + 20mm ROI paths in manifest`
- Body: Why: downstream training will read `liver_roi_path` to gate input volumes; `liver_voxel_count` is for QC (flag tiny segmentations). Path columns are populated only for the 608 transferred patients.

**Commit 6 — Gitignore + planning docs**
- Files: `.gitignore` (extended), `agent/totalseg-plan.md`, `agent/migration-plan.md`, `agent/handoff-2026-04-26.md`, `agent/handoff-2026-04-26-post-phase1.md`, `agent/convert-plan-v2.md`
- NOTE: agent docs were authored by Claude during planning sessions; commit them at the end so the repo records design rationale for posterity.
- Subject: `Extend .gitignore + add design docs under agent/`
- Body: Why: large data trees (raw/, lesion_masks/, liver_*/) are produced by the pipeline scripts and don't belong in git. The agent/ docs explain the design decisions (mnemonic naming, sub-volume selection, cohort-aware mask pairing, ROI dilation strategy) that someone re-running this in 6 months would otherwise have to reverse-engineer from the scripts.

**Important commit hygiene:**
- Use `git add <specific paths>`, never `git add -A` — there are large files in the working tree that must NOT be committed.
- Set the commit author to whatever the local config has; do NOT modify `.gitconfig`.
- After each commit, `git status` should show only un-tracked large data files (not staged).
- Co-author: append `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` per the project convention.

### 10.3 README diff for Phase G2

In `data/README.md`, under the Layout section, add two lines:
```
liver_masks/<bucket>/<cohort>/<mnemonic>.nii.gz       # binary uint8 — TotalSegmentator total_mr+liver
liver_rois/<bucket>/<cohort>/<mnemonic>.nii.gz        # binary uint8 — liver_masks dilated 20mm
```
(Replace the existing two `# ADDED LATER` lines for `liver_masks/` and the cropping rows shouldn't be touched.)

Add a new section after "sidecars.jsonl":

```markdown
### liver_rois — training-time ROI gate

`liver_rois/<bucket>/<cohort>/<mnemonic>.nii.gz` is the binary mask the training
DataLoader multiplies against the source volume to gate non-liver voxels:

```python
volume = nib.load(raw_path).get_fdata()
roi    = nib.load(liver_roi_path).get_fdata()
gated  = volume * (roi > 0)   # zero outside the dilated liver region
```

The ROI is the raw TotalSegmentator liver mask, dilated by 20 mm in physical
space (anisotropic `distance_transform_edt`). Re-generate with a different
margin via `python scripts/dilate_segmentations.py --margin-mm <N>`.
```

---

## 11. Things to NOT do (anti-patterns for the executor)

- ❌ **Don't commit `data/raw/`, `data/liver_masks/`, `data/liver_rois/`, `data/_pipeline/`, or `data/_totalseg_test/`** — `.gitignore` excludes them, but `git add -A` would override that. Stage files explicitly.
- ❌ **Don't commit if a pre-flight check failed** — the manifest must reflect what's on disk; partial state is worse than no state.
- ❌ **Don't auto-scale workers if the test phase exceeds 41 GB VRAM.** Halt and report.
- ❌ **Don't preserve the multi-label values from TotalSegmentator's `-ml` output** — both raw and ROI artifacts are binary uint8 0/1.
- ❌ **Don't compute voxel sizes from a single reference file** in the dilation script. Use per-file `img.header.get_zooms()`.
- ❌ **Don't run TotalSegmentator without `--quiet`** in the worker pool — the per-step prints from 6 concurrent workers will interleave into garbage; rely on `failures.csv` and timing CSV instead.
- ❌ **Don't re-run TotalSegmentator if `liver_masks/<...>.nii.gz` already exists with size > 1 KB** unless `--force`. The test phase produces real outputs that the production phase must not overwrite.
- ❌ **Don't change the dilation algorithm** to a non-distance-transform approach (e.g., iterative scipy `binary_dilation`) — physical-space anisotropic EDT is the locked-in approach.
- ❌ **Don't add bbox columns to the manifest.** Downstream gates with the binary mask, doesn't crop to a bbox.
- ❌ **Don't compress in `pipeline_run.log`** — append plain text. The log is for human eyes during/after the run.
- ❌ **Don't delete `data/_totalseg_test/` until Phase G**, even if the existing 10 segmentations have been copied. Keeps the comparison artifacts available if Phase B copy goes wrong.
- ❌ **Don't modify `scripts/migrate_to_home.py`, `scripts/generate_patient_names.py`, `scripts/consolidate_sidecars.py`** — they're done.

---

## 12. Pointers

- **Source manifest (post-migration):** `/home/sak185/dia-endo-conversion/data/manifest.csv` (5,060 rows, 28 cols pre-pipeline)
- **Source volumes:** `/home/sak185/dia-endo-conversion/data/raw/<bucket>/<cohort>/<mnemonic>.nii.gz`
- **TotalSegmentator install:** `~/.local/share/uv/tools/totalsegmentator/`
  - Python: `~/.local/share/uv/tools/totalsegmentator/bin/python` (3.12, torch 2.11.0+cu128)
- **Existing 10 normal-mode test segmentations:** `data/_totalseg_test/segmentations_normal/*.nii.gz`
- **Existing dilate prototype to replace:** `scripts/dilate_segmentations.py` (75-line one-off; rewrite per §6.2)
- **Reference plan style:** `agent/migration-plan.md` (especially §6 script outlines and §7 execution checklist)
- **Reference script style:** `scripts/migrate_to_home.py` (preflight checks, dry-run-default, idempotent, summary-then-execute)

---

## 13. Acceptance criteria (executor must verify before declaring "done")

- [ ] `find data/liver_masks -name '*.nii.gz' | wc -l` → **608**
- [ ] `find data/liver_rois -name '*.nii.gz' | wc -l` → **608**
- [ ] `data/_pipeline/failures.csv` has 0 rows (or only persistent post-retry failures, reported to user)
- [ ] Spot-check 5 patients: `liver_voxel_count > 1000` AND `liver_roi voxel_count >= liver_mask voxel_count` (dilation grew the mask)
- [ ] `data/manifest.csv` has 5 new columns populated for all 608 transferred patients
- [ ] `data/_totalseg_test/` has been deleted
- [ ] `git log --oneline` shows 6 new commits in the order of §10.2
- [ ] `git status` shows clean working tree (no untracked / modified files outside the gitignore)
- [ ] `peak VRAM during test < 41 GB` (recorded in `pipeline_run.log`)

---

**End of plan.** Ready for executor.
