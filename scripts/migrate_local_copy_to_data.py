"""
Migration driver: data-local-copy/ → data/ for the 108 Phase-1 positives.

Replaces the misaligned positive volumes + masks under data/ with their hand-QC'd
correctly-aligned versions from data-local-copy/. Re-runs liver_masks (TotalSeg)
and liver_rois (dilation). Updates manifest.csv. Verifies. Cleans up.

Per agent/migration_plan_local_copy.md.

Phases:
  0 — Snapshot the current state (read-only)
  1 — Move existing positive dirs to data/_pre_migration_backup/
  2 — Stage data-local-copy raws + masks into data/raw and data/lesion_masks
       (RAS-canonicalize via nib.as_closest_canonical)
  3 — Re-run TotalSegmentator on the 108 new raws (--workers 8)
  4 — Re-dilate the 108 new liver_masks at 20 mm
  5 — Update manifest.csv (108 positive rows + scanner_model backfill on all 608)
  6 — Verification suite (gates Phase 7)
  7 — Cleanup (delete backup + data-local-copy/, mark deprecated scripts)

Run:
    uv run -m scripts.migrate_local_copy_to_data --phase 0
    uv run -m scripts.migrate_local_copy_to_data --phase 1 --execute
    ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
LOCAL = PROJECT_ROOT / "data-local-copy"
BACKUP = DATA / "_pre_migration_backup"
SNAP_DIR = PROJECT_ROOT / "agent" / "migration_2026_04_27_snapshots"
EDA_OUT = PROJECT_ROOT / "eda" / "outputs"

# Patients that require a non-canonical source file (per Phase-0 audit).
NON_CANONICAL_SUFFIX = "_WATER:_COR_DIAFRAGMA_T1_LAVA_AB"
NON_CANONICAL_MNEMS = {"dapple_bunny_dome", "teak_ox_beach"}

# Five extras present in data-local-copy/ but not in our 108-positive cohort —
# user has visually QC'd that the lesion is not visualizable on the canonical;
# drop entirely.
SKIP_EXTRAS = {
    "granite-elk-quartz",
    "granite-marten-valley",
    "ivory-tern-sage",
    "steady-gorilla-crest",
    "wheat-shrew-road",
}

CANONICAL_MNEM_HYPHEN = lambda m: m.replace("_", "-")
CANONICAL_MNEM_UNDER = lambda m: m.replace("-", "_")


# =============================================================================
# Utilities
# =============================================================================

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def positives_df() -> pl.DataFrame:
    m = pl.read_csv(DATA / "manifest.csv", infer_schema_length=10000)
    return m.filter((pl.col("cohort") == "positive") & pl.col("transferred_to_home")).sort("mnemonic_id")


def negatives_df() -> pl.DataFrame:
    m = pl.read_csv(DATA / "manifest.csv", infer_schema_length=10000)
    return m.filter((pl.col("cohort") == "negative") & pl.col("transferred_to_home")).sort("mnemonic_id")


def src_paths_for(mnem_under: str) -> tuple[Path, Path]:
    """Return (src_nii, src_msk) under data-local-copy/ for a given underscore mnemonic."""
    hyphen = CANONICAL_MNEM_HYPHEN(mnem_under)
    if mnem_under in NON_CANONICAL_MNEMS:
        stem = f"{hyphen}{NON_CANONICAL_SUFFIX}"
    else:
        stem = hyphen
    return LOCAL / "nifti" / f"{stem}.nii.gz", LOCAL / "masks" / f"{stem}.nii.gz"


def dst_paths_for(row: dict) -> tuple[Path, Path, Path, Path]:
    """Return (dst_nii, dst_msk, dst_liver_mask, dst_liver_roi) for a manifest row."""
    bucket = row["bucket"]
    mnem = row["mnemonic_id"]
    return (
        DATA / "raw" / bucket / "positive" / f"{mnem}.nii.gz",
        DATA / "lesion_masks" / bucket / "positive" / f"{mnem}_mask.nii.gz",
        DATA / "liver_masks" / bucket / "positive" / f"{mnem}_liver_mask.nii.gz",
        DATA / "liver_rois" / bucket / "positive" / f"{mnem}_liver_roi.nii.gz",
    )


def log(msg: str) -> None:
    print(f"[migrate] {msg}", flush=True)


# =============================================================================
# Phase 0 — Snapshot
# =============================================================================

def phase0(execute: bool) -> int:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Phase 0: snapshot to {SNAP_DIR}")

    targets = [
        DATA / "manifest.csv",
        DATA / "sidecars.jsonl",
        DATA / "splits.json",
        DATA / "patient_id_mapping.csv",
    ]
    for p in targets:
        if p.exists():
            dst = SNAP_DIR / p.name
            if execute:
                shutil.copy2(p, dst)
            log(f"  snapshot {p.name} ({p.stat().st_size} bytes)")

    pos = positives_df()
    neg = negatives_df()
    log(f"  positive rows: {pos.height}; negative rows: {neg.height}")

    rows: list[dict] = []
    log("  hashing 108 positive raws + masks (pre-migration)")
    for r in pos.iter_rows(named=True):
        raw_p = DATA / r["raw_path"]
        msk_p = DATA / r["lesion_mask_path"]
        rec = {
            "mnemonic_id": r["mnemonic_id"],
            "anon_id": r["anon_id"],
            "bucket": r["bucket"],
            "raw_path": r["raw_path"],
            "raw_sha256": sha256_file(raw_p) if raw_p.exists() else "",
            "lesion_mask_path": r["lesion_mask_path"],
            "lesion_mask_sha256": sha256_file(msk_p) if msk_p.exists() else "",
        }
        rows.append(rec)
    log("  hashing 500 negative raws (regression baseline)")
    neg_rows: list[dict] = []
    for r in neg.iter_rows(named=True):
        raw_p = DATA / r["raw_path"]
        neg_rows.append({
            "mnemonic_id": r["mnemonic_id"],
            "raw_path": r["raw_path"],
            "raw_sha256": sha256_file(raw_p) if raw_p.exists() else "",
        })

    if execute:
        pl.DataFrame(rows).write_csv(SNAP_DIR / "before_sha_positives.csv")
        pl.DataFrame(neg_rows).write_csv(SNAP_DIR / "before_sha_negatives.csv")
        (SNAP_DIR / "phase0_completed.txt").write_text(
            f"completed {datetime.now().isoformat()}\n"
            f"positives hashed: {len(rows)}\n"
            f"negatives hashed: {len(neg_rows)}\n"
        )
        log(f"  wrote before_sha_positives.csv ({len(rows)}) and before_sha_negatives.csv ({len(neg_rows)})")
    else:
        log("  DRY-RUN — no files written")
    return 0


# =============================================================================
# Phase 1 — Backup positives
# =============================================================================

def phase1(execute: bool) -> int:
    log(f"Phase 1: backup positive dirs → {BACKUP}")
    moved = []
    for sub in ("raw", "lesion_masks", "liver_masks", "liver_rois"):
        for bucket in ("cross-validation", "holdout"):
            src = DATA / sub / bucket / "positive"
            dst = BACKUP / sub / bucket / "positive"
            if not src.exists():
                log(f"  skip {src} (does not exist)")
                continue
            if execute:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    log(f"  backup target already exists: {dst} — aborting")
                    return 1
                shutil.move(str(src), str(dst))
            n = sum(1 for _ in dst.glob("*.nii.gz")) if execute else sum(1 for _ in src.glob("*.nii.gz"))
            log(f"  moved {src} → {dst} ({n} files)")
            moved.append((src, dst, n))
    if not execute:
        log("  DRY-RUN — no files moved")
    return 0


# =============================================================================
# Phase 2 — Stage raws + masks (RAS canonicalize)
# =============================================================================

def _save_ras(src: Path, dst: Path, *, is_mask: bool) -> dict:
    """Load src, apply nib.as_closest_canonical (→ RAS), optionally binarize+uint8 for masks,
    save to dst. Returns metadata."""
    img = nib.load(src)
    img = nib.as_closest_canonical(img)
    arr = np.asarray(img.dataobj)
    if is_mask:
        arr = (arr > 0).astype(np.uint8)
        out = nib.Nifti1Image(arr, img.affine, img.header)
        out.header.set_data_dtype(np.uint8)
        out.header.set_slope_inter(1.0, 0.0)
    else:
        # Preserve dtype as-is from canonical reorientation; just rebuild image with affine.
        out = nib.Nifti1Image(arr, img.affine, img.header)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, dst)
    final = nib.load(dst)
    final_arr = np.asarray(final.dataobj)
    return {
        "shape": final.shape,
        "zooms": tuple(float(z) for z in final.header.get_zooms()[:3]),
        "axcodes": "".join(nib.aff2axcodes(final.affine)),
        "voxel_count": int((final_arr > 0).sum()) if is_mask else None,
        "sha256": sha256_file(dst),
    }


def phase2(execute: bool) -> int:
    log(f"Phase 2: stage 108 positives from {LOCAL} (RAS canonicalize)")
    # Pre-flight: confirm 5 extras are not present in our 108 cohort
    pos = positives_df()
    pos_anons = set(pos["anon_id"].to_list())
    local_map = pl.read_csv(LOCAL / "patient_id_mapping.csv")
    local_extras = local_map.filter(
        ~pl.col("anon_id").is_in(list(pos_anons))
    )["mnemonic_id"].to_list()
    if set(local_extras) != SKIP_EXTRAS:
        log(f"  WARN: data-local-copy extras {set(local_extras)} != SKIP_EXTRAS {SKIP_EXTRAS}")
    log(f"  will SKIP {len(local_extras)} extras: {sorted(local_extras)}")

    rows: list[dict] = []
    n_ok = n_fail = 0
    for r in pos.iter_rows(named=True):
        mnem = r["mnemonic_id"]
        src_nii, src_msk = src_paths_for(mnem)
        dst_nii, dst_msk, _, _ = dst_paths_for(r)

        if not src_nii.exists():
            log(f"  FAIL {mnem}: source NIfTI not found at {src_nii}")
            rows.append({"mnemonic_id": mnem, "anon_id": r["anon_id"], "status": "missing_src_nii",
                         "src_nii": str(src_nii)})
            n_fail += 1
            continue
        if not src_msk.exists():
            log(f"  FAIL {mnem}: source mask not found at {src_msk}")
            rows.append({"mnemonic_id": mnem, "anon_id": r["anon_id"], "status": "missing_src_msk",
                         "src_msk": str(src_msk)})
            n_fail += 1
            continue

        rec = {
            "mnemonic_id": mnem,
            "anon_id": r["anon_id"],
            "bucket": r["bucket"],
            "src_nii": str(src_nii),
            "src_msk": str(src_msk),
            "dst_nii": str(dst_nii),
            "dst_msk": str(dst_msk),
            "non_canonical": mnem in NON_CANONICAL_MNEMS,
        }
        if execute:
            try:
                nii_meta = _save_ras(src_nii, dst_nii, is_mask=False)
                msk_meta = _save_ras(src_msk, dst_msk, is_mask=True)
                rec.update({
                    "raw_shape": "x".join(str(s) for s in nii_meta["shape"]),
                    "raw_zooms": str(nii_meta["zooms"]),
                    "raw_axcodes": nii_meta["axcodes"],
                    "raw_sha256": nii_meta["sha256"],
                    "mask_voxels": msk_meta["voxel_count"],
                    "mask_axcodes": msk_meta["axcodes"],
                    "mask_sha256": msk_meta["sha256"],
                    "shape_eq": nii_meta["shape"] == msk_meta["shape"],
                    "zooms_eq": nii_meta["zooms"] == msk_meta["zooms"],
                    "status": "ok",
                })
                if not (rec["shape_eq"] and rec["zooms_eq"]):
                    log(f"  FAIL {mnem}: nii/msk shape or zoom mismatch after canonicalization")
                    rec["status"] = "shape_or_zoom_mismatch"
                    n_fail += 1
                else:
                    n_ok += 1
                if (n_ok + n_fail) % 20 == 0:
                    log(f"  ... {n_ok + n_fail}/108 (ok={n_ok} fail={n_fail})")
            except Exception as e:  # noqa: BLE001
                rec["status"] = f"error: {e!r}"
                n_fail += 1
                log(f"  FAIL {mnem}: {e!r}")
        else:
            rec["status"] = "dry_run"
        rows.append(rec)

    EDA_OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(EDA_OUT / "migration_phase2_report.csv")
    log(f"  wrote {EDA_OUT / 'migration_phase2_report.csv'}")
    log(f"  ok={n_ok} fail={n_fail}")
    return 0 if n_fail == 0 else 1


# =============================================================================
# Phase 3 — TotalSegmentator
# =============================================================================

def phase3(execute: bool, workers: int) -> int:
    log(f"Phase 3: TotalSegmentator (--workers {workers})")
    cmd = [
        "uv", "run", "-m", "scripts.run_totalseg",
        "--data-root", str(DATA),
        "--workers", str(workers),
    ]
    if execute:
        cmd.append("--execute")
    log(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


# =============================================================================
# Phase 4 — Dilation
# =============================================================================

def phase4(execute: bool, workers: int) -> int:
    log(f"Phase 4: Dilate liver_masks at 20 mm (--workers {workers})")
    cmd = [
        "uv", "run", "-m", "scripts.dilate_segmentations",
        "--input-dir", str(DATA / "liver_masks"),
        "--output-dir", str(DATA / "liver_rois"),
        "--margin-mm", "20",
        "--workers", str(workers),
        "--data-root", str(DATA),
    ]
    if execute:
        cmd.append("--execute")
    log(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


# =============================================================================
# Phase 5 — Manifest update + scanner_model backfill
# =============================================================================

def phase5(execute: bool) -> int:
    log("Phase 5: update manifest (positives + scanner_model backfill)")
    m = pl.read_csv(DATA / "manifest.csv", infer_schema_length=10000)

    pos_mnems = set(
        m.filter((pl.col("cohort") == "positive") & pl.col("transferred_to_home"))["mnemonic_id"].to_list()
    )
    log(f"  recomputing fields for {len(pos_mnems)} positives")

    ts = datetime.now().isoformat(timespec="seconds")
    sha_map: dict[str, str] = {}
    shape_map: dict[str, str] = {}
    nslc_map: dict[str, int] = {}
    px_map: dict[str, float] = {}
    py_map: dict[str, float] = {}
    for r in m.filter(pl.col("mnemonic_id").is_in(list(pos_mnems))).iter_rows(named=True):
        mnem = r["mnemonic_id"]
        raw_p = DATA / r["raw_path"]
        if not raw_p.exists():
            log(f"  WARN: raw missing for {mnem} at {raw_p}")
            continue
        img = nib.load(raw_p)
        sha_map[mnem] = sha256_file(raw_p)
        shape_map[mnem] = "x".join(str(s) for s in img.shape)
        nslc_map[mnem] = int(img.shape[1]) if len(img.shape) >= 2 else 0
        zooms = img.header.get_zooms()
        px_map[mnem] = float(zooms[0]) if len(zooms) >= 1 else 0.0
        py_map[mnem] = float(zooms[2]) if len(zooms) >= 3 else 0.0

    # Build scanner_model backfill from sidecars.
    log("  reading sidecars for scanner_model backfill")
    sidecar_model: dict[str, str] = {}
    with open(DATA / "sidecars.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mnem = obj.get("mnemonic_id")
            sc = obj.get("sidecar", {})
            mname = sc.get("ManufacturersModelName") or ""
            if mnem:
                sidecar_model[mnem] = mname
    log(f"  sidecar models: {len(sidecar_model)} entries")

    def update_row(r: dict) -> dict:
        out = dict(r)
        mnem = r["mnemonic_id"]
        # Backfill scanner_model for any transferred row.
        if r.get("transferred_to_home") and not (r.get("scanner_model") or "").strip():
            out["scanner_model"] = sidecar_model.get(mnem, r.get("scanner_model") or "")
        # For positives, recompute geometry + sha + ts.
        if mnem in pos_mnems:
            if mnem in sha_map:
                out["sha256_raw"] = sha_map[mnem]
                out["shape"] = shape_map[mnem]
                out["n_slices_actual"] = nslc_map[mnem]
                out["pixel_spacing_x_mm"] = px_map[mnem]
                out["pixel_spacing_y_mm"] = py_map[mnem]
                out["migration_timestamp"] = ts
        return out

    new_rows = [update_row(r) for r in m.iter_rows(named=True)]
    new = pl.DataFrame(new_rows, schema=m.schema)

    if execute:
        tmp = DATA / "manifest.csv.tmp"
        new.write_csv(tmp)
        tmp.replace(DATA / "manifest.csv")
        log("  wrote manifest.csv")
    else:
        log("  DRY-RUN — manifest.csv not written")

    # Diff vs Phase-0 snapshot for sanity.
    snap = SNAP_DIR / "manifest.csv"
    if snap.exists():
        old = pl.read_csv(snap, infer_schema_length=10000)
        merged = old.join(new, on="mnemonic_id", how="inner", suffix="_new")
        n_pos_changed = merged.filter(
            (pl.col("mnemonic_id").is_in(list(pos_mnems))) &
            (pl.col("sha256_raw") != pl.col("sha256_raw_new"))
        ).height
        n_scanner_changed = merged.filter(
            (pl.col("scanner_model").fill_null("") != pl.col("scanner_model_new").fill_null(""))
        ).height
        log(f"  diff vs snapshot: sha256_raw changed for {n_pos_changed} positives; "
            f"scanner_model changed for {n_scanner_changed} rows")
    return 0


# =============================================================================
# Phase 6 — Verification suite
# =============================================================================

def _verify_one_positive(r: dict) -> dict:
    rec = {"mnemonic_id": r["mnemonic_id"], "bucket": r["bucket"]}
    raw_p, msk_p, lm_p, lr_p = dst_paths_for(r)
    files = {"raw": raw_p, "msk": msk_p, "liver_mask": lm_p, "liver_roi": lr_p}
    for k, p in files.items():
        rec[f"{k}_exists"] = p.exists()
    if not all(p.exists() for p in files.values()):
        rec["status"] = "missing_files"
        return rec
    raw = nib.load(raw_p)
    msk = nib.load(msk_p)
    lm = nib.load(lm_p)
    lr = nib.load(lr_p)
    msk_arr = np.asarray(msk.dataobj)
    lm_arr = np.asarray(lm.dataobj)
    lr_arr = np.asarray(lr.dataobj)
    rec["shape_match"] = (raw.shape == msk.shape == lm.shape == lr.shape)
    rec["affine_match"] = bool(
        np.allclose(raw.affine, msk.affine, atol=1e-3) and
        np.allclose(raw.affine, lm.affine, atol=1e-3) and
        np.allclose(raw.affine, lr.affine, atol=1e-3)
    )
    rec["msk_binary"] = bool(set(np.unique(msk_arr).tolist()).issubset({0, 1}))
    rec["msk_uint8"] = msk_arr.dtype == np.uint8
    rec["lm_binary"] = bool(set(np.unique(lm_arr).tolist()).issubset({0, 1}))
    rec["lm_uint8"] = lm_arr.dtype == np.uint8
    rec["lr_binary"] = bool(set(np.unique(lr_arr).tolist()).issubset({0, 1}))
    rec["lr_uint8"] = lr_arr.dtype == np.uint8
    rec["liver_voxels"] = int(lm_arr.sum())
    rec["roi_voxels"] = int(lr_arr.sum())
    rec["roi_contains_liver"] = bool(((lm_arr > 0) & (lr_arr == 0)).sum() == 0)
    rec["lesion_voxels"] = int(msk_arr.sum())
    rec["axcodes"] = "".join(nib.aff2axcodes(raw.affine))
    rec["status"] = "ok"
    return rec


def phase6(execute: bool) -> int:
    log("Phase 6: verification suite")
    pos = positives_df()

    # 6a — filesystem integrity
    log("  6a — filesystem integrity for 108 positives")
    rows = []
    n_ok = n_fail = 0
    for r in pos.iter_rows(named=True):
        rec = _verify_one_positive(r)
        rows.append(rec)
        if rec["status"] != "ok":
            n_fail += 1
            log(f"    FAIL {rec['mnemonic_id']}: {rec['status']}")
            continue
        flags = [
            rec["shape_match"], rec["affine_match"],
            rec["msk_binary"], rec["msk_uint8"],
            rec["lm_binary"], rec["lm_uint8"],
            rec["lr_binary"], rec["lr_uint8"],
            rec["roi_contains_liver"],
            100_000 <= rec["liver_voxels"] <= 5_000_000,
            rec["lesion_voxels"] > 0,
        ]
        if all(flags):
            n_ok += 1
        else:
            n_fail += 1
            log(f"    FAIL {rec['mnemonic_id']}: flags={flags}")
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(EDA_OUT / "migration_phase6_filesystem.csv")
    log(f"  6a result: ok={n_ok} fail={n_fail} (of {pos.height})")
    if n_fail:
        return 1

    # 6b — manifest consistency
    log("  6b — manifest consistency")
    m = pl.read_csv(DATA / "manifest.csv", infer_schema_length=10000)
    if m.height != 5060:
        log(f"    WARN: manifest row count {m.height} != 5060")
    transferred = m.filter(pl.col("transferred_to_home"))
    log(f"    transferred rows: {transferred.height}")
    sm_present = transferred.filter(pl.col("scanner_model").is_not_null() & (pl.col("scanner_model") != ""))
    log(f"    scanner_model populated: {sm_present.height}/{transferred.height}")
    sm_values = set(transferred["scanner_model"].drop_nulls().to_list())
    log(f"    scanner_model values: {sorted(sm_values)}")
    lm_present = transferred.filter(pl.col("liver_voxel_count").is_not_null())
    log(f"    liver_voxel_count populated: {lm_present.height}/{transferred.height}")

    # 6c — splits consistency
    log("  6c — splits consistency")
    splits = json.loads((DATA / "splits.json").read_text())
    assignments = splits.get("assignments", {})
    pos_anons = set(pos["anon_id"].to_list())
    missing_in_splits = pos_anons - set(assignments.keys())
    if missing_in_splits:
        log(f"    FAIL: {len(missing_in_splits)} positive anon_ids missing from splits")
        return 1
    log(f"    all 108 positive anon_ids in splits.json ✓")

    # 6d — mask alignment quality (lesion vs ring contrast z)
    log("  6d — mask alignment quality (lesion-ring contrast z)")
    from scipy.ndimage import binary_dilation
    cz_rows = []
    for r in pos.iter_rows(named=True):
        raw_p, msk_p, _, _ = dst_paths_for(r)
        raw = np.asarray(nib.load(raw_p).dataobj).astype(float)
        msk = np.asarray(nib.load(msk_p).dataobj).astype(bool)
        if not msk.any():
            cz_rows.append({"mnemonic_id": r["mnemonic_id"], "contrast_z": None})
            continue
        dil = binary_dilation(msk, iterations=5)
        ring = dil & ~msk
        if not ring.any():
            cz_rows.append({"mnemonic_id": r["mnemonic_id"], "contrast_z": None})
            continue
        les_m = raw[msk].mean()
        ring_m = raw[ring].mean()
        ring_s = raw[ring].std()
        cz = float((les_m - ring_m) / max(ring_s, 1e-6))
        cz_rows.append({"mnemonic_id": r["mnemonic_id"], "lesion_mean": float(les_m),
                        "ring_mean": float(ring_m), "contrast_z": cz})
    cz_df = pl.DataFrame(cz_rows)
    cz_df.write_csv(EDA_OUT / "migration_phase6_alignment.csv")
    cz = cz_df["contrast_z"].drop_nulls()
    log(f"    contrast_z: median={cz.median():.3f}  P5={cz.quantile(0.05):.3f}  "
        f"min={cz.min():.3f}  weak<0.2={cz_df.filter(pl.col('contrast_z') < 0.2).height}")

    # 6e — lesion containment in 20 mm liver_roi
    log("  6e — lesion containment in 20 mm liver_roi")
    cont_rows = []
    full_in = 0
    for r in pos.iter_rows(named=True):
        _, msk_p, _, lr_p = dst_paths_for(r)
        msk = np.asarray(nib.load(msk_p).dataobj).astype(bool)
        roi = np.asarray(nib.load(lr_p).dataobj).astype(bool)
        if not msk.any():
            cont_rows.append({"mnemonic_id": r["mnemonic_id"], "lesion_voxels": 0,
                              "outside_voxels": 0, "fully_in_roi": True})
            full_in += 1
            continue
        outside = int((msk & ~roi).sum())
        cont_rows.append({"mnemonic_id": r["mnemonic_id"], "lesion_voxels": int(msk.sum()),
                          "outside_voxels": outside, "fully_in_roi": outside == 0})
        if outside == 0:
            full_in += 1
    pl.DataFrame(cont_rows).write_csv(EDA_OUT / "migration_phase6_containment.csv")
    log(f"    fully contained in 20 mm ROI: {full_in}/{pos.height}")

    # 6f — slice geometry (axcodes + 512×N×512)
    log("  6f — slice geometry")
    geom_axcodes: dict[str, int] = {}
    bad_shape = 0
    for r in pos.iter_rows(named=True):
        raw_p, _, _, _ = dst_paths_for(r)
        img = nib.load(raw_p)
        ax = "".join(nib.aff2axcodes(img.affine))
        geom_axcodes[ax] = geom_axcodes.get(ax, 0) + 1
        if not (img.shape[0] == 512 and img.shape[2] == 512):
            bad_shape += 1
    log(f"    axcodes histogram: {geom_axcodes}")
    log(f"    bad 512×N×512: {bad_shape}/{pos.height}")

    # 6h — negatives untouched (regression check)
    log("  6h — negatives untouched (SHA-256 vs Phase-0 snapshot)")
    snap_neg = SNAP_DIR / "before_sha_negatives.csv"
    if not snap_neg.exists():
        log("    WARN: no Phase-0 negative snapshot — skipping")
    else:
        old = pl.read_csv(snap_neg)
        neg = negatives_df()
        old_map = {r["mnemonic_id"]: r["raw_sha256"] for r in old.iter_rows(named=True)}
        n_diff = 0
        diffs = []
        for r in neg.iter_rows(named=True):
            mnem = r["mnemonic_id"]
            raw_p = DATA / r["raw_path"]
            if not raw_p.exists():
                n_diff += 1
                diffs.append(mnem)
                continue
            cur = sha256_file(raw_p)
            if cur != old_map.get(mnem, ""):
                n_diff += 1
                diffs.append(mnem)
        log(f"    negatives differing from snapshot: {n_diff}/{neg.height}")
        if n_diff:
            log(f"    first differing: {diffs[:5]}")
            return 1

    log("  Phase 6 PASSED")
    return 0


# =============================================================================
# Phase 7 — Cleanup
# =============================================================================

def phase7(execute: bool) -> int:
    log("Phase 7: cleanup")
    if execute:
        if BACKUP.exists():
            log(f"  removing {BACKUP}")
            shutil.rmtree(BACKUP)
        if LOCAL.exists():
            log(f"  removing {LOCAL}")
            shutil.rmtree(LOCAL)
        # Mark realign scripts as deprecated.
        for name in ("realign_masks.py", "realign_masks_v2.py"):
            p = PROJECT_ROOT / "scripts" / name
            if p.exists():
                txt = p.read_text()
                if "DEPRECATED" not in txt.split("\n", 1)[0]:
                    p.write_text(
                        '"""DEPRECATED 2026-04-27 — superseded by data-local-copy migration. '
                        'Kept as historical reference only.\n\n'
                        + txt.lstrip().lstrip('"""')
                    )
                    log(f"  marked {name} as deprecated")
        marker = SNAP_DIR / "migration_completed.txt"
        marker.write_text(
            f"completed {datetime.now().isoformat()}\n"
            f"phases 0-7 succeeded.\n"
        )
        log(f"  wrote {marker}")
    else:
        log("  DRY-RUN — no files removed")
        log(f"    would remove: {BACKUP}, {LOCAL}")
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="Migration phase to run (0..7).")
    ap.add_argument("--execute", action="store_true",
                    help="Actually do the work; otherwise dry-run where applicable.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Worker count for Phase 3 (TotalSeg) and Phase 4 (dilation). Default: 8.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log(f"phase={args.phase} execute={args.execute} workers={args.workers}")
    if args.phase == 0:
        return phase0(args.execute)
    if args.phase == 1:
        return phase1(args.execute)
    if args.phase == 2:
        return phase2(args.execute)
    if args.phase == 3:
        return phase3(args.execute, args.workers)
    if args.phase == 4:
        return phase4(args.execute, args.workers)
    if args.phase == 5:
        return phase5(args.execute)
    if args.phase == 6:
        return phase6(args.execute)
    if args.phase == 7:
        return phase7(args.execute)
    return 2


if __name__ == "__main__":
    sys.exit(main())
