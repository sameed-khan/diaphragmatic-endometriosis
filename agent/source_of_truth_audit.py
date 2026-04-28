"""Source-of-truth audit for the 108 Phase-1 positive cohort.

Outputs:
  - eda/outputs/source_of_truth_audit.csv (108 rows)
  - eda/outputs/source_of_truth_landscape.json (global landscape)
  - eda/outputs/source_of_truth_dicom_sample.csv (10 patient DICOM probe)
"""
from __future__ import annotations

import json
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
from scipy import ndimage as ndi

ROOT = Path("/scratch/pioneer/users/sak185/diaphragmatic-endometriosis")
DIAE = Path("/home/jjs374/DiaE")
NIFTI_DIR = DIAE / "nifti"
MASKS_DIR = DIAE / "masks"
DICOM_DIR = DIAE / "dicom"
OUT_DIR = ROOT / "eda" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANON_RE = re.compile(r"^(ANON[A-F0-9]+)(?:_(.+))?\.nii\.gz$")


def parse_filename(name: str):
    m = ANON_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)  # anon_id, suffix or None


def landscape_scan() -> dict:
    """Scan whole nifti/ and masks/ folders."""
    nifti_files = sorted(p.name for p in NIFTI_DIR.glob("*.nii.gz"))
    mask_files = sorted(p.name for p in MASKS_DIR.glob("*.nii.gz"))

    nifti_canon: dict[str, list[str]] = {}
    nifti_noncanon: dict[str, list[str]] = {}
    for fn in nifti_files:
        anon, suffix = parse_filename(fn)
        if anon is None:
            continue
        if suffix is None:
            nifti_canon.setdefault(anon, []).append(fn)
        else:
            nifti_noncanon.setdefault(anon, []).append(fn)

    mask_canon: dict[str, list[str]] = {}
    mask_noncanon: dict[str, list[str]] = {}
    for fn in mask_files:
        anon, suffix = parse_filename(fn)
        if anon is None:
            continue
        if suffix is None:
            mask_canon.setdefault(anon, []).append(fn)
        else:
            mask_noncanon.setdefault(anon, []).append(fn)

    all_nifti_anons = set(nifti_canon) | set(nifti_noncanon)
    all_mask_anons = set(mask_canon) | set(mask_noncanon)

    def categorise(canon: dict, noncanon: dict) -> dict:
        only_canon = sum(1 for a in canon if a not in noncanon)
        only_non = sum(1 for a in noncanon if a not in canon)
        both = sum(1 for a in canon if a in noncanon)
        return {"only_canonical": only_canon, "only_noncanonical": only_non, "both": both,
                "total_patients": len(set(canon) | set(noncanon))}

    suffix_counts_nifti: dict[str, int] = {}
    for files in nifti_noncanon.values():
        for fn in files:
            _, suf = parse_filename(fn)
            suffix_counts_nifti[suf] = suffix_counts_nifti.get(suf, 0) + 1

    suffix_counts_mask: dict[str, int] = {}
    for files in mask_noncanon.values():
        for fn in files:
            _, suf = parse_filename(fn)
            suffix_counts_mask[suf] = suffix_counts_mask.get(suf, 0) + 1

    return {
        "nifti_total_files": len(nifti_files),
        "mask_total_files": len(mask_files),
        "nifti_unique_patients": len(all_nifti_anons),
        "mask_unique_patients": len(all_mask_anons),
        "nifti_breakdown": categorise(nifti_canon, nifti_noncanon),
        "mask_breakdown": categorise(mask_canon, mask_noncanon),
        "noncanonical_suffix_counts_nifti": suffix_counts_nifti,
        "noncanonical_suffix_counts_mask": suffix_counts_mask,
    }


def header_info(path: Path):
    img = nib.load(str(path))
    return {
        "shape": tuple(int(x) for x in img.shape),
        "zooms": tuple(float(x) for x in img.header.get_zooms()[:3]),
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
        "affine": img.affine.tolist(),
    }


def lesion_ring_contrast(nifti_path: Path, mask_path: Path) -> float:
    """Compute (mean_lesion - mean_ring)/std_ring; positive => mask aligns to bright structure.

    Aligns the mask to the NIfTI grid by axis flips inferred from affine sign mismatches
    (the canonical jjs374 masks have a Z-axis sign flip relative to the canonical NIfTI).
    """
    try:
        img = nib.load(str(nifti_path))
        msk = nib.load(str(mask_path))
        if img.shape[:3] != msk.shape[:3]:
            return float("nan")
        arr = np.asarray(img.dataobj, dtype=np.float32)
        m = np.asarray(msk.dataobj) > 0
        if m.sum() == 0:
            return float("nan")

        # Detect per-axis sign mismatch between img and mask affines.
        # If the diagonal sign of mask affine differs from img affine on axis k, flip the mask along k.
        for k in range(3):
            if np.sign(img.affine[k, k]) != np.sign(msk.affine[k, k]) and abs(img.affine[k, k]) > 1e-6 and abs(msk.affine[k, k]) > 1e-6:
                m = np.flip(m, axis=k)

        ring = ndi.binary_dilation(m, iterations=5) & (~m)
        if ring.sum() == 0:
            return float("nan")
        lesion_mean = float(arr[m].mean())
        ring_vals = arr[ring]
        ring_mean = float(ring_vals.mean())
        ring_std = float(ring_vals.std())
        if ring_std < 1e-6:
            return 0.0
        return (lesion_mean - ring_mean) / ring_std
    except Exception:
        return float("nan")


def per_patient_audit(args):
    anon_id, mnemonic_id = args
    canonical_nifti = NIFTI_DIR / f"{anon_id}.nii.gz"
    canonical_mask = MASKS_DIR / f"{anon_id}.nii.gz"

    rec = {
        "mnemonic_id": mnemonic_id,
        "anon_id": anon_id,
        "canonical_nifti_exists": canonical_nifti.exists(),
        "canonical_mask_exists": canonical_mask.exists(),
        "nifti_shape": "",
        "nifti_zooms": "",
        "nifti_axcodes": "",
        "mask_shape": "",
        "mask_voxel_sum": 0,
        "shape_match": False,
        "non_canonical_nifti_count": 0,
        "non_canonical_nifti_suffixes": "",
        "non_canonical_mask_count": 0,
        "non_canonical_mask_suffixes": "",
        "lesion_ring_contrast_z": float("nan"),
        "dicom_series_count": 0,
        "dicom_series_names": "",
    }

    if canonical_nifti.exists():
        try:
            h = header_info(canonical_nifti)
            rec["nifti_shape"] = "x".join(str(x) for x in h["shape"])
            rec["nifti_zooms"] = ",".join(f"{x:.3f}" for x in h["zooms"])
            rec["nifti_axcodes"] = h["axcodes"]
        except Exception as e:
            rec["nifti_axcodes"] = f"ERR:{e}"

    if canonical_mask.exists():
        try:
            mimg = nib.load(str(canonical_mask))
            marr = np.asarray(mimg.dataobj)
            rec["mask_shape"] = "x".join(str(x) for x in mimg.shape)
            rec["mask_voxel_sum"] = int((marr > 0).sum())
            if canonical_nifti.exists():
                nimg = nib.load(str(canonical_nifti))
                rec["shape_match"] = bool(nimg.shape[:3] == mimg.shape[:3])
        except Exception as e:
            rec["mask_shape"] = f"ERR:{e}"

    # Non-canonical inventory
    nc_nifti = sorted(NIFTI_DIR.glob(f"{anon_id}_*.nii.gz"))
    rec["non_canonical_nifti_count"] = len(nc_nifti)
    rec["non_canonical_nifti_suffixes"] = ";".join(parse_filename(p.name)[1] or "" for p in nc_nifti)

    nc_mask = sorted(MASKS_DIR.glob(f"{anon_id}_*.nii.gz"))
    rec["non_canonical_mask_count"] = len(nc_mask)
    rec["non_canonical_mask_suffixes"] = ";".join(parse_filename(p.name)[1] or "" for p in nc_mask)

    # Lesion ring contrast
    if canonical_nifti.exists() and canonical_mask.exists() and rec["shape_match"]:
        rec["lesion_ring_contrast_z"] = lesion_ring_contrast(canonical_nifti, canonical_mask)

    # DICOM series listing
    pdir = DICOM_DIR / anon_id
    if pdir.exists():
        series = sorted(s.name for s in pdir.iterdir() if s.is_dir())
        rec["dicom_series_count"] = len(series)
        rec["dicom_series_names"] = " | ".join(series)

    return rec


def main():
    print("[1/4] Landscape scan...")
    ls = landscape_scan()
    (OUT_DIR / "source_of_truth_landscape.json").write_text(json.dumps(ls, indent=2))
    print(json.dumps(ls, indent=2))

    print("\n[2/4] Loading manifest + sidecars for 108 positives...")
    df = pl.read_csv(ROOT / "data/manifest.csv", infer_schema_length=10000)
    pos = df.filter((pl.col("cohort") == "positive") & (pl.col("transferred_to_home")))
    print(f"  {len(pos)} positives")

    sidecars: dict[str, dict] = {}
    with open(ROOT / "data/sidecars.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sidecars[d["anon_id"]] = d.get("sidecar", {})

    args_list = list(zip(pos["anon_id"].to_list(), pos["mnemonic_id"].to_list()))

    print("\n[3/4] Per-patient audit (parallel)...")
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(per_patient_audit, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 20 == 0:
                print(f"  {i}/{len(args_list)}")

    rec_by_anon = {r["anon_id"]: r for r in results}

    print("\n[4/4] Joining with manifest + sidecars + protocol cluster...")
    rows = []
    for row in pos.iter_rows(named=True):
        a = row["anon_id"]
        side = sidecars.get(a, {})
        rec = rec_by_anon.get(a, {})
        scanner = side.get("ManufacturersModelName") or ""
        sd = side.get("SeriesDescription") or row.get("series_description") or ""
        te = side.get("EchoTime")
        tr = side.get("RepetitionTime")
        fa = side.get("FlipAngle")
        it = "/".join(side.get("ImageType") or [])
        # Protocol clustering: scanner + canonical-series-description family
        sd_norm = sd.strip()
        if sd_norm.startswith("WATER: COR LAVA DIAF"):
            family = "LAVA_DIAF"
        elif sd_norm.startswith("WATER: COR DIAFRAGMA T1 LAVA"):
            family = "DIAFRAGMA_T1_LAVA"
        elif sd_norm.startswith("WATER: COR LAVA FLEX NAV"):
            family = "LAVA_FLEX_NAV"
        else:
            family = f"OTHER:{sd_norm}"
        cluster = f"{scanner or 'UNKNOWN'}|{family}"

        # Eligibility
        reasons = []
        if not rec.get("canonical_nifti_exists"):
            reasons.append("no_canonical_nifti")
        if not rec.get("canonical_mask_exists"):
            reasons.append("no_canonical_mask")
        if not rec.get("shape_match"):
            reasons.append("shape_mismatch")
        z = rec.get("lesion_ring_contrast_z")
        if isinstance(z, float) and (np.isnan(z) or z < 0):
            reasons.append(f"low_ring_contrast(z={z:.2f})")
        if rec.get("non_canonical_mask_count", 0) > 0:
            reasons.append("has_noncanonical_mask")
        verdict = "PASS" if not reasons else "REVIEW"

        rows.append({
            "mnemonic_id": row["mnemonic_id"],
            "anon_id": a,
            "split": row["split"],
            "bucket": row["bucket"],
            "scanner_model": scanner,
            "series_description": sd_norm,
            "image_type": it,
            "echo_time_ms": (te or 0) * 1000 if isinstance(te, (int, float)) else None,
            "repetition_time_ms": (tr or 0) * 1000 if isinstance(tr, (int, float)) else None,
            "flip_angle": fa,
            "slice_thickness_mm": side.get("SliceThickness"),
            "spacing_between_slices_mm": side.get("SpacingBetweenSlices"),
            "canonical_nifti_exists": rec.get("canonical_nifti_exists", False),
            "canonical_mask_exists": rec.get("canonical_mask_exists", False),
            "shape_match": rec.get("shape_match", False),
            "nifti_shape": rec.get("nifti_shape", ""),
            "nifti_zooms": rec.get("nifti_zooms", ""),
            "mask_voxel_sum": rec.get("mask_voxel_sum", 0),
            "non_canonical_nifti_count": rec.get("non_canonical_nifti_count", 0),
            "non_canonical_nifti_suffixes": rec.get("non_canonical_nifti_suffixes", ""),
            "non_canonical_mask_count": rec.get("non_canonical_mask_count", 0),
            "non_canonical_mask_suffixes": rec.get("non_canonical_mask_suffixes", ""),
            "lesion_ring_contrast_z": rec.get("lesion_ring_contrast_z"),
            "dicom_series_count": rec.get("dicom_series_count", 0),
            "dicom_series_names": rec.get("dicom_series_names", ""),
            "protocol_cluster": cluster,
            "protocol_family": family,
            "eligibility_verdict": verdict,
            "eligibility_reason": ";".join(reasons),
        })

    out = pl.DataFrame(rows)
    out_path = OUT_DIR / "source_of_truth_audit.csv"
    out.write_csv(out_path)
    print(f"\nWrote {out_path} with {len(out)} rows")
    print("\nProtocol cluster counts:")
    print(out.group_by("protocol_cluster").agg(pl.len().alias("n")).sort("n", descending=True))
    print("\nVerdict counts:")
    print(out["eligibility_verdict"].value_counts())
    print("\nNoncanonical mask present:", out.filter(pl.col("non_canonical_mask_count") > 0).select("mnemonic_id", "anon_id", "non_canonical_mask_suffixes"))


if __name__ == "__main__":
    main()
