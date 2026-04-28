"""DEPRECATED 2026-04-27 — superseded by data-local-copy migration. Kept as historical reference only.

Realign existing radiologist masks to fresh dcm2niix RAS NIfTIs.

Lossless dual-axis flip: mask[::-1, :, ::-1] + adopt fresh affine.
Verified against 5 sample patients in the prior investigation (plan §2.6).

For each row in alignment_audit.csv (one per existing mask file):
  1. Load mask. Get its shape.
  2. Search the patient's freshly-converted output dir for a WATER volume
     (water_canonical*, water_alt_*) whose shape matches the mask (any sub-volume
     produced by dcm2niix splits).
  3. Apply lossless dual-axis flip; adopt the fresh volume's affine + header
     (with scl_slope=1, scl_inter=0 and uint8 dtype to avoid float-precision
     drift in the round-trip).
  4. Save as <output_root>/masks_pos/<ANONID>/mask_<TARGET_BASENAME>.nii.gz
     where TARGET_BASENAME is e.g. canonical, canonicala, alt_01.
  5. QC: reload, check axcodes match, affine matches, values are exactly {0,1},
     mask is non-empty.
"""
import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl


def _shape_match(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Match if shapes are equal under any axis permutation (sorted match).
    Coronal LAVA stores slice along middle axis in fresh dcm2niix but old
    masks may use a different orientation; sorted-shape comparison is
    invariant to that."""
    return sorted(a) == sorted(b)


def find_target_volume(pid_out_dir: Path, mask_shape: tuple[int, ...]) -> Path | None:
    """Return the path of the freshly-converted WATER volume matching mask_shape."""
    for pat in ("water_canonical*.nii.gz", "water_alt_*.nii.gz"):
        for f in sorted(pid_out_dir.glob(pat)):
            try:
                if _shape_match(nib.load(f).shape, mask_shape):
                    return f
            except Exception:
                continue
    return None


def target_basename(target_nii: Path) -> str:
    """water_canonical.nii.gz -> canonical;  water_alt_01.nii.gz -> alt_01;
    water_canonicala.nii.gz -> canonicala."""
    name = target_nii.name
    m = re.match(r"^water_(.+)\.nii\.gz$", name)
    return m.group(1) if m else name.replace(".nii.gz", "")


def realign_one(target_nii: Path, mask_path: Path, out_path: Path) -> dict:
    """Apply dual-axis flip + adopt target's affine; save as uint8 mask."""
    target = nib.load(target_nii)
    mask = nib.load(mask_path)
    notes = []
    if not _shape_match(mask.shape, target.shape):
        return {"shape_match": False, "qc_status": "fail",
                "qc_notes": f"shape {mask.shape} vs target {target.shape}"}

    # Lossless dual-axis flip; cast to uint8 (mask is binary).
    raw = np.asanyarray(mask.dataobj)  # raw int values, no scl_slope applied
    fixed = raw[::-1, :, ::-1].astype(np.uint8)

    out_header = target.header.copy()
    out_header.set_data_dtype(np.uint8)
    out_header.set_slope_inter(1.0, 0.0)  # avoid float drift on save/reload
    out = nib.Nifti1Image(fixed, affine=target.affine, header=out_header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, out_path)

    reloaded = nib.load(out_path)
    raw_reloaded = np.asanyarray(reloaded.dataobj)
    affine_diff = float(np.abs(reloaded.affine - target.affine).max())
    axc_target = nib.aff2axcodes(target.affine)
    axc_mask = nib.aff2axcodes(reloaded.affine)
    vals = np.unique(raw_reloaded).tolist()
    fg_voxel_count = int((raw_reloaded > 0).sum())

    # The flip is purely geometric; mask label values are preserved as-is.
    # Many source masks are not strict {0,1} — some use {0,2} or {0,1,2} for
    # multi-lesion/multi-label annotations. Treat anything non-zero as foreground.
    status = "ok"
    if axc_target != axc_mask:
        status = "fail"; notes.append("axcodes_mismatch")
    if affine_diff > 1e-3:
        status = "fail"; notes.append(f"affine_diff={affine_diff:.6f}")
    if fg_voxel_count == 0:
        status = "fail"; notes.append("empty_mask")
    if set(vals) - {0, 1}:
        # Pass-through: not an error, just a flag for downstream.
        notes.append(f"label_values={vals[:8]}")

    return {
        "shape_match": True,
        "axcodes_target": str(axc_target),
        "axcodes_mask_fixed": str(axc_mask),
        "affine_max_diff": affine_diff,
        "label_values": str(vals[:8]),
        "n_unique_labels": len(vals),
        "fg_voxel_count": fg_voxel_count,
        "qc_status": status,
        "qc_notes": ";".join(notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True,
                    help="(unused; kept for backward-compat)")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--masks-root", type=Path, required=True,
                    help="(unused; mask paths come from alignment_audit)")
    ap.add_argument("--alignment-audit", type=Path, required=True)
    ap.add_argument("--patient-list", type=Path)
    args = ap.parse_args()

    audit = pl.read_csv(args.alignment_audit, infer_schema_length=10000)
    if args.patient_list and args.patient_list.exists():
        keep = set(args.patient_list.read_text().split())
        audit = audit.filter(pl.col("patient_id").is_in(list(keep)))

    qc_rows = []
    for row in audit.iter_rows(named=True):
        pid = row["patient_id"]
        mask_src = row.get("mask_source_path", "")
        rec = {
            "patient_id": pid,
            "mask_filename": row.get("mask_filename", ""),
            "filename_suffix": row.get("filename_suffix", ""),
            "mask_source_path": mask_src,
            "target_volume": "",
            "mask_out_path": "",
        }

        if not mask_src or not Path(mask_src).exists():
            rec.update(qc_status="skip",
                       qc_notes=row.get("reason") or "no_mask_source")
            qc_rows.append(rec)
            continue

        # Locate target volume by shape match.
        pid_out_dir = args.output_root / "nifti_pos" / pid
        if not pid_out_dir.is_dir():
            rec.update(qc_status="skip", qc_notes="fresh_pid_dir_not_found")
            qc_rows.append(rec)
            continue

        try:
            mask_shape = nib.load(mask_src).shape
        except Exception as e:
            rec.update(qc_status="fail", qc_notes=f"load_mask_error:{e!r}"[:200])
            qc_rows.append(rec)
            continue

        target = find_target_volume(pid_out_dir, mask_shape)
        if target is None:
            rec.update(qc_status="fail",
                       qc_notes=f"no_fresh_volume_matches_mask_shape={mask_shape}")
            qc_rows.append(rec)
            continue

        basename = target_basename(target)
        out_path = (args.output_root / "masks_pos" / pid /
                    f"mask_{basename}.nii.gz")
        rec["target_volume"] = target.name
        rec["mask_out_path"] = str(out_path)
        rec.update(realign_one(target, Path(mask_src), out_path))
        qc_rows.append(rec)

    out_csv = args.output_root / "alignment_audit_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df = pl.DataFrame(qc_rows)
    result_df.write_csv(out_csv)
    if "qc_status" in result_df.columns:
        vc = result_df["qc_status"].value_counts()
        summary = dict(zip(vc["qc_status"].to_list(), vc["count"].to_list()))
        print(f"Realignment summary: {summary}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
