"""Build a self-contained re-mask package under /home/sak185/manual-remask/

For each of the 23 patients with mask-on-alt:
  manual-remask/<ANONID>/
    water_canonical.nii.gz, .json
    water_canonicala.nii.gz, .json   (if dcm2niix split the canonical)
    water_canonicalb.nii.gz, .json   (if 3-way split)
    reference_existing_mask.nii.gz   (the current mask — on the wrong/alt series; for visual reference only)
    dicom_canonical/                  (original DICOM series for the canonical, for re-segmentation if preferred)

Plus a manifest.csv at the package root with shape info per sub-volume.
"""
import argparse
import csv
import shutil
from pathlib import Path

import nibabel as nib
import polars as pl


# 22 mask-on-alt + 1 unmatched mystery patient
REMASK_PIDS = [
    "ANON01042AC6BED6", "ANON04E544117D33", "ANON0CA704939E49", "ANON1E4395CE2DC8",
    "ANON25C6C345BBDA", "ANON347F8214D258", "ANON474B6A632EC1", "ANON6A3A48D35640",
    "ANON75E0F1948C42", "ANON7658F5FE016F", "ANON76F7B5163F24", "ANON8EF9FA2BE221",
    "ANON92A132502D71", "ANON98178BC6BA01", "ANONB37185FC9DAF", "ANONB875B81287C3",
    "ANONC0DC7E3FB015", "ANONC4A3AEBA378D", "ANOND6AD65AC2CC6", "ANONDB32D8579FDC",
    "ANONDC70CDAF08A1", "ANONE4156CDEDFE1", "ANONF7C4EE526DB8",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Where the freshly-converted canonicals live (output/nifti_pos/<ANONID>/...)")
    ap.add_argument("--existing-masks", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True,
                    help="audit_mask_canonical.csv from earlier")
    ap.add_argument("--package-dir", type=Path, required=True)
    ap.add_argument("--include-dicoms", action="store_true",
                    help="Also copy the canonical DICOM series (default: skip; ~6 GB if on)")
    args = ap.parse_args()

    plan = pl.read_csv(args.workplan, infer_schema_length=10000)
    audit = pl.read_csv(args.audit, infer_schema_length=10000)

    canonicals = plan.filter(
        (pl.col("cohort") == "pos") & (pl.col("role") == "canonical")
        & pl.col("patient_id").is_in(REMASK_PIDS))
    canon_by_pid = {r["patient_id"]: r for r in canonicals.iter_rows(named=True)}
    audit_by_pid = {r["patient_id"]: r for r in audit.iter_rows(named=True)}

    args.package_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for pid in REMASK_PIDS:
        canon = canon_by_pid.get(pid)
        if not canon:
            print(f"  [skip] {pid}: no canonical row in workplan")
            continue
        pkg_dir = args.package_dir / pid
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Copy fresh canonical NIfTIs (may be 1, 2, or 3 sub-volumes).
        fresh_dir = args.output_root / "nifti_pos" / pid
        sub_volumes = sorted(fresh_dir.glob("water_canonical*.nii.gz"))
        if not sub_volumes:
            print(f"  [warn] {pid}: no fresh canonical NIfTI found in {fresh_dir}")
            continue
        for nii in sub_volumes:
            shutil.copy2(nii, pkg_dir / nii.name)
            json_sib = nii.with_suffix("").with_suffix(".json")
            if json_sib.exists():
                shutil.copy2(json_sib, pkg_dir / json_sib.name)

        # Copy the current "wrong" mask for reference.
        # The mask filename pattern: ANONID.nii.gz (plain) or ANONID_<series>.nii.gz (extended).
        existing_masks = list(args.existing_masks.glob(f"{pid}*.nii.gz"))
        for m in existing_masks:
            shutil.copy2(m, pkg_dir / f"reference_existing_mask_{m.name}")

        # Optionally copy DICOMs.
        dicom_files_copied = 0
        if args.include_dicoms:
            src_series = Path(canon["source_series_path"])
            dest_dicom = pkg_dir / "dicom_canonical"
            dest_dicom.mkdir(parents=True, exist_ok=True)
            for f in src_series.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    shutil.copy2(f, dest_dicom / f.name)
                    dicom_files_copied += 1

        # Manifest row(s) — one per sub-volume.
        audit_r = audit_by_pid.get(pid, {})
        for nii in sub_volumes:
            img = nib.load(nii)
            manifest_rows.append({
                "patient_id": pid,
                "package_file": nii.name,
                "shape": "x".join(str(s) for s in img.shape),
                "n_dcm_in_canonical_source": canon["n_dcm_files_in_source"],
                "canonical_series_desc": canon["source_series_description"],
                "current_mask_series_desc": audit_r.get("mask_source_series_desc", ""),
                "current_mask_thickness_mm": audit_r.get("mask_source_thickness_mm", ""),
                "canonical_thickness_mm": audit_r.get("canonical_thickness_mm", ""),
                "n_water_series": audit_r.get("n_water_series", ""),
                "dicom_files_copied": dicom_files_copied if args.include_dicoms else "(skipped)",
                "note": ("dcm2niix split the canonical into multiple sub-volumes; "
                         "re-segment on the one(s) that show the correct anatomy."
                         if len(sub_volumes) > 1 else ""),
            })
        print(f"  [done] {pid}: {len(sub_volumes)} sub-volume(s); "
              f"dicoms={dicom_files_copied}")

    # Write manifest
    manifest_path = args.package_dir / "manifest.csv"
    fields = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nManifest: {manifest_path}")
    print(f"Package root: {args.package_dir}")


if __name__ == "__main__":
    main()
