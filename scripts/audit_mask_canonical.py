"""One-off investigation: for every positive with a radiologist mask, determine
which source series the mask was made on, and compare to the canonical pick.

Output: console table + audit_mask_canonical.csv with columns:
  patient_id, mask_source_series_desc, mask_z, canonical_series_desc, canonical_n_dcm,
  is_canonical (bool), n_water_series, alt_series_descs, slice_thickness_canonical_mm,
  slice_thickness_mask_source_mm, mask_source_inference_method
"""
import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--prescan", type=Path, required=True)
    ap.add_argument("--existing-nifti", type=Path, required=True)
    ap.add_argument("--existing-masks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    plan = pl.read_csv(args.workplan, infer_schema_length=10000)
    prescan = pl.read_csv(args.prescan, infer_schema_length=10000)

    # Per-patient: list all WATER series with their (path, desc, n_dcm, slice_thickness)
    pos_water = (prescan.filter((pl.col("cohort") == "pos")
                                & (pl.col("image_type_token") == "WATER"))
                        .select(["patient_id", "series_path", "series_folder_name",
                                 "series_description", "n_dcm_files",
                                 "slice_thickness_mm"]))
    by_pid_water: dict[str, list[dict]] = {}
    for r in pos_water.iter_rows(named=True):
        by_pid_water.setdefault(r["patient_id"], []).append(r)

    # Canonical pick per patient (from workplan)
    canonical = {r["patient_id"]: r for r in plan.filter(
        (pl.col("cohort") == "pos") & (pl.col("role") == "canonical")
    ).iter_rows(named=True)}

    nifti_files = list(args.existing_nifti.glob("*.nii.gz"))
    nifti_by_pid: dict[str, list[tuple[Path, str | None]]] = {}
    for f in nifti_files:
        m = re.match(r"^(ANON[A-F0-9]+)(?:_(.*))?\.nii\.gz$", f.name)
        if m:
            nifti_by_pid.setdefault(m.group(1), []).append((f, m.group(2)))

    rows = []
    counter = {"on_canonical": 0, "on_alt": 0, "no_match": 0,
               "no_mask": 0, "single_water_no_choice": 0}
    for pid, can in canonical.items():
        water_series = by_pid_water.get(pid, [])
        n_water = len(water_series)
        if pid not in nifti_by_pid:
            counter["no_mask"] += 1
            continue

        mask_z = None
        mask_source_desc = None
        method = None
        candidates = nifti_by_pid[pid]

        # Try: extended-style filename match
        for f, desc_suffix in candidates:
            if desc_suffix is not None:
                # Try to map the filename suffix back to a series_description.
                # The existing files were made by replacing spaces with '_'.
                wanted = desc_suffix.replace("_", " ")
                # SeriesDescriptions may start with "WATER:" — the suffix usually
                # encodes that as "WATER:" too.
                for ws in water_series:
                    desc = ws["series_description"].replace(" ", "_")
                    if desc == desc_suffix:
                        mask_source_desc = ws["series_description"]
                        # Slice axis varies; pick the dim closest to source n_dcm.
                        sh = nib.load(f).shape
                        n_dcm = int(ws["n_dcm_files"])
                        mask_z = min(sh, key=lambda d: abs(d - n_dcm))
                        method = "extended_filename_match"
                        break
                if mask_source_desc:
                    break

        if mask_source_desc is None:
            # Fall back: load nifti, get z, match against water_series n_dcm_files
            for f, desc_suffix in candidates:
                if desc_suffix is None:  # plain
                    sh = nib.load(f).shape
                    # For each candidate series, take min-axis-diff
                    def closest_diff(n_dcm: int) -> int:
                        return min(abs(d - n_dcm) for d in sh)
                    matches = [ws for ws in water_series
                               if closest_diff(int(ws["n_dcm_files"])) <= 2]
                    # The "z" we report is the matched dimension
                    z = (min(sh, key=lambda d: abs(d - int(matches[0]["n_dcm_files"])))
                         if matches else max(sh))
                    if len(matches) == 1:
                        mask_source_desc = matches[0]["series_description"]
                        mask_z = z
                        method = "plain_z_dim_match"
                    elif len(matches) > 1 and n_water == 1:
                        mask_source_desc = water_series[0]["series_description"]
                        mask_z = z
                        method = "single_water_only"
                        counter["single_water_no_choice"] += 1
                    elif n_water == 1:
                        mask_source_desc = water_series[0]["series_description"]
                        mask_z = z
                        method = "single_water_default"
                        counter["single_water_no_choice"] += 1
                    break
            if mask_source_desc is None and n_water == 1 and candidates:
                mask_source_desc = water_series[0]["series_description"]
                sh = nib.load(candidates[0][0]).shape
                mask_z = min(sh, key=lambda d: abs(d - int(water_series[0]["n_dcm_files"])))
                method = "single_water_fallback"
                counter["single_water_no_choice"] += 1

        if mask_source_desc is None:
            method = "NO_MATCH"
            counter["no_match"] += 1
        else:
            is_canon = (mask_source_desc == can["source_series_description"])
            if is_canon:
                counter["on_canonical"] += 1
            else:
                counter["on_alt"] += 1

        canonical_thickness = ""
        mask_source_thickness = ""
        for ws in water_series:
            if ws["series_description"] == can["source_series_description"]:
                canonical_thickness = str(ws["slice_thickness_mm"])
            if ws["series_description"] == mask_source_desc:
                mask_source_thickness = str(ws["slice_thickness_mm"])

        rows.append({
            "patient_id": pid,
            "n_water_series": n_water,
            "mask_inference_method": method or "",
            "mask_source_series_desc": mask_source_desc or "",
            "mask_z": mask_z if mask_z is not None else "",
            "mask_source_thickness_mm": mask_source_thickness,
            "canonical_series_desc": can["source_series_description"],
            "canonical_n_dcm": can["n_dcm_files_in_source"],
            "canonical_thickness_mm": canonical_thickness,
            "is_canonical": (mask_source_desc == can["source_series_description"]) if mask_source_desc else "",
            "alt_series_descs": " | ".join(
                sorted(ws["series_description"] for ws in water_series
                       if ws["series_description"] != can["source_series_description"])
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== Summary across {len(rows)} positives with mask ===")
    for k, v in counter.items():
        print(f"  {k}: {v}")
    print(f"\n=== Patients with mask on a NON-canonical series (n={counter['on_alt']}) ===")
    print(f"{'patient_id':<22} {'n_water':>3} {'mask_z':>6} {'mask_thk':>8} "
          f"{'can_dcm':>8} {'can_thk':>8} {'mask_series':<35} -> {'canonical_series':<35}")
    for r in rows:
        if r["is_canonical"] is False:
            print(f"{r['patient_id']:<22} {r['n_water_series']:>3} "
                  f"{str(r['mask_z']):>6} {r['mask_source_thickness_mm']:>8} "
                  f"{str(r['canonical_n_dcm']):>8} {r['canonical_thickness_mm']:>8} "
                  f"{r['mask_source_series_desc']:<35} -> {r['canonical_series_desc']:<35}")
    print(f"\n=== Patients where mask source could not be inferred (n={counter['no_match']}) ===")
    for r in rows:
        if r["mask_inference_method"] == "NO_MATCH":
            print(f"  {r['patient_id']} — n_water={r['n_water_series']}")
    print(f"\nFull audit CSV: {args.output}")


if __name__ == "__main__":
    main()
