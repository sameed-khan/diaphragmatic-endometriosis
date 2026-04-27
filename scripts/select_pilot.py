"""Auto-pick ~12 deterministic pilot patients from workplan.csv.

Picks: 5 single-WATER neg, 3 multi-WATER neg, 2 FAT-bearing patients,
2 positives (one extended-style nifti filename, one plain).

Determinism: within each category, sort by patient_id and take the first N.
"""
import argparse
import re
import sys
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--alignment-audit", type=Path, required=True)
    ap.add_argument("--existing-nifti", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    plan = pl.read_csv(args.workplan, infer_schema_length=10000)

    # WATER series counts per (cohort, patient_id).
    water_counts = (
        plan.filter(pl.col("image_type_token") == "WATER")
            .group_by(["cohort", "patient_id"])
            .agg(pl.len().alias("n_water"))
    )
    fat_pids = (
        plan.filter(pl.col("image_type_token") == "FAT")
            .select("patient_id").unique()["patient_id"].to_list()
    )

    neg_water = water_counts.filter(pl.col("cohort") == "neg")
    pos_water = water_counts.filter(pl.col("cohort") == "pos")

    single_neg = sorted(
        neg_water.filter(pl.col("n_water") == 1)["patient_id"].to_list()
    )[:5]
    multi_neg = sorted(
        neg_water.filter(pl.col("n_water") > 1)["patient_id"].to_list()
    )[:3]
    # Prefer FAT-bearing negatives, fall back to any FAT-bearing.
    neg_pids_set = set(neg_water["patient_id"].to_list())
    fat_neg = sorted(p for p in fat_pids if p in neg_pids_set)[:2]

    # Positives: pick distinct patients across categories so the pilot exercises
    # all three mask-source code paths plus the no-mask path.
    nifti_files = list(args.existing_nifti.glob("*.nii.gz"))
    plain_set, extended_set = set(), set()
    for f in nifti_files:
        m = re.match(r"^(ANON[A-F0-9]+)(?:_(.*))?\.nii\.gz$", f.name)
        if not m:
            continue
        if m.group(2) is None:
            plain_set.add(m.group(1))
        else:
            extended_set.add(m.group(1))
    pos_pids_set = set(pos_water["patient_id"].to_list())
    plain_only = sorted((plain_set - extended_set) & pos_pids_set)[:1]
    extended_only = sorted((extended_set - plain_set) & pos_pids_set)[:1]
    both_styles = sorted((plain_set & extended_set) & pos_pids_set)[:1]
    no_nifti = sorted(pos_pids_set - (plain_set | extended_set))[:1]

    pilot = sorted(set(single_neg + multi_neg + fat_neg
                       + plain_only + extended_only + both_styles + no_nifti))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(pilot) + "\n")
    print(f"Wrote {len(pilot)} pilot patient IDs to {args.output}")
    print("Selection breakdown:")
    print(f"  single-WATER neg:        {single_neg}")
    print(f"  multi-WATER neg:         {multi_neg}")
    print(f"  FAT-bearing neg:         {fat_neg}")
    print(f"  pos (plain-only nifti):  {plain_only}")
    print(f"  pos (extended-only):     {extended_only}")
    print(f"  pos (both styles):       {both_styles}")
    print(f"  pos (no nifti/mask):     {no_nifti}")


if __name__ == "__main__":
    main()
