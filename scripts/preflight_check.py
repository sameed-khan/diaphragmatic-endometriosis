"""Preflight: validate that source data referenced by the workplan is on disk.

Run this before submitting a SLURM array. It walks every workplan row whose
patient_id is in the supplied --patient-list (or every row if no list is given)
and verifies:
  - source_series_path exists and is a directory
  - it contains at least one .dcm-like file (matches DCM count vs n_dcm_files_in_source)
  - the existing-masks directory and existing-nifti directory are present
  - splits.json (if --splits-json given) covers every patient in --patient-list

Exits non-zero on the first batch of problems with a short summary so SLURM
tasks can fail fast before dcm2niix attempts.
"""
import argparse
import sys
from pathlib import Path

import polars as pl


def count_dcms(series_dir: Path) -> int:
    n = 0
    try:
        for p in series_dir.iterdir():
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() == ".dcm" or "." not in p.name:
                n += 1
    except OSError:
        return -1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--patient-list", type=Path,
                    help="Optional: scope check to patient IDs in this file")
    ap.add_argument("--existing-masks", type=Path,
                    default=Path("/scratch/pioneer/users/sak185/dia-endo-conversion/input/masks"))
    ap.add_argument("--existing-nifti", type=Path,
                    default=Path("/scratch/pioneer/users/sak185/dia-endo-conversion/input/nifti"))
    ap.add_argument("--splits-json", type=Path,
                    help="Optional: verify every patient in --patient-list "
                         "appears in splits.json assignments")
    ap.add_argument("--max-issues-shown", type=int, default=20)
    args = ap.parse_args()

    issues: list[str] = []

    # 1. Workplan must exist
    if not args.workplan.exists():
        print(f"FATAL: workplan not found: {args.workplan}", file=sys.stderr)
        return 2
    df = pl.read_csv(args.workplan, infer_schema_length=10000)

    # 2. Filter to patient list if provided
    if args.patient_list:
        if not args.patient_list.exists():
            print(f"FATAL: patient list not found: {args.patient_list}",
                  file=sys.stderr)
            return 2
        keep = {p for p in args.patient_list.read_text().split() if p}
        if not keep:
            print(f"FATAL: patient list is empty: {args.patient_list}",
                  file=sys.stderr)
            return 2
        df = df.filter(pl.col("patient_id").is_in(list(keep)))
        # Identify patient_ids in the list that have NO workplan rows
        present_pids = set(df["patient_id"].unique().to_list())
        missing_from_workplan = sorted(keep - present_pids)
        for pid in missing_from_workplan:
            issues.append(f"patient {pid}: in --patient-list but no workplan rows")

    # 3. Per-row source-data check
    n_rows = df.height
    n_missing_dir = 0
    n_empty_dir = 0
    n_count_mismatch = 0
    for r in df.iter_rows(named=True):
        src = Path(r["source_series_path"])
        if not src.is_dir():
            n_missing_dir += 1
            issues.append(f"{r['patient_id']} | {r['role']:<9} | "
                          f"source dir missing: {src}")
            continue
        actual = count_dcms(src)
        expected = int(r["n_dcm_files_in_source"] or 0)
        if actual <= 0:
            n_empty_dir += 1
            issues.append(f"{r['patient_id']} | {r['role']:<9} | "
                          f"source dir empty (0 dcm files): {src}")
        elif expected and actual != expected:
            # Not necessarily fatal — files may have been added/removed
            # since prescan — but worth surfacing.
            n_count_mismatch += 1
            issues.append(f"{r['patient_id']} | {r['role']:<9} | "
                          f"dcm count {actual} != prescan-expected {expected}: {src}")

    # 4. Flat dirs (masks, nifti) — only if positives are in scope
    has_pos = df.filter(pl.col("cohort") == "pos").height > 0
    if has_pos:
        if not args.existing_masks.is_dir():
            issues.append(f"existing-masks dir missing: {args.existing_masks}")
        elif not any(args.existing_masks.iterdir()):
            issues.append(f"existing-masks dir empty: {args.existing_masks}")
        if not args.existing_nifti.is_dir():
            issues.append(f"existing-nifti dir missing: {args.existing_nifti}")
        elif not any(args.existing_nifti.iterdir()):
            issues.append(f"existing-nifti dir empty: {args.existing_nifti}")

    # 5. splits.json coverage check
    if args.splits_json:
        if not args.splits_json.exists():
            issues.append(f"splits-json missing: {args.splits_json}")
        else:
            import json
            doc = json.loads(args.splits_json.read_text())
            assigned = set(doc.get("assignments", {}).keys())
            patient_pids = set(df["patient_id"].unique().to_list())
            unassigned = sorted(patient_pids - assigned)
            for pid in unassigned:
                issues.append(f"patient {pid}: not in splits.json assignments")

    # === Summary ===
    print(f"Preflight checked {n_rows} workplan rows "
          f"({df['patient_id'].n_unique()} unique patients).")
    if not issues:
        print("OK — all source data present.")
        return 0

    fatal = n_missing_dir + n_empty_dir
    print(f"\n{len(issues)} issue(s) found "
          f"({n_missing_dir} missing dirs, {n_empty_dir} empty dirs, "
          f"{n_count_mismatch} dcm-count mismatches).")
    print(f"\nFirst {min(args.max_issues_shown, len(issues))} issue(s):")
    for line in issues[:args.max_issues_shown]:
        print(f"  {line}")
    if len(issues) > args.max_issues_shown:
        print(f"  ... and {len(issues) - args.max_issues_shown} more")

    # Exit non-zero only on fatal issues (missing/empty source dirs).
    # Count mismatches are warnings.
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
