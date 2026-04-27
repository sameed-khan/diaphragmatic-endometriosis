"""
Rename all patient files from ANON IDs to mnemonic IDs.

Renames files in:
- nifti/*.nii.gz
- masks/*.nii.gz
- masks/*.csv

Uses the mapping from patient_id_mapping.csv.
For WATER files, replaces only the patient ID prefix, preserving the WATER suffix.

Dry-run by default. Pass --execute to actually rename.
"""

from pathlib import Path
import polars as pl
import sys

NIFTI_DIR = Path("nifti")
MASK_DIR = Path("masks")
MAPPING_PATH = Path("patient_id_mapping.csv")

def main():
    execute = "--execute" in sys.argv
    if not execute:
        print("=== DRY RUN (pass --execute to rename) ===\n")

    mapping = pl.read_csv(MAPPING_PATH)
    id_map = dict(zip(
        mapping["anon_id"].to_list(),
        mapping["mnemonic_id"].to_list(),
    ))

    renames = []  # list of (old_path, new_path)

    # Process all target directories and extensions
    targets = [
        (NIFTI_DIR, "*.nii.gz"),
        (MASK_DIR, "*.nii.gz"),
        (MASK_DIR, "*.csv"),
    ]

    for directory, pattern in targets:
        for fpath in sorted(directory.glob(pattern)):
            old_name = fpath.name
            # Find which ANON ID this file belongs to
            matched_pid = None
            for anon_id in id_map:
                if old_name.startswith(anon_id):
                    matched_pid = anon_id
                    break

            if matched_pid is None:
                print(f"  WARNING: No mapping for {fpath} — skipping")
                continue

            mnemonic = id_map[matched_pid]
            new_name = old_name.replace(matched_pid, mnemonic, 1)
            new_path = fpath.parent / new_name
            renames.append((fpath, new_path))

    # Print summary
    print(f"Total renames: {len(renames)}")
    for old, new in renames:
        print(f"  {old.name}  ->  {new.name}")

    # Check for conflicts within each directory
    by_dir: dict[Path, list[str]] = {}
    for _, new in renames:
        by_dir.setdefault(new.parent, []).append(new.name)
    for d, names in by_dir.items():
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            print(f"\nERROR: Duplicate targets in {d}: {set(dupes)}")
            sys.exit(1)

    # Check no target already exists
    for old, new in renames:
        if new.exists() and old != new:
            print(f"\nERROR: Target already exists: {new}")
            sys.exit(1)

    if execute:
        print(f"\nExecuting {len(renames)} renames...")
        for old, new in renames:
            old.rename(new)
        print("Done.")
    else:
        print(f"\nDry run complete. Run with --execute to apply.")


if __name__ == "__main__":
    main()
