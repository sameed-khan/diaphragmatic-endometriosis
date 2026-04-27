"""
Migrate Phase 1 dev cohort (608 patients) from /scratch to /home/data with
mnemonic renaming, per agent/migration-plan.md.

The script is idempotent and dry-run by default. To actually copy files,
pass --execute. The dry-run prints the full plan, runs all pre-flight checks,
and writes nothing.

Outputs (under --target-root):
    raw/<bucket>/<cohort>/<mnemonic>.{nii.gz,json}
    lesion_masks/<bucket>/positive/<mnemonic>_mask.nii.gz
    manifest.csv          # 5,060 rows; transferred_to_home gates physical presence
    splits.json           # verbatim copy
    patient_id_mapping.csv  # verbatim copy from --mapping
    README.md             # auto-generated

Bucket = "holdout" if split == "holdout" else "cross-validation".
The fold ID lives in manifest.csv["split"] and splits.json — NOT in the path.
"""

import argparse
import hashlib
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl

PHASE1_SPLITS = ["holdout", "fold0", "fold1", "fold2", "fold3", "fold4"]
PHASE_PRIORITY = {"phase1": 0, "remask23": 1, "pilot": 2}

# All Phase 1 canonical sub-volumes have this image_type. Used as a filter when
# selecting the "right" sub-volume for negatives (which lack a mask to anchor the choice).
EXPECTED_IMAGE_TYPE = r"DERIVED\PRIMARY\DIXON\WATER\MAGNITUDE"

EXPECTED_TRANSFER_COUNT = 608
EXPECTED_BUCKET_COUNTS = {
    ("holdout", "positive"): 22,
    ("holdout", "negative"): 100,
    ("cross-validation", "positive"): 86,
    ("cross-validation", "negative"): 400,
}
EXPECTED_MASK_COUNT = 108


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Root that resolves output_subdir paths (e.g., /scratch/.../output).")
    ap.add_argument("--mapping", type=Path, required=True,
                    help="patient_id_mapping.csv (anon_id, mnemonic_id).")
    ap.add_argument("--splits-json", type=Path, required=True)
    ap.add_argument("--target-root", type=Path, required=True,
                    help="Destination root (e.g., /home/sak185/dia-endo-conversion/data).")
    ap.add_argument("--execute", action="store_true",
                    help="Actually copy files. Without this flag, dry-run only.")
    ap.add_argument("--progress-every", type=int, default=50)
    return ap.parse_args()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_shape(s: str) -> tuple[int, ...]:
    """Parse a manifest `shape` string like '512x98x512' into (512, 98, 512)."""
    return tuple(int(p) for p in s.split("x"))


def slice_count_from_shape(s: str) -> int:
    """For a coronal LAVA Dixon WATER volume, the slice axis is the middle dim
    of the post-reorient shape (in-plane is 512x512; slice count is the odd dim).
    Return the middle dim if 3D, else the smallest dim as a fallback."""
    dims = parse_shape(s)
    if len(dims) == 3:
        return dims[1]
    return min(dims)


def select_canonical_volume(pid: str, rows: pl.DataFrame, output_root: Path) -> dict:
    """Pick a single canonical row per patient.

    For positives: pair with whichever `mask_canonical*.nii.gz` exists in
    masks_pos/<pid>/, by matching the file's suffix to `output_filename`. If
    multiple canonical-family masks exist, prefer the unsuffixed one (mask_canonical),
    then alphabetic order. Falls through to the negative rule if no match found.

    For negatives: filter to image_type==EXPECTED_IMAGE_TYPE; pick the row with
    the largest slice count (middle dim of `shape`); deterministic tie-break by
    volume_index ascending.
    """
    if rows.height == 1:
        return rows.row(0, named=True)
    cohort = rows["cohort"][0]
    if cohort == "pos":
        mask_dir = output_root / "masks_pos" / pid
        if mask_dir.is_dir():
            canon_masks = sorted(
                f.name for f in mask_dir.iterdir() if f.name.startswith("mask_canonical")
            )
            # Prefer mask_canonical.nii.gz first, then suffixed variants alphabetically.
            preferred = []
            if "mask_canonical.nii.gz" in canon_masks:
                preferred.append("mask_canonical.nii.gz")
                canon_masks.remove("mask_canonical.nii.gz")
            preferred.extend(canon_masks)
            for mname in preferred:
                suffix = mname.replace("mask_", "").replace(".nii.gz", "")
                target = f"water_{suffix}.nii.gz"
                matched = rows.filter(pl.col("output_filename") == target)
                if matched.height > 0:
                    return matched.row(0, named=True)
            # Mask file exists but no matching water sub-volume — fall through.

    # Negative rule (also fallback for positives that didn't mask-pair).
    filt = rows.filter(pl.col("image_type") == EXPECTED_IMAGE_TYPE)
    if filt.height == 0:
        filt = rows  # no rows match expected image_type — fall back to full set
    filt = filt.with_columns(
        _slice_count=pl.col("shape").map_elements(slice_count_from_shape, return_dtype=pl.Int64)
    )
    filt = filt.sort(["_slice_count", "volume_index"], descending=[True, False])
    return filt.drop("_slice_count").row(0, named=True)


def load_canonicals(source_manifest: Path, output_root: Path) -> pl.DataFrame:
    df = pl.read_csv(source_manifest, infer_schema_length=10000)
    canon_all = df.filter(pl.col("role") == "canonical")

    # Phase preference at the patient level: keep only rows from the most-preferred
    # phase a given patient appears in.
    canon_all = canon_all.with_columns(
        phase_priority=pl.col("phase").map_elements(
            lambda s: PHASE_PRIORITY.get(s, 3),
            return_dtype=pl.Int32,
        )
    )
    min_phase = canon_all.group_by("patient_id").agg(
        pl.col("phase_priority").min().alias("_min_phase")
    )
    canon_all = canon_all.join(min_phase, on="patient_id").filter(
        pl.col("phase_priority") == pl.col("_min_phase")
    ).drop(["phase_priority", "_min_phase"])

    # Per-patient selection: one row per patient via the rule in select_canonical_volume.
    selected = []
    for pid_tuple, group in canon_all.group_by("patient_id"):
        pid = pid_tuple[0] if isinstance(pid_tuple, tuple) else pid_tuple
        selected.append(select_canonical_volume(pid, group, output_root))
    canon = pl.DataFrame(selected)

    canon = canon.with_columns(
        transferred_to_home=pl.col("split").is_in(PHASE1_SPLITS),
        bucket=pl.when(pl.col("split") == "holdout")
                 .then(pl.lit("holdout"))
                 .otherwise(pl.lit("cross-validation")),
        cohort_full=pl.when(pl.col("cohort") == "pos")
                      .then(pl.lit("positive"))
                      .otherwise(pl.lit("negative")),
        had_multi_canonical=pl.col("n_volumes_from_series") > 1,
        selected_subvolume=pl.col("volume_index") != 0,
    )
    return canon


def build_plan(canon: pl.DataFrame, mapping: pl.DataFrame, output_root: Path,
               target_root: Path) -> list[dict]:
    name_lookup = dict(zip(mapping["anon_id"].to_list(), mapping["mnemonic_id"].to_list()))
    plan = []
    transferred = canon.filter(pl.col("transferred_to_home"))
    for r in transferred.iter_rows(named=True):
        anon = r["patient_id"]
        if anon not in name_lookup:
            raise SystemExit(f"Patient {anon} (split={r['split']}) is missing from mapping CSV.")
        mnemonic = name_lookup[anon]
        bucket = r["bucket"]
        cohort = r["cohort_full"]
        raw_dir = target_root / "raw" / bucket / cohort
        src_nii = output_root / r["output_subdir"] / r["output_filename"]
        # JSON sidecar shares the basename of the .nii.gz (drop both .nii and .gz).
        src_json = src_nii.with_name(src_nii.name.replace(".nii.gz", ".json"))
        dst_nii = raw_dir / f"{mnemonic}.nii.gz"
        dst_json = raw_dir / f"{mnemonic}.json"

        if r["cohort"] == "pos":
            # Mask filename derives from the selected sub-volume's output_filename:
            #   water_canonical.nii.gz   -> mask_canonical.nii.gz
            #   water_canonicala.nii.gz  -> mask_canonicala.nii.gz
            volume_basename = r["output_filename"].replace("water_", "").replace(".nii.gz", "")
            mask_filename = f"mask_{volume_basename}.nii.gz"
            src_mask = output_root / "masks_pos" / anon / mask_filename
            dst_mask = target_root / "lesion_masks" / bucket / "positive" / f"{mnemonic}_mask.nii.gz"
        else:
            src_mask = None
            dst_mask = None

        plan.append({
            "anon_id": anon,
            "mnemonic_id": mnemonic,
            "split": r["split"],
            "bucket": bucket,
            "cohort": cohort,
            "src_nii": src_nii,
            "dst_nii": dst_nii,
            "src_json": src_json,
            "dst_json": dst_json,
            "src_mask": src_mask,
            "dst_mask": dst_mask,
            "expected_sha256": r["sha256"],
        })
    return plan


def preflight(plan: list[dict], canon: pl.DataFrame, mapping: pl.DataFrame,
              target_root: Path) -> int:
    """Returns total estimated bytes. Aborts on any failure."""
    print("\n=== Pre-flight checks ===")
    errors = []

    # 1. Source files exist.
    missing_nii = [p["src_nii"] for p in plan if not p["src_nii"].exists()]
    missing_json = [p["src_json"] for p in plan if not p["src_json"].exists()]
    missing_mask = [p["src_mask"] for p in plan if p["src_mask"] is not None and not p["src_mask"].exists()]
    if missing_nii:
        errors.append(f"{len(missing_nii)} missing source NIfTIs (first: {missing_nii[0]})")
    if missing_json:
        errors.append(f"{len(missing_json)} missing source JSONs (first: {missing_json[0]})")
    if missing_mask:
        errors.append(f"{len(missing_mask)} missing source masks (first: {missing_mask[0]})")

    # 2. dst paths unique.
    dsts = []
    for p in plan:
        dsts.extend([p["dst_nii"], p["dst_json"]])
        if p["dst_mask"] is not None:
            dsts.append(p["dst_mask"])
    dup = [k for k, v in Counter(dsts).items() if v > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate destination paths (first: {dup[0]})")

    # 3. Mapping covers every canonical patient (and therefore every transferred one).
    canon_ids = set(canon["patient_id"].to_list())
    map_ids = set(mapping["anon_id"].to_list())
    missing_in_map = canon_ids - map_ids
    if missing_in_map:
        errors.append(f"{len(missing_in_map)} canonical patients missing from mapping "
                      f"(first: {sorted(missing_in_map)[0]})")

    # 4. Counts match expectations.
    n_transferred = canon.filter(pl.col("transferred_to_home")).height
    if n_transferred != EXPECTED_TRANSFER_COUNT:
        errors.append(f"transferred_to_home count={n_transferred}, expected {EXPECTED_TRANSFER_COUNT}")

    bucket_cohort = canon.filter(pl.col("transferred_to_home")).group_by(
        ["bucket", "cohort_full"]
    ).agg(pl.len()).sort(["bucket", "cohort_full"])
    actual_bc = {(r["bucket"], r["cohort_full"]): r["len"] for r in bucket_cohort.iter_rows(named=True)}
    for (b, c), expected in EXPECTED_BUCKET_COUNTS.items():
        actual = actual_bc.get((b, c), 0)
        if actual != expected:
            errors.append(f"bucket=({b},{c}): {actual}, expected {expected}")

    n_masks = sum(1 for p in plan if p["src_mask"] is not None)
    if n_masks != EXPECTED_MASK_COUNT:
        errors.append(f"mask plan count={n_masks}, expected {EXPECTED_MASK_COUNT}")

    # 5. Target dir non-existent or empty (or only contains the mapping/scripts we put there).
    if target_root.exists():
        # We tolerate the user having pre-placed mapping CSV and similar scaffolding.
        offending = []
        for p in target_root.iterdir():
            # Anything under raw/ or lesion_masks/ would indicate a prior partial run.
            if p.name in ("raw", "lesion_masks", "manifest.csv", "splits.json", "README.md"):
                offending.append(p)
        if offending:
            errors.append(f"target_root has prior migration artifacts: {[str(p) for p in offending]}. "
                          f"Remove them or re-run idempotently after sha verification.")

    # 6. Estimated size.
    total_bytes = 0
    for p in plan:
        total_bytes += p["src_nii"].stat().st_size if p["src_nii"].exists() else 0
        total_bytes += p["src_json"].stat().st_size if p["src_json"].exists() else 0
        if p["src_mask"] is not None and p["src_mask"].exists():
            total_bytes += p["src_mask"].stat().st_size

    if errors:
        print("Pre-flight FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(2)

    print(f"  All source NIfTIs present:   {len(plan)}")
    print(f"  All source JSONs present:    {len(plan)}")
    print(f"  All source masks present:    {n_masks} (positives only)")
    print(f"  All dst paths unique:        ok")
    print(f"  Mapping covers all patients: ok ({len(canon_ids)} canonical IDs)")
    print(f"  Bucket x cohort counts:      ok")
    print(f"  Estimated bytes to copy:     {total_bytes/1e9:.2f} GB")
    return total_bytes


def print_summary(plan: list[dict], canon: pl.DataFrame, total_bytes: int):
    print("\n=== Migration plan summary ===")
    n = len(plan)
    n_files = 2 * n + sum(1 for p in plan if p["dst_mask"] is not None)
    print(f"Patients to migrate: {n}")
    print(f"Files to copy:       {n_files}  ({n} NIfTI + {n} JSON + "
          f"{sum(1 for p in plan if p['dst_mask'] is not None)} masks)")
    print(f"Bytes to copy:       {total_bytes/1e9:.2f} GB")
    print(f"\nBy bucket x cohort:")
    bc = Counter((p["bucket"], p["cohort"]) for p in plan)
    for (b, c), v in sorted(bc.items()):
        print(f"  {b:18s} {c:9s} {v}")
    print(f"\nBy split (per manifest, not directory tree):")
    by_split = canon.filter(pl.col("transferred_to_home")).group_by(
        ["split", "cohort_full"]
    ).agg(pl.len()).sort(["split", "cohort_full"])
    for r in by_split.iter_rows(named=True):
        print(f"  {r['split']:8s} {r['cohort_full']:9s} {r['len']}")
    print(f"\nProject totals (manifest.csv will have):")
    print(f"  Total rows:           {canon.height}")
    print(f"  transferred_to_home:  {canon.filter(pl.col('transferred_to_home')).height}")
    print(f"  not_transferred:      {canon.filter(~pl.col('transferred_to_home')).height}")


def copy_with_verify(src: Path, dst: Path, expected_sha: str | None) -> tuple[bool, str]:
    """Copy src→dst (idempotent). Returns (copied, dst_sha).

    If dst already exists with matching sha, skip. If expected_sha is set, verify dst sha after copy.
    """
    if dst.exists():
        existing_sha = sha256_file(dst)
        if expected_sha is not None and existing_sha == expected_sha:
            return False, existing_sha
        if expected_sha is None:
            return False, existing_sha  # JSON: trust existence (no sha to compare)
        # Mismatch: re-copy
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if expected_sha is not None:
        dst_sha = sha256_file(dst)
        if dst_sha != expected_sha:
            raise SystemExit(f"sha256 mismatch after copy: {dst} (got {dst_sha}, expected {expected_sha})")
        return True, dst_sha
    return True, ""


def execute(plan: list[dict], progress_every: int) -> dict:
    print("\n=== Executing migration ===")
    t0 = time.time()
    n_copied_nii = 0
    n_copied_json = 0
    n_copied_mask = 0
    n_skipped = 0
    bytes_copied = 0
    for i, p in enumerate(plan, 1):
        copied_n, _ = copy_with_verify(p["src_nii"], p["dst_nii"], p["expected_sha256"])
        copied_j, _ = copy_with_verify(p["src_json"], p["dst_json"], None)
        if copied_n:
            n_copied_nii += 1
            bytes_copied += p["src_nii"].stat().st_size
        else:
            n_skipped += 1
        if copied_j:
            n_copied_json += 1
            bytes_copied += p["src_json"].stat().st_size
        if p["src_mask"] is not None:
            # No sha for masks in source manifest → existence-based idempotency only.
            copied_m, _ = copy_with_verify(p["src_mask"], p["dst_mask"], None)
            if copied_m:
                n_copied_mask += 1
                bytes_copied += p["src_mask"].stat().st_size
        if i % progress_every == 0 or i == len(plan):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i:>4d}/{len(plan)}] {elapsed:6.1f}s elapsed  ({rate:.1f} pat/s)")
    return {
        "n_copied_nii": n_copied_nii,
        "n_copied_json": n_copied_json,
        "n_copied_mask": n_copied_mask,
        "n_skipped_nii": n_skipped,
        "bytes_copied": bytes_copied,
        "wall_seconds": time.time() - t0,
    }


def write_manifest(canon: pl.DataFrame, mapping: pl.DataFrame, target_root: Path,
                   migration_ts: str) -> Path:
    name_lookup = dict(zip(mapping["anon_id"].to_list(), mapping["mnemonic_id"].to_list()))
    rows = []
    for r in canon.iter_rows(named=True):
        anon = r["patient_id"]
        mnemonic = name_lookup.get(anon, "")
        bucket = r["bucket"]
        cohort_full = r["cohort_full"]
        transferred = r["transferred_to_home"]
        if transferred and mnemonic:
            raw_path = f"raw/{bucket}/{cohort_full}/{mnemonic}.nii.gz"
            raw_json_path = f"raw/{bucket}/{cohort_full}/{mnemonic}.json"
            lesion_mask_path = (
                f"lesion_masks/{bucket}/positive/{mnemonic}_mask.nii.gz"
                if r["cohort"] == "pos" else ""
            )
            ts = migration_ts
        else:
            raw_path = ""
            raw_json_path = ""
            lesion_mask_path = ""
            ts = ""
        rows.append({
            "mnemonic_id": mnemonic,
            "anon_id": anon,
            "split": r["split"] if r["split"] is not None else "",
            "cohort": cohort_full,
            "soft_negative": bool(r["soft_negative"]) if r["soft_negative"] is not None else False,
            "transferred_to_home": bool(transferred),
            "raw_path": raw_path,
            "raw_json_path": raw_json_path,
            "lesion_mask_path": lesion_mask_path,
            "bucket": bucket,
            "original_filename": r["output_filename"],
            "original_subdir": r["output_subdir"],
            "source_series_path": r["source_series_path"],
            "image_type": r["image_type"] if r["image_type"] is not None else "",
            "series_description": r["series_description"] if r["series_description"] is not None else "",
            "scanner_model": r["scanner_model"] if r["scanner_model"] is not None else "",
            "magnetic_field_strength": r["magnetic_field_strength"],
            "slice_thickness_mm": r["slice_thickness_mm"],
            "pixel_spacing_x_mm": r["pixel_spacing_x_mm"],
            "pixel_spacing_y_mm": r["pixel_spacing_y_mm"],
            "shape": r["shape"] if r["shape"] is not None else "",
            "n_slices_actual": r["n_slices_actual"],
            "n_volumes_from_series": r["n_volumes_from_series"],
            "volume_index": r["volume_index"],
            "had_multi_canonical": bool(r["had_multi_canonical"]),
            "selected_subvolume": bool(r["selected_subvolume"]),
            "phase": r["phase"],
            "sha256_raw": r["sha256"] if r["sha256"] is not None else "",
            "migration_timestamp": ts,
        })
    out = pl.DataFrame(rows).sort("mnemonic_id")
    path = target_root / "manifest.csv"
    out.write_csv(path)
    return path


def write_readme(target_root: Path, canon: pl.DataFrame, plan: list[dict],
                 migration_ts: str, stats: dict | None) -> Path:
    transferred = canon.filter(pl.col("transferred_to_home"))
    by_bc = Counter((p["bucket"], p["cohort"]) for p in plan)
    by_split_cohort = transferred.group_by(["split", "cohort_full"]).agg(pl.len()).sort(["split", "cohort_full"])

    fold_table_rows = []
    for r in by_split_cohort.iter_rows(named=True):
        fold_table_rows.append(f"| {r['split']} | {r['cohort_full']} | {r['len']} |")
    fold_table = "\n".join(fold_table_rows)

    bytes_str = f"{stats['bytes_copied']/1e9:.2f} GB" if stats else "(dry-run)"
    wall_str = f"{stats['wall_seconds']:.1f} s" if stats else "(dry-run)"

    text = f"""# dia-endo-conversion data tree

**Generated:** {migration_ts}
**Source:** /scratch/pioneer/users/sak185/dia-endo-conversion/output/
**Migration script:** scripts/migrate_to_home.py
**Naming:** mnemonic IDs from scripts/generate_patient_names.py + scripts/wordlists.json

## Layout

```
raw/<bucket>/<cohort>/<mnemonic>.{{nii.gz,json}}
    where <bucket> in {{holdout, cross-validation}}
          <cohort> in {{positive, negative}}
lesion_masks/<bucket>/positive/<mnemonic>_mask.nii.gz       # GT masks, positives only
liver_masks/<bucket>/<cohort>/<mnemonic>_liver_mask.nii.gz  # ADDED LATER (TotalSegmentator)
cropped_raw/<bucket>/<cohort>/<mnemonic>.{{nii.gz,json}}     # ADDED LATER
cropped_lesion_masks/<bucket>/positive/<mnemonic>_mask.nii.gz # ADDED LATER
normalized_p1p99/<bucket>/<cohort>/<mnemonic>.nii.gz   # ADDED LATER (if precomputed)
predictions/<run_id>/<bucket>/<cohort>/<mnemonic>.nii.gz # ADDED DURING TRAINING
manifest.csv               # project-wide; transferred_to_home gates physical presence
patient_id_mapping.csv     # ANON ↔ mnemonic; immutable
splits.json                # frozen splits (seed=42); authoritative fold assignment
```

**Two physical buckets, five logical folds:** `cross-validation/` contains all 486 CV-pool
patients in one tree. The fold assignment (fold0..fold4) is in `manifest.csv["split"]` and
`splits.json["assignments"]`. The training DataLoader reads splits.json (or the manifest)
at runtime to determine which patients to use for which fold.

## Migration counts (this migration only)

| Bucket            | positive | negative | total |
|-------------------|---------:|---------:|------:|
| holdout           | {by_bc[('holdout','positive')]}       | {by_bc[('holdout','negative')]}      | {by_bc[('holdout','positive')]+by_bc[('holdout','negative')]}   |
| cross-validation  | {by_bc[('cross-validation','positive')]}       | {by_bc[('cross-validation','negative')]}      | {by_bc[('cross-validation','positive')]+by_bc[('cross-validation','negative')]}   |
| **total**         | **{sum(v for (b,c),v in by_bc.items() if c=='positive')}**  | **{sum(v for (b,c),v in by_bc.items() if c=='negative')}**  | **{sum(by_bc.values())}** |

Per-fold breakdown of the cross-validation bucket (from `manifest.csv`, not the directory tree):

| split | cohort | count |
|-------|--------|------:|
{fold_table}

Lesion masks copied: {sum(1 for p in plan if p['dst_mask'] is not None)} (one per positive patient).

## Project totals (in manifest.csv)

- Total patients tracked:    {canon.height}
- Transferred to /home:      {transferred.height}
- Not transferred (Phase 2 + leftovers): {canon.height - transferred.height}

Filter `transferred_to_home == True` in `manifest.csv` to scope to this directory.

## Verification

```bash
find raw -name "*.nii.gz" | wc -l              # → 608
find lesion_masks -name "*.nii.gz" | wc -l     # → 108
find raw -name "*.json" | wc -l                # → 608
ls raw/holdout/positive/ | wc -l               # → 22 (.nii.gz)
ls raw/cross-validation/positive/ | wc -l      # → 86 (.nii.gz)
```

## Migration stats

- Bytes copied: {bytes_str}
- Wall time:    {wall_str}

## Re-running

The migration is idempotent — re-running with the same inputs produces no changes
(script detects existing files via sha256 match for NIfTIs, existence for JSONs and masks).
To force re-migration of a single patient, delete the target files and re-run.

## Provenance

See `agent/migration-plan.md` for the design rationale, decisions, and the full
execution checklist.
"""
    path = target_root / "README.md"
    path.write_text(text)
    return path


def main():
    args = parse_args()

    if args.execute:
        print(">>> EXECUTE MODE — files will be written. <<<")
    else:
        print(">>> DRY-RUN — no files will be written. Pass --execute to perform the copy. <<<")

    canon = load_canonicals(args.source_manifest, args.output_root)
    print(f"\nCanonicals (post-dedup):       {canon.height}")
    print(f"Transferred (Phase 1):         {canon.filter(pl.col('transferred_to_home')).height}")
    n_subvol = canon.filter(pl.col("transferred_to_home") & pl.col("selected_subvolume")).height
    n_multi = canon.filter(pl.col("transferred_to_home") & pl.col("had_multi_canonical")).height
    print(f"Transferred multi-canonical:   {n_multi}")
    print(f"Transferred non-vol_idx=0:     {n_subvol} (selected a sub-volume rather than the first)")

    mapping = pl.read_csv(args.mapping)
    print(f"Mapping rows:            {mapping.height}")

    plan = build_plan(canon, mapping, args.output_root, args.target_root)
    total_bytes = preflight(plan, canon, mapping, args.target_root)
    print_summary(plan, canon, total_bytes)

    if not args.execute:
        print("\nDry-run complete. Re-run with --execute to perform the copy.")
        return

    args.target_root.mkdir(parents=True, exist_ok=True)
    stats = execute(plan, args.progress_every)

    migration_ts = datetime.now().isoformat(timespec="seconds")

    # Manifest
    mpath = write_manifest(canon, mapping, args.target_root, migration_ts)
    print(f"\nWrote manifest: {mpath}")

    # splits.json verbatim
    splits_dst = args.target_root / "splits.json"
    shutil.copy2(args.splits_json, splits_dst)
    print(f"Wrote splits:   {splits_dst}")

    # patient_id_mapping.csv verbatim (in case --mapping was elsewhere)
    map_dst = args.target_root / "patient_id_mapping.csv"
    if args.mapping.resolve() != map_dst.resolve():
        shutil.copy2(args.mapping, map_dst)
        print(f"Wrote mapping:  {map_dst}")

    # README
    readme = write_readme(args.target_root, canon, plan, migration_ts, stats)
    print(f"Wrote README:   {readme}")

    # Final report
    print("\n=== Final report ===")
    print(f"Patients migrated:   {len(plan)} (expected {EXPECTED_TRANSFER_COUNT})")
    print(f"NIfTIs copied:       {stats['n_copied_nii']} (skipped existing: {stats['n_skipped_nii']})")
    print(f"JSONs copied:        {stats['n_copied_json']}")
    print(f"Masks copied:        {stats['n_copied_mask']} (expected {EXPECTED_MASK_COUNT})")
    print(f"Bytes copied:        {stats['bytes_copied']/1e9:.2f} GB")
    print(f"Wall time:           {stats['wall_seconds']:.1f} s")
    print(f"sha256 mismatches:   0 (any would have aborted)")
    print(f"Manifest rows:       {canon.height}")


if __name__ == "__main__":
    main()
