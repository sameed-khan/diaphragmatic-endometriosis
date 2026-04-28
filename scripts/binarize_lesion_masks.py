"""Destructively binarize lesion masks in-place.

Collapses any nonzero voxel value to 1, preserving NIfTI affine/header geometry.
Writes a per-file CSV report to eda/outputs/binarize_lesion_masks_report.csv.

Run with:  uv run -m scripts.binarize_lesion_masks
"""

from __future__ import annotations

import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
REPORT_PATH = PROJECT_ROOT / "eda" / "outputs" / "binarize_lesion_masks_report.csv"

EXPECTED_LABELS = {0, 1, 2}
MAX_WORKERS = 16


def _process_one(mnemonic_id: str, rel_path: str) -> dict:
    """Load one mask, binarize in place if needed, return a record dict."""
    record: dict = {
        "mnemonic_id": mnemonic_id,
        "path": rel_path,
        "was_already_binary": False,
        "unique_values_before": "",
        "voxels_label_1_before": 0,
        "voxels_label_2_before": 0,
        "voxels_other_before": 0,
        "total_nonzero_before": 0,
        "total_nonzero_after": 0,
        "shape": "",
        "dtype_before": "",
        "dtype_after": "",
        "write_succeeded": False,
        "verification_passed": False,
        "error": "",
    }

    abs_path = DATA_DIR / rel_path
    try:
        if not abs_path.exists():
            record["error"] = f"file not found: {abs_path}"
            return record

        img = nib.load(str(abs_path))
        arr = np.asarray(img.dataobj)

        unique_vals, counts = np.unique(arr, return_counts=True)
        unique_set = {int(v) for v in unique_vals.tolist()}
        record["unique_values_before"] = ",".join(str(v) for v in sorted(unique_set))
        record["shape"] = "x".join(str(s) for s in arr.shape)
        record["dtype_before"] = str(arr.dtype)

        per_label = {int(v): int(c) for v, c in zip(unique_vals.tolist(), counts.tolist())}
        record["voxels_label_1_before"] = per_label.get(1, 0)
        record["voxels_label_2_before"] = per_label.get(2, 0)
        record["voxels_other_before"] = sum(
            c for v, c in per_label.items() if v not in (0, 1, 2)
        )
        record["total_nonzero_before"] = int(np.count_nonzero(arr))

        # Already binary?
        if unique_set.issubset({0, 1}):
            record["was_already_binary"] = True
            record["dtype_after"] = record["dtype_before"]
            record["total_nonzero_after"] = record["total_nonzero_before"]
            # Verify it really is binary (sanity)
            record["verification_passed"] = True
            record["write_succeeded"] = False  # no write needed
            return record

        # Safety: refuse if unexpected labels present
        if not unique_set.issubset(EXPECTED_LABELS):
            unexpected = sorted(unique_set - EXPECTED_LABELS)
            record["error"] = (
                f"unexpected label values present: {unexpected}; skipping write"
            )
            return record

        # Binarize
        new_arr = (arr > 0).astype(np.uint8)

        # Build new image preserving affine + header geometry; force dtype uint8
        new_header = img.header.copy()
        new_header.set_data_dtype(np.uint8)
        new_img = nib.Nifti1Image(new_arr, img.affine, new_header)

        nib.save(new_img, str(abs_path))
        record["write_succeeded"] = True

        # Verify by reloading
        reloaded = nib.load(str(abs_path))
        r_arr = np.asarray(reloaded.dataobj)
        r_unique = {int(v) for v in np.unique(r_arr).tolist()}
        r_shape = tuple(r_arr.shape)
        if r_unique.issubset({0, 1}) and r_shape == arr.shape:
            record["verification_passed"] = True
        else:
            record["error"] = (
                f"verification failed: unique={sorted(r_unique)} shape={r_shape}"
            )

        record["total_nonzero_after"] = int(np.count_nonzero(r_arr))
        record["dtype_after"] = str(r_arr.dtype)
        return record

    except Exception as e:  # noqa: BLE001
        record["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return record


def main() -> int:
    df = pl.read_csv(str(MANIFEST_PATH), infer_schema_length=2000)
    sel = df.filter(
        (pl.col("transferred_to_home") == True)  # noqa: E712
        & (pl.col("cohort") == "positive")
    ).select(["mnemonic_id", "lesion_mask_path"])

    n = sel.height
    print(f"[binarize] selected {n} lesion masks from manifest", flush=True)
    if n != 108:
        print(f"[binarize] WARNING: expected 108 rows, got {n}", flush=True)

    rows = sel.iter_rows(named=True)
    tasks = [(r["mnemonic_id"], r["lesion_mask_path"]) for r in rows]

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_process_one, mid, p): (mid, p) for mid, p in tasks
        }
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"[binarize] processed {done}/{len(futures)}", flush=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame(records).sort("mnemonic_id")
    out_df.write_csv(str(REPORT_PATH))
    print(f"[binarize] report written to {REPORT_PATH}", flush=True)

    # Summary
    total = len(records)
    already_binary = sum(1 for r in records if r["was_already_binary"])
    had_label2 = sum(1 for r in records if r["voxels_label_2_before"] > 0)
    had_unexpected = sum(1 for r in records if r["voxels_other_before"] > 0)
    skipped_unexpected = sum(
        1
        for r in records
        if (not r["was_already_binary"])
        and (not r["write_succeeded"])
        and r["error"]
        and "unexpected label" in r["error"]
    )
    failures = [r for r in records if r["error"] and "unexpected label" not in r["error"]]
    verification_failures = [
        r for r in records if (r["write_succeeded"] and not r["verification_passed"])
    ]
    wrote = sum(1 for r in records if r["write_succeeded"])

    print("\n========== BINARIZATION SUMMARY ==========", flush=True)
    print(f"total processed        : {total}", flush=True)
    print(f"had label-2 voxels     : {had_label2}", flush=True)
    print(f"already binary {{0,1}}   : {already_binary}", flush=True)
    print(f"had unexpected labels  : {had_unexpected}", flush=True)
    print(f"masks rewritten        : {wrote}", flush=True)
    print(f"skipped (unexpected)   : {skipped_unexpected}", flush=True)
    print(f"verification failures  : {len(verification_failures)}", flush=True)
    print(f"errors (other)         : {len(failures)}", flush=True)

    if skipped_unexpected:
        print(
            "\n!!! WARNING: some masks had UNEXPECTED label values and were NOT written. "
            "Inspect these in the CSV report !!!",
            flush=True,
        )
        for r in records:
            if (
                (not r["was_already_binary"])
                and (not r["write_succeeded"])
                and r["error"]
                and "unexpected label" in r["error"]
            ):
                print(
                    f"  - {r['mnemonic_id']}: unique_before={r['unique_values_before']}",
                    flush=True,
                )

    if failures:
        print("\n!!! ERRORS encountered:", flush=True)
        for r in failures:
            print(f"  - {r['mnemonic_id']} ({r['path']}): {r['error'].splitlines()[0]}", flush=True)
        return 1

    if verification_failures:
        print("\n!!! VERIFICATION failures (write succeeded but reload mismatch):", flush=True)
        for r in verification_failures:
            print(f"  - {r['mnemonic_id']}: {r['error']}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
