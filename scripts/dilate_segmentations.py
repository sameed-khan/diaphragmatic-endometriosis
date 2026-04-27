"""
Dilate the binary liver masks in liver_masks/ by a physical-space margin (default
20 mm) using an anisotropic distance transform, writing binary uint8 ROIs to
liver_rois/ that mirror the input directory structure.

The script is dry-run by default. Pass --execute to actually write ROIs.

Per-file voxel sizes are read from each NIfTI's header (do NOT assume a single
voxel grid across the cohort).

Outputs (under --output-dir):
    <bucket>/<cohort>/<mnemonic>_liver_roi.nii.gz   # binary uint8 0/1, dilated by --margin-mm

After running over the full set with no subset filter, the manifest gets two
columns populated for each transferred patient: liver_roi_path, liver_roi_margin_mm.

Per agent/totalseg-plan.md §6.2.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

DEFAULT_WORKERS = 6
DEFAULT_MARGIN_MM = 20
MIN_NON_TRIVIAL_FILE_BYTES = 1024


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory containing binary liver masks (e.g., data/liver_masks).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write dilated ROIs (e.g., data/liver_rois).")
    ap.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM,
                    help="Physical-space dilation radius in millimetres (default: 20).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="Number of CPU workers (default: 6).")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Project data root, used for the manifest update. If omitted, the manifest is not updated.")
    ap.add_argument("--force", action="store_true",
                    help="Re-dilate patients whose output already exists.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write outputs. Without this flag, dry-run only.")
    return ap.parse_args()


def _dilate_one(item: dict) -> dict:
    src = Path(item["src"])
    dst = Path(item["dst"])
    margin_mm = float(item["margin_mm"])

    t0 = time.time()
    img = nib.load(src)
    data = np.asarray(img.dataobj) > 0
    n_in = int(data.sum())
    if n_in == 0:
        return {
            "src": str(src), "dst": str(dst),
            "voxel_in": 0, "voxel_out": 0,
            "wall_seconds": time.time() - t0,
            "status": "skipped_empty",
            "error": "",
        }
    voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
    try:
        dist = distance_transform_edt(~data, sampling=voxel_sizes)
        dilated = (dist <= margin_mm).astype(np.uint8)
        n_out = int(dilated.sum())
        out = nib.Nifti1Image(dilated, img.affine, img.header)
        out.header.set_data_dtype(np.uint8)
        out.header.set_slope_inter(1.0, 0.0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out, dst)
        return {
            "src": str(src), "dst": str(dst),
            "voxel_in": n_in, "voxel_out": n_out,
            "wall_seconds": time.time() - t0,
            "status": "ok", "error": "",
        }
    except Exception as e:  # noqa: BLE001 — surface any failure as a row, don't crash the pool
        return {
            "src": str(src), "dst": str(dst),
            "voxel_in": n_in, "voxel_out": 0,
            "wall_seconds": time.time() - t0,
            "status": "error", "error": repr(e),
        }


def update_manifest(data_root: Path, output_dir: Path, margin_mm: float) -> None:
    manifest_path = data_root / "manifest.csv"
    m = pl.read_csv(manifest_path, infer_schema_length=10000)

    if "liver_roi_path" not in m.columns:
        m = m.with_columns(pl.lit("", dtype=pl.Utf8).alias("liver_roi_path"))
    if "liver_roi_margin_mm" not in m.columns:
        m = m.with_columns(pl.lit(None, dtype=pl.Float64).alias("liver_roi_margin_mm"))

    paths: list[str] = []
    margins: list[float | None] = []

    output_rel = output_dir.relative_to(data_root)

    for r in m.iter_rows(named=True):
        if not r["transferred_to_home"]:
            paths.append(r.get("liver_roi_path") or "")
            margins.append(r.get("liver_roi_margin_mm"))
            continue
        bucket = r["bucket"]
        cohort = r["cohort"]
        mnem = r["mnemonic_id"]
        rel = f"{output_rel}/{bucket}/{cohort}/{mnem}_liver_roi.nii.gz"
        full = data_root / rel
        if full.exists() and full.stat().st_size > MIN_NON_TRIVIAL_FILE_BYTES:
            paths.append(rel)
            margins.append(float(margin_mm))
        else:
            paths.append("")
            margins.append(None)

    m = m.with_columns([
        pl.Series("liver_roi_path", paths, dtype=pl.Utf8),
        pl.Series("liver_roi_margin_mm", margins, dtype=pl.Float64),
    ])
    tmp = manifest_path.with_suffix(".csv.tmp")
    m.write_csv(tmp)
    tmp.replace(manifest_path)


def main():
    args = parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if args.execute:
        print(">>> EXECUTE — ROIs will be written. <<<")
    else:
        print(">>> DRY-RUN — no ROIs will be written. Pass --execute to proceed. <<<")

    if not input_dir.exists():
        raise SystemExit(f"input dir does not exist: {input_dir}")

    src_files = sorted(input_dir.rglob("*.nii.gz"))

    pairs: list[dict] = []
    for src in src_files:
        rel = src.relative_to(input_dir)
        # Convention: source liver mask <mnem>_liver_mask.nii.gz -> dilated ROI <mnem>_liver_roi.nii.gz.
        # Fall back to preserving the basename if it doesn't follow the convention (e.g., older runs).
        if rel.name.endswith("_liver_mask.nii.gz"):
            new_name = rel.name[: -len("_liver_mask.nii.gz")] + "_liver_roi.nii.gz"
            rel = rel.with_name(new_name)
        dst = output_dir / rel
        if not args.force and dst.exists() and dst.stat().st_size > MIN_NON_TRIVIAL_FILE_BYTES:
            continue
        pairs.append({"src": str(src), "dst": str(dst), "margin_mm": args.margin_mm})

    print(f"\n=== Plan ===")
    print(f"  input dir:    {input_dir}")
    print(f"  output dir:   {output_dir}")
    print(f"  --margin-mm:  {args.margin_mm}")
    print(f"  --workers:    {args.workers}")
    print(f"  source files: {len(src_files)}")
    print(f"  to process:   {len(pairs)} (skipped existing: {len(src_files) - len(pairs)})")
    if pairs:
        print(f"  first 3:")
        for p in pairs[:3]:
            print(f"    {Path(p['src']).name}  ->  {p['dst']}")

    if not args.execute:
        print("\nDry-run complete.")
        return 0

    if not pairs:
        print("\nNothing to do.")
        return 0

    n_ok = n_skip = n_err = 0
    voxel_in_sum = voxel_out_sum = 0
    wall_sum = 0.0
    errors: list[dict] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        iterator = pool.imap_unordered(_dilate_one, pairs)
        for result in tqdm(iterator, total=len(pairs), unit="pat", desc="Dilate"):
            wall_sum += result["wall_seconds"]
            if result["status"] == "ok":
                n_ok += 1
                voxel_in_sum += result["voxel_in"]
                voxel_out_sum += result["voxel_out"]
            elif result["status"] == "skipped_empty":
                n_skip += 1
            else:
                n_err += 1
                errors.append(result)

    print("\n=== Summary ===")
    print(f"  ok:           {n_ok}")
    print(f"  skipped_empty:{n_skip}")
    print(f"  errors:       {n_err}")
    if n_ok:
        print(f"  mean wall/file: {wall_sum / n_ok:.1f} s")
        print(f"  mean voxel growth ratio: {voxel_out_sum / max(1, voxel_in_sum):.2f}x")
    for e in errors[:5]:
        print(f"  ERROR {Path(e['src']).name}: {e['error']}")

    # Manifest update only when --data-root is provided and we processed against the full input dir.
    if args.data_root is not None:
        print("\n=== Updating manifest ===")
        update_manifest(args.data_root.resolve(), output_dir, args.margin_mm)
        print(f"  Updated liver_roi_path / liver_roi_margin_mm in "
              f"{(args.data_root / 'manifest.csv').resolve()}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
