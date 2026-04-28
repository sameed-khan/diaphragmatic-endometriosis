"""DEPRECATED 2026-04-27 — superseded by data-local-copy migration. Kept as historical reference only.

Realign radiologist source masks onto the freshly-converted home raw NIfTI.

This is a fixed replacement for `scripts/realign_masks.py`. The original script
did a hard-coded `mask[::-1, :, ::-1]` flip and adopted the target's affine —
which is correct ONLY when (a) the source mask's true orientation is LAI and
(b) the target's voxel grid covers the same physical region with the same
spacing. When dcm2niix produces multiple `water_canonical*.nii.gz`
sub-volumes with the same shape but different physical extents, the original
script picks the alphabetically first shape-match, which can be the wrong
sub-volume — placing the mask at incorrect physical positions.

This v2 instead uses **physical-space resampling** via
`nibabel.processing.resample_from_to` so the mask voxels always land on the
target's voxel grid at the correct physical positions, regardless of:
- orientation differences (handled by nibabel's affine math)
- spacing differences (handled by nearest-neighbour resampling)
- origin/translation differences

Usage:
  uv run python scripts/realign_masks_v2.py --execute

Inputs (driven from `manifest.csv`):
  - source mask: /scratch/.../input/masks/<anon>.nii.gz (or *<anon>*.nii.gz)
  - target raw:  data/<raw_path>  (already migrated as the canonical home volume)

Outputs:
  - data/<lesion_mask_path>  (overwritten in-place if --execute, dry-run otherwise)
  - eda/outputs/realign_masks_v2_report.csv  (audit log)

If the home raw doesn't physically cover all of the source mask, voxels
outside the overlap are silently lost (and reported in the audit log).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_MASKS = Path("/scratch/pioneer/users/sak185/dia-endo-conversion/input/masks")
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "eda" / "outputs" / "realign_masks_v2_report.csv"


def find_source_mask(anon: str) -> Path | None:
    for cand in [INPUT_MASKS / f"{anon}.nii.gz", *sorted(INPUT_MASKS.glob(f"{anon}*.nii.gz"))]:
        if cand.exists():
            return cand
    return None


def realign_one(anon: str, raw_path: Path, lesion_out: Path, execute: bool) -> dict:
    src = find_source_mask(anon)
    if src is None:
        return {"anon_id": anon, "status": "no_source", "src_voxels": 0, "out_voxels": 0}
    raw = nib.load(raw_path)
    src_img = nib.load(src)
    src_voxels = int((np.asanyarray(src_img.dataobj) > 0).sum())

    # Binarize source first (some have label values 1, 2, 3 — treat all > 0 as lesion).
    src_arr = np.asanyarray(src_img.dataobj)
    src_bin = (src_arr > 0).astype(np.uint8)
    src_bin_img = nib.Nifti1Image(src_bin, affine=src_img.affine, header=src_img.header)

    # Physical-space resample onto the home raw's voxel grid (nearest-neighbor).
    resampled = nib_proc.resample_from_to(src_bin_img, (raw.shape, raw.affine), order=0)
    out_arr = np.asanyarray(resampled.dataobj).astype(np.uint8)
    out_voxels = int((out_arr > 0).sum())

    overlap_frac = out_voxels / max(src_voxels, 1)

    if execute:
        out_header = raw.header.copy()
        out_header.set_data_dtype(np.uint8)
        out_header.set_slope_inter(1.0, 0.0)
        out_img = nib.Nifti1Image(out_arr, affine=raw.affine, header=out_header)
        lesion_out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, lesion_out)

    return {
        "anon_id": anon,
        "status": "ok",
        "src_voxels": src_voxels,
        "out_voxels": out_voxels,
        "voxels_lost_to_crop": src_voxels - out_voxels,
        "overlap_frac": overlap_frac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Write outputs in-place; default is dry-run")
    args = ap.parse_args()

    manifest = pl.read_csv(DATA / "manifest.csv").filter(
        (pl.col("transferred_to_home"))
        & (pl.col("cohort") == "positive")
    )
    print(f"Realigning {manifest.height} lesion masks via physical-space resampling...")

    rows = manifest.select(["mnemonic_id", "anon_id", "raw_path", "lesion_mask_path"]).to_dicts()
    records = []
    for r in rows:
        rec = realign_one(
            r["anon_id"],
            DATA / r["raw_path"],
            DATA / r["lesion_mask_path"],
            execute=args.execute,
        )
        rec["mnemonic_id"] = r["mnemonic_id"]
        records.append(rec)

    df = pl.DataFrame(records)
    df.write_csv(OUT)

    print(f"\nResults written to {OUT}")
    print(f"\nSummary:")
    print(f"  Total: {df.height}")
    print(f"  Status ok: {df.filter(pl.col('status') == 'ok').height}")
    print(f"  Voxels lost to crop (worst): {df['voxels_lost_to_crop'].max()}")
    print(f"  Overlap fraction: median={df['overlap_frac'].median():.3f} P5={df['overlap_frac'].quantile(0.05):.3f}")
    bad = df.filter(pl.col("overlap_frac") < 0.95).sort("overlap_frac")
    print(f"  Patients with overlap < 95%: {bad.height}")
    if bad.height:
        print(bad.select(["mnemonic_id", "src_voxels", "out_voxels", "overlap_frac"]))


if __name__ == "__main__":
    main()
