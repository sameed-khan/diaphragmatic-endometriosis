"""Stage 6: concatenate per-task manifests, harvest BIDS sidecars, sanity-check."""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from _common import MIN_SLICES_QC_FLAG


def sha256_of(path: Path, blocksize=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b: break
            h.update(b)
    return h.hexdigest()


HARVEST_FIELDS = (
    "n_slices_actual", "shape", "scanner_model", "magnetic_field_strength",
    "slice_thickness_mm", "spacing_between_slices_mm",
    "pixel_spacing_x_mm", "pixel_spacing_y_mm", "image_type",
    "series_description", "conversion_software_version", "sha256",
)
EMPTY_HARVEST = {k: "" for k in HARVEST_FIELDS}


def harvest(nifti_path: Path) -> dict:
    img = nib.load(nifti_path)
    json_path = nifti_path.with_suffix("").with_suffix(".json")
    side = json.loads(json_path.read_text()) if json_path.exists() else {}
    pix = side.get("PixelSpacing", [None, None])
    return {
        "n_slices_actual": str(img.shape[2]),
        "shape": "x".join(str(s) for s in img.shape),
        "scanner_model": str(side.get("ManufacturerModelName", "")),
        "magnetic_field_strength": str(side.get("MagneticFieldStrength", "")),
        "slice_thickness_mm": str(side.get("SliceThickness", "")),
        "spacing_between_slices_mm": str(side.get("SpacingBetweenSlices", "")),
        "pixel_spacing_x_mm": str(pix[0]) if pix and pix[0] is not None else "",
        "pixel_spacing_y_mm": str(pix[1]) if pix and len(pix) > 1 and pix[1] is not None else "",
        "image_type": "\\".join(side.get("ImageType", [])),
        "series_description": str(side.get("SeriesDescription", "")),
        "conversion_software_version": str(side.get("ConversionSoftwareVersion", "")),
        "sha256": sha256_of(nifti_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--skipped", type=Path, required=True)
    ap.add_argument("--alignment-audit", type=Path)
    ap.add_argument("--splits-json", type=Path,
                    help="splits.json from build_splits.py; joins `split` "
                         "and `soft_negative` columns into manifest")
    args = ap.parse_args()

    parts = sorted(glob.glob(str(args.output_root / "manifest_part_*.csv")))
    if not parts:
        raise SystemExit("No manifest_part_*.csv found in output root")

    # Tag each row with the manifest_part it came from, then derive `phase`.
    # The part filename is the SLURM-job-level provenance: more authoritative
    # than `split` alone, since pilot/remask23 patients later got `split` values
    # assigned by build_splits.py but were physically converted in earlier runs.
    def derive_phase(part_name: str) -> str:
        # manifest_part_phase1_<task>.csv -> "phase1"
        # manifest_part_phase2_<task>.csv -> "phase2"
        # manifest_part_pilot.csv         -> "pilot"
        # manifest_part_remask23.csv      -> "remask23"
        stem = part_name.removeprefix("manifest_part_").removesuffix(".csv")
        if stem.startswith("phase1_"): return "phase1"
        if stem.startswith("phase2_"): return "phase2"
        return stem  # "pilot", "remask23", or anything else verbatim

    tagged = []
    for p in parts:
        sub = pl.read_csv(p, infer_schema_length=10000)
        part_name = Path(p).name
        sub = sub.with_columns(
            pl.lit(part_name).alias("manifest_part_source"),
            pl.lit(derive_phase(part_name)).alias("phase"),
        )
        tagged.append(sub)
    df = pl.concat(tagged, how="diagonal_relaxed")

    rows = []
    for r in df.iter_rows(named=True):
        files = [f for f in str(r["produced_files"] or "").split(";") if f]
        base = {k: r[k] for k in r}
        if not files:
            rows.append({**base, **EMPTY_HARVEST, "output_filename": "",
                         "volume_index": 0, "n_volumes_from_series": 0,
                         "conversion_status": ("failed" if r["exit_code"] != 0
                                               else "no_output")})
            continue
        for i, fname in enumerate(files):
            row = {**base, **EMPTY_HARVEST, "output_filename": fname,
                   "volume_index": i,
                   "n_volumes_from_series": len(files),
                   "conversion_status": ("ok" if r["exit_code"] == 0 else "failed")}
            nifti_path = args.output_root / r["output_subdir"] / fname
            if nifti_path.exists():
                row.update(harvest(nifti_path))
                try:
                    n_slices = int(row.get("n_slices_actual") or 0)
                except (TypeError, ValueError):
                    n_slices = 0
                if n_slices < MIN_SLICES_QC_FLAG:
                    row["conversion_status"] = "qc_flag:low_slice_count"
            rows.append(row)

    manifest = pl.from_dicts(rows, infer_schema_length=None)

    # Join split + soft_negative annotations from splits.json. Patients not in
    # splits.json (e.g. early pilot output before splits existed) get null
    # `split` and False `soft_negative`.
    if args.splits_json and args.splits_json.exists():
        doc = json.loads(args.splits_json.read_text())
        split_map = doc.get("assignments", {})
        soft_neg = set(doc.get("soft_negative_pids", []))
        split_df = pl.DataFrame({
            "patient_id": list(split_map.keys()),
            "split": list(split_map.values()),
        })
        manifest = manifest.join(split_df, on="patient_id", how="left")
        manifest = manifest.with_columns(
            pl.col("patient_id").is_in(list(soft_neg)).alias("soft_negative")
        )

    manifest.write_csv(args.output_root / "manifest.csv")
    qc_flags = manifest.filter(
        pl.col("conversion_status").str.starts_with("qc_flag")
        | (pl.col("conversion_status") != "ok"))
    qc_flags.write_csv(args.output_root / "qc_flags.csv")

    n_ok = manifest.filter(pl.col("conversion_status") == "ok").height
    n_failed = manifest.filter(pl.col("conversion_status") == "failed").height
    n_qc = manifest.filter(
        pl.col("conversion_status").str.starts_with("qc_flag")).height
    n_skipped = pl.read_csv(args.skipped)["patient_id"].n_unique() if args.skipped.exists() else 0

    summary = (
        f"Manifest rows:    {manifest.height}\n"
        f"OK:               {n_ok}\n"
        f"Failed:           {n_failed}\n"
        f"QC-flagged:       {n_qc}\n"
        f"Skipped patients: {n_skipped}\n"
    )
    (args.output_root / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
