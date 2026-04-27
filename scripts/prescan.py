"""Stage 1: scan every series subfolder, read one DICOM, write pre_scan_index.csv.

Operates on the consolidated layout:
  <input_root>/positive/<ANONID>/<series>/...
  <input_root>/negative/<ANONID>/<series>/...
"""
import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import pydicom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _common import COHORT_DIRS, first_dcm_in

CSV_FIELDS = [
    "cohort", "patient_id", "series_path", "series_folder_name",
    "n_dcm_files", "image_type_full", "image_type_token",
    "series_description", "manufacturer", "manufacturer_model_name",
    "magnetic_field_strength", "slice_thickness_mm",
    "spacing_between_slices_mm", "pixel_spacing_x_mm", "pixel_spacing_y_mm",
    "rows", "cols", "sequence_name", "scanning_sequence",
    "read_status", "read_error",
]


def list_series(input_root: Path):
    """Yield (cohort, patient_id, series_dir) tuples."""
    for cohort_dir, cohort_label in COHORT_DIRS.items():
        base = input_root / cohort_dir
        if not base.is_dir():
            continue
        for patient in sorted(p for p in base.iterdir() if p.is_dir()):
            for series in sorted(s for s in patient.iterdir() if s.is_dir()):
                yield cohort_label, patient.name, series


def scan_one(args):
    cohort, pid, series = args
    n_dcm = sum(1 for p in series.iterdir() if p.is_file() and not p.name.startswith("."))
    row = {f: "" for f in CSV_FIELDS}
    row.update(cohort=cohort, patient_id=pid,
               series_path=str(series), series_folder_name=series.name,
               n_dcm_files=n_dcm)
    sample = first_dcm_in(series)
    if sample is None:
        row["read_status"] = "empty_folder"
        return row
    try:
        ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        it = list(ds.get("ImageType", []))
        row["image_type_full"] = "\\".join(it)
        row["image_type_token"] = it[3] if len(it) >= 4 else ""
        row["series_description"] = str(ds.get("SeriesDescription", ""))
        row["manufacturer"] = str(ds.get("Manufacturer", ""))
        row["manufacturer_model_name"] = str(ds.get("ManufacturerModelName", ""))
        row["magnetic_field_strength"] = str(ds.get("MagneticFieldStrength", ""))
        row["slice_thickness_mm"] = str(ds.get("SliceThickness", ""))
        row["spacing_between_slices_mm"] = str(ds.get("SpacingBetweenSlices", ""))
        ps = ds.get("PixelSpacing", [None, None])
        row["pixel_spacing_x_mm"] = str(ps[0]) if ps and ps[0] is not None else ""
        row["pixel_spacing_y_mm"] = str(ps[1]) if ps and len(ps) > 1 and ps[1] is not None else ""
        row["rows"] = str(ds.get("Rows", ""))
        row["cols"] = str(ds.get("Columns", ""))
        row["sequence_name"] = str(ds.get("SequenceName", ""))
        row["scanning_sequence"] = str(ds.get("ScanningSequence", ""))
        row["read_status"] = "ok"
    except Exception as e:
        row["read_status"] = "error"
        row["read_error"] = repr(e)[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True,
                    help="Path containing positive/ and negative/ subdirs (post-consolidation)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    series_list = list(list_series(args.input_root))
    print(f"Scanning {len(series_list)} series across {args.workers} workers...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool, args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in tqdm(pool.imap_unordered(scan_one, series_list, chunksize=10),
                        total=len(series_list)):
            w.writerow(row)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
