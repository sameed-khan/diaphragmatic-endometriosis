"""DICOM-level probe of a representative sample of 10 of the 108 positives."""
from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pydicom

ROOT = Path("/scratch/pioneer/users/sak185/diaphragmatic-endometriosis")
DICOM_DIR = Path("/home/jjs374/DiaE/dicom")
OUT_DIR = ROOT / "eda" / "outputs"


def probe_series(series_dir: Path) -> dict:
    dcms = sorted(p for p in series_dir.iterdir() if p.suffix == ".dcm")
    if not dcms:
        return {"error": "no_dcm"}
    ds = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True, force=True)
    def g(tag, default=""):
        v = getattr(ds, tag, default)
        if hasattr(v, "value"):
            v = v.value
        return v
    return {
        "n_files": len(dcms),
        "SeriesDescription": str(g("SeriesDescription")),
        "SeriesNumber": str(g("SeriesNumber")),
        "ImageType": "/".join(list(g("ImageType", []) or [])),
        "EchoTime_ms": float(g("EchoTime") or 0),
        "RepetitionTime_ms": float(g("RepetitionTime") or 0),
        "FlipAngle": float(g("FlipAngle") or 0),
        "MRAcquisitionType": str(g("MRAcquisitionType")),
        "SliceThickness": str(g("SliceThickness")),
        "SpacingBetweenSlices": str(g("SpacingBetweenSlices")),
        "PixelSpacing": str(g("PixelSpacing")),
        "ManufacturerModelName": str(g("ManufacturerModelName")),
        "ProtocolName": str(g("ProtocolName")),
        "ScanOptions": str(g("ScanOptions")),
    }


def main():
    audit = pl.read_csv(OUT_DIR / "source_of_truth_audit.csv")
    # Sample: a mix of Artist + Explorer + DIAFRAGMA_T1_LAVA + FLEX_NAV
    sample_anons = []
    # 4 SIGNA Artist | LAVA_DIAF
    art_lava = audit.filter(pl.col("protocol_cluster") == "SIGNA Artist|LAVA_DIAF").head(3)
    sample_anons.extend(art_lava["anon_id"].to_list())
    # 3 SIGNA Explorer | LAVA_DIAF
    exp_lava = audit.filter(pl.col("protocol_cluster") == "SIGNA Explorer|LAVA_DIAF").head(3)
    sample_anons.extend(exp_lava["anon_id"].to_list())
    # 3 SIGNA Explorer | DIAFRAGMA_T1_LAVA
    exp_diaf = audit.filter(pl.col("protocol_cluster") == "SIGNA Explorer|DIAFRAGMA_T1_LAVA").head(3)
    sample_anons.extend(exp_diaf["anon_id"].to_list())
    # 1 SIGNA Artist | LAVA_FLEX_NAV
    flex = audit.filter(pl.col("protocol_cluster") == "SIGNA Artist|LAVA_FLEX_NAV").head(1)
    sample_anons.extend(flex["anon_id"].to_list())

    rows = []
    for a in sample_anons:
        pdir = DICOM_DIR / a
        if not pdir.exists():
            rows.append({"anon_id": a, "series_dir": "MISSING", "error": "no_dicom_dir"})
            continue
        for sdir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            try:
                info = probe_series(sdir)
            except Exception as e:
                info = {"error": str(e)}
            rows.append({"anon_id": a, "series_dir": sdir.name, **info})

    df = pl.DataFrame(rows)
    out = OUT_DIR / "source_of_truth_dicom_sample.csv"
    df.write_csv(out)
    print(f"Wrote {out}")
    print(df.select("anon_id", "series_dir", "SeriesDescription", "ImageType",
                    "EchoTime_ms", "RepetitionTime_ms", "FlipAngle", "MRAcquisitionType",
                    "SliceThickness", "ManufacturerModelName"))


if __name__ == "__main__":
    main()
