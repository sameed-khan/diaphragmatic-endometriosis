"""Analyze in-plane voxel spacing across the cohort and recommend TARGET_SPACING.

Reads the NIfTI header (no voxel data) for every patient in the manifest and
extracts ``header.get_zooms()[0]`` (X) and ``header.get_zooms()[2]`` (Z).
Histograms each axis at 0.01 mm resolution. If any single bin contains > 50%
of the cohort, that bin's value is chosen for that axis; otherwise the cohort
median is used. Writes a single-file analysis with the recommended
``TARGET_SPACING`` constant ready to paste into ``scripts/preprocess.py``.

CLI:
    uv run python scripts/analyze_inplane_spacing.py \\
        --manifest data/manifest.jsonl \\
        --raw-root data/ \\
        --output agent/complete_spec/analysis_inplane_spacing.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bin_value(value: float) -> float:
    """Quantize to 0.01 mm bins (rounded)."""
    return round(round(float(value) / 0.01) * 0.01, 2)


def _decide(values: list[float]) -> tuple[float, list[tuple[float, int, float]]]:
    """Decision rule: dominant bin (>50%) else cohort median.

    Returns (chosen_value, top_5_bins_with_pct).
    """
    bins = [_bin_value(v) for v in values]
    counter = Counter(bins)
    n = len(values)
    top = counter.most_common(5)
    top_with_pct = [(val, cnt, 100.0 * cnt / n) for val, cnt in top]

    most_common_val, most_common_cnt = counter.most_common(1)[0]
    if most_common_cnt / n > 0.5:
        chosen = most_common_val
    else:
        chosen = float(np.median(np.array(values, dtype=np.float64)))
        chosen = _bin_value(chosen)
    return chosen, top_with_pct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    xs: list[float] = []
    zs: list[float] = []
    skipped: list[tuple[str, str]] = []

    for r in rows:
        pid = r["patient_id"]
        rel = r["paths"]["raw"]
        path = (args.raw_root / rel).resolve()
        try:
            img = nib.load(str(path))
            zooms = img.header.get_zooms()
            xs.append(float(zooms[0]))
            zs.append(float(zooms[2]))
        except Exception as e:  # noqa: BLE001
            skipped.append((pid, repr(e)))

    if len(xs) == 0:
        raise SystemExit("No volumes successfully read; aborting.")

    chosen_x, top_x = _decide(xs)
    chosen_z, top_z = _decide(zs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# In-plane spacing analysis")
    lines.append(f"n_volumes_read: {len(xs)}")
    lines.append(f"n_skipped: {len(skipped)}")
    lines.append("")
    lines.append("## X-axis (header.get_zooms()[0]) top 5 bins")
    for val, cnt, pct in top_x:
        lines.append(f"  {val:.2f} mm  count={cnt}  ({pct:.2f}%)")
    lines.append(f"  median: {float(np.median(xs)):.4f} mm")
    lines.append(f"  chosen: {chosen_x:.2f} mm")
    lines.append("")
    lines.append("## Z-axis (header.get_zooms()[2]) top 5 bins")
    for val, cnt, pct in top_z:
        lines.append(f"  {val:.2f} mm  count={cnt}  ({pct:.2f}%)")
    lines.append(f"  median: {float(np.median(zs)):.4f} mm")
    lines.append(f"  chosen: {chosen_z:.2f} mm")
    lines.append("")
    if skipped:
        lines.append("## Skipped volumes")
        for pid, err in skipped:
            lines.append(f"  {pid}: {err}")
        lines.append("")
    lines.append("## Paste into scripts/preprocess.py")
    lines.append(f"TARGET_SPACING = ({chosen_x:.2f}, 1.5, {chosen_z:.2f})  # mm; from analyze_inplane_spacing.py")
    out_text = "\n".join(lines) + "\n"
    args.output.write_text(out_text)
    print(out_text)


if __name__ == "__main__":
    main()
