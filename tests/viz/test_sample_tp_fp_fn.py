"""Tests for endo.viz.run_viz.sample_tp_fp_fn."""

from __future__ import annotations

import csv
from pathlib import Path

from endo.viz.run_viz import sample_tp_fp_fn


def _make_viz_dir(tmp_path: Path, n_per_event: int = 30) -> Path:
    out = tmp_path / "viz"
    out.mkdir()
    rows = []
    for ev in ("tp", "fp", "fn"):
        for i in range(n_per_event):
            png_name = f"pid{i:03d}_{ev}_slice42.png"
            (out / png_name).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
            rows.append({
                "patient_id": f"pid{i:03d}",
                "slice_y": 42,
                "event_type": ev,
                "score": 0.5,
                "gt_iou": 0.0,
                "png_path": str(out / png_name),
            })
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["patient_id", "slice_y", "event_type", "score", "gt_iou", "png_path"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def test_sample_tp_fp_fn_caps_at_n(tmp_path: Path) -> None:
    viz_dir = _make_viz_dir(tmp_path, n_per_event=30)
    sample_dir = sample_tp_fp_fn(viz_dir, n_tp=20, n_fp=20, n_fn=20, seed=42)
    pngs = list(sample_dir.glob("*.png"))
    assert len(pngs) == 60
    by_event = {"tp": 0, "fp": 0, "fn": 0}
    for p in pngs:
        for ev in by_event:
            if p.name.startswith(ev + "_"):
                by_event[ev] += 1
    assert by_event == {"tp": 20, "fp": 20, "fn": 20}


def test_sample_tp_fp_fn_deterministic(tmp_path: Path) -> None:
    viz_dir = _make_viz_dir(tmp_path, n_per_event=30)
    s1 = sample_tp_fp_fn(viz_dir, n_tp=5, n_fp=5, n_fn=5, seed=99)
    files1 = sorted(p.name for p in s1.glob("*.png"))
    s2 = sample_tp_fp_fn(viz_dir, n_tp=5, n_fp=5, n_fn=5, seed=99)
    files2 = sorted(p.name for p in s2.glob("*.png"))
    assert files1 == files2


def test_sample_tp_fp_fn_handles_empty_event(tmp_path: Path) -> None:
    out = tmp_path / "viz"
    out.mkdir()
    # only TP rows
    (out / "a_tp_slice0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["patient_id", "slice_y", "event_type", "score", "gt_iou", "png_path"],
        )
        w.writeheader()
        w.writerow({
            "patient_id": "a", "slice_y": 0, "event_type": "tp",
            "score": 0.5, "gt_iou": 0.0, "png_path": str(out / "a_tp_slice0.png"),
        })
    sample_dir = sample_tp_fp_fn(out, n_tp=20, n_fp=20, n_fn=20, seed=0)
    pngs = list(sample_dir.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name.startswith("tp_")
