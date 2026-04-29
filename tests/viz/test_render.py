"""Smoke test for ``endo.viz.render``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from endo.viz.render import render_slice_overlay, save_slice_png


def test_render_smoke(tmp_path: Path) -> None:
    """Render a synthetic slice + boxes; assert PNG is created and non-empty."""
    rng = np.random.default_rng(0)
    # Synthetic 5-channel slice (C, H, W) — render path should pick the center.
    volume = rng.standard_normal((5, 64, 64)).astype(np.float32)
    pred_boxes = np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
    pred_scores = np.array([0.8], dtype=np.float32)
    gt_boxes = np.array([[8.0, 8.0, 28.0, 28.0]], dtype=np.float32)

    img = render_slice_overlay(
        volume=volume,
        slice_y=0,  # ignored for 5-channel input
        lesion_mask_center=None,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        event_type="tp",
        patient_id="synthetic_pid",
    )
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8

    out = tmp_path / "test_render.png"
    save_slice_png(img, out)
    assert out.exists()
    assert out.stat().st_size > 0
