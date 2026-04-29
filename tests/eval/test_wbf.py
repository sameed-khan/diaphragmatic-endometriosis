"""Tests E1, E3 — WBF aggregation and box-size threshold (PRD §11.8)."""

from __future__ import annotations

import numpy as np

from endo.eval.wbf import weighted_box_fusion_3d
from endo.inference_pass import SliceScore


def _ss(slice_y: int, boxes, scores) -> SliceScore:
    return SliceScore(
        patient_id="p1",
        slice_y=int(slice_y),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32).reshape(-1),
        aux_seg_max=0.0,
    )


def test_wbf_aggregates_overlapping():
    """E1: 3 highly overlapping boxes on the same slice fuse to 1."""
    boxes = [
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 51.0, 49.0],
        [9.0, 13.0, 52.0, 50.0],
    ]
    scores = [0.6, 0.5, 0.7]
    fused = weighted_box_fusion_3d(
        [_ss(20, boxes, scores)],
        image_size=(384, 384),
        iou_thr=0.3,
    )
    assert fused["fused_boxes"].shape[0] == 1
    # WBF score is a weighted average; should be in [min, max] of inputs.
    fused_score = float(fused["fused_scores"][0])
    assert 0.4 <= fused_score <= 0.8
    # slice_y is preserved in the 5th column.
    assert int(fused["fused_boxes"][0, 4]) == 20


def test_wbf_box_size_threshold():
    """E3: large box at 0.06 (≥ large_thr=0.05) keeps; small box at 0.06
    (< small_thr=0.30) drops."""
    # 0.82 mm/pixel → 12.2 px ≈ 10 mm. Large box: 20×20 px = 16.4 mm > 10mm
    # threshold. Small box: 5×5 px = 4.1 mm < 10mm threshold.
    large_box = [50.0, 50.0, 70.0, 70.0]   # 20×20 px ≈ 16 mm
    small_box = [200.0, 200.0, 205.0, 205.0]  # 5×5 px ≈ 4 mm
    fused = weighted_box_fusion_3d(
        [_ss(10, [large_box, small_box], [0.06, 0.06])],
        image_size=(384, 384),
        iou_thr=0.3,
        large_threshold=0.05,
        small_threshold=0.30,
        box_size_threshold_mm=10.0,
        inplane_mm=0.82,
    )
    boxes = fused["fused_boxes"]
    # Only the large box should survive.
    assert boxes.shape[0] == 1
    surviving_x1 = float(boxes[0, 0])
    assert abs(surviving_x1 - 50.0) < 5.0  # the large box's x1


def test_wbf_keeps_disjoint_per_slice():
    """Two non-overlapping boxes on different slices both survive."""
    fused = weighted_box_fusion_3d(
        [
            _ss(10, [[10, 10, 30, 30]], [0.8]),
            _ss(50, [[200, 200, 250, 250]], [0.7]),
        ],
        image_size=(384, 384),
        iou_thr=0.5,
    )
    assert fused["fused_boxes"].shape[0] == 2
    slice_ys = sorted(int(b[4]) for b in fused["fused_boxes"])
    assert slice_ys == [10, 50]


def test_wbf_empty_returns_empty():
    fused = weighted_box_fusion_3d([], image_size=(384, 384))
    assert fused["fused_boxes"].shape == (0, 5)
    assert fused["fused_scores"].shape == (0,)
