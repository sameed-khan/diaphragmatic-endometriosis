"""Unit tests for ``endo.viz.tagging.tag_slice_events`` (PRD §11.9)."""

from __future__ import annotations

import numpy as np

from endo.viz.tagging import tag_slice_events


def _box(x1, y1, x2, y2) -> list[float]:
    return [float(x1), float(y1), float(x2), float(y2)]


def test_event_tagging_tp() -> None:
    """V1: pred IoU=0.5 with GT → tagged tp."""
    gt = np.array([_box(0, 0, 10, 10)], dtype=np.float32)
    # Box that is shifted but with high IoU (≈ 1.0 here).
    pred = np.array([_box(0, 0, 10, 10)], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)

    out = tag_slice_events(pred, scores, gt, iou_threshold=0.3)
    assert len(out["tp"]) == 1
    assert len(out["fp"]) == 0
    assert len(out["fn"]) == 0
    box, score, gt_idx = out["tp"][0]
    assert score == float(np.float32(0.9))
    assert gt_idx == 0
    np.testing.assert_allclose(box, pred[0])


def test_event_tagging_fp_low_iou() -> None:
    """V3: GT exists, pred IoU<0.3 → tagged fp."""
    gt = np.array([_box(0, 0, 10, 10)], dtype=np.float32)
    pred = np.array([_box(50, 50, 60, 60)], dtype=np.float32)  # disjoint from GT
    scores = np.array([0.7], dtype=np.float32)

    out = tag_slice_events(pred, scores, gt, iou_threshold=0.3)
    assert len(out["tp"]) == 0
    assert len(out["fp"]) == 1
    # Unmatched GT becomes an FN.
    assert len(out["fn"]) == 1


def test_event_tagging_fn() -> None:
    """V4: GT exists, no pred → tagged fn."""
    gt = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)], dtype=np.float32)
    pred = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)

    out = tag_slice_events(pred, scores, gt, iou_threshold=0.3)
    assert len(out["tp"]) == 0
    assert len(out["fp"]) == 0
    assert len(out["fn"]) == 2


def test_event_tagging_mixed_slice() -> None:
    """V5: slice has 1 TP + 1 FP + 1 FN → all 3 categories represented."""
    gt = np.array(
        [
            _box(0, 0, 10, 10),       # will be matched (TP)
            _box(100, 100, 110, 110), # will be unmatched (FN)
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            _box(0, 0, 10, 10),       # high IoU with GT[0] → TP
            _box(200, 200, 210, 210), # disjoint from any GT → FP
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.6], dtype=np.float32)

    out = tag_slice_events(pred, scores, gt, iou_threshold=0.3)
    assert len(out["tp"]) == 1
    assert len(out["fp"]) == 1
    assert len(out["fn"]) == 1

    # The matched pair is the high-score pred + GT[0].
    matched_box, matched_score, matched_idx = out["tp"][0]
    assert matched_score == float(np.float32(0.9))
    assert matched_idx == 0

    # The unmatched GT must be GT[1] at (100, 100, 110, 110).
    np.testing.assert_allclose(out["fn"][0], gt[1])
