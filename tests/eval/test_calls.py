"""Per-call JSONL extraction tests (audit 2026-04-29 §3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from endo.eval.calls import (
    build_call_records,
    build_detection_map,
    extract_gt_lesions,
    extract_pred_calls,
    match_calls_to_gt,
    write_calls_jsonl,
)


def test_extract_pred_calls_simple_two_components():
    """Two disconnected boxes on disjoint slices → two pred calls."""
    boxes = np.array(
        [
            [10.0, 10.0, 30.0, 30.0, 5],
            [200.0, 200.0, 220.0, 220.0, 60],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.7, 0.3], dtype=np.float32)
    det_map = build_detection_map(boxes, scores, volume_shape=(120, 256, 256))
    calls = extract_pred_calls(det_map)
    assert len(calls) == 2
    # Higher-score component first or second — doesn't matter, but both
    # scores must round-trip.
    out_scores = sorted(c.score for c in calls)
    assert np.isclose(out_scores[0], 0.3)
    assert np.isclose(out_scores[1], 0.7)


def test_match_centroid_in_mask_tp_fp_fn():
    """One GT lesion, two pred calls: one with centroid inside, one outside.

    Inside-call → TP (highest score wins); outside-call → FP. Add a second
    GT lesion that gets no matching call → FN.
    """
    Y, Z, X = 60, 128, 128
    gt_mask = np.zeros((Y, Z, X), dtype=np.uint8)
    # Lesion 1 at (30, 64, 64) - 4x6x6 cuboid.
    gt_mask[28:32, 60:68, 60:68] = 1
    # Lesion 2 at (10, 20, 20) - 4x6x6 cuboid; will be FN.
    gt_mask[8:12, 16:24, 16:24] = 1

    # Pred call 1: a 4x6x6 box overlapping lesion 1 (TP).
    # Pred call 2: a small box at (50, 100, 100) — outside both lesions (FP).
    boxes = np.array(
        [
            [60.0, 60.0, 68.0, 68.0, 30],  # over lesion 1, slice 30
            [98.0, 98.0, 104.0, 104.0, 50],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.8, 0.4], dtype=np.float32)
    det_map = build_detection_map(boxes, scores, volume_shape=(Y, Z, X))
    pred_calls = extract_pred_calls(det_map)
    gt_lesions = extract_gt_lesions(gt_mask)
    assert len(gt_lesions) == 2

    tp_match, fp_idxs, fn_ids = match_calls_to_gt(pred_calls, gt_lesions, gt_mask)
    assert len(tp_match) == 1
    assert len(fp_idxs) == 1
    assert len(fn_ids) == 1

    records = build_call_records(
        run_id="r",
        entrypoint="cv",
        fold=0,
        pid="p1",
        pred_calls=pred_calls,
        gt_lesions=gt_lesions,
        tp_match=tp_match,
        fp_call_idxs=fp_idxs,
        fn_lesion_ids=fn_ids,
        large_thr=0.05,
        small_thr=0.30,
        box_size_split_mm=10.0,
    )
    types = sorted(r["call_type"] for r in records)
    assert types == ["fn", "fp", "tp"]
    # Volume in mm^3 is positive for every record.
    for r in records:
        assert r["volume_mm3"] > 0


def test_jsonl_roundtrip(tmp_path: Path):
    rec = [
        {"call_id": "p1_pred_1", "call_type": "tp", "volume_mm3": 12.3},
        {"call_id": "p1_fn_2", "call_type": "fn", "volume_mm3": 4.5},
    ]
    p = tmp_path / "calls.jsonl"
    write_calls_jsonl(p, rec)
    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert lines == rec
