"""Test E9 — threshold grid search recovers known optimum (PRD §11.8)."""

from __future__ import annotations

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.threshold_search import grid_search_threshold


def _vol(score: float, slice_y: int = 80, large: bool = True) -> dict:
    """Per-volume prediction with one box.

    The default canvas is (Y=160, Z=384, X=384) and picai_eval treats a
    positive volume's GT as a central voxel at (Y/2, Z/2, X/2). When ``large``
    we draw a 30×30 box centered on (192, 192) at slice 80 (overlaps GT).
    When small, we draw a 4×4 box far off-center (won't overlap GT — useful
    as a "false-positive candidate")."""
    if large:
        box = [177.0, 177.0, 207.0, 207.0]
    else:
        box = [10.0, 10.0, 14.0, 14.0]
    fb = np.asarray([box + [slice_y]], dtype=np.float32)
    fs = np.asarray([score], dtype=np.float32)
    return {"fused_boxes": fb, "fused_scores": fs, "score": float(score)}


def test_threshold_grid_search_finds_optimum():
    """E9: Build a synthetic dataset where the (large_thr, small_thr) grid
    optimum is forced by the score / size structure.

    Setup (4 pos + 4 neg):
      - Each positive volume has one **large** box at the GT central voxel
        with score 0.5 (must be kept by ``large_thr``).
      - Each negative volume has one **small** box overlapping the GT center
        too — but with a higher confidence (0.6). Without size-aware
        filtering, the negatives outrank the positives.

    Because the small boxes are far smaller than ``box_size_split_mm``, they
    get gated by ``small_thr``. To recover sens=1 the optimum must be:
      - large_thr ≤ 0.50  (keeps the positives)
      - small_thr > 0.60  (drops the high-score negatives)
    """
    preds: dict[str, dict] = {}
    labels: dict[str, int] = {}
    # Slightly varied per-volume scores so picai_eval has > 1 threshold to
    # scan (it bails out at len(thresholds) < 2).
    rng = np.random.default_rng(0)
    for i in range(4):
        s = float(0.50 + 0.01 * (i + 1))  # 0.51, 0.52, 0.53, 0.54
        v = _vol(score=s, slice_y=80, large=True)
        v["fused_scores"] = np.asarray([s], dtype=np.float32)
        v["score"] = s
        preds[f"pos_{i}"] = v
        labels[f"pos_{i}"] = 1
    for i in range(4):
        s = float(0.60 + 0.01 * (i + 1))  # 0.61–0.64
        small_box = [188.0, 188.0, 196.0, 196.0]
        fb = np.asarray([small_box + [80]], dtype=np.float32)
        fs = np.asarray([s], dtype=np.float32)
        preds[f"neg_{i}"] = {"fused_boxes": fb, "fused_scores": fs, "score": s}
        labels[f"neg_{i}"] = 0

    cfg = EvalConfig(
        large_threshold_grid=[0.05, 0.40, 0.70],
        small_threshold_grid=[0.10, 0.50, 0.70],
        box_size_split_mm=10.0,
    )
    result = grid_search_threshold(preds, labels, eval_cfg=cfg, target_fp=0.1)
    # large_thr=0.70 kills the 0.5x-score positives → sens drops to 0.
    assert result["best_large_thr"] < 0.70
    # small_thr=0.10 or 0.50 keeps the 0.6x-score negatives → FPs at FP=0.1
    # crowd out positives. Only small_thr=0.70 drops them.
    assert result["best_small_thr"] >= 0.70
    # The optimum sens should be the *maximum* observed in the grid.
    grid_sens = [g["sens_at_0.1fp"] for g in result["grid_table"]]
    assert result["best_sens_at_0.1fp"] == max(grid_sens)
    assert result["best_sens_at_0.1fp"] > 0.5  # at least half the positives recovered.
    # Grid table has every combo.
    assert len(result["grid_table"]) == 3 * 3
