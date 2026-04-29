"""Per-fold and CV-pooled WBF threshold grid search (Component 7 §6.4).

Maximizes ``sens@2FP/vol`` over the cartesian product of
``(large_threshold_grid, small_threshold_grid)``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.froc import compute_froc
from endo.eval.wbf import _box_max_dim_mm


def _apply_thresholds_inplace(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    large_thr: float,
    small_thr: float,
    box_size_threshold_mm: float,
    inplane_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter the (already fused) boxes by score, gated on physical size."""
    if fused_boxes.size == 0:
        return fused_boxes, fused_scores
    boxes_xz = fused_boxes[:, :4]
    max_dim_mm = _box_max_dim_mm(boxes_xz, inplane_mm=inplane_mm)
    is_large = max_dim_mm >= float(box_size_threshold_mm)
    keep = np.where(
        is_large, fused_scores >= float(large_thr), fused_scores >= float(small_thr)
    )
    return fused_boxes[keep], fused_scores[keep]


def grid_search_threshold(
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    large_grid: list[float] | None = None,
    small_grid: list[float] | None = None,
    eval_cfg: EvalConfig | None = None,
    target_fp: float = 2.0,
) -> dict:
    """Grid-search ``(large_thr, small_thr)`` to maximize sens@``target_fp``.

    ``per_volume_predictions[pid]`` is the *unfiltered* WBF output:
    ``{'fused_boxes': (M,5), 'fused_scores': (M,)}``. Each grid cell rebuilds
    the per-volume score (max of surviving fused scores) and recomputes FROC.

    Returns:
        ``{'best_large_thr': float, 'best_small_thr': float,
            'best_sens_at_2fp': float, 'grid_table': list[dict]}``.
    """
    cfg = eval_cfg if eval_cfg is not None else EvalConfig()
    if large_grid is None:
        large_grid = list(cfg.large_threshold_grid)
    if small_grid is None:
        small_grid = list(cfg.small_threshold_grid)

    inplane_mm = 0.82
    box_size_split_mm = float(cfg.box_size_split_mm)

    pids = sorted(per_volume_predictions.keys())
    grid_table: list[dict] = []
    best_score = -1.0
    best_large = float(large_grid[0])
    best_small = float(small_grid[0])

    for lt in large_grid:
        for st in small_grid:
            preds_filtered: dict[str, dict] = {}
            for pid in pids:
                src = per_volume_predictions[pid]
                fb = np.asarray(src.get("fused_boxes", np.zeros((0, 5))), dtype=np.float32)
                fs = np.asarray(src.get("fused_scores", np.zeros((0,))), dtype=np.float32)
                fb2, fs2 = _apply_thresholds_inplace(
                    fb, fs, lt, st, box_size_split_mm, inplane_mm
                )
                vol_score = float(fs2.max()) if fs2.size > 0 else 0.0
                preds_filtered[pid] = {
                    "fused_boxes": fb2,
                    "fused_scores": fs2,
                    "score": vol_score,
                }
            froc = compute_froc(
                preds_filtered, per_volume_labels, fp_per_volume_levels=(target_fp,)
            )
            sens = float(froc.get(f"sensitivity_at_{target_fp}", float("nan")))
            grid_table.append(
                {"large_thr": float(lt), "small_thr": float(st), f"sens_at_{target_fp}fp": sens}
            )
            if not np.isnan(sens) and sens > best_score:
                best_score = sens
                best_large = float(lt)
                best_small = float(st)

    return {
        "best_large_thr": best_large,
        "best_small_thr": best_small,
        f"best_sens_at_{target_fp}fp": float(best_score) if best_score >= 0 else float("nan"),
        "grid_table": grid_table,
    }
