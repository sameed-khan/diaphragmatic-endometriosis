"""Lesion-level FROC with real GT masks (audit 2026-04-29 §3.4)."""

from __future__ import annotations

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.metrics import compute_volume_metrics


def _make_pred_for_pid(box_xz_y: tuple[float, float, float, float, int], score: float) -> dict:
    boxes = np.array([list(box_xz_y)], dtype=np.float32)
    scores = np.array([score], dtype=np.float32)
    return {"fused_boxes": boxes, "fused_scores": scores, "score": float(score)}


def test_lesion_mask_path_yields_finite_sensitivity():
    Y, Z, X = 60, 128, 128
    # One positive volume with a 4x6x6 lesion mask at (30, 64, 64); pred box overlaps it.
    pos_mask = np.zeros((Y, Z, X), dtype=np.uint8)
    pos_mask[28:32, 60:68, 60:68] = 1

    preds = {
        "pos1": _make_pred_for_pid((60.0, 60.0, 68.0, 68.0, 30), 0.8),
        "neg1": {
            "fused_boxes": np.zeros((0, 5), dtype=np.float32),
            "fused_scores": np.zeros((0,), dtype=np.float32),
            "score": 0.0,
        },
    }
    labels = {"pos1": 1, "neg1": 0}
    masks = {"pos1": pos_mask, "neg1": np.zeros((Y, Z, X), dtype=np.uint8)}

    cfg = EvalConfig()
    cfg.bootstrap_n = 1
    cfg.froc_fp_points = [0.5, 1.0]

    metrics = compute_volume_metrics(preds, labels, eval_cfg=cfg, gt_masks=masks)
    # Sensitivity at FP=0.5 should be 1.0 (one positive, one matching candidate).
    sens = metrics.get("sens_at_0.5fp", {}).get("value", float("nan"))
    assert sens == 1.0 or np.isclose(sens, 1.0)
