"""Tests E5, E6 — volume metrics + bootstrap CI (PRD §11.8)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from endo.config.eval import EvalConfig
from endo.eval.metrics import bootstrap_ci, compute_volume_metrics


def _synth_predictions(n_pos: int, n_neg: int, seed: int = 0) -> tuple[dict, dict]:
    """Generate easy-to-classify synthetic per-volume predictions:
    positives have score ~ N(0.7, 0.1), negatives ~ N(0.3, 0.1)."""
    rng = np.random.default_rng(seed)
    preds: dict[str, dict] = {}
    labels: dict[str, int] = {}
    for i in range(n_pos):
        pid = f"pos_{i}"
        preds[pid] = {
            "fused_boxes": np.zeros((0, 5), dtype=np.float32),
            "fused_scores": np.zeros((0,), dtype=np.float32),
            "score": float(np.clip(rng.normal(0.7, 0.1), 0, 1)),
        }
        labels[pid] = 1
    for i in range(n_neg):
        pid = f"neg_{i}"
        preds[pid] = {
            "fused_boxes": np.zeros((0, 5), dtype=np.float32),
            "fused_scores": np.zeros((0,), dtype=np.float32),
            "score": float(np.clip(rng.normal(0.3, 0.1), 0, 1)),
        }
        labels[pid] = 0
    return preds, labels


def test_compute_volume_metrics_smoke():
    """E5: 10 vols (5 pos, 5 neg) → all expected keys, no NaN in point estimates."""
    preds, labels = _synth_predictions(5, 5)
    cfg = EvalConfig(bootstrap_n=50)  # small n for speed
    metrics = compute_volume_metrics(preds, labels, eval_cfg=cfg)
    assert "volume_auroc" in metrics
    assert "ap" in metrics
    # Every configured FP point should yield a sens_at_<fp>fp entry.
    for fp in cfg.froc_fp_points:
        assert f"sens_at_{fp}fp" in metrics
    # Point estimates should be finite for AUROC (synthetic separable data).
    val = metrics["volume_auroc"]["value"]
    assert not math.isnan(val)
    assert val > 0.5


def test_bootstrap_ci_widens_with_fewer_patients():
    """E6: 50 patients → wider CI than 200 patients."""
    preds_small, labels_small = _synth_predictions(25, 25, seed=1)
    preds_big, labels_big = _synth_predictions(100, 100, seed=2)
    cfg = EvalConfig(bootstrap_n=200)

    m_small = compute_volume_metrics(preds_small, labels_small, eval_cfg=cfg)
    m_big = compute_volume_metrics(preds_big, labels_big, eval_cfg=cfg)

    width_small = m_small["volume_auroc"]["ci_upper"] - m_small["volume_auroc"]["ci_lower"]
    width_big = m_big["volume_auroc"]["ci_upper"] - m_big["volume_auroc"]["ci_lower"]
    assert width_small > width_big


def test_bootstrap_ci_basic():
    values = [(float(i), 0) for i in range(100)]
    lo, hi = bootstrap_ci(values, lambda items: float(np.mean([v for v, _ in items])), n=200)
    # Mean of 0..99 is 49.5; CI should bracket it.
    assert lo <= 49.5 <= hi
