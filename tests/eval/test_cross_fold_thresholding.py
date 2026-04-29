"""Cross-fold threshold tuning (audit 2026-04-29 §3.2).

Verifies that for fold f, thresholds are tuned only on the union of the
OTHER folds' raw preds — no leakage from fold f's own val set.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.run_eval import _ensemble_threshold, _tune_thresholds_cross_fold


def _fake_fold_preds(n_folds: int = 5, per_fold: int = 4):
    raw_by_fold = {}
    labels_by_fold = {}
    for f in range(n_folds):
        preds = {}
        labels = {}
        for i in range(per_fold):
            pid = f"f{f}_p{i}"
            preds[pid] = {
                "fused_boxes": np.array([[10, 10, 30, 30, 5]], dtype=np.float32),
                "fused_scores": np.array([0.6], dtype=np.float32),
                "score": 0.6,
            }
            labels[pid] = i % 2
        raw_by_fold[f] = preds
        labels_by_fold[f] = labels
    return raw_by_fold, labels_by_fold


def test_tune_excludes_target_fold():
    raw_by_fold, labels_by_fold = _fake_fold_preds()
    cfg = EvalConfig()
    cfg.large_threshold_grid = [0.05, 0.1]
    cfg.small_threshold_grid = [0.30]
    cfg.bootstrap_n = 1

    seen_pids: list[set[str]] = []

    def _spy_grid_search(per_volume_predictions, per_volume_labels, *, eval_cfg):
        seen_pids.append(set(per_volume_predictions.keys()))
        return {"best_large_thr": 0.05, "best_small_thr": 0.30, "grid_table": []}

    with patch(
        "endo.eval.run_eval.grid_search_threshold", side_effect=_spy_grid_search
    ):
        _tune_thresholds_cross_fold(raw_by_fold, labels_by_fold, cfg)

    # 5 calls, one per fold. For fold f, none of the seen pids should belong to fold f.
    assert len(seen_pids) == 5
    for f, pids in enumerate(seen_pids):
        for pid in pids:
            assert not pid.startswith(f"f{f}_"), (
                f"fold {f} threshold tuning saw its own pid {pid}"
            )


def test_ensemble_threshold_is_mean():
    per_fold = {
        0: {"large": 0.04, "small": 0.20},
        1: {"large": 0.06, "small": 0.30},
        2: {"large": 0.08, "small": 0.40},
    }
    ens = _ensemble_threshold(per_fold)
    assert np.isclose(ens["large"], 0.06)
    assert np.isclose(ens["small"], 0.30)
