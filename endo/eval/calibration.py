"""Volume-level calibration metrics: Brier score, ECE, and reliability curve.

These are computed from the same ``(score, label)`` pairs that drive
``compute_volume_metrics`` and rely on the same patient-level bootstrap
primitive. Reliability curve data is persisted as a JSON sidecar so the
abstract figure can render it later without re-running inference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def brier_score(score_label_pairs: Sequence[tuple[float, int]]) -> float:
    """Mean squared error between predicted score and binary label.

    Lower is better. A constant predictor at the prevalence p yields
    p(1-p); a perfect predictor yields 0.
    """
    if not score_label_pairs:
        return float("nan")
    s = np.asarray([float(x[0]) for x in score_label_pairs], dtype=np.float64)
    y = np.asarray([float(x[1]) for x in score_label_pairs], dtype=np.float64)
    return float(np.mean((s - y) ** 2))


def expected_calibration_error(
    score_label_pairs: Sequence[tuple[float, int]],
    n_bins: int = 10,
) -> float:
    """Equal-width-bin ECE across [0, 1].

    Empty bins contribute 0. Returns NaN when input is empty.
    """
    if not score_label_pairs:
        return float("nan")
    s = np.asarray([float(x[0]) for x in score_label_pairs], dtype=np.float64)
    y = np.asarray([float(x[1]) for x in score_label_pairs], dtype=np.float64)
    n = s.size
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        if i == int(n_bins) - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not mask.any():
            continue
        conf = float(s[mask].mean())
        acc = float(y[mask].mean())
        weight = float(mask.sum()) / float(n)
        ece += weight * abs(conf - acc)
    return float(ece)


def reliability_curve(
    score_label_pairs: Sequence[tuple[float, int]],
    n_bins: int = 10,
) -> list[dict]:
    """Equal-width reliability curve.

    Each entry: ``{bin_low, bin_high, mean_pred, frac_pos, count}``.
    Empty bins are emitted with ``count=0`` and NaN ``mean_pred``/``frac_pos``
    so the figure can plot bin centers consistently.
    """
    if not score_label_pairs:
        return []
    s = np.asarray([float(x[0]) for x in score_label_pairs], dtype=np.float64)
    y = np.asarray([float(x[1]) for x in score_label_pairs], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out: list[dict] = []
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        if i == int(n_bins) - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        count = int(mask.sum())
        if count == 0:
            out.append({
                "bin_low": float(lo),
                "bin_high": float(hi),
                "mean_pred": float("nan"),
                "frac_pos": float("nan"),
                "count": 0,
            })
            continue
        out.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "mean_pred": float(s[mask].mean()),
            "frac_pos": float(y[mask].mean()),
            "count": count,
        })
    return out
