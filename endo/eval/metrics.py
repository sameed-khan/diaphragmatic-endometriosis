"""Volume-level metrics + bootstrap CIs (Component 7 §6.2, §6.3).

Per-patient resampling, sklearn AUROC/AP, and a thin bridge to FROC
(``compute_froc``) for sensitivity-at-FP points. All bootstraps are
patient-level (PRD I.9.6).
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.froc import compute_froc


# ----------------------------------------------------------------------------
# Bootstrap


def bootstrap_ci(
    values: Sequence,
    statistic_fn: Callable[[Sequence], float],
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Patient-level bootstrap CI (sample-with-replacement at the unit level).

    ``values`` is a 1D iterable of arbitrary per-patient items (tuples, dicts,
    floats, ...) that ``statistic_fn`` knows how to consume. Returns
    ``(low, high)`` at confidence ``1 - alpha``.
    """
    items = list(values)
    n_items = len(items)
    if n_items == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(int(n)):
        idx = rng.integers(0, n_items, size=n_items)
        sample = [items[i] for i in idx]
        try:
            v = float(statistic_fn(sample))
        except Exception:
            v = float("nan")
        if not math.isnan(v):
            stats.append(v)
    if not stats:
        return (float("nan"), float("nan"))
    arr = np.asarray(stats, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return (lo, hi)


# ----------------------------------------------------------------------------
# Metric primitives


def _volume_auroc(items: Sequence[tuple[float, int]]) -> float:
    """Items are ``(score, label)`` tuples (one per patient)."""
    if not items:
        return float("nan")
    scores = np.asarray([float(s) for s, _ in items], dtype=np.float64)
    labels = np.asarray([int(y) for _, y in items], dtype=np.int64)
    if labels.min() == labels.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _volume_ap(items: Sequence[tuple[float, int]]) -> float:
    if not items:
        return float("nan")
    scores = np.asarray([float(s) for s, _ in items], dtype=np.float64)
    labels = np.asarray([int(y) for _, y in items], dtype=np.int64)
    if labels.sum() == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def _bootstrap_fp_curves(
    pids: list[str],
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    fp_points: Sequence[float],
    n: int,
    seed: int,
) -> dict[float, tuple[float, float]]:
    """Single bootstrap pass yielding sens@<fp> CIs for *all* fp_points at
    once. We cache each unique pid's per-volume max-score-per-detection
    contribution; resampling becomes a re-aggregation over a multiset of
    pids — no picai_eval per-resample, no 3D canvas allocation per resample.

    Sensitivity at FP/vol is computed by the hand-rolled patient-level
    threshold sweep (the same primitive that powers ``_hand_rolled_froc``).
    """
    rng = np.random.default_rng(seed)
    n_items = len(pids)
    # Pre-extract score + label per pid.
    score_by_pid = {p: float(per_volume_predictions[p].get("score", 0.0)) for p in pids}
    label_by_pid = {p: int(per_volume_labels.get(p, 0)) for p in pids}

    out: dict[float, list[float]] = {fp: [] for fp in fp_points}

    for _ in range(int(n)):
        idx = rng.integers(0, n_items, size=n_items)
        sample_pids = [pids[i] for i in idx]
        scores = np.asarray([score_by_pid[p] for p in sample_pids], dtype=np.float64)
        labels = np.asarray([label_by_pid[p] for p in sample_pids], dtype=np.int64)
        n_total = len(sample_pids)
        n_pos = int(labels.sum())
        if n_pos == 0 or n_total == 0:
            for fp in fp_points:
                out[fp].append(float("nan"))
            continue
        order = np.argsort(-scores, kind="stable")
        tp = 0
        fp_count = 0
        fp_curve = []
        sens_curve = []
        for j in order:
            if labels[j] == 1:
                tp += 1
            else:
                fp_count += 1
            fp_curve.append(fp_count / n_total)
            sens_curve.append(tp / n_pos)
        fp_arr = np.asarray(fp_curve)
        sens_arr = np.asarray(sens_curve)
        for fp in fp_points:
            below = fp_arr <= float(fp)
            if below.any():
                out[fp].append(float(sens_arr[below][-1]))
            else:
                out[fp].append(0.0)

    cis: dict[float, tuple[float, float]] = {}
    for fp, vals in out.items():
        clean = [v for v in vals if not math.isnan(v)]
        if not clean:
            cis[fp] = (float("nan"), float("nan"))
        else:
            arr = np.asarray(clean, dtype=np.float64)
            cis[fp] = (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))
    return cis


# ----------------------------------------------------------------------------
# Top-level


def compute_volume_metrics(
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    eval_cfg: EvalConfig | None = None,
) -> dict:
    """Compute AUROC, AP, sens@{fp_points} with patient-level bootstrap CIs.

    Args:
        per_volume_predictions: ``{pid: {'score': float, 'fused_boxes': (M,5),
            'fused_scores': (M,), 'label': int (optional)}}``. ``label`` is
            sourced from ``per_volume_labels`` if missing.
        per_volume_labels: ``{pid: 0|1}``.
        eval_cfg: optional :class:`EvalConfig`; defaults are used if ``None``.

    Returns a dict mapping metric → ``{'value', 'ci_lower', 'ci_upper'}``.
    """
    cfg = eval_cfg if eval_cfg is not None else EvalConfig()
    pids = sorted(per_volume_predictions.keys())
    if not pids:
        return {}

    score_label_pairs: list[tuple[float, int]] = []
    for pid in pids:
        pred = per_volume_predictions[pid]
        score = float(pred.get("score", 0.0))
        label = int(per_volume_labels.get(pid, pred.get("label", 0)))
        score_label_pairs.append((score, label))

    # Point estimates.
    auroc = _volume_auroc(score_label_pairs)
    ap = _volume_ap(score_label_pairs)
    froc = compute_froc(
        {pid: per_volume_predictions[pid] for pid in pids},
        {pid: int(per_volume_labels.get(pid, 0)) for pid in pids},
        fp_per_volume_levels=tuple(cfg.froc_fp_points),
    )

    # Bootstrap.
    auroc_lo, auroc_hi = bootstrap_ci(
        score_label_pairs, _volume_auroc, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed
    )
    ap_lo, ap_hi = bootstrap_ci(
        score_label_pairs, _volume_ap, n=cfg.bootstrap_n, seed=cfg.bootstrap_seed + 1
    )

    out: dict[str, dict[str, float]] = {
        "volume_auroc": {"value": auroc, "ci_lower": auroc_lo, "ci_upper": auroc_hi},
        "ap": {"value": ap, "ci_lower": ap_lo, "ci_upper": ap_hi},
    }

    # Per-fp-point sensitivities + bootstrap (single pass over all FP points).
    fp_cis = _bootstrap_fp_curves(
        pids,
        per_volume_predictions,
        per_volume_labels,
        cfg.froc_fp_points,
        n=cfg.bootstrap_n,
        seed=cfg.bootstrap_seed + 7,
    )
    for fp_target in cfg.froc_fp_points:
        key = f"sens_at_{fp_target}fp"
        value = float(froc.get(f"sensitivity_at_{fp_target}", float("nan")))
        lo, hi = fp_cis.get(fp_target, (float("nan"), float("nan")))
        out[key] = {"value": value, "ci_lower": lo, "ci_upper": hi}

    out["n_patients"] = {"value": float(len(pids)), "ci_lower": float("nan"), "ci_upper": float("nan")}
    return out
