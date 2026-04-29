"""Lesion-level sensitivity stratified by lesion volume.

Operates on the per-call records produced by :func:`endo.eval.calls.build_call_records`.
For each volume bin we compute lesion-level sensitivity = TP / (TP + FN),
plus a bootstrap CI by resampling lesions (TP+FN units) with replacement.

Bins are defined in mm³. Defaults span the diaphragmatic-endometriosis
range observed in the dataset (median lesion ≈ a few hundred mm³, with
a long tail). They can be overridden by callers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


# Volume bin edges in mm³ (right-open except the last). Tuned to give roughly
# balanced lesion counts across the cohort (verify before reporting).
DEFAULT_VOLUME_EDGES_MM3: tuple[float, ...] = (0.0, 200.0, 1000.0, 5000.0, float("inf"))


def _bin_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">{lo:g}mm3"
    if lo == 0.0:
        return f"<={hi:g}mm3"
    return f"{lo:g}-{hi:g}mm3"


def _bin_index(volume_mm3: float, edges: Sequence[float]) -> int:
    v = float(volume_mm3)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if lo <= v <= hi:
                return i
        else:
            if lo <= v < hi:
                return i
    return -1


def lesion_units_from_calls(
    call_records: Iterable[dict],
    edges_mm3: Sequence[float] = DEFAULT_VOLUME_EDGES_MM3,
) -> list[tuple[int, int]]:
    """Reduce call records to ``(bin_index, hit)`` lesion-level units.

    A *lesion unit* is one GT lesion: TP records (one per matched lesion) yield
    ``hit=1``, FN records yield ``hit=0``. FP records are ignored — this is a
    *recall* (sensitivity) breakdown, not precision.
    """
    units: list[tuple[int, int]] = []
    for r in call_records:
        ct = r.get("call_type")
        if ct not in ("tp", "fn"):
            continue
        v = r.get("volume_mm3")
        if v is None:
            continue
        bi = _bin_index(float(v), edges_mm3)
        if bi < 0:
            continue
        units.append((bi, 1 if ct == "tp" else 0))
    return units


def _sensitivity(items: Sequence[tuple[int, int]]) -> float:
    if not items:
        return float("nan")
    hits = sum(int(h) for _, h in items)
    return float(hits) / float(len(items))


def compute_lesion_volume_strata(
    call_records: Iterable[dict],
    edges_mm3: Sequence[float] = DEFAULT_VOLUME_EDGES_MM3,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Compute per-volume-bin lesion sensitivity with bootstrap CIs.

    Returns a list of dicts ready to feed the eval-report row builder:

    .. code-block::

        [
          {
            "stratum_kind": "lesion_volume_bin",
            "stratum_value": "<=200mm3",
            "metrics": {
              "lesion_sensitivity": {"value", "ci_lower", "ci_upper"},
            },
            "n_lesions": int,
            "n_tp": int,
            "n_fn": int,
          },
          ...
        ]
    """
    edges = list(edges_mm3)
    units = lesion_units_from_calls(call_records, edges_mm3=edges)
    rng = np.random.default_rng(int(seed))

    by_bin: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(edges) - 1)}
    for bi, h in units:
        by_bin.setdefault(bi, []).append((bi, int(h)))

    results: list[dict] = []
    for i in range(len(edges) - 1):
        items = by_bin.get(i, [])
        lo, hi = edges[i], edges[i + 1]
        label = _bin_label(lo, hi)
        n_lesions = len(items)
        n_tp = sum(int(h) for _, h in items)
        n_fn = n_lesions - n_tp

        point = _sensitivity(items)
        if n_lesions == 0:
            ci_lo, ci_hi = float("nan"), float("nan")
        else:
            stats: list[float] = []
            arr_idx = np.arange(n_lesions)
            hits_arr = np.asarray([h for _, h in items], dtype=np.float64)
            for _ in range(int(bootstrap_n)):
                idx = rng.integers(0, n_lesions, size=n_lesions)
                stats.append(float(hits_arr[idx].mean()))
            arr = np.asarray(stats, dtype=np.float64)
            ci_lo = float(np.quantile(arr, 0.025))
            ci_hi = float(np.quantile(arr, 0.975))

        results.append({
            "stratum_kind": "lesion_volume_bin",
            "stratum_value": label,
            "metrics": {
                "lesion_sensitivity": {
                    "value": point,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                },
            },
            "n_lesions": n_lesions,
            "n_tp": n_tp,
            "n_fn": n_fn,
        })
    return results
