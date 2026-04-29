"""Stratified breakdowns (Component 7 §7).

For each ``(stratum_kind, stratum_value)`` we recompute volume-level metrics
on the subset of patients matching the stratum. Bootstrap is restricted to
patients within the stratum (PRD I.9.7).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from endo.config.eval import EvalConfig
from endo.eval.metrics import compute_volume_metrics


def _slice_thickness_bin(row: dict) -> str:
    """Bin the slice thickness from a manifest row to ``<=2mm`` / ``>2mm``."""
    st = row.get("slice_thickness_mm")
    if st is None:
        # fall back to variant: A is 1.5 mm reconstructed, B is 3.6 mm.
        variant = row.get("variant")
        if variant == "A":
            return "<=2mm"
        if variant == "B":
            return ">2mm"
        return "unknown"
    try:
        return "<=2mm" if float(st) <= 2.0 else ">2mm"
    except Exception:
        return "unknown"


def _stratum_key(row: dict, kind: str) -> str:
    if kind == "scanner_model" or kind == "scanner":
        return str(row.get("scanner_model") or row.get("scanner") or "unknown")
    if kind == "variant":
        return str(row.get("variant") or "unknown")
    if kind == "slice_thickness_bin":
        return _slice_thickness_bin(row)
    return str(row.get(kind, "unknown"))


def stratify_metrics(
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    manifest_rows: Sequence[dict] | Mapping[str, dict],
    strata: list[str] | None = None,
    eval_cfg: EvalConfig | None = None,
    *,
    raw_predictions: Mapping[str, dict] | None = None,
    gt_masks: Mapping[str, "np.ndarray"] | None = None,
) -> list[dict]:
    """Compute per-stratum volume metrics.

    ``manifest_rows`` may be a list (each with ``patient_id`` field) or a
    ``{patient_id: row}`` mapping.

    Returns a list of dicts, one per ``(stratum_kind, stratum_value)``:
    ``{'stratum_kind', 'stratum_value', 'metrics': {metric: {value, ci_lower,
    ci_upper}}, 'n_patients'}``.

    Raw vs thresholded split (audit 2026-04-29 §3.3): AUROC/AP are computed
    from ``raw_predictions`` (unfiltered fused scores) when provided, while
    FROC/sens@FP use the (thresholded) ``per_volume_predictions``.
    """
    cfg = eval_cfg if eval_cfg is not None else EvalConfig()
    if strata is None:
        strata = list(cfg.stratify_keys)

    if isinstance(manifest_rows, Mapping):
        lookup: dict[str, dict] = dict(manifest_rows)
    else:
        lookup = {r["patient_id"]: r for r in manifest_rows}

    out: list[dict] = []
    for kind in strata:
        # Bucket pids by stratum_value.
        buckets: dict[str, list[str]] = {}
        for pid in per_volume_predictions.keys():
            row = lookup.get(pid)
            if row is None:
                continue
            value = _stratum_key(row, kind)
            buckets.setdefault(value, []).append(pid)

        for value, pids in buckets.items():
            sub_preds = {p: per_volume_predictions[p] for p in pids}
            sub_labels = {p: int(per_volume_labels.get(p, 0)) for p in pids}
            sub_raw = (
                {p: raw_predictions[p] for p in pids if p in raw_predictions}
                if raw_predictions is not None
                else None
            )
            sub_masks = (
                {p: gt_masks[p] for p in pids if p in gt_masks}
                if gt_masks is not None
                else None
            )
            metrics = compute_volume_metrics(
                sub_preds,
                sub_labels,
                eval_cfg=cfg,
                raw_predictions=sub_raw,
                gt_masks=sub_masks,
            )
            out.append(
                {
                    "stratum_kind": kind,
                    "stratum_value": value,
                    "metrics": metrics,
                    "n_patients": len(pids),
                }
            )
    return out
