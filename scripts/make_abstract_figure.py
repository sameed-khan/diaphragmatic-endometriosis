"""RSNA abstract figure renderer.

Composes a 6-panel figure summarizing the 5-fold CV + holdout result for an
experiment: ROC, FROC, sens-by-scanner, sens-by-thickness, sens-by-lesion-volume,
and reliability diagram.

Usage::

    uv run scripts/make_abstract_figure.py --exp baseline-rtmdet-p2 \\
        --out runs/baseline-rtmdet-p2_*/eval/abstract_figure.pdf

Inputs (all auto-discovered under ``runs/<exp>_*/``):
  * eval/eval_report.csv            — CV metrics (per_fold + cv_pooled + strata)
  * eval/raw_preds_fold{0..4}.json  — per-fold per-volume max scores
  * eval/raw_preds_cv_pooled.json   — pooled per-volume max scores
  * eval/reliability_cv_pooled.json — reliability curve data
  * holdout/run_*/eval_report.csv             — holdout metrics
  * holdout/run_*/raw_preds_holdout.json      — holdout per-volume scores
  * holdout/run_*/reliability_holdout.json    — holdout reliability

Outputs the figure as PDF + PNG. Also writes a sibling ``abstract_numbers.json``
with the headline metrics + their 95 % CIs so the abstract text can be
populated without re-reading the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


# ----------------------------------------------------------------------------
# Path discovery


def _resolve_run_dir(runs_root: Path, exp_name: str) -> Path:
    candidates = sorted(runs_root.glob(f"{exp_name}_*"))
    if not candidates:
        raise SystemExit(f"No run dir matching {exp_name}_* under {runs_root}")
    if len(candidates) > 1:
        # Pick the most recently modified.
        candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _latest_holdout(run_dir: Path) -> Path | None:
    holds = sorted((run_dir / "holdout").glob("run_*"))
    if not holds:
        return None
    return holds[-1]


# ----------------------------------------------------------------------------
# IO


def _read_eval_report(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _read_raw_preds(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        return np.zeros((0,)), np.zeros((0,), dtype=np.int64)
    payload = json.loads(json_path.read_text())
    items = payload.get("predictions", {})
    scores = np.asarray([float(v["score"]) for v in items.values()], dtype=np.float64)
    labels = np.asarray([int(v["label"]) for v in items.values()], dtype=np.int64)
    return scores, labels


def _read_reliability(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text()).get("curve", [])


# ----------------------------------------------------------------------------
# Metric helpers


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _filter_rows(
    rows: Sequence[dict],
    *,
    metric: str | None = None,
    scope: str | None = None,
    stratum_kind: str | None = None,
    rescored: bool | None = None,
) -> list[dict]:
    out = []
    for r in rows:
        if metric is not None and r.get("metric") != metric:
            continue
        if scope is not None and r.get("scope") != scope:
            continue
        if stratum_kind is not None and r.get("stratum_kind") != stratum_kind:
            continue
        if rescored is not None:
            r_rescored = (str(r.get("rescored", "")).lower() == "true")
            if r_rescored != rescored:
                continue
        out.append(r)
    return out


def _per_fold_values(rows: Sequence[dict], metric: str) -> list[tuple[int, float, float, float]]:
    """Return (fold, value, ci_lo, ci_hi) for per_fold rows of a given metric."""
    out = []
    for r in _filter_rows(rows, metric=metric, scope="per_fold", stratum_kind=""):
        f = r.get("fold")
        if f in (None, ""):
            continue
        out.append(
            (
                int(f),
                _safe_float(r.get("value", "nan")),
                _safe_float(r.get("ci_lower_95", "nan")),
                _safe_float(r.get("ci_upper_95", "nan")),
            )
        )
    return sorted(out, key=lambda t: t[0])


def _pooled_value(rows: Sequence[dict], metric: str, scope: str = "cv_pooled") -> tuple[float, float, float]:
    matches = _filter_rows(rows, metric=metric, scope=scope, stratum_kind="")
    if not matches:
        return float("nan"), float("nan"), float("nan")
    r = matches[-1]
    return (
        _safe_float(r.get("value", "nan")),
        _safe_float(r.get("ci_lower_95", "nan")),
        _safe_float(r.get("ci_upper_95", "nan")),
    )


def _stratified_values(
    rows: Sequence[dict],
    metric: str,
    stratum_kind: str,
    scope: str = "cv_pooled",
) -> list[tuple[str, float, float, float]]:
    out = []
    for r in _filter_rows(rows, metric=metric, scope=scope, stratum_kind=stratum_kind):
        out.append(
            (
                str(r.get("stratum_value", "")),
                _safe_float(r.get("value", "nan")),
                _safe_float(r.get("ci_lower_95", "nan")),
                _safe_float(r.get("ci_upper_95", "nan")),
            )
        )
    return out


# ----------------------------------------------------------------------------
# Panel renderers


def _panel_roc(ax, fold_scores: list[np.ndarray], fold_labels: list[np.ndarray],
               pooled_scores: np.ndarray, pooled_labels: np.ndarray,
               holdout_scores: np.ndarray, holdout_labels: np.ndarray) -> None:
    grid = np.linspace(0.0, 1.0, 101)
    fold_tprs = []
    for s, y in zip(fold_scores, fold_labels):
        if y.size == 0 or y.max() == y.min():
            continue
        fpr, tpr, _ = roc_curve(y, s)
        tpr_i = np.interp(grid, fpr, tpr)
        tpr_i[0] = 0.0
        fold_tprs.append(tpr_i)
    if fold_tprs:
        arr = np.vstack(fold_tprs)
        mean_tpr = arr.mean(axis=0)
        ax.fill_between(grid, arr.min(axis=0), arr.max(axis=0), alpha=0.15,
                        color="C0", label="CV folds (min–max)")
        ax.plot(grid, mean_tpr, color="C0", lw=2, label="CV mean")
    if pooled_scores.size > 0 and pooled_labels.max() != pooled_labels.min():
        fpr, tpr, _ = roc_curve(pooled_labels, pooled_scores)
        ax.plot(fpr, tpr, color="C1", lw=2, ls="--", label="CV pooled")
    if holdout_scores.size > 0 and holdout_labels.max() != holdout_labels.min():
        fpr, tpr, _ = roc_curve(holdout_labels, holdout_scores)
        ax.plot(fpr, tpr, color="C3", lw=2, label="Holdout")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.5, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(A) Volume-level ROC")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)


def _panel_froc(ax, cv_rows: list[dict], holdout_rows: list[dict]) -> None:
    fp_levels: list[float] = []
    cv_vals: list[tuple[float, float, float]] = []
    holdout_vals: list[tuple[float, float, float]] = []

    metric_rows_cv = sorted(
        [r for r in _filter_rows(cv_rows, scope="cv_pooled", stratum_kind="")
         if r.get("metric", "").startswith("sens_at_") and r.get("metric", "").endswith("fp")],
        key=lambda r: float(r["metric"].replace("sens_at_", "").replace("fp", "")),
    )
    for r in metric_rows_cv:
        fp = float(r["metric"].replace("sens_at_", "").replace("fp", ""))
        fp_levels.append(fp)
        cv_vals.append((_safe_float(r.get("value")), _safe_float(r.get("ci_lower_95")),
                        _safe_float(r.get("ci_upper_95"))))

    holdout_lookup = {
        float(r["metric"].replace("sens_at_", "").replace("fp", "")): r
        for r in _filter_rows(holdout_rows, scope="holdout", stratum_kind="")
        if r.get("metric", "").startswith("sens_at_") and r.get("metric", "").endswith("fp")
    }
    for fp in fp_levels:
        r = holdout_lookup.get(fp)
        if r is None:
            holdout_vals.append((float("nan"), float("nan"), float("nan")))
        else:
            holdout_vals.append((_safe_float(r.get("value")), _safe_float(r.get("ci_lower_95")),
                                 _safe_float(r.get("ci_upper_95"))))

    if not fp_levels:
        ax.set_axis_off(); return

    fp_arr = np.asarray(fp_levels)
    cv_arr = np.asarray(cv_vals)
    ho_arr = np.asarray(holdout_vals)

    ax.errorbar(fp_arr, cv_arr[:, 0],
                yerr=[cv_arr[:, 0] - cv_arr[:, 1], cv_arr[:, 2] - cv_arr[:, 0]],
                fmt="o-", color="C0", lw=2, label="CV pooled (95 % CI)")
    if not np.all(np.isnan(ho_arr[:, 0])):
        ax.errorbar(fp_arr, ho_arr[:, 0],
                    yerr=[np.nan_to_num(ho_arr[:, 0] - ho_arr[:, 1]),
                          np.nan_to_num(ho_arr[:, 2] - ho_arr[:, 0])],
                    fmt="s--", color="C3", lw=2, label="Holdout")
    ax.set_xscale("log")
    ax.set_xlabel("FP per volume")
    ax.set_ylabel("Lesion sensitivity")
    ax.set_title("(B) FROC")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)


def _panel_strata_bars(ax, cv_rows: list[dict], holdout_rows: list[dict],
                       *, metric: str, stratum_kind: str, title: str,
                       order: list[str] | None = None) -> None:
    cv = _stratified_values(cv_rows, metric=metric, stratum_kind=stratum_kind, scope="cv_pooled")
    ho = _stratified_values(holdout_rows, metric=metric, stratum_kind=stratum_kind, scope="holdout")
    if order is None:
        order = sorted({s[0] for s in cv} | {s[0] for s in ho})
    if not order:
        ax.set_axis_off(); return
    cv_lookup = {s[0]: s for s in cv}
    ho_lookup = {s[0]: s for s in ho}
    x = np.arange(len(order))
    w = 0.38
    cv_vals = np.asarray([cv_lookup.get(k, (k, np.nan, np.nan, np.nan))[1] for k in order])
    cv_lo = np.asarray([cv_lookup.get(k, (k, np.nan, np.nan, np.nan))[2] for k in order])
    cv_hi = np.asarray([cv_lookup.get(k, (k, np.nan, np.nan, np.nan))[3] for k in order])
    ho_vals = np.asarray([ho_lookup.get(k, (k, np.nan, np.nan, np.nan))[1] for k in order])
    ho_lo = np.asarray([ho_lookup.get(k, (k, np.nan, np.nan, np.nan))[2] for k in order])
    ho_hi = np.asarray([ho_lookup.get(k, (k, np.nan, np.nan, np.nan))[3] for k in order])

    ax.bar(x - w / 2, np.nan_to_num(cv_vals), w, color="C0", label="CV pooled",
           yerr=[np.nan_to_num(cv_vals - cv_lo), np.nan_to_num(cv_hi - cv_vals)],
           capsize=3)
    ax.bar(x + w / 2, np.nan_to_num(ho_vals), w, color="C3", label="Holdout",
           yerr=[np.nan_to_num(ho_vals - ho_lo), np.nan_to_num(ho_hi - ho_vals)],
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)


def _panel_reliability(ax, cv_curve: list[dict], ho_curve: list[dict]) -> None:
    if cv_curve:
        cv_pred = np.asarray([b["mean_pred"] for b in cv_curve], dtype=np.float64)
        cv_obs = np.asarray([b["frac_pos"] for b in cv_curve], dtype=np.float64)
        cv_n = np.asarray([b["count"] for b in cv_curve], dtype=np.int64)
        m = ~np.isnan(cv_pred) & ~np.isnan(cv_obs) & (cv_n > 0)
        if m.any():
            ax.plot(cv_pred[m], cv_obs[m], "o-", color="C0", lw=2, label="CV pooled")
    if ho_curve:
        ho_pred = np.asarray([b["mean_pred"] for b in ho_curve], dtype=np.float64)
        ho_obs = np.asarray([b["frac_pos"] for b in ho_curve], dtype=np.float64)
        ho_n = np.asarray([b["count"] for b in ho_curve], dtype=np.int64)
        m = ~np.isnan(ho_pred) & ~np.isnan(ho_obs) & (ho_n > 0)
        if m.any():
            ax.plot(ho_pred[m], ho_obs[m], "s--", color="C3", lw=2, label="Holdout")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("(F) Calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)


# ----------------------------------------------------------------------------
# Top-level


def render(run_dir: Path, holdout_dir: Path | None, out_path: Path) -> dict:
    eval_dir = run_dir / "eval"
    cv_rows = _read_eval_report(eval_dir / "eval_report.csv")
    holdout_rows = _read_eval_report(holdout_dir / "eval_report.csv") if holdout_dir else []

    fold_scores: list[np.ndarray] = []
    fold_labels: list[np.ndarray] = []
    for f in range(5):
        s, y = _read_raw_preds(eval_dir / f"raw_preds_fold{f}.json")
        if s.size > 0:
            fold_scores.append(s); fold_labels.append(y)

    pooled_s, pooled_y = _read_raw_preds(eval_dir / "raw_preds_cv_pooled.json")
    holdout_s, holdout_y = (
        _read_raw_preds(holdout_dir / "raw_preds_holdout.json") if holdout_dir else (np.zeros((0,)), np.zeros((0,), dtype=np.int64))
    )

    cv_curve = _read_reliability(eval_dir / "reliability_cv_pooled.json")
    ho_curve = _read_reliability(holdout_dir / "reliability_holdout.json") if holdout_dir else []

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _panel_roc(axes[0, 0], fold_scores, fold_labels, pooled_s, pooled_y, holdout_s, holdout_y)
    _panel_froc(axes[0, 1], cv_rows, holdout_rows)
    _panel_strata_bars(axes[0, 2], cv_rows, holdout_rows,
                       metric="sens_at_2.0fp", stratum_kind="scanner_model",
                       title="(C) Sens@2FP by scanner")
    _panel_strata_bars(axes[1, 0], cv_rows, holdout_rows,
                       metric="sens_at_2.0fp", stratum_kind="slice_thickness_bin",
                       title="(D) Sens@2FP by slice thickness")
    _panel_strata_bars(axes[1, 1], cv_rows, holdout_rows,
                       metric="lesion_sensitivity", stratum_kind="lesion_volume_bin",
                       title="(E) Lesion sens by volume",
                       order=["<=200mm3", "200-1000mm3", "1000-5000mm3", ">5000mm3"])
    _panel_reliability(axes[1, 2], cv_curve, ho_curve)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    headline = {
        "cv_pooled": {
            "volume_auroc": _pooled_value(cv_rows, "volume_auroc", "cv_pooled"),
            "ap": _pooled_value(cv_rows, "ap", "cv_pooled"),
            "sens_at_2.0fp": _pooled_value(cv_rows, "sens_at_2.0fp", "cv_pooled"),
            "brier": _pooled_value(cv_rows, "brier", "cv_pooled"),
            "ece": _pooled_value(cv_rows, "ece", "cv_pooled"),
        },
        "holdout": {
            "volume_auroc": _pooled_value(holdout_rows, "volume_auroc", "holdout"),
            "ap": _pooled_value(holdout_rows, "ap", "holdout"),
            "sens_at_2.0fp": _pooled_value(holdout_rows, "sens_at_2.0fp", "holdout"),
            "brier": _pooled_value(holdout_rows, "brier", "holdout"),
            "ece": _pooled_value(holdout_rows, "ece", "holdout"),
        },
        "per_fold_volume_auroc": _per_fold_values(cv_rows, "volume_auroc"),
    }
    def _sanitize(obj):
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    out_path.with_name("abstract_numbers.json").write_text(json.dumps(_sanitize(headline), indent=2, default=str))
    return headline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exp", required=True, help="experiment name (e.g., baseline-rtmdet-p2)")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=None,
                   help="output PDF path (defaults to runs/<exp>_*/eval/abstract_figure.pdf)")
    args = p.parse_args(argv)

    run_dir = _resolve_run_dir(args.runs_root, args.exp)
    holdout_dir = _latest_holdout(run_dir)
    out_path = args.out if args.out is not None else (run_dir / "eval" / "abstract_figure.pdf")
    headline = render(run_dir, holdout_dir, out_path)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.png')}")
    print(f"Wrote {out_path.with_name('abstract_numbers.json')}")
    print(json.dumps(headline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
