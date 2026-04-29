"""Smoke test for scripts/make_abstract_figure.py — runs end-to-end against
synthetic CSV/JSON inputs and verifies the figure + JSON sidecar are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pytest


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    cols = (
        "run_id,entrypoint,metric,scope,fold,stratum_kind,stratum_value,"
        "rescored,value,ci_lower_95,ci_upper_95,n_patients,n_lesions,code_version"
    ).split(",")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def _make_preds(n_pos: int, n_neg: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    preds = {}
    for i in range(n_pos):
        preds[f"p{i}"] = {"score": float(np.clip(rng.normal(0.7, 0.1), 0, 1)), "label": 1}
    for i in range(n_neg):
        preds[f"n{i}"] = {"score": float(np.clip(rng.normal(0.3, 0.1), 0, 1)), "label": 0}
    return {"predictions": preds}


def test_figure_renders_to_pdf_and_png(tmp_path):
    """End-to-end render of the 6-panel figure with full inputs present."""
    from scripts.make_abstract_figure import render

    run_dir = tmp_path / "runs" / "exp_00000000"
    eval_dir = run_dir / "eval"
    holdout_dir = run_dir / "holdout" / "run_x"

    cv_rows = [
        ["r1", "cv", "volume_auroc", "cv_pooled", "", "", "", False, 0.85, 0.78, 0.91, 486, 86, "abc"],
        ["r1", "cv", "ap", "cv_pooled", "", "", "", False, 0.62, 0.5, 0.74, 486, 86, "abc"],
        ["r1", "cv", "sens_at_2.0fp", "cv_pooled", "", "", "", False, 0.7, 0.6, 0.8, 486, 86, "abc"],
        ["r1", "cv", "sens_at_0.5fp", "cv_pooled", "", "", "", False, 0.45, 0.35, 0.55, 486, 86, "abc"],
        ["r1", "cv", "brier", "cv_pooled", "", "", "", False, 0.10, 0.08, 0.13, 486, 86, "abc"],
        ["r1", "cv", "ece", "cv_pooled", "", "", "", False, 0.05, 0.03, 0.07, 486, 86, "abc"],
        ["r1", "cv", "volume_auroc", "per_fold", 0, "", "", False, 0.83, 0.70, 0.93, 100, 18, "abc"],
        ["r1", "cv", "volume_auroc", "per_fold", 1, "", "", False, 0.86, 0.74, 0.95, 99, 18, "abc"],
        ["r1", "cv", "sens_at_2.0fp", "cv_pooled", "", "scanner_model", "SIGNA Artist", False, 0.78, 0.65, 0.88, 200, 40, "abc"],
        ["r1", "cv", "sens_at_2.0fp", "cv_pooled", "", "slice_thickness_bin", "<=2mm", False, 0.75, 0.62, 0.85, 250, 42, "abc"],
        ["r1", "cv", "lesion_sensitivity", "cv_pooled", "", "lesion_volume_bin", "<=200mm3", False, 0.45, 0.32, 0.58, 0, 30, "abc"],
        ["r1", "cv", "lesion_sensitivity", "cv_pooled", "", "lesion_volume_bin", "200-1000mm3", False, 0.7, 0.55, 0.85, 0, 40, "abc"],
    ]
    holdout_rows = [
        ["h1", "holdout", "volume_auroc", "holdout", "", "", "", False, 0.82, 0.72, 0.90, 122, 22, "abc"],
        ["h1", "holdout", "sens_at_2.0fp", "holdout", "", "", "", False, 0.68, 0.55, 0.78, 122, 22, "abc"],
        ["h1", "holdout", "sens_at_0.5fp", "holdout", "", "", "", False, 0.4, 0.28, 0.52, 122, 22, "abc"],
        ["h1", "holdout", "brier", "holdout", "", "", "", False, 0.12, 0.09, 0.15, 122, 22, "abc"],
    ]
    _write_csv(eval_dir / "eval_report.csv", cv_rows)
    _write_csv(holdout_dir / "eval_report.csv", holdout_rows)

    eval_dir.mkdir(parents=True, exist_ok=True)
    for f in range(3):
        (eval_dir / f"raw_preds_fold{f}.json").write_text(json.dumps(_make_preds(18, 80, seed=f)))
    (eval_dir / "raw_preds_cv_pooled.json").write_text(json.dumps(_make_preds(86, 400, seed=42)))
    (holdout_dir / "raw_preds_holdout.json").write_text(json.dumps(_make_preds(22, 100, seed=99)))
    (eval_dir / "reliability_cv_pooled.json").write_text(json.dumps({"curve": [
        {"bin_low": i * 0.1, "bin_high": (i + 1) * 0.1, "mean_pred": i * 0.1 + 0.05,
         "frac_pos": min(1.0, max(0.0, (i + 1) / 10)), "count": 5}
        for i in range(10)
    ]}))
    (holdout_dir / "reliability_holdout.json").write_text(json.dumps({"curve": []}))

    out = eval_dir / "abstract_figure.pdf"
    headline = render(run_dir, holdout_dir, out)

    assert out.exists() and out.stat().st_size > 1000
    assert out.with_suffix(".png").exists()
    sidecar = out.with_name("abstract_numbers.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert "cv_pooled" in payload and "holdout" in payload
    assert payload["cv_pooled"]["volume_auroc"][0] == pytest.approx(0.85)
    assert payload["holdout"]["volume_auroc"][0] == pytest.approx(0.82)


def test_figure_handles_missing_holdout(tmp_path):
    """Render must succeed even if holdout artifacts are absent."""
    from scripts.make_abstract_figure import render

    run_dir = tmp_path / "runs" / "exp_00000000"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)

    cv_rows = [
        ["r1", "cv", "volume_auroc", "cv_pooled", "", "", "", False, 0.85, 0.78, 0.91, 486, 86, "abc"],
        ["r1", "cv", "sens_at_2.0fp", "cv_pooled", "", "", "", False, 0.7, 0.6, 0.8, 486, 86, "abc"],
    ]
    _write_csv(eval_dir / "eval_report.csv", cv_rows)
    (eval_dir / "raw_preds_cv_pooled.json").write_text(json.dumps(_make_preds(86, 400)))

    out = eval_dir / "abstract_figure.pdf"
    headline = render(run_dir, holdout_dir=None, out_path=out)
    assert out.exists()
    payload = json.loads(out.with_name("abstract_numbers.json").read_text())
    # Holdout entries are present but null when unavailable.
    assert payload["holdout"]["volume_auroc"][0] is None
