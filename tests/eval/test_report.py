"""Test E15 — eval_report.csv is append-only across runs."""

from __future__ import annotations

import csv
from pathlib import Path

from endo.eval.report import (
    EVAL_REPORT_COLUMNS,
    EvalReportRow,
    append_eval_report,
    write_eval_thresholds_json,
)


def _make_row(run_id: str, metric: str = "volume_auroc", value: float = 0.8) -> EvalReportRow:
    return EvalReportRow(
        run_id=run_id,
        entrypoint="cv",
        metric=metric,
        scope="per_fold",
        fold=0,
        rescored=False,
        value=value,
        ci_lower_95=value - 0.05,
        ci_upper_95=value + 0.05,
        n_patients=100,
        n_lesions=20,
        code_version="abc1234",
    )


def test_eval_csv_append_only(tmp_path: Path):
    """E15: Write twice; second write preserves first run's rows."""
    csv_path = tmp_path / "eval_report.csv"

    # Run 1.
    rows1 = [_make_row("run_001", "volume_auroc", 0.80), _make_row("run_001", "ap", 0.65)]
    append_eval_report(csv_path, rows1)

    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert all(r["run_id"] == "run_001" for r in reader)
    assert reader[0]["metric"] == "volume_auroc"

    # Run 2.
    rows2 = [_make_row("run_002", "volume_auroc", 0.82), _make_row("run_002", "ap", 0.70)]
    append_eval_report(csv_path, rows2)

    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 4
    run_ids = {r["run_id"] for r in reader}
    assert run_ids == {"run_001", "run_002"}

    # First-run rows still intact.
    run1_rows = [r for r in reader if r["run_id"] == "run_001"]
    assert len(run1_rows) == 2
    metrics_run1 = {r["metric"] for r in run1_rows}
    assert metrics_run1 == {"volume_auroc", "ap"}


def test_eval_csv_header_columns(tmp_path: Path):
    csv_path = tmp_path / "eval_report.csv"
    append_eval_report(csv_path, [_make_row("run_x")])
    with csv_path.open() as f:
        first_line = f.readline().strip()
    assert first_line == ",".join(EVAL_REPORT_COLUMNS)


def test_write_eval_thresholds_json(tmp_path: Path):
    path = tmp_path / "eval_thresholds.json"
    write_eval_thresholds_json(
        path,
        {
            "run_id": "cv_x",
            "per_fold_thresholds": {"0": {"large": 0.05, "small": 0.30}},
            "ensemble_threshold": {"large": 0.04, "small": 0.28},
        },
    )
    import json

    data = json.loads(path.read_text())
    assert data["run_id"] == "cv_x"
    assert data["ensemble_threshold"]["large"] == 0.04
