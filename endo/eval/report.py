"""``eval_report.csv`` writer + ``eval_thresholds.json`` writer (Component 7 §4).

The CSV is **append-only**: existing rows are preserved across runs, and each
new run appends additional rows with a fresh ``run_id``. Atomic write via
temp-file-rename so a crash mid-write doesn't corrupt earlier rows.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


EVAL_REPORT_COLUMNS: list[str] = [
    "run_id",
    "entrypoint",
    "metric",
    "scope",
    "fold",
    "stratum_kind",
    "stratum_value",
    "rescored",
    "value",
    "ci_lower_95",
    "ci_upper_95",
    "n_patients",
    "n_lesions",
    "code_version",
]


@dataclass
class EvalReportRow:
    """One row of ``eval_report.csv`` (Component 7 §4.1)."""

    run_id: str
    entrypoint: str  # 'cv' | 'holdout'
    metric: str
    scope: str  # 'per_fold' | 'cv_pooled' | 'holdout'
    fold: int | None = None
    stratum_kind: str | None = None
    stratum_value: str | None = None
    rescored: bool = False
    value: float = float("nan")
    ci_lower_95: float = float("nan")
    ci_upper_95: float = float("nan")
    n_patients: int = 0
    n_lesions: int = 0
    code_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def append_eval_report(path: Path | str, rows: list[EvalReportRow]) -> None:
    """Append rows to ``eval_report.csv`` atomically.

    If the file does not exist, the header is written. Existing files are
    appended in place; on crash, the temp file is left behind (atomic rename
    only fires after a successful full write of new content).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    # Read existing content into memory (small files), append new rows, atomic
    # rename. This is the simplest way to keep the file sane across multiple
    # subprocesses (we serialize at the row level).
    existing_lines: list[str] = []
    if file_exists:
        with path.open("r", newline="") as f:
            existing_lines = f.readlines()

    tmp_dir = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", newline="") as f:
            writer = csv.writer(f)
            if not existing_lines:
                writer.writerow(EVAL_REPORT_COLUMNS)
            else:
                # Re-emit existing content verbatim so we preserve formatting.
                f.writelines(existing_lines)
            for row in rows:
                d = row.as_dict()
                writer.writerow([_format_value(d[c]) for c in EVAL_REPORT_COLUMNS])
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_eval_thresholds_json(path: Path | str, results: dict) -> None:
    """Write ``eval_thresholds.json`` (Component 7 §4.2).

    ``results`` shape:
        ``{'run_id': str, 'per_fold_thresholds': {'0': {'large': float,
        'small': float}, ...}, 'ensemble_threshold': {'large': float,
        'small': float}}``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str))
    os.replace(tmp, path)
