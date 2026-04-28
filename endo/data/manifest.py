"""Manifest + cohort readers (post-Phase-0a unified format).

Authoritative inputs:
  - data/manifest.jsonl  — 608 rows, mnemonic-keyed
  - data/cohort.json     — splits + stratification metadata

Both are produced by ``scripts/build_unified_manifest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read ``data/manifest.jsonl`` into a list of dicts (one per patient)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_cohort_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return json.load(f)


def manifest_by_pid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = r["patient_id"]
        if pid in out:
            raise ValueError(f"Duplicate patient_id in manifest: {pid}")
        out[pid] = r
    return out


def fold_split(
    rows: list[dict[str, Any]], fold: int
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(train_pids, val_pids, holdout_pids)`` for the given fold.

    Train = CV patients with ``r["fold"] != fold``; val = CV with ``r["fold"] == fold``;
    holdout = all ``cohort == "holdout"`` patients (loaded only by ``predict_holdout``).
    """
    train, val, holdout = [], [], []
    for r in rows:
        pid = r["patient_id"]
        if r["cohort"] == "holdout":
            holdout.append(pid)
        elif r["cohort"] == "cross-validation":
            if r["fold"] == fold:
                val.append(pid)
            else:
                train.append(pid)
        else:
            raise ValueError(f"Unknown cohort {r['cohort']!r} for {pid}")
    return train, val, holdout
