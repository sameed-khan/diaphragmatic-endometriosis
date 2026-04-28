"""Shared fixtures for Component 1 (preprocessing) tests.

Real-data fixtures pin the smallest positive and smallest negative volumes (by
raw NIfTI file size) from ``data/manifest.jsonl``. Resolved at collection time.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"


def _read_manifest() -> list[dict]:
    rows: list[dict] = []
    with MANIFEST_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _smallest(rows: list[dict], predicate) -> dict:
    candidates = [r for r in rows if predicate(r)]
    return min(candidates, key=lambda r: (DATA_ROOT / r["paths"]["raw"]).stat().st_size)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def real_fixtures():
    """Return dict with smallest-positive (CV) and smallest-negative (holdout)
    real cohort patients. Skipped if data/manifest.jsonl is absent.
    """
    if not MANIFEST_PATH.exists():
        pytest.skip("data/manifest.jsonl not present")
    rows = _read_manifest()
    pos_cv = _smallest(rows, lambda r: r["label"] == "positive" and r["cohort"] == "cross-validation")
    neg_holdout_candidates = [r for r in rows if r["label"] == "negative" and r["cohort"] == "holdout"]
    if neg_holdout_candidates:
        neg = min(neg_holdout_candidates, key=lambda r: (DATA_ROOT / r["paths"]["raw"]).stat().st_size)
    else:
        neg = _smallest(rows, lambda r: r["label"] == "negative")
    return {"positive_cv": pos_cv, "negative_holdout": neg}


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    p = tmp_path / "cache" / "v1"
    p.mkdir(parents=True, exist_ok=True)
    return p
