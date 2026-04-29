"""Smoke-test integration tests (PRD §11.10).

The integration test SM1-SM4 — actually running the smoke training — needs
the preprocessed cache. It's gated on cache existence so we can run the
test suite without a full preprocessing run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CACHE_ROOT = Path("cache/v1")
SMOKE_REQUIRES = (
    CACHE_ROOT / "preprocessed_manifest.jsonl",
    CACHE_ROOT / "gt_boxes.parquet",
)


def _cache_ready() -> bool:
    return all(p.exists() for p in SMOKE_REQUIRES)


def test_smoke_module_importable():
    from scripts.smoke_train import run_smoke, pick_smoke_pids, write_smoke_manifest  # noqa: F401


def test_pick_smoke_pids_synthetic(tmp_path: Path):
    """Unit test for the pid picker on synthetic manifest rows."""
    from scripts.smoke_train import pick_smoke_pids

    # Build cache stub: per-pid volume.npy of varying sizes so smallest is picked.
    cache = tmp_path / "v1"
    (cache / "volumes").mkdir(parents=True)
    rows = []
    for i in range(20):
        pid = f"p{i:02d}"
        (cache / "volumes" / pid).mkdir()
        (cache / "volumes" / pid / "volume.npy").write_bytes(b"x" * (1000 + i))
        rows.append({
            "patient_id": pid,
            "cohort": "cross-validation",
            "label": "positive" if i < 10 else "negative",
            "fold": i % 5,
        })
    pids = pick_smoke_pids(rows, cache, n_pos=2, n_neg=3)
    assert len(pids) == 5
    # At least one of the 2 positives must have fold=0 (val) and another fold!=0.
    chosen = [r for r in rows if r["patient_id"] in pids]
    pos = [r for r in chosen if r["label"] == "positive"]
    folds = {r["fold"] for r in pos}
    assert 0 in folds, f"expected at least one positive in fold 0, got {folds}"
    assert any(f != 0 for f in folds), f"expected at least one positive in another fold, got {folds}"


@pytest.mark.skipif(not _cache_ready(), reason="cache/v1 not built yet")
def test_smoke_runs_to_completion_real_cache():
    """SM1-SM4 in one go. Requires real cache."""
    from scripts.smoke_train import run_smoke

    res = run_smoke(keep_artifacts=False)
    # SM2 — loss decreases
    assert res["last10_loss"] < res["first10_loss"]
    # SM3 — no NaN
    assert res["finite"]
    # SM4 — val/slice_auroc logged
    assert res["val_slice_auroc"] is not None
    # SM1 — completes (implicit; if not it raised already)
    assert res["n_steps"] >= 20
