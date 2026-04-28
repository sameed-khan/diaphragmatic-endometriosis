"""Integration tests for Component 1 — runs on a 2-volume real fixture.

Marker convention: real-cohort tests have ``real`` in the test name and rely on
the ``real_fixtures`` fixture which auto-skips if data is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preprocess as pp  # noqa: E402


def _run_two_volume_fixture(real_fixtures, cache_dir: Path):
    pos = real_fixtures["positive_cv"]
    neg = real_fixtures["negative_holdout"]
    cfg = pp.PreprocessConfig(
        manifest_path=REPO_ROOT / "data" / "manifest.jsonl",
        cohort_path=REPO_ROOT / "data" / "cohort.json",
        raw_root=REPO_ROOT / "data",
        cache_root=cache_dir,
        workers=1,
        force=True,
        dry_run=False,
        code_version="test",
    )
    summary = pp.preprocess_cohort(cfg, patient_filter=[pos["patient_id"], neg["patient_id"]])
    return pos, neg, summary


def test_real_int1_end_to_end(real_fixtures, tmp_cache):
    pos, neg, summary = _run_two_volume_fixture(real_fixtures, tmp_cache)
    assert summary["n_failed"] == 0, summary["failures"]
    # manifest written
    rows = []
    with (tmp_cache / "preprocessed_manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    pids = {r["patient_id"] for r in rows}
    assert pos["patient_id"] in pids
    assert neg["patient_id"] in pids
    # gt_boxes parquet written
    bp = tmp_cache / "gt_boxes.parquet"
    assert bp.exists()
    df = pl.read_parquet(bp)
    if df.height > 0:
        for col in ["patient_id", "slice_y", "cc_id", "x1", "z1", "x2", "z2", "box_max_dim_mm"]:
            assert col in df.columns


def test_real_int2_volume_shape(real_fixtures, tmp_cache):
    pos, neg, _ = _run_two_volume_fixture(real_fixtures, tmp_cache)
    for r in (pos, neg):
        v = np.load(tmp_cache / "volumes" / r["patient_id"] / "volume.npy")
        assert v.shape == (408, 174, 408)
        assert v.dtype == np.float16


def test_real_int3_no_liver_mask_in_cache(real_fixtures, tmp_cache):
    pos, neg, _ = _run_two_volume_fixture(real_fixtures, tmp_cache)
    for r in (pos, neg):
        d = tmp_cache / "volumes" / r["patient_id"]
        assert not (d / "liver_mask.npy").exists()


def test_real_int5_lesion_vs_ring_z(real_fixtures, tmp_cache):
    pos, _neg, _ = _run_two_volume_fixture(real_fixtures, tmp_cache)
    rows = []
    with (tmp_cache / "preprocessed_manifest.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    pos_row = next(r for r in rows if r["patient_id"] == pos["patient_id"])
    assert pos_row["lesion_vs_ring_z"] is not None
    assert pos_row["lesion_vs_ring_z"] >= pp.LESION_VS_RING_Z_FLOOR


def test_real_int7_holdout_no_border_band(real_fixtures, tmp_cache):
    pos, neg, _ = _run_two_volume_fixture(real_fixtures, tmp_cache)
    # neg is holdout in the fixture; should have NO border_band
    band = tmp_cache / "border_bands" / f"{neg['patient_id']}.npy"
    assert neg["cohort"] == "holdout"
    assert not band.exists()
    # pos is CV — should have a border_band file
    if pos["cohort"] == "cross-validation":
        band_pos = tmp_cache / "border_bands" / f"{pos['patient_id']}.npy"
        assert band_pos.exists()
