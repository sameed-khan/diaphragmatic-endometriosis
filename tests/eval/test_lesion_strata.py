"""Tests for endo.eval.lesion_strata — lesion-volume-binned sensitivity."""

from __future__ import annotations

import math

import pytest

from endo.eval.lesion_strata import (
    DEFAULT_VOLUME_EDGES_MM3,
    compute_lesion_volume_strata,
    lesion_units_from_calls,
)


def _records(*entries):
    """Helper: build minimal call records from (call_type, volume_mm3) pairs."""
    out = []
    for ct, v in entries:
        rec = {"call_type": ct, "volume_mm3": float(v) if v is not None else None}
        out.append(rec)
    return out


def test_fp_records_are_ignored():
    """FP records must not contribute to lesion-level units (recall denominator)."""
    recs = _records(("fp", 1.0), ("fp", 100.0))
    units = lesion_units_from_calls(recs)
    assert units == []


def test_units_split_by_bin():
    recs = _records(
        ("tp", 50.0),    # <=200
        ("fn", 50.0),    # <=200
        ("tp", 500.0),   # 200-1000
        ("tp", 2000.0),  # 1000-5000
        ("fn", 9999.0),  # >5000
    )
    units = lesion_units_from_calls(recs)
    # 5 lesions in 4 distinct bins.
    assert len(units) == 5
    bins = sorted(set(bi for bi, _ in units))
    assert bins == [0, 1, 2, 3]


def test_sensitivity_per_bin():
    recs = _records(
        ("tp", 50.0), ("fn", 50.0),  # <=200: sens = 0.5
        ("tp", 500.0),                # 200-1000: sens = 1.0
        ("fn", 2000.0),               # 1000-5000: sens = 0.0
    )
    out = compute_lesion_volume_strata(recs, bootstrap_n=50, seed=0)
    by_bin = {s["stratum_value"]: s for s in out}
    assert by_bin["<=200mm3"]["metrics"]["lesion_sensitivity"]["value"] == pytest.approx(0.5)
    assert by_bin["200-1000mm3"]["metrics"]["lesion_sensitivity"]["value"] == pytest.approx(1.0)
    assert by_bin["1000-5000mm3"]["metrics"]["lesion_sensitivity"]["value"] == pytest.approx(0.0)


def test_empty_bin_yields_nan_with_zero_count():
    recs = _records(("tp", 50.0))  # only <=200 populated
    out = compute_lesion_volume_strata(recs, bootstrap_n=50)
    by_bin = {s["stratum_value"]: s for s in out}
    # All four default bins are emitted regardless.
    assert set(by_bin) == {"<=200mm3", "200-1000mm3", "1000-5000mm3", ">5000mm3"}
    assert math.isnan(by_bin[">5000mm3"]["metrics"]["lesion_sensitivity"]["value"])
    assert by_bin[">5000mm3"]["n_lesions"] == 0


def test_ci_brackets_point_estimate():
    """Bootstrap CI should contain the point estimate."""
    recs = _records(*([("tp", 50.0)] * 8 + [("fn", 50.0)] * 2))  # 8/10 = 0.8
    out = compute_lesion_volume_strata(recs, bootstrap_n=500, seed=42)
    s = next(x for x in out if x["stratum_value"] == "<=200mm3")
    pt = s["metrics"]["lesion_sensitivity"]["value"]
    lo = s["metrics"]["lesion_sensitivity"]["ci_lower"]
    hi = s["metrics"]["lesion_sensitivity"]["ci_upper"]
    assert pt == pytest.approx(0.8)
    assert lo <= pt <= hi
    # Realistic CI half-width for n=10 binomial(0.8) bootstrap is ~0.2.
    assert (hi - lo) > 0.1


def test_record_with_missing_volume_skipped():
    recs = [
        {"call_type": "tp", "volume_mm3": None},
        {"call_type": "tp", "volume_mm3": 50.0},
    ]
    units = lesion_units_from_calls(recs)
    assert len(units) == 1


def test_default_edges_are_finite_increasing():
    edges = DEFAULT_VOLUME_EDGES_MM3
    assert edges[0] == 0.0
    for i in range(len(edges) - 1):
        assert edges[i] < edges[i + 1]
    assert edges[-1] == float("inf")
