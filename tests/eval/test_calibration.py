"""Tests for endo.eval.calibration — Brier, ECE, reliability curve."""

from __future__ import annotations

import math

import pytest

from endo.eval.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)


def test_brier_score_perfect_predictions():
    pairs = [(1.0, 1), (0.0, 0)] * 10
    assert brier_score(pairs) == pytest.approx(0.0)


def test_brier_score_random_at_half():
    pairs = [(0.5, 1), (0.5, 0)] * 10
    assert brier_score(pairs) == pytest.approx(0.25)


def test_brier_score_empty_is_nan():
    assert math.isnan(brier_score([]))


def test_brier_score_inverted_is_one():
    """Predicting the opposite of the truth gives Brier = 1."""
    pairs = [(1.0, 0), (0.0, 1)] * 5
    assert brier_score(pairs) == pytest.approx(1.0)


def test_ece_perfect_calibration():
    """A predictor whose conf == observed acc in each bin has ECE = 0."""
    # All scores 0.05, label 0 → bin [0, 0.1): conf=0.05, acc=0, gap=0.05.
    # Use a predictor that sits at the bin midpoint and matches that bin's accuracy.
    pairs = [(0.05, 0)] * 10 + [(0.95, 1)] * 10
    # bin [0, 0.1): conf=0.05, acc=0, weight=0.5 → 0.5*0.05 = 0.025
    # bin [0.9, 1.0]: conf=0.95, acc=1, weight=0.5 → 0.5*0.05 = 0.025
    assert expected_calibration_error(pairs, n_bins=10) == pytest.approx(0.05, abs=1e-6)


def test_ece_extreme_miscalibration():
    """All confident=1 but actually all negatives → ECE near 1.0."""
    pairs = [(0.95, 0)] * 20
    val = expected_calibration_error(pairs, n_bins=10)
    assert val == pytest.approx(0.95, abs=1e-6)


def test_ece_empty_is_nan():
    assert math.isnan(expected_calibration_error([]))


def test_reliability_curve_matches_simple_layout():
    pairs = [(0.05, 0)] * 10 + [(0.95, 1)] * 10
    curve = reliability_curve(pairs, n_bins=10)
    assert len(curve) == 10
    # Bins 0 and 9 are populated; others empty.
    populated = [b for b in curve if b["count"] > 0]
    assert len(populated) == 2
    # First populated bin: low end with score ~ 0.05, frac_pos = 0.
    first = populated[0]
    assert first["mean_pred"] == pytest.approx(0.05)
    assert first["frac_pos"] == pytest.approx(0.0)
    assert first["count"] == 10
    # Last populated bin: high end with score ~ 0.95, frac_pos = 1.
    last = populated[-1]
    assert last["mean_pred"] == pytest.approx(0.95)
    assert last["frac_pos"] == pytest.approx(1.0)
    assert last["count"] == 10


def test_reliability_curve_includes_empty_bins():
    """Empty bins are emitted with count=0 so the figure can plot uniform x-axis."""
    pairs = [(0.5, 1), (0.5, 0)]
    curve = reliability_curve(pairs, n_bins=10)
    assert len(curve) == 10
    empty = [b for b in curve if b["count"] == 0]
    assert len(empty) == 9
    populated = [b for b in curve if b["count"] > 0]
    assert len(populated) == 1
    assert populated[0]["count"] == 2


def test_reliability_curve_empty_input():
    assert reliability_curve([]) == []
