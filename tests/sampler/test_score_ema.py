"""Unit tests for ScoreEMATracker (S8, S9, S10)."""

from __future__ import annotations

import math

import pytest

from endo.sampler.score_ema import ScoreEMATracker


def test_loss_ema_initialization_first_then_decay() -> None:
    """S8: first update sets value, subsequent uses EMA recurrence."""
    tracker = ScoreEMATracker(decay=0.9)
    key = ("p001", 42)

    tracker.update(key, score=0.5, is_positive_slice=False)
    assert tracker.get(key) == pytest.approx(0.5)

    tracker.update(key, score=1.0, is_positive_slice=False)
    # 0.9*0.5 + 0.1*1.0 = 0.55
    assert tracker.get(key) == pytest.approx(0.55)

    tracker.update(key, score=0.0, is_positive_slice=False)
    # 0.9*0.55 + 0.1*0.0 = 0.495
    assert tracker.get(key) == pytest.approx(0.495)


def test_loss_ema_skips_positive_slices() -> None:
    """S9: PRD I.8.3 — is_positive_slice=True is a no-op."""
    tracker = ScoreEMATracker(decay=0.9)
    tracker.update(("p001", 10), score=0.99, is_positive_slice=True)
    assert len(tracker) == 0
    assert tracker.get(("p001", 10)) is None

    # And does not affect existing entries either.
    tracker.update(("p001", 11), score=0.3, is_positive_slice=False)
    tracker.update(("p001", 11), score=10.0, is_positive_slice=True)
    assert tracker.get(("p001", 11)) == pytest.approx(0.3)


def test_loss_ema_top_k_returns_highest() -> None:
    """S10: top_k(N) returns the N keys with highest EMA, descending."""
    tracker = ScoreEMATracker(decay=0.9)
    keys = [("p001", i) for i in range(10)]
    for i, k in enumerate(keys):
        tracker.update(k, score=float(i) / 10.0, is_positive_slice=False)

    top3 = tracker.top_k(k=3)
    assert top3 == [("p001", 9), ("p001", 8), ("p001", 7)]

    # k larger than population returns all entries.
    full = tracker.top_k(k=100)
    assert len(full) == 10
    # k=0 returns empty.
    assert tracker.top_k(k=0) == []


def test_loss_ema_state_dict_roundtrip() -> None:
    tracker = ScoreEMATracker(decay=0.85)
    tracker.update(("p001", 1), 0.5, is_positive_slice=False)
    tracker.update(("p002", 7), 0.2, is_positive_slice=False)

    sd = tracker.state_dict()
    restored = ScoreEMATracker(decay=0.5)
    restored.load_state_dict(sd)

    assert restored.decay == pytest.approx(0.85)
    assert restored.get(("p001", 1)) == pytest.approx(0.5)
    assert restored.get(("p002", 7)) == pytest.approx(0.2)


def test_loss_ema_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError):
        ScoreEMATracker(decay=0.0)
    with pytest.raises(ValueError):
        ScoreEMATracker(decay=1.0)
