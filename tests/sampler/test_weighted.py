"""Unit tests for WeightedScheduledSampler (S1-S6)."""

from __future__ import annotations

from collections import Counter

import pytest

from endo.config.sampler import SamplerConfig
from endo.sampler.weighted import WeightedScheduledSampler


def _make_slice_index(
    n_pos: int = 200,
    n_nipv: int = 200,
    n_ninv: int = 600,
) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    sy = 0
    for i in range(n_pos):
        out.append((f"pos_{i:04d}", sy, "pos_slice"))
        sy += 1
    for i in range(n_nipv):
        out.append((f"pos_{i:04d}", sy, "neg_slice_pos_vol"))
        sy += 1
    for i in range(n_ninv):
        out.append((f"neg_{i:04d}", sy, "neg_slice_neg_vol"))
        sy += 1
    return out


def _kind_counts(sampler: WeightedScheduledSampler, n: int = 10_000) -> Counter[str]:
    sampler.cfg.samples_per_epoch = n
    counts: Counter[str] = Counter()
    for idx in iter(sampler):
        kind = sampler._slice_index[idx][2]
        counts[kind] += 1
    return counts


def test_S1_p_pos_decay_schedule() -> None:
    """S1: p_pos at epochs 0, 10, 20, 30, 60 matches the linear schedule."""
    cfg = SamplerConfig(
        pos_frac_start=0.50,
        pos_frac_end=0.25,
        decay_epochs=30,
        samples_per_epoch=100,
    )
    s = WeightedScheduledSampler(_make_slice_index(), cfg, seed=0)

    expected = {0: 0.500, 10: 0.500 - (10 / 30) * 0.25, 20: 0.500 - (20 / 30) * 0.25, 30: 0.250, 60: 0.250}
    for ep, want in expected.items():
        s.set_epoch(ep)
        assert s.current_p_pos() == pytest.approx(want, abs=1e-9), f"epoch {ep}"


def test_S2_mix_at_epoch_0() -> None:
    """S2: empirical sampling distribution at epoch 0 ≈ {.50, .25, .25}."""
    cfg = SamplerConfig(
        pos_frac_start=0.50,
        pos_frac_end=0.25,
        decay_epochs=30,
        neg_in_pos_vol_share=0.50,
        hard_pool_substitution_rate=0.0,
        hard_pool_start_epoch=999,  # disabled
        samples_per_epoch=10_000,
    )
    s = WeightedScheduledSampler(_make_slice_index(), cfg, seed=42)
    s.set_epoch(0)
    counts = _kind_counts(s, 10_000)

    pos = counts["pos_slice"] / 10_000
    nipv = counts["neg_slice_pos_vol"] / 10_000
    ninv = counts["neg_slice_neg_vol"] / 10_000
    assert pos == pytest.approx(0.50, abs=0.03)
    assert nipv == pytest.approx(0.25, abs=0.03)
    assert ninv == pytest.approx(0.25, abs=0.03)


def test_S3_mix_at_epoch_30() -> None:
    """S3: empirical sampling distribution at epoch 30 ≈ {.25, .375, .375}."""
    cfg = SamplerConfig(
        pos_frac_start=0.50,
        pos_frac_end=0.25,
        decay_epochs=30,
        neg_in_pos_vol_share=0.50,
        hard_pool_substitution_rate=0.0,
        hard_pool_start_epoch=999,
        samples_per_epoch=10_000,
    )
    s = WeightedScheduledSampler(_make_slice_index(), cfg, seed=7)
    s.set_epoch(30)
    counts = _kind_counts(s, 10_000)
    pos = counts["pos_slice"] / 10_000
    nipv = counts["neg_slice_pos_vol"] / 10_000
    ninv = counts["neg_slice_neg_vol"] / 10_000
    assert pos == pytest.approx(0.25, abs=0.03)
    assert nipv == pytest.approx(0.375, abs=0.03)
    assert ninv == pytest.approx(0.375, abs=0.03)


def test_S4_seeded_reproducible() -> None:
    """S4: same seed + epoch yields identical sequences."""
    cfg = SamplerConfig(samples_per_epoch=5_000)
    s1 = WeightedScheduledSampler(_make_slice_index(), cfg, seed=123)
    s2 = WeightedScheduledSampler(_make_slice_index(), cfg, seed=123)
    s1.set_epoch(7)
    s2.set_epoch(7)
    seq1 = list(iter(s1))
    seq2 = list(iter(s2))
    assert seq1 == seq2
    # Different epoch → different sequence (with overwhelming probability).
    s2.set_epoch(8)
    seq3 = list(iter(s2))
    assert seq1 != seq3


def test_S5_hard_pool_off_pre_start_epoch() -> None:
    """S5: hard pool substitution off before hard_pool_start_epoch."""
    cfg = SamplerConfig(
        pos_frac_start=0.0,  # everything is a negative draw
        pos_frac_end=0.0,
        neg_in_pos_vol_share=0.0,  # everything routed to ninv branch
        hard_pool_substitution_rate=1.0,  # would always substitute if active
        hard_pool_start_epoch=5,
        samples_per_epoch=2_000,
    )
    slice_index = _make_slice_index()
    s = WeightedScheduledSampler(slice_index, cfg, seed=0)
    # Hard pool: only positive-volume neg slices (all NIPV).
    nipv_dataset_indices = [i for i, (_, _, k) in enumerate(slice_index) if k == "neg_slice_pos_vol"]
    s.set_hard_pool(nipv_dataset_indices)
    s.set_epoch(4)  # below start
    seq = list(iter(s))
    kinds = Counter(slice_index[i][2] for i in seq)
    # Every draw should be NINV; 0 substitutions => 0 NIPV samples.
    assert kinds["neg_slice_pos_vol"] == 0
    assert kinds["neg_slice_neg_vol"] == 2_000


def test_S6_hard_pool_substitution_rate_post_start_epoch() -> None:
    """S6: with hard pool active, ~30% of NINV draws come from the pool."""
    cfg = SamplerConfig(
        pos_frac_start=0.0,
        pos_frac_end=0.0,
        neg_in_pos_vol_share=0.0,
        hard_pool_substitution_rate=0.30,
        hard_pool_start_epoch=5,
        samples_per_epoch=10_000,
    )
    slice_index = _make_slice_index()
    s = WeightedScheduledSampler(slice_index, cfg, seed=0)
    nipv_idx = [i for i, (_, _, k) in enumerate(slice_index) if k == "neg_slice_pos_vol"]
    s.set_hard_pool(nipv_idx)
    s.set_epoch(10)
    seq = list(iter(s))
    kinds = Counter(slice_index[i][2] for i in seq)
    sub_rate = kinds["neg_slice_pos_vol"] / 10_000
    assert sub_rate == pytest.approx(0.30, abs=0.05)


def test_len_matches_samples_per_epoch() -> None:
    cfg = SamplerConfig(samples_per_epoch=4321)
    s = WeightedScheduledSampler(_make_slice_index(), cfg, seed=0)
    assert len(s) == 4321
    assert sum(1 for _ in iter(s)) == 4321


def test_full_pass_mode_len_equals_dataset_size() -> None:
    cfg = SamplerConfig(epoch_mode="full_pass", samples_per_epoch=999)
    sl = _make_slice_index()
    s = WeightedScheduledSampler(sl, cfg, seed=0)
    assert len(s) == len(sl)
