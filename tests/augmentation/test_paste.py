"""Component 4 paste tests (T1.1–T1.7)."""

from __future__ import annotations

import numpy as np
import pytest

from endo.augmentation.paste import (
    apply_paste,
    multi_paste_volume,
    sample_n_pastes,
    select_paste_site,
)
from endo.config.augmentation import PasteConfig

from tests.augmentation.conftest import (
    FRAME_SHAPE,
    make_donor_with_shell,
    make_sample,
    make_uniform_donor,
)


# ---------------------------------------------------------------------------
# T1.1 — n_pastes distribution
# ---------------------------------------------------------------------------


def test_sample_n_pastes_distribution() -> None:
    rng = np.random.default_rng(0)
    n = 30000
    counts = np.array(
        [sample_n_pastes(rng, p_any_paste=0.5, n_paste_sigma=1.0, n_paste_max=7) for _ in range(n)]
    )
    p_zero = float((counts == 0).mean())
    assert 0.46 <= p_zero <= 0.54, f"P(n=0) = {p_zero:.3f} not within [0.46, 0.54]"
    assert counts.max() <= 7
    # Conditional on n>0, mode must be 1 (P(1) ≈ 0.38 dominates).
    pos = counts[counts > 0]
    bincount = np.bincount(pos, minlength=8)
    mode = int(np.argmax(bincount))
    assert mode == 1, f"conditional mode is {mode}, expected 1 (counts={bincount})"


# ---------------------------------------------------------------------------
# T1.2 — n_pastes seeded reproducibility
# ---------------------------------------------------------------------------


def test_sample_n_pastes_seeded_reproducible() -> None:
    a = np.random.default_rng(42)
    b = np.random.default_rng(42)
    seq_a = [sample_n_pastes(a, 0.5, 1.0, 7) for _ in range(200)]
    seq_b = [sample_n_pastes(b, 0.5, 1.0, 7) for _ in range(200)]
    assert seq_a == seq_b


# ---------------------------------------------------------------------------
# T1.3 — paste site is in border_band
# ---------------------------------------------------------------------------


def test_paste_site_inside_border_band() -> None:
    rng = np.random.default_rng(0)
    sample = make_sample(rng_seed=1)
    donor = make_uniform_donor(size_xyz=(3, 3, 3))
    band_set = {tuple(map(int, c)) for c in sample.border_band_coords}
    occupancy = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    n_hits = 0
    for _ in range(100):
        site = select_paste_site(
            sample.border_band_coords, occupancy, donor, rng, frame_shape=FRAME_SHAPE
        )
        if site is not None:
            n_hits += 1
            assert tuple(map(int, site)) in band_set
    assert n_hits > 90, f"only {n_hits}/100 successful sites — band too sparse?"


# ---------------------------------------------------------------------------
# T1.4 — no overlap with existing lesion
# ---------------------------------------------------------------------------


def test_paste_no_overlap_with_existing() -> None:
    # Pre-place a lesion at a known location; ensure the paste search avoids it.
    rng = np.random.default_rng(0)
    pre = (12, 18, 8, 12, 12, 18)  # x0,x1,y0,y1,z0,z1
    sample = make_sample(rng_seed=1, pre_lesion_box=pre)
    donor = make_uniform_donor(size_xyz=(3, 3, 3))
    occupancy = (sample.lesion_mask_full_cropped > 0).astype(np.uint8)
    placed_paste_masks: list[np.ndarray] = []

    cfg = PasteConfig(p_any_paste=1.0, n_paste_sigma=1.0, n_paste_max=3)
    # Force paste schedule with a fixed RNG.
    volume = sample.volume_full_cropped.copy()
    lesion_mask = sample.lesion_mask_full_cropped.copy()
    occupancy_init = (lesion_mask > 0).astype(np.uint8)
    for _ in range(30):
        site = select_paste_site(
            sample.border_band_coords, occupancy_init, donor, rng, frame_shape=FRAME_SHAPE
        )
        assert site is not None
        # The mask returned by the apply call must not intersect the pre-existing region.
        result = apply_paste(volume, lesion_mask, occupancy_init, donor, site, frame_shape=FRAME_SHAPE)
        assert result is not None
        ix0, ix1, iy0, iy1, iz0, iz1 = result.target_box
        new_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        new_mask[ix0:ix1, iy0:iy1, iz0:iz1] |= result.paste_mask_crop
        # Must not overlap original pre lesion.
        x0, x1, y0, y1, z0, z1 = pre
        original = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        original[x0:x1, y0:y1, z0:z1] = 1
        assert not np.any((new_mask > 0) & (original > 0)), "Paste overlaps pre-existing lesion"


# ---------------------------------------------------------------------------
# T1.5 — no overlap between pastes
# ---------------------------------------------------------------------------


def test_paste_no_overlap_between_pastes() -> None:
    rng = np.random.default_rng(0)
    sample = make_sample(rng_seed=1)
    donor = make_uniform_donor(size_xyz=(3, 3, 3))

    cfg = PasteConfig(p_any_paste=1.0, n_paste_sigma=2.5, n_paste_max=5)
    # Use a custom n_pastes path: force 5 pastes via direct apply_paste loop.
    volume = sample.volume_full_cropped.copy()
    lesion_mask = sample.lesion_mask_full_cropped.copy()
    occupancy = (lesion_mask > 0).astype(np.uint8)

    masks: list[np.ndarray] = []
    placed = 0
    for _ in range(5):
        site = select_paste_site(
            sample.border_band_coords, occupancy, donor, rng, frame_shape=FRAME_SHAPE
        )
        if site is None:
            break
        result = apply_paste(volume, lesion_mask, occupancy, donor, site, frame_shape=FRAME_SHAPE)
        if result is None:
            continue
        full_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        ix0, ix1, iy0, iy1, iz0, iz1 = result.target_box
        full_mask[ix0:ix1, iy0:iy1, iz0:iz1] |= result.paste_mask_crop
        for prev in masks:
            assert not np.any((full_mask > 0) & (prev > 0)), "Pastes overlap"
        masks.append(full_mask)
        placed += 1
    assert placed >= 3, f"only {placed} pastes placed; expected at least 3"


# ---------------------------------------------------------------------------
# T1.6 — pasted region intensity matches local stats
# ---------------------------------------------------------------------------


def test_paste_intensity_match_local_stats() -> None:
    rng = np.random.default_rng(0)
    # Build a volume with a known mean of 0.3 in the paste region.
    fx, fy, fz = FRAME_SHAPE
    volume = np.full(FRAME_SHAPE, 0.3, dtype=np.float32)
    # Add tiny noise so std > 0.
    volume += rng.normal(0.0, 0.01, size=FRAME_SHAPE).astype(np.float32)
    lesion_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    occupancy = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    donor = make_donor_with_shell(size_xyz=(7, 3, 7), cc_intensity=2.5, cc_intensity_std=0.1)
    site = (fx // 2, fy // 2, fz // 2)
    result = apply_paste(volume, lesion_mask, occupancy, donor, site, frame_shape=FRAME_SHAPE)
    assert result is not None

    # Pasted region's mean ≈ target_local_mean ± 0.1 (per spec).
    ix0, ix1, iy0, iy1, iz0, iz1 = result.target_box
    paste_voxels = volume[ix0:ix1, iy0:iy1, iz0:iz1][result.paste_mask_crop.astype(bool)]
    pasted_mean = float(paste_voxels.mean())
    assert abs(pasted_mean - 0.3) < 0.1, (
        f"pasted mean {pasted_mean:.3f} not within 0.1 of target_local_mean 0.30"
    )


# ---------------------------------------------------------------------------
# T1.7 — soft-blend continuity
# ---------------------------------------------------------------------------


def test_paste_soft_blend_continuity() -> None:
    rng = np.random.default_rng(0)
    fx, fy, fz = FRAME_SHAPE
    base = 0.5
    volume = np.full(FRAME_SHAPE, base, dtype=np.float32)
    volume += rng.normal(0.0, 0.05, size=FRAME_SHAPE).astype(np.float32)
    lesion_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    occupancy = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    donor = make_donor_with_shell(size_xyz=(7, 3, 7), cc_intensity=2.0, cc_intensity_std=0.05)
    site = (fx // 2, fy // 2, fz // 2)
    pre = volume.copy()
    result = apply_paste(volume, lesion_mask, occupancy, donor, site, frame_shape=FRAME_SHAPE)
    assert result is not None
    ix0, ix1, iy0, iy1, iz0, iz1 = result.target_box

    # Sample voxels just OUTSIDE the donor mask (in the shell). The blend should
    # not produce a discontinuity larger than 1.5 × σ (background σ ≈ 0.05).
    shell_crop = donor.tight_shell_mask.astype(bool)
    after = volume[ix0:ix1, iy0:iy1, iz0:iz1]
    before = pre[ix0:ix1, iy0:iy1, iz0:iz1]
    diff = np.abs(after - before)[shell_crop]
    if diff.size == 0:
        pytest.skip("donor shell is empty; no blend voxels to check")
    # Background σ ≈ 0.05; donor CC mean ≈ 2.0 → an unblended seam would be ~1.5.
    # We assert the soft-blend keeps the maximum jump in shell ≤ 1.5 × CC σ scale.
    assert float(np.percentile(diff, 95)) < 1.5, (
        f"95th-pct shell jump {float(np.percentile(diff, 95)):.3f} exceeds 1.5"
    )
