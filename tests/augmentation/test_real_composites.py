"""Component 4 metric tests — synthetic stand-ins for T2.1, T2.4, T2.5.

These tests are normally Tier 2 (real cache + real bank). For local CI we
exercise the same invariants on synthetic samples + an inline 3-entry bank.
"""

from __future__ import annotations

import numpy as np

from endo.augmentation.paste import multi_paste_volume
from endo.config.augmentation import PasteConfig

from tests.augmentation.conftest import FRAME_SHAPE, make_sample, make_uniform_donor


def _build_inline_bank() -> list:
    return [
        make_uniform_donor(size_xyz=(3, 3, 3), donor_patient_id="d0"),
        make_uniform_donor(size_xyz=(5, 3, 5), donor_patient_id="d1"),
        make_uniform_donor(size_xyz=(4, 3, 4), donor_patient_id="d2"),
    ]


# ---------------------------------------------------------------------------
# T2.1 — paste centroid near border-band (here: AT a border-band voxel)
# ---------------------------------------------------------------------------


def test_paste_centroid_in_border_band_set() -> None:
    rng = np.random.default_rng(0)
    bank = _build_inline_bank()

    cfg = PasteConfig(p_any_paste=1.0, n_paste_sigma=2.0, n_paste_max=3)
    n_pastes_total = 0
    n_in_band = 0
    for trial in range(20):
        sample = make_sample(rng_seed=trial + 1)
        band_set = {tuple(map(int, c)) for c in sample.border_band_coords}
        _, _, results = multi_paste_volume(
            sample.volume_full_cropped.copy(),
            sample.lesion_mask_full_cropped.copy(),
            sample.border_band_coords,
            bank,
            cfg,
            rng,
            frame_shape=FRAME_SHAPE,
        )
        for r in results:
            n_pastes_total += 1
            if tuple(map(int, r.site)) in band_set:
                n_in_band += 1

    # Every paste must originate from a border-band voxel by construction.
    assert n_pastes_total > 0
    assert n_in_band == n_pastes_total, (
        f"only {n_in_band}/{n_pastes_total} paste centroids lie in border_band"
    )


# ---------------------------------------------------------------------------
# T2.4 — paste right-side only (synthetic stand-in: band restricted to x>cx)
# ---------------------------------------------------------------------------


def test_paste_right_side_only_synthetic() -> None:
    """If the border_band is restricted to x>cx, every paste site must have x>cx."""
    rng = np.random.default_rng(0)
    sample = make_sample(rng_seed=1)
    fx, fy, fz = FRAME_SHAPE
    cx = fx // 2
    band = sample.border_band_coords
    right_band = band[band[:, 0] > cx]
    sample.border_band_coords = right_band.astype(np.int16, copy=False)

    bank = _build_inline_bank()
    cfg = PasteConfig(p_any_paste=1.0, n_paste_sigma=2.0, n_paste_max=3)

    _, _, results = multi_paste_volume(
        sample.volume_full_cropped.copy(),
        sample.lesion_mask_full_cropped.copy(),
        sample.border_band_coords,
        bank,
        cfg,
        rng,
        frame_shape=FRAME_SHAPE,
    )
    assert len(results) > 0
    for r in results:
        assert int(r.site[0]) > cx, f"paste site x={r.site[0]} not on right side"


# ---------------------------------------------------------------------------
# T2.5 — no paste outside volume bounds
# ---------------------------------------------------------------------------


def test_no_paste_outside_volume_bounds() -> None:
    rng = np.random.default_rng(0)
    bank = _build_inline_bank()
    cfg = PasteConfig(p_any_paste=1.0, n_paste_sigma=2.0, n_paste_max=4)

    fx, fy, fz = FRAME_SHAPE
    for trial in range(8):
        sample = make_sample(rng_seed=trial + 10)
        volume = sample.volume_full_cropped.copy()
        lesion_mask = sample.lesion_mask_full_cropped.copy()
        _, lm_out, results = multi_paste_volume(
            volume,
            lesion_mask,
            sample.border_band_coords,
            bank,
            cfg,
            rng,
            frame_shape=FRAME_SHAPE,
        )
        # All updated mask voxels must be in [0, fx)×[0, fy)×[0, fz).
        xs, ys, zs = np.where(lm_out > 0)
        if xs.size:
            assert int(xs.min()) >= 0 and int(xs.max()) < fx
            assert int(ys.min()) >= 0 and int(ys.max()) < fy
            assert int(zs.min()) >= 0 and int(zs.max()) < fz
        for r in results:
            ix0, ix1, iy0, iy1, iz0, iz1 = r.target_box
            assert ix0 >= 0 and ix1 <= fx
            assert iy0 >= 0 and iy1 <= fy
            assert iz0 >= 0 and iz1 <= fz
