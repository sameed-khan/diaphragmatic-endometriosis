"""Component 4 box re-derivation + 5-channel extraction tests
(T1.16, T1.17, T1.18, T1.19)."""

from __future__ import annotations

import numpy as np

from endo.augmentation.boxes import (
    clamp_box_to_frame,
    derive_boxes_from_mask,
)
from endo.augmentation.transform import TrainAugmentation
from endo.config.augmentation import AugmentationConfig, PasteConfig

from tests.augmentation.conftest import FRAME_SHAPE, make_sample, make_uniform_donor


# ---------------------------------------------------------------------------
# T1.16 — box re-derivation matches mask
# ---------------------------------------------------------------------------


def test_box_rederivation_matches_mask() -> None:
    fx, fz = 40, 40
    mask = np.zeros((fx, fz), dtype=np.uint8)
    # Two non-overlapping CCs.
    mask[5:11, 5:9] = 1  # x∈[5,11), z∈[5,9)
    mask[20:28, 30:34] = 1  # x∈[20,28), z∈[30,34)
    boxes = derive_boxes_from_mask(mask, connectivity=26, min_dim=2)
    assert len(boxes) == 2
    boxes_set = {tuple(b) for b in boxes}
    assert (5, 5, 11, 9) in boxes_set
    assert (20, 30, 28, 34) in boxes_set


# ---------------------------------------------------------------------------
# T1.17 — sub-pixel CCs are dropped
# ---------------------------------------------------------------------------


def test_box_skip_subpixel_artifacts() -> None:
    fx, fz = 40, 40
    mask = np.zeros((fx, fz), dtype=np.uint8)
    mask[10, 10] = 1  # 1×1 CC — should be dropped (min_dim=2)
    mask[20:25, 20:25] = 1  # 5×5 CC — kept
    boxes = derive_boxes_from_mask(mask, connectivity=26, min_dim=2)
    assert len(boxes) == 1
    assert boxes[0] == (20, 20, 25, 25)


def test_clamp_box_to_frame_drops_tiny() -> None:
    assert clamp_box_to_frame((0.0, 0.0, 1.0, 1.0), (10, 10), min_dim=2) is None
    assert clamp_box_to_frame((0.0, 0.0, 5.0, 5.0), (10, 10), min_dim=2) == (0, 0, 5, 5)
    # Out-of-frame clamping.
    assert clamp_box_to_frame((-3.0, -3.0, 5.0, 5.0), (10, 10), min_dim=2) == (0, 0, 5, 5)


# ---------------------------------------------------------------------------
# T1.18, T1.19 — 5-channel slice extraction
# ---------------------------------------------------------------------------


def _make_minimal_train_aug(tmp_path) -> TrainAugmentation:
    cache_root = tmp_path / "cache" / "v1"
    (cache_root / "runtime").mkdir(parents=True, exist_ok=True)
    # Pre-write a cohort_local_std so we don't trigger the cohort scan.
    (cache_root / "runtime" / "cohort_local_std.json").write_text(
        '{"cohort_median_local_std": 1.0, "n_volumes_sampled": 0, '
        '"samples_per_volume": 0, "computed_at": "1970-01-01T00:00:00Z", '
        '"code_version": "test"}'
    )
    cfg = AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.0),  # disable paste
    )
    aug = TrainAugmentation(
        cfg=cfg,
        cache_root=cache_root,
        bank_path=cache_root / "lesion_banks" / "current.pkl",  # missing → empty bank
        rng_seed=0,
    )
    return aug


def test_5ch_slice_extraction_shape(tmp_path) -> None:
    aug = _make_minimal_train_aug(tmp_path)
    sample = make_sample(rng_seed=2)
    fx, fy, fz = FRAME_SHAPE
    sample.slice_y = fy // 2
    out = aug(sample)
    assert out.volume_5ch.shape == (5, fz, fx)
    assert out.volume_5ch.dtype == np.float32
    assert out.lesion_mask_center.shape == (fz, fx)


def test_5ch_center_channel_alignment(tmp_path) -> None:
    """Channel 2 of volume_5ch should equal volume[:, slice_y, :].T (post-augment)."""
    aug = _make_minimal_train_aug(tmp_path)
    # Use a config that produces the IDENTITY transform: zero rotation, scale=1,
    # zero translation, zero elastic, zero noise.
    aug.cfg.geometric.rotation_deg = 0.0
    aug.cfg.geometric.scale_min = 1.0
    aug.cfg.geometric.scale_max = 1.0
    aug.cfg.geometric.translation_frac = 0.0
    aug.cfg.geometric.elastic_sigma = 0.0
    aug.cfg.geometric.elastic_control_points = 4
    aug.cfg.geometric.p_elastic = 0.0
    aug.cfg.intensity.gamma_min = 1.0
    aug.cfg.intensity.gamma_max = 1.0
    aug.cfg.intensity.bias_min = 1.0
    aug.cfg.intensity.bias_max = 1.0
    aug.cfg.intensity.noise_sigma = 0.0
    aug.cfg.paste.p_any_paste = 0.0

    sample = make_sample(rng_seed=3)
    sample.slice_y = FRAME_SHAPE[1] // 2
    expected_center = sample.volume_full_cropped[:, sample.slice_y, :].T.astype(np.float32)
    out = aug(sample)
    # Channel 2 == expected.
    assert np.allclose(out.volume_5ch[2], expected_center, atol=1e-5)
