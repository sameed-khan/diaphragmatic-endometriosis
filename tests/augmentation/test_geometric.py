"""Component 4 geometric tests (T1.11–T1.13)."""

from __future__ import annotations

import numpy as np

from endo.augmentation.geometric import (
    apply_affine_lockstep,
    apply_elastic_lockstep,
    geometric_aug,
    random_affine_2d,
    random_elastic_2d,
)
from endo.config.augmentation import GeometricConfig

from tests.augmentation.conftest import FRAME_SHAPE


# ---------------------------------------------------------------------------
# T1.11 — lockstep
# ---------------------------------------------------------------------------


def test_geometric_lockstep() -> None:
    """An identity-near affine + elastic must keep volume foreground aligned with mask."""
    rng = np.random.default_rng(0)
    fx, fy, fz = FRAME_SHAPE
    # A volume with a sharp foreground "blob" placed under the mask.
    volume = np.zeros(FRAME_SHAPE, dtype=np.float32)
    lesion_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    volume[15:25, 8:12, 15:25] = 1.0
    lesion_mask[15:25, 8:12, 15:25] = 1

    cfg = GeometricConfig(
        rotation_deg=2.0,
        scale_min=0.99,
        scale_max=1.01,
        translation_frac=0.01,
        elastic_sigma=0.5,
        elastic_control_points=4,
        p_elastic=1.0,
    )
    out_vol, out_msk = geometric_aug(volume.copy(), lesion_mask.copy(), cfg, rng)

    # Where the mask is set, the volume should be > 0.5 (allowing for blur).
    msk_bool = out_msk.astype(bool)
    if msk_bool.sum() > 0:
        # Coarse alignment: mean intensity inside post-aug mask should be >> mean outside.
        mean_in = float(out_vol[msk_bool].mean())
        mean_out = float(out_vol[~msk_bool].mean())
        assert mean_in > 0.4, f"post-aug volume mean inside mask {mean_in:.3f} too low"
        assert mean_in > mean_out + 0.3, (
            f"volume foreground not aligned with mask post-aug "
            f"(in={mean_in:.3f}, out={mean_out:.3f})"
        )


# ---------------------------------------------------------------------------
# T1.12 — in-plane only (no Y movement)
# ---------------------------------------------------------------------------


def test_geometric_in_plane_only() -> None:
    """A volume with an indicator on a single Y slice must stay on that slice."""
    rng = np.random.default_rng(123)
    fx, fy, fz = FRAME_SHAPE
    target_y = fy // 2

    volume = np.zeros(FRAME_SHAPE, dtype=np.float32)
    volume[:, target_y, :] = 1.0  # all-foreground on one y slice
    lesion_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    lesion_mask[:, target_y, :] = 1

    cfg = GeometricConfig(p_elastic=1.0)
    out_vol, out_msk = geometric_aug(volume, lesion_mask, cfg, rng)

    # All foreground (>0) voxels in volume must stay on y == target_y.
    nonzero_y = np.unique(np.where(out_vol > 0.5)[1])
    assert set(nonzero_y.tolist()).issubset({target_y}), (
        f"Y movement detected: foreground appears on Y slices {nonzero_y.tolist()}"
    )
    nonzero_y_msk = np.unique(np.where(out_msk > 0)[1])
    assert set(nonzero_y_msk.tolist()).issubset({target_y})


# ---------------------------------------------------------------------------
# T1.13 — Y-coherent elastic field
# ---------------------------------------------------------------------------


def test_geometric_y_coherent() -> None:
    """The same elastic field applied at every Y must produce the same in-plane warp."""
    rng = np.random.default_rng(42)
    fx, fy, fz = FRAME_SHAPE

    # Same in-plane image at every Y.
    rng_img = np.random.default_rng(0)
    img2d = rng_img.standard_normal((fx, fz)).astype(np.float32)
    volume = np.broadcast_to(img2d[:, None, :], FRAME_SHAPE).copy()
    lesion_mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    field = random_elastic_2d(
        rng, alpha=1.0, sigma=2.0, shape_xz=(fx, fz), n_control_points=8
    )
    out_vol, _ = apply_elastic_lockstep(volume, lesion_mask, field)
    # Every Y slice of the output must equal the first slice (T1.13).
    for y in range(fy):
        assert np.allclose(out_vol[:, y, :], out_vol[:, 0, :], atol=1e-5), (
            f"Y={y} slice differs from Y=0"
        )
