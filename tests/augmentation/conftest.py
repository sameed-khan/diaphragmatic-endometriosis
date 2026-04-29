"""Synthetic fixtures for Component 4 unit tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from endo.data.samples import Sample
from endo.lesion_bank import LesionBankEntry


# Use a small frame to keep tests fast. Geometry math is identical to (384,160,384).
FRAME_SHAPE: tuple[int, int, int] = (40, 20, 40)


def make_uniform_donor(
    *,
    size_xyz: tuple[int, int, int] = (5, 3, 5),
    intensity: float = 0.7,
    centroid: tuple[int, int, int] | None = None,
    donor_patient_id: str = "donor_0",
    donor_cc_id: int = 1,
) -> LesionBankEntry:
    """Build a fake :class:`LesionBankEntry` with a uniform-intensity tight CC.

    The mask covers the entire ``size_xyz`` volume (a small cuboid lesion).
    Shell mask is one-voxel thick on the outer face; suitable for blend tests.
    """
    sx, sy, sz = size_xyz
    tight_mask = np.ones((sx, sy, sz), dtype=np.uint8)
    tight_intensities = np.full((sx, sy, sz), float(intensity), dtype=np.float32)
    # Shell: zero (we'll use a tiny synthetic shell mask of zero size to
    # exercise the no-shell branch in some tests). For others, a 0-voxel shell
    # is fine because cuboid + zero-padding leaves no exterior voxels in the
    # tight bbox.
    tight_shell_mask = np.zeros((sx, sy, sz), dtype=np.uint8)
    if centroid is None:
        centroid = (sx // 2, sy // 2, sz // 2)
    return LesionBankEntry(
        donor_patient_id=donor_patient_id,
        donor_cc_id=int(donor_cc_id),
        tight_mask=tight_mask,
        tight_intensities=tight_intensities,
        tight_shell_mask=tight_shell_mask,
        centroid_offset_in_tight=tuple(int(c) for c in centroid),
        z_extent_voxels=int(sy),
        intensity_mean=float(intensity),
        intensity_std=1e-6,
        physical_extent_mm=(sx * 0.82, sy * 1.5, sz * 0.82),
    )


def make_donor_with_shell(
    *,
    size_xyz: tuple[int, int, int] = (5, 3, 5),
    cc_intensity: float = 1.0,
    cc_intensity_std: float = 0.1,
) -> LesionBankEntry:
    """Donor whose tight bbox is large enough to include a 1-voxel shell ring."""
    sx, sy, sz = size_xyz
    tight_mask = np.zeros((sx, sy, sz), dtype=np.uint8)
    # Inner CC: 1-voxel margin on all sides so the bbox includes the shell.
    tight_mask[1 : sx - 1, :, 1 : sz - 1] = 1
    tight_intensities = np.zeros((sx, sy, sz), dtype=np.float32)
    rng = np.random.default_rng(0)
    cc_bool = tight_mask.astype(bool)
    tight_intensities[cc_bool] = (
        rng.normal(0.0, 1.0, size=int(cc_bool.sum())).astype(np.float32) * cc_intensity_std
        + cc_intensity
    )
    # Shell: one-voxel ring around the CC, in-plane only (to mimic the real
    # 1 mm anisotropic shell).
    tight_shell_mask = np.zeros((sx, sy, sz), dtype=np.uint8)
    tight_shell_mask[0, :, :] = 1
    tight_shell_mask[sx - 1, :, :] = 1
    tight_shell_mask[:, :, 0] = 1
    tight_shell_mask[:, :, sz - 1] = 1
    tight_shell_mask &= 1 - tight_mask  # exclude any overlap with CC
    centroid = (sx // 2, sy // 2, sz // 2)
    return LesionBankEntry(
        donor_patient_id="donor_shell",
        donor_cc_id=1,
        tight_mask=tight_mask,
        tight_intensities=tight_intensities,
        tight_shell_mask=tight_shell_mask,
        centroid_offset_in_tight=centroid,
        z_extent_voxels=int(sy),
        intensity_mean=float(tight_intensities[cc_bool].mean()),
        intensity_std=float(tight_intensities[cc_bool].std() + 1e-6),
        physical_extent_mm=(sx * 0.82, sy * 1.5, sz * 0.82),
    )


def make_sample(
    *,
    rng_seed: int = 0,
    frame: tuple[int, int, int] = FRAME_SHAPE,
    border_band_size: int = 200,
    pre_lesion_box: tuple[int, int, int, int, int, int] | None = None,
    patient_id: str = "p0",
    slice_y: int | None = None,
) -> Sample:
    """Build a synthetic :class:`Sample` with full-cropped arrays populated."""
    rng = np.random.default_rng(rng_seed)
    fx, fy, fz = frame
    volume = rng.standard_normal(frame, dtype=np.float32) * 0.5
    lesion_mask = np.zeros(frame, dtype=np.uint8)
    if pre_lesion_box is not None:
        x0, x1, y0, y1, z0, z1 = pre_lesion_box
        lesion_mask[x0:x1, y0:y1, z0:z1] = 1

    # Border band: random voxels in the volume (kept inside-bounds).
    n = int(border_band_size)
    xs = rng.integers(5, fx - 5, size=n, dtype=np.int16)
    ys = rng.integers(2, fy - 2, size=n, dtype=np.int16)
    zs = rng.integers(5, fz - 5, size=n, dtype=np.int16)
    border_band = np.stack([xs, ys, zs], axis=1).astype(np.int16)

    if slice_y is None:
        slice_y = fy // 2

    return Sample(
        volume_5ch=np.zeros((5, fz, fx), dtype=np.float32),
        lesion_mask_center=np.zeros((fz, fx), dtype=np.uint8),
        boxes=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        patient_id=patient_id,
        slice_y=int(slice_y),
        is_positive_volume=False,
        is_positive_slice=False,
        pad_offset=(0, 0, 0),
        volume_full_cropped=volume,
        lesion_mask_full_cropped=lesion_mask,
        border_band_coords=border_band,
    )


@pytest.fixture
def synth_donor() -> LesionBankEntry:
    return make_uniform_donor()


@pytest.fixture
def synth_sample() -> Sample:
    return make_sample()
