"""Lesion copy-paste augmentation (Component 4 §5).

Pastes donor lesion CCs from a global :class:`endo.lesion_bank.LesionBankEntry`
bank into a target volume's right-side liver/diaphragm border band, with:

  - Half-Gaussian-clipped multi-paste schedule (P(n=0) = 1 - p_any_paste).
  - Border-band-only site selection with collision avoidance.
  - Target-local intensity rescaling (donor stats → target shell stats).
  - Soft 1-mm-shell linear blend at the lesion boundary.

Out-of-bounds rejection: a paste whose translated mask has > 25% voxels
clipped is rejected (donor centroid too close to the (384, 160, 384) frame).

All array axes are ``(X, Y, Z)`` matching the cached volume layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.ndimage as ndi

from endo.config.augmentation import PasteConfig
from endo.lesion_bank import LesionBankEntry, SPACING_MM


# Conservative upper bound on placement attempts per requested paste.
MAX_ATTEMPTS_PER_PASTE: int = 50


# ---------------------------------------------------------------------------
# n_pastes scheduler
# ---------------------------------------------------------------------------


def sample_n_pastes(
    rng: np.random.Generator,
    p_any_paste: float,
    n_paste_sigma: float,
    n_paste_max: int,
) -> int:
    """Half-Gaussian-clipped multi-paste count.

    With probability ``1 - p_any_paste`` returns 0. Otherwise draws
    ``int(abs(N(0, sigma))) + 1`` and clips to ``[1, n_paste_max]``. The mode
    of the conditional-on-positive distribution is 1.
    """
    if rng.random() >= float(p_any_paste):
        return 0
    x = abs(rng.normal(0.0, float(n_paste_sigma)))
    n = int(x) + 1
    return int(min(n, int(n_paste_max)))


# ---------------------------------------------------------------------------
# Site selection
# ---------------------------------------------------------------------------


def _donor_extent(donor: LesionBankEntry) -> tuple[int, int, int]:
    return tuple(int(s) for s in donor.tight_mask.shape)  # (Δx, Δy, Δz)


def select_paste_site(
    border_band_coords: np.ndarray,
    occupancy_mask: np.ndarray,
    donor: LesionBankEntry,
    rng: np.random.Generator,
    *,
    frame_shape: tuple[int, int, int] = (384, 160, 384),
    max_attempts: int = MAX_ATTEMPTS_PER_PASTE,
    max_oob_clip_frac: float = 0.25,
) -> tuple[int, int, int] | None:
    """Pick a candidate site (in voxel coords) for ``donor`` in ``frame_shape``.

    The site is a *centroid* coordinate in the target frame; the donor's tight
    bbox is then translated so its ``centroid_offset_in_tight`` lands at the
    site. We require:

      1. Site sampled uniformly from ``border_band_coords``.
      2. The translated donor mask has ≤ ``max_oob_clip_frac`` of its voxels
         clipped by the target frame.
      3. The translated donor mask does not intersect ``occupancy_mask``.

    Returns ``None`` if ``max_attempts`` candidates all fail.
    """
    if border_band_coords is None or border_band_coords.shape[0] == 0:
        return None

    n_band = int(border_band_coords.shape[0])
    cx_off, cy_off, cz_off = donor.centroid_offset_in_tight
    dx_size, dy_size, dz_size = _donor_extent(donor)

    for _ in range(int(max_attempts)):
        idx = int(rng.integers(0, n_band))
        sx, sy, sz = (int(c) for c in border_band_coords[idx])

        # tight-bbox start in target coords (donor centroid → site).
        x0 = sx - cx_off
        y0 = sy - cy_off
        z0 = sz - cz_off
        x1 = x0 + dx_size
        y1 = y0 + dy_size
        z1 = z0 + dz_size

        # Clip frame intersection.
        ix0, iy0, iz0 = max(x0, 0), max(y0, 0), max(z0, 0)
        ix1 = min(x1, int(frame_shape[0]))
        iy1 = min(y1, int(frame_shape[1]))
        iz1 = min(z1, int(frame_shape[2]))
        if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
            continue

        # Donor-local crop indices.
        dxs0, dys0, dzs0 = ix0 - x0, iy0 - y0, iz0 - z0
        dxs1, dys1, dzs1 = dxs0 + (ix1 - ix0), dys0 + (iy1 - iy0), dzs0 + (iz1 - iz0)

        donor_crop = donor.tight_mask[dxs0:dxs1, dys0:dys1, dzs0:dzs1]
        donor_total = int(donor.tight_mask.sum())
        donor_keep = int(donor_crop.sum())
        if donor_total > 0:
            clip_frac = 1.0 - donor_keep / donor_total
            if clip_frac > float(max_oob_clip_frac):
                continue

        # Collision check against already-occupied voxels.
        occ_crop = occupancy_mask[ix0:ix1, iy0:iy1, iz0:iz1]
        if np.any((donor_crop > 0) & (occ_crop > 0)):
            continue

        return (sx, sy, sz)

    return None


# ---------------------------------------------------------------------------
# Single-paste apply
# ---------------------------------------------------------------------------


@dataclass
class _PasteResult:
    site: tuple[int, int, int]
    target_box: tuple[int, int, int, int, int, int]  # (x0,x1,y0,y1,z0,z1) in target frame
    donor_box: tuple[int, int, int, int, int, int]  # corresponding donor-local crop
    paste_mask_crop: np.ndarray  # (Δx', Δy', Δz') uint8 inside the cropped sub-bbox


def apply_paste(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    occupancy_mask: np.ndarray,
    donor: LesionBankEntry,
    site: tuple[int, int, int],
    *,
    spacing_mm: tuple[float, float, float] = SPACING_MM,
    frame_shape: tuple[int, int, int] | None = None,
) -> _PasteResult | None:
    """Composite ``donor`` at ``site`` into ``volume`` and ``lesion_mask`` in place.

    - ``volume``: float32 ``(X, Y, Z)`` array — modified in-place.
    - ``lesion_mask``: uint8 ``(X, Y, Z)`` array — modified in-place
      (donor mask OR'd in).
    - ``occupancy_mask``: uint8 ``(X, Y, Z)`` — also OR'd in-place so that
      subsequent pastes can detect collisions.
    - ``donor``: bank entry whose ``tight_mask`` will be translated so that
      ``donor.centroid_offset_in_tight`` lands on ``site``.
    - ``site``: target-frame centroid voxel ``(sx, sy, sz)``.

    Returns ``None`` if the resulting bbox does not intersect the frame.
    """
    if frame_shape is None:
        frame_shape = tuple(int(s) for s in volume.shape)
    fx, fy, fz = frame_shape

    cx_off, cy_off, cz_off = donor.centroid_offset_in_tight
    dx_size, dy_size, dz_size = _donor_extent(donor)
    sx, sy, sz = (int(c) for c in site)

    x0, y0, z0 = sx - cx_off, sy - cy_off, sz - cz_off
    x1, y1, z1 = x0 + dx_size, y0 + dy_size, z0 + dz_size

    ix0, iy0, iz0 = max(x0, 0), max(y0, 0), max(z0, 0)
    ix1, iy1, iz1 = min(x1, fx), min(y1, fy), min(z1, fz)
    if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
        return None

    dxs0, dys0, dzs0 = ix0 - x0, iy0 - y0, iz0 - z0
    dxs1 = dxs0 + (ix1 - ix0)
    dys1 = dys0 + (iy1 - iy0)
    dzs1 = dzs0 + (iz1 - iz0)

    donor_mask_crop = donor.tight_mask[dxs0:dxs1, dys0:dys1, dzs0:dzs1].astype(bool)
    donor_int_crop = donor.tight_intensities[dxs0:dxs1, dys0:dys1, dzs0:dzs1].astype(
        np.float32
    )
    donor_shell_crop = donor.tight_shell_mask[dxs0:dxs1, dys0:dys1, dzs0:dzs1].astype(
        bool
    )

    # Build target-shell ⇒ compute target-local stats.
    # The shell already excludes the CC interior (per Component 2 §5.1).
    target_view = volume[ix0:ix1, iy0:iy1, iz0:iz1]
    if donor_shell_crop.any():
        # Restrict to voxels that exist inside frame; the crop already does that.
        shell_vals = target_view[donor_shell_crop]
        target_local_mean = float(shell_vals.mean())
        target_local_std = float(shell_vals.std()) if shell_vals.size > 1 else 1.0
    else:
        # Degenerate; fall back to neighbourhood mean of paste-mask voxels.
        target_local_mean = float(target_view[donor_mask_crop].mean()) if donor_mask_crop.any() else 0.0
        target_local_std = 1.0

    # Rescale donor intensities into target stats.
    d_mean = float(donor.intensity_mean)
    d_std = float(donor.intensity_std)
    if d_std <= 1e-6:
        d_std = 1.0
    target_local_std = max(target_local_std, 1e-6)

    # Inside the donor mask, z-score donor intensities and rescale to target.
    donor_normed = np.zeros_like(donor_int_crop, dtype=np.float32)
    donor_normed[donor_mask_crop] = (
        donor_int_crop[donor_mask_crop] - d_mean
    ) / d_std
    injected = donor_normed * float(target_local_std) + float(target_local_mean)

    # 1) Hard-paste interior of the lesion.
    target_view_writable = volume[ix0:ix1, iy0:iy1, iz0:iz1]
    target_view_writable[donor_mask_crop] = injected[donor_mask_crop]

    # 2) Soft-blend the 1 mm shell. We use a linear ramp from 1 at the donor
    #    boundary to 0 at the outer shell edge. Because the shell mask itself is
    #    already a 1-mm anisotropic dilation, we approximate the ramp with the
    #    distance from each shell voxel to the nearest donor voxel (in mm).
    if donor_shell_crop.any():
        # Distance (in mm) from non-CC voxels to the nearest CC voxel,
        # restricted to the cropped shell region.
        dist_inside = ndi.distance_transform_edt(
            ~donor_mask_crop, sampling=spacing_mm
        )
        # Per-voxel α: 1 at the boundary, 0 at >= 1 mm away.
        alpha = np.clip(1.0 - dist_inside, 0.0, 1.0).astype(np.float32)
        alpha[~donor_shell_crop] = 0.0
        # For the shell, blend injected ⇆ original. We need an "injected" value
        # at shell voxels too — extrapolate via the rescaled donor intensities
        # (which are valid for the entire crop window). Where donor_int_crop
        # equals 0 (outside CC), use target_local_mean as a soft fill.
        shell_injected = injected.copy()
        outside_mask = ~donor_mask_crop
        shell_injected[outside_mask] = float(target_local_mean)
        blended = (
            alpha * shell_injected + (1.0 - alpha) * target_view_writable
        ).astype(np.float32)
        sel = donor_shell_crop & (alpha > 0)
        target_view_writable[sel] = blended[sel]

    # 3) Update lesion + occupancy masks.
    lesion_mask[ix0:ix1, iy0:iy1, iz0:iz1] |= donor_mask_crop.astype(np.uint8)
    occupancy_mask[ix0:ix1, iy0:iy1, iz0:iz1] |= donor_mask_crop.astype(np.uint8)

    return _PasteResult(
        site=(sx, sy, sz),
        target_box=(ix0, ix1, iy0, iy1, iz0, iz1),
        donor_box=(dxs0, dxs1, dys0, dys1, dzs0, dzs1),
        paste_mask_crop=donor_mask_crop.astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Multi-paste orchestration
# ---------------------------------------------------------------------------


def multi_paste_volume(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    border_band_coords: np.ndarray | None,
    bank: Sequence[LesionBankEntry],
    cfg: PasteConfig,
    rng: np.random.Generator,
    *,
    frame_shape: tuple[int, int, int] | None = None,
    spacing_mm: tuple[float, float, float] = SPACING_MM,
) -> tuple[np.ndarray, np.ndarray, list[_PasteResult]]:
    """Multi-paste driver.

    Returns ``(volume, lesion_mask, paste_results)`` — the first two are the
    same arrays passed in (modified in-place; returned for ergonomics).
    """
    if frame_shape is None:
        frame_shape = tuple(int(s) for s in volume.shape)

    n_pastes = sample_n_pastes(
        rng,
        p_any_paste=cfg.p_any_paste,
        n_paste_sigma=cfg.n_paste_sigma,
        n_paste_max=cfg.n_paste_max,
    )
    results: list[_PasteResult] = []
    if n_pastes == 0 or len(bank) == 0:
        return volume, lesion_mask, results
    if border_band_coords is None or border_band_coords.shape[0] == 0:
        return volume, lesion_mask, results

    # Occupancy mask seeded from the existing native lesion mask. Subsequent
    # pastes OR their paste_masks into this so non-overlap is enforced across
    # both native lesions and prior synthetic pastes.
    occupancy = (lesion_mask > 0).astype(np.uint8)

    for _ in range(n_pastes):
        donor_idx = int(rng.integers(0, len(bank)))
        donor = bank[donor_idx]

        site = select_paste_site(
            border_band_coords,
            occupancy,
            donor,
            rng,
            frame_shape=frame_shape,
            max_attempts=cfg.max_paste_attempts if hasattr(cfg, "max_paste_attempts") else MAX_ATTEMPTS_PER_PASTE,
            max_oob_clip_frac=cfg.max_oob_clip_frac
            if hasattr(cfg, "max_oob_clip_frac")
            else 0.25,
        )
        if site is None:
            continue
        result = apply_paste(
            volume,
            lesion_mask,
            occupancy,
            donor,
            site,
            spacing_mm=spacing_mm,
            frame_shape=frame_shape,
        )
        if result is not None:
            results.append(result)

    return volume, lesion_mask, results
