"""Geometric augmentation (Component 4 §6).

In-plane (X-Z) affine + elastic deformation, applied lockstep to the volume
and lesion mask, **coherent across all Y slices** (invariant T1.13).

The convention here matches the cached volume layout ``(X, Y, Z)`` so the
in-plane operates on axes 0 and 2 (Y is axis 1).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from endo.config.augmentation import GeometricConfig


# ---------------------------------------------------------------------------
# Affine
# ---------------------------------------------------------------------------


def random_affine_2d(
    rng: np.random.Generator,
    *,
    max_rot_deg: float,
    scale_min: float,
    scale_max: float,
    max_translate_px_x: float,
    max_translate_px_z: float,
) -> np.ndarray:
    """Sample a 2x3 affine that maps OUTPUT (x, z) → INPUT (x, z) coords.

    The form returned is a single 2x3 matrix ``[[a, b, tx], [c, d, tz]]``
    such that ``(x_in, z_in)^T = M @ (x_out, z_out, 1)^T``. This is exactly
    the inverse-mapping convention used by ``scipy.ndimage.affine_transform``
    and ``map_coordinates``.
    """
    theta = float(rng.uniform(-float(max_rot_deg), float(max_rot_deg))) * np.pi / 180.0
    s = float(rng.uniform(float(scale_min), float(scale_max)))
    tx = float(rng.uniform(-float(max_translate_px_x), float(max_translate_px_x)))
    tz = float(rng.uniform(-float(max_translate_px_z), float(max_translate_px_z)))

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # FORWARD transform (output coords <- input coords):
    #   [x_out]   [s*cos  -s*sin] [x_in - cx]   [cx + tx]
    #   [z_out] = [s*sin   s*cos] [z_in - cz] + [cz + tz]
    # We need INVERSE for ndimage.
    # inverse: F^{-1}(x_out) = R^{-1} (x_out - cx - t) / s + cx
    #                        = (1/s) [ cos  sin] (x_out - cx - tx) + cx
    #                                [-sin  cos] (z_out - cz - tz)   cz
    inv_s = 1.0 / s
    a, b = inv_s * cos_t, inv_s * sin_t
    c, d = -inv_s * sin_t, inv_s * cos_t
    # Centred at frame middle, applied in pixel coords downstream.
    M = np.array(
        [
            [a, b, 0.0],
            [c, d, 0.0],
        ],
        dtype=np.float64,
    )
    # Pack the translation channel separately as (tx_post, tz_post). The actual
    # offset depends on frame size (computed later in apply_affine_lockstep).
    # We stash the FORWARD translation here and synthesize the offset there.
    M_forward = np.array(
        [
            [s * cos_t, -s * sin_t, tx],
            [s * sin_t, s * cos_t, tz],
        ],
        dtype=np.float64,
    )
    # Combine inverse linear with a placeholder offset column we'll overwrite
    # in apply_affine_lockstep when we know the frame center.
    M[0, 2] = M_forward[0, 2]  # smuggle forward tx
    M[1, 2] = M_forward[1, 2]  # smuggle forward tz
    return M


def _affine_inverse_offset(M_smuggle: np.ndarray, shape_xz: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert the (smuggled) forward-translation matrix into the proper
    inverse 2x2 + offset that ``ndimage.affine_transform`` expects."""
    # Recover forward parameters from smuggled matrix.
    a_inv, b_inv = float(M_smuggle[0, 0]), float(M_smuggle[0, 1])
    c_inv, d_inv = float(M_smuggle[1, 0]), float(M_smuggle[1, 1])
    tx_fwd = float(M_smuggle[0, 2])
    tz_fwd = float(M_smuggle[1, 2])

    inv_lin = np.array([[a_inv, b_inv], [c_inv, d_inv]], dtype=np.float64)

    cx = (shape_xz[0] - 1) / 2.0
    cz = (shape_xz[1] - 1) / 2.0
    centre = np.array([cx, cz], dtype=np.float64)
    fwd_t = np.array([tx_fwd, tz_fwd], dtype=np.float64)
    # x_in = inv_lin @ (x_out - centre - fwd_t) + centre
    offset = centre - inv_lin @ (centre + fwd_t)
    return inv_lin, offset


def apply_affine_lockstep(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    affine_2x3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same in-plane affine to every Y slice of (volume, lesion_mask).

    Volume uses bilinear (order=1); lesion_mask uses nearest (order=0).
    """
    fx, fy, fz = volume.shape
    inv_lin, offset = _affine_inverse_offset(affine_2x3, (fx, fz))

    # Build full 3D inverse mapping by leaving Y as identity (no displacement
    # along axis 1). For ndimage.affine_transform, the matrix maps OUTPUT →
    # INPUT coords.
    M3 = np.eye(3, dtype=np.float64)
    M3[0, 0], M3[0, 2] = inv_lin[0, 0], inv_lin[0, 1]
    M3[2, 0], M3[2, 2] = inv_lin[1, 0], inv_lin[1, 1]
    off3 = np.array([offset[0], 0.0, offset[1]], dtype=np.float64)

    out_vol = ndi.affine_transform(
        volume,
        matrix=M3,
        offset=off3,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    out_msk = ndi.affine_transform(
        lesion_mask.astype(np.uint8),
        matrix=M3,
        offset=off3,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.uint8)
    return out_vol.astype(volume.dtype, copy=False), out_msk


# ---------------------------------------------------------------------------
# Elastic
# ---------------------------------------------------------------------------


def random_elastic_2d(
    rng: np.random.Generator,
    *,
    alpha: float,
    sigma: float,
    shape_xz: tuple[int, int],
    n_control_points: int = 8,
) -> np.ndarray:
    """Generate a 2-D displacement field of shape ``(2, X, Z)``.

    Implementation: draw ``(n_control_points, n_control_points)`` Gaussian
    samples for each of dx/dz and bicubic-zoom up to ``shape_xz``. The
    field is then returned as ``np.stack([dx, dz], axis=0)``.

    The ``alpha`` parameter scales the displacement magnitude. The scheme
    (Gaussian × bicubic upsample) is equivalent to a low-pass-smoothed
    random field; ``sigma`` here controls the per-control-point noise σ.
    """
    n = int(n_control_points)
    fx, fz = int(shape_xz[0]), int(shape_xz[1])
    dx_lo = rng.normal(0.0, float(sigma), size=(n, n)).astype(np.float64)
    dz_lo = rng.normal(0.0, float(sigma), size=(n, n)).astype(np.float64)

    zoom_x = fx / float(n)
    zoom_z = fz / float(n)
    dx = ndi.zoom(dx_lo, [zoom_x, zoom_z], order=3, mode="nearest")[:fx, :fz]
    dz = ndi.zoom(dz_lo, [zoom_x, zoom_z], order=3, mode="nearest")[:fx, :fz]

    # Pad/truncate to exact shape (zoom can sometimes round off-by-one).
    if dx.shape != (fx, fz):
        dx_full = np.zeros((fx, fz), dtype=np.float64)
        dx_full[: dx.shape[0], : dx.shape[1]] = dx[:fx, :fz]
        dx = dx_full
    if dz.shape != (fx, fz):
        dz_full = np.zeros((fx, fz), dtype=np.float64)
        dz_full[: dz.shape[0], : dz.shape[1]] = dz[:fx, :fz]
        dz = dz_full

    field = np.stack([dx * float(alpha), dz * float(alpha)], axis=0).astype(np.float64)
    return field


def apply_elastic_lockstep(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same in-plane elastic field to every Y slice.

    ``field`` shape: ``(2, X, Z)`` with ``field[0]=dx`` and ``field[1]=dz``.

    Implementation uses ``map_coordinates`` per-slice (cheaper than building
    a full 3D coordinate grid). The same field is reused across slices →
    Y-coherence (invariant T1.13).
    """
    fx, fy, fz = volume.shape
    fxz = (fx, fz)
    dx, dz = field[0], field[1]
    if dx.shape != fxz or dz.shape != fxz:
        raise ValueError(
            f"elastic field shape {dx.shape} does not match volume in-plane {fxz}"
        )

    grid_x, grid_z = np.meshgrid(
        np.arange(fx, dtype=np.float64),
        np.arange(fz, dtype=np.float64),
        indexing="ij",
    )
    src_x = grid_x + dx
    src_z = grid_z + dz
    coords_2d = np.stack([src_x, src_z], axis=0)  # (2, X, Z)

    out_vol = np.empty_like(volume)
    out_msk = np.empty_like(lesion_mask)
    for y in range(fy):
        out_vol[:, y, :] = ndi.map_coordinates(
            volume[:, y, :], coords_2d, order=1, mode="constant", cval=0.0
        )
        out_msk[:, y, :] = ndi.map_coordinates(
            lesion_mask[:, y, :].astype(np.uint8),
            coords_2d,
            order=0,
            mode="constant",
            cval=0,
        ).astype(np.uint8)
    return out_vol, out_msk


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def geometric_aug(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    cfg: GeometricConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose affine + elastic. In-plane only (no Y movement)."""
    fx, _fy, fz = volume.shape
    M = random_affine_2d(
        rng,
        max_rot_deg=cfg.rotation_deg,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        max_translate_px_x=cfg.translation_frac * fx,
        max_translate_px_z=cfg.translation_frac * fz,
    )
    volume, lesion_mask = apply_affine_lockstep(volume, lesion_mask, M)

    if rng.random() < float(getattr(cfg, "p_elastic", 1.0)):
        field = random_elastic_2d(
            rng,
            alpha=1.0,
            sigma=cfg.elastic_sigma,
            shape_xz=(fx, fz),
            n_control_points=cfg.elastic_control_points,
        )
        volume, lesion_mask = apply_elastic_lockstep(volume, lesion_mask, field)

    return volume, lesion_mask
