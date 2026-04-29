"""Box re-derivation from the augmented lesion mask (Component 4 §8).

After paste + geometric + intensity, we re-derive 2D ``(x1, z1, x2, z2)`` boxes
from the (warped) ``lesion_mask_full_cropped`` using the locked CC connectivity
(read from ``cache/v1/runtime/connectivity_lock.json``; default 26 if missing).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _structure_for_connectivity(connectivity: int) -> np.ndarray:
    if int(connectivity) == 6:
        return ndi.generate_binary_structure(3, 1)
    if int(connectivity) == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")


def read_connectivity(connectivity_lock_path: Path | None) -> int:
    """Read the locked connectivity (6 or 26) from disk; default 26."""
    if connectivity_lock_path is None:
        return 26
    p = Path(connectivity_lock_path)
    if not p.exists():
        return 26
    try:
        data = json.loads(p.read_text())
        c = int(data.get("connectivity", 26))
        if c not in (6, 26):
            return 26
        return c
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to read %s: %s; defaulting to 26-connectivity", p, exc)
        return 26


# ---------------------------------------------------------------------------
# Per-slice CC bbox derivation
# ---------------------------------------------------------------------------


def clamp_box_to_frame(
    box: tuple[float, float, float, float],
    frame_xz: tuple[int, int],
    *,
    min_dim: int = 2,
) -> tuple[int, int, int, int] | None:
    """Clamp a single 2D box to the frame; drop tiny boxes.

    Box format: ``(x1, z1, x2, z2)`` with ``x2/z2`` exclusive.
    Returns ``None`` if either dimension is below ``min_dim``.
    """
    fx, fz = frame_xz
    x1 = int(max(0, min(int(box[0]), fx)))
    z1 = int(max(0, min(int(box[1]), fz)))
    x2 = int(max(0, min(int(box[2]), fx)))
    z2 = int(max(0, min(int(box[3]), fz)))
    if x2 - x1 < int(min_dim) or z2 - z1 < int(min_dim):
        return None
    return (x1, z1, x2, z2)


def derive_boxes_from_mask(
    lesion_mask_2d: np.ndarray,
    *,
    connectivity: int = 26,
    min_dim: int = 2,
) -> list[tuple[int, int, int, int]]:
    """2D CC → bbox derivation for a single ``(X, Z)`` slice.

    The 2D structure is the in-plane projection of the 3D structure: 8-conn for
    26-connectivity, 4-conn for 6-connectivity (axis-aligned only).
    """
    if int(connectivity) == 6:
        struct2d = ndi.generate_binary_structure(2, 1)  # 4-conn
    else:
        struct2d = np.ones((3, 3), dtype=np.uint8)  # 8-conn

    binary = (lesion_mask_2d > 0).astype(np.uint8)
    if not binary.any():
        return []
    labels, n_cc = ndi.label(binary, structure=struct2d)
    if n_cc == 0:
        return []
    objects = ndi.find_objects(labels)
    fx, fz = lesion_mask_2d.shape
    boxes: list[tuple[int, int, int, int]] = []
    for obj in objects:
        if obj is None:
            continue
        x_slc, z_slc = obj
        x1, x2 = int(x_slc.start), int(x_slc.stop)
        z1, z2 = int(z_slc.start), int(z_slc.stop)
        clamped = clamp_box_to_frame((x1, z1, x2, z2), (fx, fz), min_dim=min_dim)
        if clamped is None:
            _LOGGER.debug(
                "Dropped sub-pixel box at (%d,%d)-(%d,%d) (mask shape %s)",
                x1, z1, x2, z2, lesion_mask_2d.shape,
            )
            continue
        boxes.append(clamped)
    return boxes


def derive_all_boxes(
    lesion_mask_3d: np.ndarray,
    *,
    connectivity: int = 26,
    min_dim: int = 2,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """3D CC label → per-slice 2D bboxes.

    Boxes are split per slice_y from a single 3D label op (so a single CC
    spanning multiple slices contributes to each slice it intersects).
    """
    structure = _structure_for_connectivity(connectivity)
    binary = (lesion_mask_3d > 0).astype(np.uint8)
    if not binary.any():
        return {}
    labels_3d, n_cc = ndi.label(binary, structure=structure)
    if n_cc == 0:
        return {}

    out: dict[int, list[tuple[int, int, int, int]]] = {}
    fx, _fy, fz = lesion_mask_3d.shape
    for cc_id in range(1, n_cc + 1):
        mask_cc = labels_3d == cc_id
        if not mask_cc.any():
            continue
        # Find which Y slices this CC intersects.
        ys = np.where(mask_cc.any(axis=(0, 2)))[0]
        for y in ys:
            # 2D bbox of this CC at this y.
            slc = mask_cc[:, int(y), :]
            xs_idx, zs_idx = np.where(slc)
            x1, x2 = int(xs_idx.min()), int(xs_idx.max()) + 1
            z1, z2 = int(zs_idx.min()), int(zs_idx.max()) + 1
            clamped = clamp_box_to_frame((x1, z1, x2, z2), (fx, fz), min_dim=min_dim)
            if clamped is None:
                continue
            out.setdefault(int(y), []).append(clamped)
    return out
