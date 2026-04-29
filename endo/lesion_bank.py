"""Lesion Bank — donor-side payload for the copy-paste augmentation.

Implements Component 2 (`agent/complete_spec/02_lesion_bank.md`) and
the PRD §6.4 dataclass schema.

The bank holds, for every connected component (CC) in every cross-validation
positive donor, a tight bounding-box payload comprising:

  - the CC mask
  - the post-z-score intensity values inside the CC (zero outside)
  - a 1 mm anisotropic outer shell of the CC
  - the centroid (in tight-bbox local coords)
  - simple intensity stats and physical extent

The bank is built once by ``scripts/build_lesion_bank.py`` and consumed by
the augmentation transform via :func:`load_bank`.

Anisotropic spacing is fixed at ``(0.82, 1.5, 0.82) mm`` per the
preprocessed cache contract (PRD §5.2.x).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.ndimage as ndi


SPACING_MM: tuple[float, float, float] = (0.82, 1.5, 0.82)
SHELL_THICKNESS_MM: float = 1.0
# Padding for the shell distance transform; chosen to comfortably cover a
# 1 mm dilation along the most-resolved (0.82 mm) axes.
_SHELL_PAD: tuple[int, int, int] = (2, 1, 2)


@dataclass(frozen=True)
class LesionBankEntry:
    """One CC's donor-side payload. See PRD §6.4.

    Spatial axes are ``(x, y, z)`` matching the cached volume layout:
    ``y`` is the slice axis (axis 1), ``x`` and ``z`` are in-plane.
    """

    donor_patient_id: str
    donor_cc_id: int
    tight_mask: np.ndarray  # (Δx, Δy, Δz) uint8
    tight_intensities: np.ndarray  # (Δx, Δy, Δz) float32, zero outside CC
    tight_shell_mask: np.ndarray  # (Δx, Δy, Δz) uint8
    centroid_offset_in_tight: tuple[int, int, int]
    z_extent_voxels: int
    intensity_mean: float
    intensity_std: float
    physical_extent_mm: tuple[float, float, float]


# ---------------------------------------------------------------------------
# CC extraction
# ---------------------------------------------------------------------------


def _structure_for_connectivity(connectivity: int) -> np.ndarray:
    """Return the 3D ``structure`` array for ``scipy.ndimage.label``.

    ``connectivity == 6``  → face-only neighbours (cross structure).
    ``connectivity == 26`` → all neighbours (full ``np.ones((3, 3, 3))``).
    """
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")


def extract_entries_from_arrays(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    patient_id: str,
    connectivity: int,
    spacing_mm: tuple[float, float, float] = SPACING_MM,
    shell_mm: float = SHELL_THICKNESS_MM,
) -> list[LesionBankEntry]:
    """Extract one :class:`LesionBankEntry` per CC in ``lesion_mask``.

    ``volume`` and ``lesion_mask`` are expected to share shape
    ``(X, Y, Z)`` — the standard preprocessed cache layout.
    """
    if volume.shape != lesion_mask.shape:
        raise ValueError(
            f"volume.shape={volume.shape} != lesion_mask.shape={lesion_mask.shape}"
        )

    structure = _structure_for_connectivity(connectivity)
    cc_labels, n_cc = ndi.label(lesion_mask.astype(np.uint8), structure=structure)
    if n_cc == 0:
        return []

    objects = ndi.find_objects(cc_labels)
    entries: list[LesionBankEntry] = []
    for cc_idx, bbox in enumerate(objects):
        if bbox is None:
            continue
        cc_id = cc_idx + 1  # 1-indexed; matches gt_boxes.parquet cc_id

        tight_label_block = cc_labels[bbox]
        tight_mask = (tight_label_block == cc_id).astype(np.uint8)
        tight_vol = np.asarray(volume[bbox], dtype=np.float32)

        tight_intensities = (tight_vol * tight_mask).astype(np.float32)

        # Shell: 1 mm anisotropic outer dilation, exclusive of the CC.
        padded_mask = np.pad(tight_mask, [(p, p) for p in _SHELL_PAD])
        dist_outside = ndi.distance_transform_edt(
            ~padded_mask.astype(bool),
            sampling=spacing_mm,
        )
        shell_padded = ((dist_outside > 0) & (dist_outside <= shell_mm)).astype(np.uint8)
        shell_tight = shell_padded[
            _SHELL_PAD[0] : shell_padded.shape[0] - _SHELL_PAD[0],
            _SHELL_PAD[1] : shell_padded.shape[1] - _SHELL_PAD[1],
            _SHELL_PAD[2] : shell_padded.shape[2] - _SHELL_PAD[2],
        ]

        # Stats over CC voxels.
        cc_bool = tight_mask.astype(bool)
        cc_vals = tight_vol[cc_bool]
        intensity_mean = float(cc_vals.mean())
        intensity_std = float(cc_vals.std())

        coords = np.argwhere(cc_bool)
        centroid = coords.mean(axis=0)
        centroid_offset = (
            int(round(float(centroid[0]))),
            int(round(float(centroid[1]))),
            int(round(float(centroid[2]))),
        )

        z_extent_voxels = int(tight_mask.any(axis=(0, 2)).sum())
        physical_extent_mm = (
            float(tight_mask.shape[0] * spacing_mm[0]),
            float(tight_mask.shape[1] * spacing_mm[1]),
            float(tight_mask.shape[2] * spacing_mm[2]),
        )

        entries.append(
            LesionBankEntry(
                donor_patient_id=patient_id,
                donor_cc_id=cc_id,
                tight_mask=tight_mask,
                tight_intensities=tight_intensities,
                tight_shell_mask=shell_tight,
                centroid_offset_in_tight=centroid_offset,
                z_extent_voxels=z_extent_voxels,
                intensity_mean=intensity_mean,
                intensity_std=intensity_std,
                physical_extent_mm=physical_extent_mm,
            )
        )

    return entries


def extract_entries_for_donor(
    patient_id: str,
    cache_root: Path,
    *,
    connectivity: int,
) -> list[LesionBankEntry]:
    """Memory-mapped extraction for a single donor patient."""
    cache_root = Path(cache_root)
    pdir = cache_root / "volumes" / patient_id
    volume = np.load(pdir / "volume.npy", mmap_mode="r")
    lesion_mask = np.load(pdir / "lesion_mask.npy", mmap_mode="r")
    # Materialize lesion_mask (small) to avoid mmap weirdness in the label op.
    lesion_mask = np.asarray(lesion_mask)
    return extract_entries_from_arrays(
        np.asarray(volume),
        lesion_mask,
        patient_id=patient_id,
        connectivity=connectivity,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bank(entries: Iterable[LesionBankEntry], path: str | Path) -> Path:
    """Pickle ``entries`` to ``path`` atomically. Returns the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(list(entries), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_bank(path: str | Path) -> list[LesionBankEntry]:
    """Load a pickled list of :class:`LesionBankEntry` from ``path``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def current_bank_path(cache_root: str | Path) -> Path:
    """Resolve ``<cache_root>/lesion_banks/current.pkl`` (follows symlink)."""
    p = Path(cache_root) / "lesion_banks" / "current.pkl"
    return p.resolve() if p.exists() or p.is_symlink() else p
