"""``TrainAugmentation`` — top-level callable for the training augment path.

Pipeline (Component 4 §4):

    paste → geometric → intensity → re-derive boxes → extract 5-ch slice

Built once per ``LesionDataModule`` and passed as ``augment_train`` to the
:class:`endo.data.dataset.LesionDataset`. Per-call deterministic given a
seed and the sample's ``(patient_id, slice_y)``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.ndimage as ndi

from endo.augmentation.boxes import derive_boxes_from_mask, read_connectivity
from endo.augmentation.geometric import geometric_aug
from endo.augmentation.intensity import intensity_aug
from endo.augmentation.paste import multi_paste_volume
from endo.config.augmentation import AugmentationConfig
from endo.data.samples import Sample
from endo.lesion_bank import LesionBankEntry, load_bank


_LOGGER = logging.getLogger(__name__)

_DEFAULT_BANK_REL = Path("lesion_banks") / "current.pkl"
_DEFAULT_CONN_LOCK_REL = Path("runtime") / "connectivity_lock.json"
_DEFAULT_LOCAL_STD_REL = Path("runtime") / "cohort_local_std.json"


# ---------------------------------------------------------------------------
# Cohort-local-std cache (lazy, one-time)
# ---------------------------------------------------------------------------


def _compute_local_std_3x3x1(volume: np.ndarray) -> np.ndarray:
    """Return per-voxel std of a (3, 3, 1) box.

    Equivalent to ``sqrt(E[X^2] - E[X]^2)`` with a uniform filter of size
    ``(3, 3, 1)`` over ``(X, Y, Z)``.
    """
    v = volume.astype(np.float32, copy=False)
    mean = ndi.uniform_filter(v, size=(3, 3, 1), mode="reflect")
    mean_sq = ndi.uniform_filter(v * v, size=(3, 3, 1), mode="reflect")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(var)


def _sample_local_std_at(
    volume: np.ndarray,
    coords: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``n_samples`` random voxels from ``coords`` and return local stds.

    Vectorised: build the full 3x3x1 std map once, then index.
    """
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    n = int(min(n_samples, coords.shape[0]))
    pick = rng.choice(coords.shape[0], size=n, replace=False)
    sel = coords[pick]
    std_map = _compute_local_std_3x3x1(volume)
    xs, ys, zs = sel[:, 0], sel[:, 1], sel[:, 2]
    return std_map[xs, ys, zs].astype(np.float32, copy=False)


def compute_cohort_local_std(
    cache_root: Path,
    *,
    samples_per_volume: int = 100,
    n_volumes_max: int | None = None,
    rng_seed: int = 0,
) -> dict[str, object]:
    """One-time scan over the cache: per-volume sample of local stds.

    Returns the schema described in PRD §5.2.6.
    """
    cache_root = Path(cache_root)
    pre_path = cache_root / "preprocessed_manifest.jsonl"
    if not pre_path.exists():
        # Nothing to scan; return an empty / sentinel record so callers don't crash.
        return {
            "cohort_median_local_std": 1.0,
            "n_volumes_sampled": 0,
            "samples_per_volume": int(samples_per_volume),
            "computed_at": _now_iso(),
            "code_version": "unknown",
        }

    rows: list[dict] = []
    for line in pre_path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    rng = np.random.default_rng(rng_seed)
    all_stds: list[np.ndarray] = []
    n_volumes = 0
    cv_neg_rows = [
        r for r in rows
        if r.get("cohort") == "cross-validation" and r.get("label") == "negative"
    ]
    if not cv_neg_rows:
        cv_neg_rows = [r for r in rows if r.get("cohort") == "cross-validation"]

    if n_volumes_max is not None:
        cv_neg_rows = cv_neg_rows[: int(n_volumes_max)]

    for r in cv_neg_rows:
        vol_path = cache_root / r["cache_volume_path"]
        band_rel = r.get("cache_border_band_path")
        if not vol_path.exists() or not band_rel:
            continue
        band_path = cache_root / band_rel
        if not band_path.exists():
            continue
        volume = np.load(vol_path).astype(np.float32, copy=False)
        coords = np.load(band_path).astype(np.int32, copy=False)
        if coords.shape[0] == 0:
            continue
        stds = _sample_local_std_at(
            volume, coords, n_samples=samples_per_volume, rng=rng
        )
        all_stds.append(stds)
        n_volumes += 1

    if not all_stds:
        median_std = 1.0
    else:
        flat = np.concatenate(all_stds)
        median_std = float(np.median(flat))

    return {
        "cohort_median_local_std": float(median_std),
        "n_volumes_sampled": int(n_volumes),
        "samples_per_volume": int(samples_per_volume),
        "computed_at": _now_iso(),
        "code_version": "v1",
    }


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_cohort_local_std(
    cache_root: Path,
    cohort_local_std_path: Path,
    *,
    rng_seed: int = 0,
) -> dict[str, object]:
    """Return cached cohort local-std, computing + caching if missing."""
    p = Path(cohort_local_std_path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Corrupt %s (%s); recomputing.", p, exc)
    record = compute_cohort_local_std(cache_root, rng_seed=rng_seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2))
    return record


# ---------------------------------------------------------------------------
# Per-call seeding
# ---------------------------------------------------------------------------


def _per_sample_seed(base_seed: int, sample: Sample) -> int:
    """Stable per-(seed, patient_id, slice_y) integer seed."""
    h = hashlib.sha256()
    h.update(int(base_seed).to_bytes(8, "little", signed=False))
    h.update(str(sample.patient_id).encode("utf-8"))
    h.update(int(sample.slice_y).to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest()[:8], "little", signed=False)


# ---------------------------------------------------------------------------
# TrainAugmentation
# ---------------------------------------------------------------------------


class TrainAugmentation:
    """Composable augmentation callable for the training path."""

    def __init__(
        self,
        cfg: AugmentationConfig,
        cache_root: Path,
        *,
        bank_path: Path | None = None,
        connectivity_lock_path: Path | None = None,
        cohort_local_std_path: Path | None = None,
        rng_seed: int = 42,
        bank_entries: Sequence[LesionBankEntry] | None = None,
    ) -> None:
        self.cfg = cfg
        self.cache_root = Path(cache_root)
        self.rng_seed = int(rng_seed)

        # 1) Resolve and load the bank.
        if bank_entries is not None:
            self.bank: list[LesionBankEntry] = list(bank_entries)
            self.bank_path = bank_path
        else:
            if bank_path is None:
                bank_path = self.cache_root / _DEFAULT_BANK_REL
            self.bank_path = Path(bank_path)
            if self.bank_path.exists():
                self.bank = list(load_bank(self.bank_path))
            else:
                _LOGGER.warning(
                    "Lesion bank not found at %s; paste will be a no-op.",
                    self.bank_path,
                )
                self.bank = []

        # 2) Resolve connectivity (default 26).
        if connectivity_lock_path is None:
            connectivity_lock_path = self.cache_root / _DEFAULT_CONN_LOCK_REL
        self.connectivity_lock_path = Path(connectivity_lock_path)
        self.connectivity = read_connectivity(self.connectivity_lock_path)

        # 3) Resolve / lazily build cohort_local_std.
        if cohort_local_std_path is None:
            cohort_local_std_path = self.cache_root / _DEFAULT_LOCAL_STD_REL
        self.cohort_local_std_path = Path(cohort_local_std_path)
        self.cohort_local_std_record = _ensure_cohort_local_std(
            self.cache_root, self.cohort_local_std_path, rng_seed=self.rng_seed
        )
        self.cohort_median_local_std = float(
            self.cohort_local_std_record.get("cohort_median_local_std", 1.0)
        )

    # ------------------------------------------------------------------
    # Callable
    # ------------------------------------------------------------------

    def __call__(self, sample: Sample) -> Sample:
        if sample.volume_full_cropped is None or sample.lesion_mask_full_cropped is None:
            # Validation/inference path; nothing to do. Should not normally
            # happen because the dataset only invokes augment on the training
            # branch.
            return sample

        seed = _per_sample_seed(self.rng_seed, sample)
        rng = np.random.default_rng(seed)

        volume = np.ascontiguousarray(sample.volume_full_cropped, dtype=np.float32)
        lesion_mask = np.ascontiguousarray(
            sample.lesion_mask_full_cropped, dtype=np.uint8
        )
        border_band_coords = sample.border_band_coords

        # 1) Paste.
        if self.bank and border_band_coords is not None and border_band_coords.shape[0] > 0:
            volume, lesion_mask, _paste_results = multi_paste_volume(
                volume,
                lesion_mask,
                border_band_coords.astype(np.int32, copy=False),
                self.bank,
                self.cfg.paste,
                rng,
                frame_shape=tuple(int(s) for s in volume.shape),
            )

        # 2) Geometric (in-plane affine + elastic; lockstep, Y-coherent).
        volume, lesion_mask = geometric_aug(volume, lesion_mask, self.cfg.geometric, rng)

        # 3) Intensity (volume only).
        volume = intensity_aug(volume, self.cfg.intensity, rng)

        # 4) Re-derive 2D boxes for the center slice (slice_y).
        slice_y = int(sample.slice_y)
        center_mask_xz = lesion_mask[:, slice_y, :]  # (X, Z)
        boxes_xz = derive_boxes_from_mask(
            center_mask_xz,
            connectivity=self.connectivity,
            min_dim=int(getattr(self.cfg, "skip_subpixel_voxel_threshold", 2)),
        )
        if boxes_xz:
            boxes = np.asarray(boxes_xz, dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

        # 5) Extract 5-channel slice tensor centred at slice_y.
        half = 2  # 5-channel window
        triplet_xyz = volume[:, slice_y - half : slice_y + half + 1, :]  # (X, 5, Z)
        # tensor[c, z, x] = volume[x, slice_y - 2 + c, z] → (5, Z, X)
        volume_5ch = np.ascontiguousarray(
            np.transpose(triplet_xyz, (1, 2, 0)).astype(np.float32, copy=False)
        )
        # Sanity: channel 2 == volume[:, slice_y, :].T
        # (We do not assert in production for cost reasons.)

        lesion_mask_center_xz = lesion_mask[:, slice_y, :]  # (X, Z)
        lesion_mask_center = np.ascontiguousarray(
            lesion_mask_center_xz.T.astype(np.uint8, copy=False)
        )

        # 6) Build the post-aug Sample. Drop the consumed full arrays.
        sample.volume_5ch = volume_5ch
        sample.lesion_mask_center = lesion_mask_center
        sample.boxes = boxes
        sample.labels = labels
        sample.volume_full_cropped = None
        sample.lesion_mask_full_cropped = None
        sample.border_band_coords = None
        return sample
