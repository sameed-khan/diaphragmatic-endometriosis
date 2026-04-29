"""Slice-level ``LesionDataset`` (Component 3).

See ``agent/complete_spec/03_dataset_datamodule.md`` for the full spec.

The dataset operates over an in-RAM cache of preprocessed volumes. Per
``__getitem__`` it produces a ``Sample`` carrying:

  - The 5-channel center-slice triplet ``(C=5, H=Z=384, W=X=384)`` float32.
  - The center-slice 2D lesion mask ``(H=Z=384, W=X=384)`` uint8.
  - The 2D ``(x1, z1, x2, z2)`` boxes for the slice (``(N, 4)`` float32).
  - When ``augment is not None`` (training path), the full cropped volume +
    lesion mask + jitter-translated border-band coordinates so Component 4 can
    paste/transform before the 5-channel slice is finally extracted.

Coordinate frame (mirrors PRD §5.2 + spec §5):

    cache:   (X=408, Y=174, Z=408)   pad_offset = (12, 7, 12)
    target:  (X=384, Y=160, Z=384)
    jitter:  per-axis uniform in [-12, +12] x [-7, +7] x [-12, +12] at train,
             zero (== centered crop) at val/inference. The crop start in cache
             coords is ``(12 - jx, 7 - jy, 12 - jz)``.

The ``slice_y`` carried in the emitted ``Sample`` is in the *cropped* frame,
i.e. ``slice_y_target = slice_y_cached - (7 - jy)``.

The constructor accepts a tunable ``target_input_shape`` so synthetic-cache
fixtures can use a smaller stand-in (the caching pad offset is then
``((cache - target) // 2)`` per axis and the jitter half-extent matches).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from endo.data.samples import Sample


class LesionDataset(Dataset):
    """Slice-level dataset over a fold's worth of cached patients."""

    def __init__(
        self,
        patient_ids: list[str],
        cache: dict[str, dict[str, Any]],
        gt_boxes_by_pid_slice: dict[tuple[str, int], np.ndarray],
        slice_index: list[tuple[str, int, bool, str]],
        target_input_shape: tuple[int, int, int] = (384, 160, 384),
        slice_window: int = 5,
        augment: Callable[[Sample], Sample] | None = None,
        rng_seed: int = 42,
        cache_shape: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        patient_ids
            Patients this dataset draws from. (The ``slice_index`` is
            authoritative for actual sampling; this is mostly informational.)
        cache
            Per-patient dict of in-RAM arrays:
            ``{pid: {"volume": np.ndarray (cache_shape) fp16,
                     "lesion_mask": np.ndarray | None  (cache_shape) uint8,
                     "border_band": np.ndarray | None  (M, 3) int16,
                     "manifest_row": dict}}``.
        gt_boxes_by_pid_slice
            Pre-built lookup ``(pid, slice_y_cached) -> (N, 4) float32`` boxes
            in the *cached* frame ``(0..cache_X)``.
        slice_index
            List of ``(pid, slice_y_cached, is_positive_slice, kind)`` entries.
            ``__len__`` and ``__getitem__`` index this list directly.
        target_input_shape
            ``(X, Y, Z)`` of the cropped frame returned to the model. Default
            ``(384, 160, 384)``.
        slice_window
            Number of channels in the center-slice triplet (5 ⇒ k-2..k+2).
        augment
            If ``None``, validation/inference path: jitter is centered, full
            arrays are dropped from the ``Sample``. If callable, training path:
            random jitter, full arrays + border-band coords populated, callable
            is invoked on the produced ``Sample`` and its return is yielded.
        rng_seed
            Seed for the per-instance RNG used for jitter sampling.
        cache_shape
            ``(X, Y, Z)`` of the underlying cached volumes. If ``None``, peek
            the first available cache entry.
        """
        if slice_window % 2 == 0:
            raise ValueError(f"slice_window must be odd, got {slice_window}")

        self.patient_ids = list(patient_ids)
        self.cache = cache
        self.gt_lookup = gt_boxes_by_pid_slice
        self.slice_index = slice_index
        self.target_shape = tuple(target_input_shape)
        self.slice_window = slice_window
        self.augment = augment
        self._half = slice_window // 2

        if cache_shape is None:
            # Peek any patient that is actually loaded
            any_pid = next(iter(cache))
            cache_shape = tuple(cache[any_pid]["volume"].shape)
        self.cache_shape = tuple(cache_shape)

        # Pad / jitter geometry
        cx, cy, cz = self.cache_shape
        tx, ty, tz = self.target_shape
        if (cx - tx) % 2 or (cy - ty) % 2 or (cz - tz) % 2:
            raise ValueError(
                f"cache_shape {self.cache_shape} - target_shape {self.target_shape} "
                "must be even per-axis"
            )
        self.pad_offset = ((cx - tx) // 2, (cy - ty) // 2, (cz - tz) // 2)
        # half-extent of jitter on each axis (centered crop ± these values)
        self.jitter_max = self.pad_offset  # (12, 7, 12) for default shapes

        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Sample:
        pid, slice_y_cached, is_positive_slice, _kind = self.slice_index[idx]
        entry = self.cache[pid]

        # 1. Determine jitter offsets.
        if self.augment is None:
            jx, jy, jz = 0, 0, 0
        else:
            jx_max, jy_max, jz_max = self.jitter_max
            jx = int(self._rng.integers(-jx_max, jx_max + 1))
            jy = int(self._rng.integers(-jy_max, jy_max + 1))
            jz = int(self._rng.integers(-jz_max, jz_max + 1))

            # Clamp jy so the center-slice window stays in bounds.
            # slice_y_target = slice_y_cached - py + jy ∈ [half, ty - half)
            tx, ty, tz = self.target_shape
            px, py, pz = self.pad_offset
            target_unjittered = slice_y_cached - py
            jy_lo = self._half - target_unjittered
            jy_hi = (ty - self._half - 1) - target_unjittered
            jy = max(jy_lo, min(jy, jy_hi))
            # Clamp jx, jz so the (X, Z) crop window stays inside the cache.
            cx, cy, cz = self.cache_shape
            jx = max(-(cx - tx - px), min(jx, px))
            jz = max(-(cz - tz - pz), min(jz, pz))

        return self._build_sample(pid, slice_y_cached, is_positive_slice, entry, jx, jy, jz)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        pid: str,
        slice_y_cached: int,
        is_positive_slice: bool,
        entry: dict[str, Any],
        jx: int,
        jy: int,
        jz: int,
    ) -> Sample:
        tx, ty, tz = self.target_shape
        px, py, pz = self.pad_offset

        # Crop start in cache coordinates.
        x_start = px - jx
        y_start = py - jy
        z_start = pz - jz
        x_end, y_end, z_end = x_start + tx, y_start + ty, z_start + tz

        volume_full: np.ndarray = entry["volume"]
        lesion_full: np.ndarray | None = entry.get("lesion_mask")

        # 3. Sub-crop the cached volume to the target frame.
        volume_cropped = volume_full[x_start:x_end, y_start:y_end, z_start:z_end]
        if lesion_full is not None:
            lesion_cropped = lesion_full[x_start:x_end, y_start:y_end, z_start:z_end]
        else:
            lesion_cropped = None

        # 4. Map slice_y from cache frame to crop frame and check window validity.
        slice_y_target = slice_y_cached - (py - jy)
        if not (self._half <= slice_y_target < ty - self._half):
            raise IndexError(
                f"slice_y_cached={slice_y_cached} jy={jy} maps to "
                f"slice_y_target={slice_y_target} which is outside the valid "
                f"window [{self._half}, {ty - self._half})"
            )

        # 5. Extract 5-channel triplet. Cache is (X, Y, Z); we want (C=window, H=Z, W=X).
        triplet_xyz = volume_cropped[
            :, slice_y_target - self._half : slice_y_target + self._half + 1, :
        ]  # (X, C, Z)
        # Reorder to (C, Z, X) == (C, H, W) per Sample contract (H=Z, W=X).
        volume_5ch = np.transpose(triplet_xyz, (1, 2, 0)).astype(np.float32, copy=False)
        # Make sure underlying buffer is contiguous; .astype on a view returns a copy
        # only when the dtype changes — cast to float32 forces it. If already float32
        # we still want C-contiguous for downstream torch.from_numpy.
        if not volume_5ch.flags.c_contiguous:
            volume_5ch = np.ascontiguousarray(volume_5ch)

        # 6. Lesion-mask center: (X, Z) at slice_y_target → (Z, X) per (H, W).
        if lesion_cropped is not None:
            lesion_mask_center_xz = lesion_cropped[:, slice_y_target, :]  # (X, Z)
            lesion_mask_center = np.ascontiguousarray(lesion_mask_center_xz.T).astype(
                np.uint8, copy=False
            )
        else:
            lesion_mask_center = np.zeros((tz, tx), dtype=np.uint8)

        # 7. Boxes for this slice. Cached coords are in cache frame (0..cache_X);
        #    translate to crop frame and clip to [0, tx) / [0, tz).
        boxes_cached = self.gt_lookup.get((pid, slice_y_cached))
        if boxes_cached is None or boxes_cached.shape[0] == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = boxes_cached.astype(np.float32, copy=True)
            # x1, x2 → x_axis; z1, z2 → z_axis.
            boxes[:, 0] -= x_start
            boxes[:, 2] -= x_start
            boxes[:, 1] -= z_start
            boxes[:, 3] -= z_start
            # Clip to [0, target).
            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, tx)
            boxes[:, 2] = np.clip(boxes[:, 2], 0.0, tx)
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, tz)
            boxes[:, 3] = np.clip(boxes[:, 3], 0.0, tz)
            # Drop degenerate boxes (fully outside the crop).
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

        is_positive_volume = entry["manifest_row"].get("label") == "positive"

        # 8. Carry full arrays only when training-aug path is active.
        if self.augment is None:
            volume_full_cropped = None
            lesion_mask_full_cropped = None
            border_band_coords = None
        else:
            volume_full_cropped = np.ascontiguousarray(volume_cropped).astype(
                np.float32, copy=False
            )
            if lesion_cropped is not None:
                lesion_mask_full_cropped = np.ascontiguousarray(lesion_cropped).astype(
                    np.uint8, copy=False
                )
            else:
                lesion_mask_full_cropped = np.zeros(self.target_shape, dtype=np.uint8)

            band_full: np.ndarray | None = entry.get("border_band")
            if band_full is None or band_full.shape[0] == 0:
                border_band_coords = np.zeros((0, 3), dtype=np.int16)
            else:
                shifted = band_full.astype(np.int32, copy=True)
                shifted[:, 0] -= x_start
                shifted[:, 1] -= y_start
                shifted[:, 2] -= z_start
                in_range = (
                    (shifted[:, 0] >= 0)
                    & (shifted[:, 0] < tx)
                    & (shifted[:, 1] >= 0)
                    & (shifted[:, 1] < ty)
                    & (shifted[:, 2] >= 0)
                    & (shifted[:, 2] < tz)
                )
                border_band_coords = shifted[in_range].astype(np.int16, copy=False)

        sample = Sample(
            volume_5ch=volume_5ch,
            lesion_mask_center=lesion_mask_center,
            boxes=boxes,
            labels=labels,
            patient_id=pid,
            slice_y=int(slice_y_target),
            is_positive_volume=bool(is_positive_volume),
            is_positive_slice=bool(is_positive_slice),
            pad_offset=tuple(self.pad_offset),
            volume_full_cropped=volume_full_cropped,
            lesion_mask_full_cropped=lesion_mask_full_cropped,
            border_band_coords=border_band_coords,
        )

        if self.augment is not None:
            sample = self.augment(sample)

        return sample
