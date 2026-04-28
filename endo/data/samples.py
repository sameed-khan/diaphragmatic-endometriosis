"""Per-sample / per-batch dataclasses produced by the dataset and collate fn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Sample:
    """A single training/inference sample emitted by ``LesionDataset``.

    Spatial conventions match Component 4 §9:

      - ``volume_5ch`` is in PyTorch (C, H, W) layout where ``H = Z`` and
        ``W = X`` (anatomical I-S and R-L respectively).
      - ``boxes`` are in slice-2D ``(x1, z1, x2, z2)`` ≡ ``(W_min, H_min, W_max, H_max)``.
    """

    volume_5ch: np.ndarray  # (5, 384, 384) float32
    lesion_mask_center: np.ndarray  # (384, 384) uint8
    boxes: np.ndarray  # (N, 4) float32, (x1, z1, x2, z2)
    labels: np.ndarray  # (N,) int64
    patient_id: str
    slice_y: int
    is_positive_volume: bool
    is_positive_slice: bool
    pad_offset: tuple[int, int, int]

    # Forwarded only for the augmentation path; ``None`` at val/inference.
    volume_full_cropped: np.ndarray | None = None  # (384, 160, 384) float32
    lesion_mask_full_cropped: np.ndarray | None = None  # (384, 160, 384) uint8
    border_band_coords: np.ndarray | None = None  # (M, 3) int16, cropped frame


@dataclass
class Batch:
    volume_5ch: torch.Tensor  # (B, 5, 384, 384) float32
    lesion_mask_center: torch.Tensor  # (B, 384, 384) uint8
    boxes: list[torch.Tensor]  # length B; per-image (N_i, 4)
    labels: list[torch.Tensor]  # length B; per-image (N_i,)
    patient_ids: list[str]
    slice_ys: torch.Tensor  # (B,) int64
    is_positive_volume: torch.Tensor  # (B,) bool
    is_positive_slice: torch.Tensor  # (B,) bool

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "Batch":
        return Batch(
            volume_5ch=self.volume_5ch.to(device, non_blocking=non_blocking),
            lesion_mask_center=self.lesion_mask_center.to(device, non_blocking=non_blocking),
            boxes=[b.to(device, non_blocking=non_blocking) for b in self.boxes],
            labels=[ll.to(device, non_blocking=non_blocking) for ll in self.labels],
            patient_ids=self.patient_ids,
            slice_ys=self.slice_ys.to(device, non_blocking=non_blocking),
            is_positive_volume=self.is_positive_volume.to(device, non_blocking=non_blocking),
            is_positive_slice=self.is_positive_slice.to(device, non_blocking=non_blocking),
        )
