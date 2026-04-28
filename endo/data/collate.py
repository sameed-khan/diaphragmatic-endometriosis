"""Custom collate fn that produces ``Batch`` from a list of ``Sample``.

Boxes/labels are returned as ``list[Tensor]`` because ``N_i`` (per-image box
count) varies. RTMDet's head accepts this format directly (see PRD §6.3).
"""

from __future__ import annotations

import numpy as np
import torch

from endo.data.samples import Batch, Sample


def collate_fn(samples: list[Sample]) -> Batch:
    if not samples:
        raise ValueError("collate_fn received an empty list")

    volume_5ch = torch.from_numpy(np.stack([s.volume_5ch for s in samples], axis=0)).float()
    lesion_mask_center = torch.from_numpy(
        np.stack([s.lesion_mask_center for s in samples], axis=0)
    ).to(torch.uint8)

    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for s in samples:
        if s.boxes.shape[0] == 0:
            boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            labels.append(torch.zeros((0,), dtype=torch.long))
        else:
            boxes.append(torch.from_numpy(np.ascontiguousarray(s.boxes)).float())
            labels.append(torch.from_numpy(np.ascontiguousarray(s.labels)).long())

    slice_ys = torch.tensor([s.slice_y for s in samples], dtype=torch.long)
    is_positive_volume = torch.tensor(
        [s.is_positive_volume for s in samples], dtype=torch.bool
    )
    is_positive_slice = torch.tensor(
        [s.is_positive_slice for s in samples], dtype=torch.bool
    )

    return Batch(
        volume_5ch=volume_5ch,
        lesion_mask_center=lesion_mask_center,
        boxes=boxes,
        labels=labels,
        patient_ids=[s.patient_id for s in samples],
        slice_ys=slice_ys,
        is_positive_volume=is_positive_volume,
        is_positive_slice=is_positive_slice,
    )
