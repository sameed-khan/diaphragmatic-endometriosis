"""Shared fixtures for model tests.

Synthetic batches keep the test cohort cost zero. Image size is 384x384 to
match the production input contract. The detector itself is the production
class but built from a tiny ``ModelConfig`` shim where possible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from endo.config import ExperimentConfig, ModelConfig, TrainingConfig
from endo.config.experiment import ExperimentConfig as _ExpCfg  # alias
from endo.data.samples import Batch


def _exp_config() -> ExperimentConfig:
    return _ExpCfg(
        uuid="b3a7f1e9-4c8a-4d2b-9f1c-0e6a8b9c1d2e",
        name="test-baseline",
        model=ModelConfig(),
        training=TrainingConfig(max_epochs=2, warmup_epochs=1, base_lr=2e-4, min_lr=1e-6),
    )


@pytest.fixture(scope="module")
def exp_cfg() -> ExperimentConfig:
    return _exp_config()


def make_synthetic_batch(B: int = 2, H: int = 384, W: int = 384, n_pos: int = 1) -> Batch:
    """Build a tiny synthetic Batch with B-2 negatives + n_pos positives.

    Box format: (x1, z1, x2, z2) per Sample contract.
    """
    rng = np.random.default_rng(0)
    vol = torch.from_numpy(rng.standard_normal((B, 5, H, W)).astype(np.float32))
    mask = torch.zeros((B, H, W), dtype=torch.uint8)
    boxes_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    is_pos = torch.zeros(B, dtype=torch.bool)

    for i in range(B):
        if i < n_pos:
            # one box, well inside the image, ~32-px on a side.
            x1 = float(rng.integers(64, W - 96))
            z1 = float(rng.integers(64, H - 96))
            x2, z2 = x1 + 32.0, z1 + 32.0
            boxes_list.append(torch.tensor([[x1, z1, x2, z2]], dtype=torch.float32))
            labels_list.append(torch.tensor([0], dtype=torch.int64))
            is_pos[i] = True
            mask[i, int(z1) : int(z2), int(x1) : int(x2)] = 1
        else:
            boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
            labels_list.append(torch.zeros((0,), dtype=torch.int64))

    return Batch(
        volume_5ch=vol,
        lesion_mask_center=mask,
        boxes=boxes_list,
        labels=labels_list,
        patient_ids=[f"PID{i:03d}" for i in range(B)],
        slice_ys=torch.arange(B, dtype=torch.int64),
        is_positive_volume=is_pos.clone(),
        is_positive_slice=is_pos,
    )


@pytest.fixture
def synthetic_batch() -> Batch:
    return make_synthetic_batch(B=2, n_pos=1)
