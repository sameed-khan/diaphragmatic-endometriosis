"""Training loop configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    max_epochs: int = 60
    batch_size: int = 8
    num_workers: int = 8

    base_lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 1

    aux_seg_weight: float = 0.3
    ema_decay: float = 0.999

    precision: Literal["bf16-mixed", "16-mixed", "32-true"] = "bf16-mixed"
    gradient_clip_val: float = 1.0

    log_every_n_steps: int = 10

    target_input_shape: tuple[int, int, int] = (384, 160, 384)
    slice_window: int = 5
