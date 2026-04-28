"""Sampler / hard-negative-mining configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SamplerConfig(BaseModel):
    epoch_mode: Literal["fixed_count", "full_pass"] = "fixed_count"
    samples_per_epoch: int = 6000

    # Class-mix decay (linear from epoch 0 to decay_epochs).
    pos_frac_start: float = 0.50
    pos_frac_end: float = 0.25
    decay_epochs: int = 30

    # Within the negative pool, how much weight to neg-in-pos-volume vs neg-in-neg-volume.
    neg_in_pos_vol_share: float = 0.50

    # Hard-pool substitution applies to neg-in-neg-volume draws.
    hard_pool_substitution_rate: float = 0.30
    hard_pool_start_epoch: int = 5

    deep_eval_refresh_every_epochs: int = 10
    deep_eval_start_epoch: int = 10

    # Loss-EMA tracker decay.
    score_ema_decay: float = 0.9

    hard_pool_top_k: int = 1000
