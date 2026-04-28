"""Seed utilities — fork-safe RNG derivation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(*parts: int) -> int:
    """Combine integer parts into a 32-bit seed; deterministic, order-sensitive."""
    h = 0xCBF29CE484222325
    for p in parts:
        h ^= int(p) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return int(h & 0x7FFFFFFF)
