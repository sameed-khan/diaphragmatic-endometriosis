"""Intensity augmentation (Component 4 §7).

All transforms operate on the volume only. Lesion mask is unchanged.
"""

from __future__ import annotations

import numpy as np

from endo.config.augmentation import IntensityConfig


def random_gamma(
    volume: np.ndarray, rng: np.random.Generator, *, gamma_min: float, gamma_max: float
) -> np.ndarray:
    """Sign-preserving gamma. ``volume`` is z-scored so includes negatives."""
    g = float(rng.uniform(float(gamma_min), float(gamma_max)))
    sign = np.sign(volume)
    mag = np.abs(volume)
    out = sign * np.power(mag, g)
    return out.astype(volume.dtype, copy=False)


def random_brightness_contrast(
    volume: np.ndarray,
    rng: np.random.Generator,
    *,
    bias_min: float,
    bias_max: float,
) -> np.ndarray:
    """Multiplicative bias (a.k.a. contrast) in [bias_min, bias_max]."""
    b = float(rng.uniform(float(bias_min), float(bias_max)))
    return (volume * b).astype(volume.dtype, copy=False)


def random_gaussian_noise(
    volume: np.ndarray, rng: np.random.Generator, *, noise_sigma: float
) -> np.ndarray:
    """Add iid N(0, σ) noise to ``volume``."""
    if float(noise_sigma) <= 0.0:
        return volume
    noise = rng.normal(0.0, float(noise_sigma), size=volume.shape).astype(volume.dtype)
    return (volume + noise).astype(volume.dtype, copy=False)


def intensity_aug(
    volume: np.ndarray, cfg: IntensityConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply mult-bias → gamma → noise (matches spec §7)."""
    out = random_brightness_contrast(
        volume, rng, bias_min=cfg.bias_min, bias_max=cfg.bias_max
    )
    out = random_gamma(out, rng, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max)
    out = random_gaussian_noise(out, rng, noise_sigma=cfg.noise_sigma)
    return out
