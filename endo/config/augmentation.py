"""Augmentation pipeline configuration (paste, geometric, intensity)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PasteConfig(BaseModel):
    p_any_paste: float = 0.5
    n_paste_sigma: float = 1.0
    n_paste_max: int = 7
    site_local_std_threshold: float = 2.0
    overlap_buffer_voxels: int = 0
    max_paste_attempts: int = 20
    max_oob_clip_frac: float = 0.25


class GeometricConfig(BaseModel):
    rotation_deg: float = 10.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    translation_frac: float = 0.05
    elastic_sigma: float = 2.0
    elastic_control_points: int = 8
    p_elastic: float = 0.5


class IntensityConfig(BaseModel):
    gamma_min: float = 0.8
    gamma_max: float = 1.2
    bias_min: float = 0.9
    bias_max: float = 1.1
    noise_sigma: float = 0.01


class AugmentationConfig(BaseModel):
    paste: PasteConfig = Field(default_factory=PasteConfig)
    geometric: GeometricConfig = Field(default_factory=GeometricConfig)
    intensity: IntensityConfig = Field(default_factory=IntensityConfig)
    skip_subpixel_voxel_threshold: int = 2
