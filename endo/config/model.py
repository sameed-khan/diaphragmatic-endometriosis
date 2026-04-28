"""Detector model configuration."""

from __future__ import annotations

from pydantic import BaseModel


class ModelConfig(BaseModel):
    backbone_name: str = "convnext_tiny.fb_in22k"
    in_channels: int = 5
    fpn_channels: int = 256
    fpn_strides: tuple[int, ...] = (4, 8, 16, 32)
    head_n_classes: int = 1
    head_stacked_convs: int = 2
    head_feat_channels: int = 256
    aux_seg_channels: int = 64
    aux_seg_target_size: int = 384
