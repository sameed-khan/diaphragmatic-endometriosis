"""Auxiliary segmentation head for stride-1 dense supervision.

Component 6 §4.2: takes the FPN's P2 (stride 4) feature, runs two
``ConvTranspose2d`` stages (each upsample 2x with GroupNorm+SiLU), then a
1x1 conv to a single-channel logit map. Final output ``(B, 1, 384, 384)``
when fed P2 of shape ``(B, 256, 96, 96)``.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class AuxSegHead(nn.Module):
    """Lightweight UNet-style decoder from stride 4 to stride 1."""

    def __init__(self, in_channels: int = 256, mid_channels: int = 64) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(8, mid_channels), mid_channels)
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(min(8, mid_channels), mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, p2: Tensor) -> Tensor:
        """``p2`` is stride-4 feature ``(B, in_channels, H/4, W/4)``.

        Returns logits ``(B, 1, H, W)``.
        """
        x = self.act(self.norm1(self.up1(p2)))   # stride 4 -> 2
        x = self.act(self.norm2(self.up2(x)))    # stride 2 -> 1
        return self.out_conv(x)
