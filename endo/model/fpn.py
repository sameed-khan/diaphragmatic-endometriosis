"""4-level top-down FPN with P2 (strides 4, 8, 16, 32).

Per Component 6 §4.1: lateral 1x1 -> top-down nearest upsample + add ->
3x3 Conv-GroupNorm-SiLU smoothing per level. Input is the 4 ConvNeXt-tiny
``features_only`` stages at strides {4, 8, 16, 32}. Output is a list
``[P2, P3, P4, P5]`` each ``(B, out_channels, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gn_groups(channels: int) -> int:
    """Standard convention used by the rest of the codebase: min(32, C)."""
    return min(32, channels)


class FPN(nn.Module):
    """Top-down 4-level FPN with lateral 1x1 + 3x3 smoothing."""

    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                f"FPN expects exactly 4 input levels, got {len(in_channels)}"
            )
        self.in_channels = list(in_channels)
        self.out_channels = int(out_channels)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_gn_groups(out_channels), out_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """``feats[0]`` = stride 4 ... ``feats[3]`` = stride 32."""
        if len(feats) != len(self.lateral_convs):
            raise ValueError(
                f"FPN got {len(feats)} feats, expected {len(self.lateral_convs)}"
            )
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        # Top-down: P5 (highest index) flows down to P2.
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], scale_factor=2.0, mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        outs = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        return outs
