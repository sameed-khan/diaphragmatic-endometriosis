"""Composed detector: backbone + FPN + RTMDet head + aux seg head.

Component 6 §4. Backbone is a timm ConvNeXt-tiny in ``features_only`` mode
producing 4 stages at strides {4, 8, 16, 32}. timm's built-in 5-channel
conv1 surgery replicates the 3-channel pretrained weights and rescales by
``3/in_chans``; we verify this happened (stride-4 stem weight ratio matches
``3/5`` of a fresh 3-channel reference) and fall back to an explicit override
if not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from torch import Tensor

from endo.config.model import ModelConfig

from .aux_seg_head import AuxSegHead
from .fpn import FPN
from .rtmdet_head import RTMDetHead


class LesionDetector(nn.Module):
    """Backbone -> FPN -> (RTMDet head, AuxSegHead)."""

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = model_cfg
        self._backbone = self._build_backbone(model_cfg)
        in_channels = list(self._backbone.feature_info.channels())
        self.fpn = FPN(in_channels=in_channels, out_channels=model_cfg.fpn_channels)
        self._head = RTMDetHead(
            num_classes=model_cfg.head_n_classes,
            in_channels=model_cfg.fpn_channels,
            feat_channels=model_cfg.head_feat_channels,
            stacked_convs=model_cfg.head_stacked_convs,
            strides=tuple(model_cfg.fpn_strides),
            share_conv=False,
        )
        self.aux_seg_head = AuxSegHead(
            in_channels=model_cfg.fpn_channels,
            mid_channels=model_cfg.aux_seg_channels,
        )

    # ------------------------------------------------------------------
    # Public attribute accessors (also needed for typed external use).
    # ------------------------------------------------------------------
    @property
    def head(self) -> RTMDetHead:
        return self._head

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    # ------------------------------------------------------------------
    # Construction helpers.
    # ------------------------------------------------------------------
    def _build_backbone(self, cfg: ModelConfig) -> nn.Module:
        backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            in_chans=cfg.in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Verify timm's 5-channel surgery; override if it appears to have been
        # skipped (e.g. random init).
        if cfg.in_channels != 3:
            self._maybe_fix_input_conv(backbone, cfg)
        return backbone

    @staticmethod
    def _find_input_conv(backbone: nn.Module) -> nn.Conv2d | None:
        for m in backbone.modules():
            if isinstance(m, nn.Conv2d) and m.weight.dim() == 4:
                return m
        return None

    def _maybe_fix_input_conv(self, backbone: nn.Module, cfg: ModelConfig) -> None:
        """If the 5-channel stem weights are not a sensible scaled replication
        of the 3-channel pretrained weights, redo the surgery explicitly.

        Heuristic: build a fresh 3-channel reference backbone, compare its
        stem-weight magnitude. The expected magnitude after timm's surgery is
        ``ref * 3 / in_chans``; tolerate up to 25% drift before overriding.
        """
        in_conv = self._find_input_conv(backbone)
        if in_conv is None or in_conv.weight.shape[1] != cfg.in_channels:
            return
        try:
            ref = timm.create_model(
                cfg.backbone_name,
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception:
            return
        ref_conv = self._find_input_conv(ref)
        if ref_conv is None:
            return

        with torch.no_grad():
            cur_mag = float(in_conv.weight.abs().mean())
            ref_mag = float(ref_conv.weight.abs().mean())
            expected = ref_mag * (3.0 / cfg.in_channels)
            if expected <= 0:
                return
            ratio = cur_mag / expected
            if 0.75 <= ratio <= 1.25:
                return  # timm did the right thing.

            # Override with the documented surgery: replicate 3ch weights twice
            # along the input-channel axis, slice to ``in_chans``, scale by
            # ``3 / in_chans``.
            w3 = ref_conv.weight.detach()  # (out, 3, k, k)
            repeats = (cfg.in_channels + 2) // 3
            wn = w3.repeat(1, repeats, 1, 1)[:, : cfg.in_channels].clone()
            wn.mul_(3.0 / cfg.in_channels)
            in_conv.weight.copy_(wn.to(dtype=in_conv.weight.dtype))
            if in_conv.bias is not None and ref_conv.bias is not None:
                in_conv.bias.copy_(ref_conv.bias.detach().to(dtype=in_conv.bias.dtype))

    # ------------------------------------------------------------------
    # Forward / predict.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """Run backbone -> FPN -> heads.

        Returns ``(cls_scores, bbox_preds, aux_seg_logits)``. The detection
        outputs are per-FPN-level lists; the aux seg logits are stride-1
        ``(B, 1, H, W)``.
        """
        feats = self._backbone(x)
        pyramid = self.fpn(feats)
        cls_scores, bbox_preds = self._head(pyramid)
        aux_seg_logits = self.aux_seg_head(pyramid[0])
        return cls_scores, bbox_preds, aux_seg_logits

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        image_size: tuple[int, int],
        score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.5,
        max_per_image: int = 100,
    ) -> list[dict]:
        feats = self._backbone(x)
        pyramid = self.fpn(feats)
        cls_scores, bbox_preds = self._head(pyramid)
        return self._head.predict(
            cls_scores,
            bbox_preds,
            image_size=image_size,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_per_image=max_per_image,
        )
