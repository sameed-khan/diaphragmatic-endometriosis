"""Vendored from mmdetection (rtmdet_head.py / dynamic_soft_label_assigner.py)
on 2026-04-28.

Source repo: https://github.com/open-mmlab/mmdetection (commit:
cfd5d3a985b0249de009b67d04f37263e11cdf3d)

Modifications:
  - dropped all mmcv/mmengine/mmdet imports; replaced with stdlib + torch +
    torchvision equivalents (see endo/model/__init__.py docstring for the map)
  - simplified loss to focal-cls (gamma=1.5, alpha=0.25) + CIoU bbox; dropped
    QualityFocalLoss / DistributionFocalLoss / GIoULoss variants
  - dropped ``with_objectness`` branch
  - replaced BaseDenseHead inheritance with nn.Module
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, box_iou, complete_box_iou_loss

from .assigner import DynamicSoftLabelAssigner


# -----------------------------------------------------------------------------
# Small helpers (replacing mmcv / mmengine / mmdet utilities)
# -----------------------------------------------------------------------------


def _is_norm(m: nn.Module) -> bool:
    return isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d))


def _bias_init_with_prob(prior_prob: float) -> float:
    """Inverse-sigmoid of prior_prob; used for the cls bias."""
    return float(-math.log((1 - prior_prob) / prior_prob))


def _reduce_mean(t: Tensor) -> Tensor:
    """Average a tensor across distributed ranks (no-op if not init'd)."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    t = t.clone()
    dist.all_reduce(t.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return t


def _distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Decode ``(l, t, r, b)`` distances into ``(x1, y1, x2, y2)`` boxes.

    Args:
        points: ``(..., 2)`` of ``(x, y)``.
        distance: ``(..., 4)`` of ``(left, top, right, bottom)``.
    """
    x1y1 = points - distance[..., :2]
    x2y2 = points + distance[..., 2:]
    return torch.cat([x1y1, x2y2], dim=-1)


def _bbox2distance(points: Tensor, bbox: Tensor) -> Tensor:
    """Encode ``(x1, y1, x2, y2)`` boxes as ``(l, t, r, b)`` distances."""
    lt = points - bbox[..., :2]
    rb = bbox[..., 2:] - points
    return torch.cat([lt, rb], dim=-1)


def _conv_gn_silu(in_ch: int, out_ch: int, kernel_size: int = 3,
                  padding: int = 1) -> nn.Sequential:
    """Conv-GroupNorm-SiLU block (replaces mmcv's ConvModule)."""
    num_groups = min(32, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding,
                  bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.SiLU(inplace=True),
    )


class _Scale(nn.Module):
    """Learnable scalar (replaces mmcv's Scale)."""

    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def _grid_points(featmap_size: tuple[int, int], stride: int,
                 device: torch.device, dtype: torch.dtype = torch.float32,
                 offset: float = 0.0, with_stride: bool = False) -> Tensor:
    """Cell-center points of a single feature level.

    Returns ``(num_pts, 2)`` of ``(x, y)`` if ``with_stride`` else
    ``(num_pts, 4)`` of ``(x, y, stride_w, stride_h)``.
    """
    feat_h, feat_w = featmap_size
    shift_x = (torch.arange(feat_w, device=device, dtype=dtype) + offset) * stride
    shift_y = (torch.arange(feat_h, device=device, dtype=dtype) + offset) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    if not with_stride:
        return torch.stack([xx, yy], dim=-1)
    sw = xx.new_full((xx.shape[0],), float(stride))
    sh = xx.new_full((xx.shape[0],), float(stride))
    return torch.stack([xx, yy, sw, sh], dim=-1)


def _sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 1.5,
    reduction: str = "sum",
) -> Tensor:
    """Standard sigmoid focal loss (RetinaNet)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


# -----------------------------------------------------------------------------
# RTMDet head (SepBN variant)
# -----------------------------------------------------------------------------


class RTMDetHead(nn.Module):
    """RTMDet-style detection head with separated BN per FPN level.

    Layout follows ``RTMDetSepBNHead``: each level has its own conv tower
    (cls and reg), the conv weights can optionally be shared across levels
    (``share_conv``), and per-level ``Scale`` modules scale the regressed
    distances before they are multiplied by the level stride.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int = 2,
        strides: tuple[int, ...] = (8, 16, 32),
        share_conv: bool = False,
        pred_kernel_size: int = 1,
        # loss / assigner hyper-parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 1.5,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = tuple(strides)
        self.share_conv = share_conv
        self.pred_kernel_size = pred_kernel_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight

        self.cls_out_channels = num_classes  # sigmoid head, no bg class
        self.assigner = DynamicSoftLabelAssigner()

        self._init_layers()
        self._init_weights()

    # ---- construction --------------------------------------------------------

    def _init_layers(self) -> None:
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        pad = self.pred_kernel_size // 2

        for _ in range(len(self.strides)):
            cls_tower = nn.ModuleList()
            reg_tower = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_tower.append(_conv_gn_silu(chn, self.feat_channels, 3, 1))
                reg_tower.append(_conv_gn_silu(chn, self.feat_channels, 3, 1))
            self.cls_convs.append(cls_tower)
            self.reg_convs.append(reg_tower)
            self.rtm_cls.append(nn.Conv2d(
                self.feat_channels, self.cls_out_channels,
                self.pred_kernel_size, padding=pad))
            self.rtm_reg.append(nn.Conv2d(
                self.feat_channels, 4, self.pred_kernel_size, padding=pad))

        if self.share_conv:
            # Tie the underlying nn.Conv2d weights across levels.
            for n in range(len(self.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i][0] = self.cls_convs[0][i][0]
                    self.reg_convs[n][i][0] = self.reg_convs[0][i][0]

        self.scales = nn.ModuleList([_Scale(1.0) for _ in self.strides])

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif _is_norm(m):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
        bias_cls = _bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            nn.init.normal_(rtm_cls.weight, std=0.01)
            nn.init.constant_(rtm_cls.bias, bias_cls)
            nn.init.normal_(rtm_reg.weight, std=0.01)
            nn.init.constant_(rtm_reg.bias, 0.0)

    # ---- forward -------------------------------------------------------------

    def forward(self, feats: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Per-level cls logits and (already-scaled) bbox distances.

        Returns:
            cls_scores: list of ``(B, num_classes, H, W)`` raw logits.
            bbox_preds: list of ``(B, 4, H, W)`` ``(l, t, r, b)`` distances
                already multiplied by the level stride.
        """
        assert len(feats) == len(self.strides)
        cls_scores: list[Tensor] = []
        bbox_preds: list[Tensor] = []
        for idx, (x, stride) in enumerate(zip(feats, self.strides)):
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)
            cls_score = self.rtm_cls[idx](cls_feat)
            reg_dist = self.scales[idx](self.rtm_reg[idx](reg_feat)) * float(stride)
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return cls_scores, bbox_preds

    # ---- prior helpers -------------------------------------------------------

    def _build_priors(
        self,
        featmap_sizes: list[tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
        with_stride: bool = False,
    ) -> list[Tensor]:
        return [
            _grid_points(fs, s, device=device, dtype=dtype, offset=0.0,
                         with_stride=with_stride)
            for fs, s in zip(featmap_sizes, self.strides)
        ]

    # ---- loss ----------------------------------------------------------------

    def loss(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        gt_boxes_per_image: list[Tensor],
        gt_labels_per_image: list[Tensor],
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute classification (focal) and bbox (CIoU) losses."""
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [tuple(c.shape[-2:]) for c in cls_scores]

        # Per-level priors as (num_pts, 4): (cx, cy, sw, sh).
        priors_with_stride = self._build_priors(
            featmap_sizes, device, dtype, with_stride=True
        )
        # Flat priors across levels.
        flat_priors = torch.cat(priors_with_stride, dim=0)  # (P, 4)
        flat_points = flat_priors[:, :2]                    # (P, 2)

        # Flatten predictions across levels (per image).
        # cls_scores[l]: (B, C, H, W) -> (B, P_l, C); bbox_preds[l]: (B, 4, H, W) -> (B, P_l, 4)
        flat_cls = torch.cat([
            c.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for c in cls_scores
        ], dim=1)  # (B, P, C)
        flat_dist = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ], dim=1)  # (B, P, 4)
        flat_decoded = _distance2bbox(
            flat_points.unsqueeze(0).expand(num_imgs, -1, -1), flat_dist
        )  # (B, P, 4)

        H, W = image_size
        total_cls_loss = flat_cls.new_zeros(())
        total_bbox_loss = flat_cls.new_zeros(())
        total_pos = flat_cls.new_zeros(())

        for i in range(num_imgs):
            gt_b = gt_boxes_per_image[i].to(device=device, dtype=dtype)
            gt_l = gt_labels_per_image[i].to(device=device, dtype=torch.long)
            scores_i = flat_cls[i].detach()
            decoded_i = flat_decoded[i].detach()

            assign = self.assigner(
                pred_scores=scores_i,
                priors=flat_priors,
                decoded_bboxes=decoded_i,
                gt_bboxes=gt_b,
                gt_labels=gt_l,
            )

            # Build classification targets: one-hot at assigned positives, 0 elsewhere.
            cls_targets = flat_cls.new_zeros(flat_cls.shape[1:])  # (P, C)
            pos_mask = assign.gt_inds > 0
            num_pos = int(pos_mask.sum().item())
            total_pos = total_pos + float(num_pos)
            if num_pos > 0:
                pos_labels = assign.labels[pos_mask].clamp_(min=0, max=self.num_classes - 1)
                cls_targets[pos_mask, pos_labels] = 1.0

            cls_loss_i = _sigmoid_focal_loss(
                flat_cls[i], cls_targets,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum",
            )
            total_cls_loss = total_cls_loss + cls_loss_i

            if num_pos > 0:
                pos_decoded = flat_decoded[i][pos_mask]
                # GT box per matched prior (gt_inds is 1-indexed).
                matched_gt = gt_b[assign.gt_inds[pos_mask] - 1]
                # Clamp predicted boxes inside the image to keep CIoU finite.
                pos_decoded = torch.stack([
                    pos_decoded[:, 0].clamp(0, W - 1),
                    pos_decoded[:, 1].clamp(0, H - 1),
                    pos_decoded[:, 2].clamp(0, W - 1),
                    pos_decoded[:, 3].clamp(0, H - 1),
                ], dim=-1)
                # CIoU contains arctan + sqrt that overflow under bf16 — promote
                # to fp32 for numeric stability. If the output is still
                # non-finite (rare degenerate boxes from random init), fall
                # back to a normalized L1 (each per-box term in [0, 4], same
                # scale as CIoU) so the gradient doesn't explode.
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    pos_f = pos_decoded.float()
                    gt_f = matched_gt.float()
                    bbox_loss_i = complete_box_iou_loss(pos_f, gt_f, reduction="sum")
                    if not torch.isfinite(bbox_loss_i):
                        norm = float(max(W, H))
                        per_coord = (pos_f - gt_f).abs() / norm
                        bbox_loss_i = per_coord.clamp_max(1.0).sum()
                total_bbox_loss = total_bbox_loss + bbox_loss_i.to(total_bbox_loss.dtype)

        # Normalize by world-averaged number of positives (clamped at 1).
        avg_factor = _reduce_mean(total_pos.detach()).clamp_(min=1.0)
        loss_cls = self.loss_cls_weight * total_cls_loss / avg_factor
        loss_bbox = self.loss_bbox_weight * total_bbox_loss / avg_factor
        return {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

    # ---- predict -------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        image_size: tuple[int, int],
        score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.5,
        max_per_image: int = 100,
    ) -> list[dict]:
        """Decode -> threshold -> per-class NMS, per image.

        Returns a list of length B; each element is a dict with
        ``boxes (N,4)``, ``scores (N,)``, ``labels (N,)``.
        """
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [tuple(c.shape[-2:]) for c in cls_scores]
        H, W = image_size

        # Per-level points (no stride channels needed for decode).
        priors_per_level = self._build_priors(
            featmap_sizes, device, dtype, with_stride=False
        )
        flat_points = torch.cat(priors_per_level, dim=0)  # (P, 2)

        flat_cls = torch.cat([
            c.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for c in cls_scores
        ], dim=1)  # (B, P, C)
        flat_dist = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ], dim=1)  # (B, P, 4)
        flat_decoded = _distance2bbox(
            flat_points.unsqueeze(0).expand(num_imgs, -1, -1), flat_dist
        )  # (B, P, 4)

        results: list[dict] = []
        for i in range(num_imgs):
            scores = flat_cls[i].sigmoid()  # (P, C)
            boxes = flat_decoded[i]         # (P, 4)
            # Clip to image.
            boxes = torch.stack([
                boxes[:, 0].clamp(0, W - 1),
                boxes[:, 1].clamp(0, H - 1),
                boxes[:, 2].clamp(0, W - 1),
                boxes[:, 3].clamp(0, H - 1),
            ], dim=-1)

            # Flatten (P, C) -> (P*C,) so each (prior, class) is a candidate.
            P, C = scores.shape
            flat_scores = scores.reshape(-1)
            labels = torch.arange(C, device=device).repeat(P)
            box_idx = torch.arange(P, device=device).repeat_interleave(C)
            keep_pre = flat_scores >= score_threshold
            if keep_pre.sum() == 0:
                results.append({
                    "boxes": boxes.new_zeros((0, 4)),
                    "scores": flat_scores.new_zeros((0,)),
                    "labels": labels.new_zeros((0,), dtype=torch.long),
                })
                continue
            sel_scores = flat_scores[keep_pre]
            sel_labels = labels[keep_pre]
            sel_boxes = boxes[box_idx[keep_pre]]

            keep = batched_nms(sel_boxes, sel_scores, sel_labels, nms_iou_threshold)
            keep = keep[:max_per_image]
            results.append({
                "boxes": sel_boxes[keep],
                "scores": sel_scores[keep],
                "labels": sel_labels[keep].long(),
            })
        return results


# Optional convenience: kept available because the spec mentions it.
def sigmoid_geometric_mean(x: Tensor, y: Tensor) -> Tensor:
    return torch.sqrt(torch.sigmoid(x) * torch.sigmoid(y))
