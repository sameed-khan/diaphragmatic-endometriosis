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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_iou

INF = 100000000
EPS = 1.0e-7


@dataclass
class AssignResult:
    """Output of :class:`DynamicSoftLabelAssigner`.

    Attributes:
        num_gts: number of ground-truth boxes for the image.
        gt_inds: 1-indexed assignment per prior; 0 means background. Shape
            ``(num_priors,)`` of dtype ``long``.
        max_overlaps: IoU between each prior and its assigned GT (or
            ``-INF`` for unmatched valid priors that fell outside any GT
            and ``0`` for trivial-empty cases). Shape ``(num_priors,)``.
        labels: class label of the assigned GT, ``-1`` if unassigned.
            Shape ``(num_priors,)`` of dtype ``long``.
    """

    num_gts: int
    gt_inds: Tensor
    max_overlaps: Tensor
    labels: Tensor

    @property
    def assigned_gt_inds_per_pred(self) -> Tensor:
        """Alias matching the spec (1-indexed; 0 == bg)."""
        return self.gt_inds


class DynamicSoftLabelAssigner(torch.nn.Module):
    """Computes matching between predictions and ground truth with dynamic
    soft label assignment, as in RTMDet.

    Args:
        soft_center_radius: Radius of the soft center prior. Defaults to 3.0.
        topk: Select top-k IoUs for dynamic-k calculation. Defaults to 13.
        iou_weight: Scale factor of the IoU cost. Defaults to 3.0.
    """

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> AssignResult:
        """Assign GT boxes to priors.

        Args:
            pred_scores: ``(num_priors, num_classes)`` raw classification
                logits (NOT sigmoided).
            priors: ``(num_priors, 4)`` of ``(cx, cy, stride_w, stride_h)``
                — first two cols are point centers, last two are strides
                (used to normalize the center-prior distance).
            decoded_bboxes: ``(num_priors, 4)`` predicted boxes in
                ``(x1, y1, x2, y2)``.
            gt_bboxes: ``(num_gt, 4)`` GT boxes in ``(x1, y1, x2, y2)``.
            gt_labels: ``(num_gt,)`` GT class indices.
        """
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        num_classes = pred_scores.size(-1)
        device = decoded_bboxes.device

        assigned_gt_inds = decoded_bboxes.new_zeros((num_bboxes,), dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full(
                (num_bboxes,), -1, dtype=torch.long
            )
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

        # Center-in-gt mask (treat priors as horizontal points).
        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[None, :, :2]
        rb_ = gt_bboxes[None, :, 2:] - prior_center[:, None]
        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0  # (num_priors, num_gt)
        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full(
                (num_bboxes,), -1, dtype=torch.long
            )
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

        # Soft center prior: 10 ** (normalized_distance - radius).
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (
            (valid_prior[:, None, :2] - gt_center[None, :, :])
            .pow(2)
            .sum(-1)
            .sqrt()
            / strides[:, None]
        )
        soft_center_prior = torch.pow(
            torch.tensor(10.0, device=device, dtype=distance.dtype),
            distance - self.soft_center_radius,
        )

        pairwise_ious = box_iou(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), num_classes)
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        rep_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - rep_pred_scores.sigmoid()
        soft_cls_cost = (
            F.binary_cross_entropy_with_logits(
                rep_pred_scores, soft_label, reduction="none"
            )
            * scale_factor.abs().pow(2.0)
        )
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self._dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # Convert to AssignResult format.
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

    # The original mmdet API.
    def assign(
        self,
        pred_scores: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> AssignResult:
        return self.forward(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels)

    def _dynamic_k_matching(
        self,
        cost: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """SimOTA-style dynamic-k assignment.

        Args:
            cost: ``(num_valid, num_gt)`` total cost.
            pairwise_ious: ``(num_valid, num_gt)`` IoU matrix.
            num_gt: number of GT boxes.
            valid_mask: ``(num_priors,)`` bool mask of priors inside any GT;
                mutated in-place to reflect the final foreground priors.

        Returns:
            (matched_pred_ious, matched_gt_inds):
              - matched_pred_ious: IoU of each matched prior with its GT,
                shape ``(num_fg,)``.
              - matched_gt_inds: GT index per matched prior, shape
                ``(num_fg,)``.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=int(dynamic_ks[gt_idx].item()), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1

        # Resolve priors matched to multiple GTs by taking the lowest-cost GT.
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        fg_mask_inboxes = matching_matrix.sum(1) > 0
        # Mutate the caller's valid_mask so it lines up with FG priors.
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
