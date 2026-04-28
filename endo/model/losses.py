"""Loss helpers for the lesion detector.

Component 6 §6 / PRD §6.9: total = loss_cls + loss_bbox + aux_seg_weight * loss_aux_seg.
``loss_cls`` and ``loss_bbox`` are produced by the RTMDet head. The aux seg
loss is BCE-with-logits + soft Dice on a stride-1 (B, 1, H, W) logit map
against a (B, H, W) uint8 mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_bce_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """BCE-with-logits + soft Dice on a single-channel logit map.

    Args:
        logits: ``(B, H, W)`` or ``(B, 1, H, W)`` raw logits.
        target: ``(B, H, W)`` uint8/float mask in [0, 1].
        smooth: Dice numerator/denominator smoothing constant.
    """
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target_f = target.to(dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, target_f, reduction="mean")
    probs = torch.sigmoid(logits)
    intersection = (probs * target_f).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + target_f.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return bce + dice.mean()


def compute_total_loss(
    det_losses: dict[str, Tensor],
    aux_seg_logits: Tensor,
    aux_seg_target: Tensor,
    aux_seg_weight: float = 0.3,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine detection + aux-seg losses into a single scalar to backprop.

    Returns ``(total, components)`` where ``components`` has detached entries
    suitable for logging: ``{'loss_cls', 'loss_bbox', 'loss_aux_seg', 'loss_total'}``.
    """
    loss_aux_seg = dice_bce_loss(aux_seg_logits, aux_seg_target)
    total = det_losses["loss_cls"] + det_losses["loss_bbox"] + aux_seg_weight * loss_aux_seg
    components = {
        "loss_cls": det_losses["loss_cls"].detach(),
        "loss_bbox": det_losses["loss_bbox"].detach(),
        "loss_aux_seg": loss_aux_seg.detach(),
        "loss_total": total.detach(),
    }
    return total, components
