"""3D Weighted Box Fusion over per-slice 2D detections (Component 7 §6.1).

Per-slice ``SliceScore`` lists are folded into a single per-volume detection
set: WBF is run independently on each slice (since ``ensemble_boxes`` operates
in 2D), and the per-slice fused boxes are concatenated and tagged with their
``slice_y``. Box-size-dependent confidence thresholds (``large_threshold``,
``small_threshold``) are then applied; per-volume aggregate score is
``max(scores)`` over surviving boxes.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from endo.inference_pass import SliceScore

# Resampled in-plane spacing in mm (PRD §1.3, target spacing (0.82, 1.5, 0.82)).
DEFAULT_INPLANE_MM = 0.82


def _box_max_dim_mm(boxes_xz: np.ndarray, inplane_mm: float = DEFAULT_INPLANE_MM) -> np.ndarray:
    """Per-box max((x2-x1)*mm, (z2-z1)*mm)."""
    if boxes_xz.size == 0:
        return np.zeros((0,), dtype=np.float32)
    dx = (boxes_xz[:, 2] - boxes_xz[:, 0]) * inplane_mm
    dz = (boxes_xz[:, 3] - boxes_xz[:, 1]) * inplane_mm
    return np.maximum(dx, dz).astype(np.float32)


def weighted_box_fusion_3d(
    slice_scores: Iterable[SliceScore],
    image_size: tuple[int, int],
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.001,
    large_threshold: float | None = None,
    small_threshold: float | None = None,
    box_size_threshold_mm: float = 10.0,
    inplane_mm: float = DEFAULT_INPLANE_MM,
) -> dict:
    """Fuse per-slice 2D boxes into per-volume detections.

    Args:
        slice_scores: iterable of :class:`SliceScore` from one volume.
        image_size: ``(H, W)`` of the slice (H = z-axis, W = x-axis).
        iou_thr: WBF IoU threshold.
        skip_box_thr: WBF discard threshold pre-fusion.
        large_threshold: confidence floor for boxes with max_dim_mm
            ``≥ box_size_threshold_mm``. ``None`` skips the size filter.
        small_threshold: confidence floor for boxes with max_dim_mm
            ``< box_size_threshold_mm``. ``None`` skips the size filter.
        box_size_threshold_mm: split between "large" and "small" boxes.
        inplane_mm: physical spacing per pixel; ``0.82`` post-Phase-1.

    Returns:
        ``{'fused_boxes': (M, 5)=(x1, z1, x2, z2, slice_y),
            'fused_scores': (M,) float32}``.
    """
    H, W = image_size
    H = float(H)
    W = float(W)

    out_boxes: list[np.ndarray] = []
    out_scores: list[np.ndarray] = []
    out_slice_ys: list[np.ndarray] = []

    for s in slice_scores:
        if s.boxes is None or s.boxes.size == 0 or s.scores.size == 0:
            continue
        boxes = np.asarray(s.boxes, dtype=np.float32)
        scores = np.asarray(s.scores, dtype=np.float32)
        # Normalize to [0, 1]; clip to handle minor float drift.
        norm = np.empty_like(boxes)
        norm[:, 0] = np.clip(boxes[:, 0] / W, 0.0, 1.0)
        norm[:, 1] = np.clip(boxes[:, 1] / H, 0.0, 1.0)
        norm[:, 2] = np.clip(boxes[:, 2] / W, 0.0, 1.0)
        norm[:, 3] = np.clip(boxes[:, 3] / H, 0.0, 1.0)
        labels = np.zeros((boxes.shape[0],), dtype=np.int32)

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            [norm.tolist()],
            [scores.tolist()],
            [labels.tolist()],
            weights=None,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        if len(fused_boxes) == 0:
            continue
        fused_boxes = np.asarray(fused_boxes, dtype=np.float32)
        fused_scores = np.asarray(fused_scores, dtype=np.float32)
        # Denormalize back to pixel coords.
        denorm = np.empty_like(fused_boxes)
        denorm[:, 0] = fused_boxes[:, 0] * W
        denorm[:, 1] = fused_boxes[:, 1] * H
        denorm[:, 2] = fused_boxes[:, 2] * W
        denorm[:, 3] = fused_boxes[:, 3] * H

        out_boxes.append(denorm)
        out_scores.append(fused_scores)
        out_slice_ys.append(np.full((denorm.shape[0],), int(s.slice_y), dtype=np.int32))

    if not out_boxes:
        return {
            "fused_boxes": np.zeros((0, 5), dtype=np.float32),
            "fused_scores": np.zeros((0,), dtype=np.float32),
        }

    boxes_xz = np.concatenate(out_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(out_scores, axis=0).astype(np.float32)
    slice_ys = np.concatenate(out_slice_ys, axis=0).astype(np.int32)

    # Size-dependent threshold filter.
    if large_threshold is not None and small_threshold is not None:
        max_dim_mm = _box_max_dim_mm(boxes_xz, inplane_mm=inplane_mm)
        is_large = max_dim_mm >= float(box_size_threshold_mm)
        keep = np.where(
            is_large,
            scores >= float(large_threshold),
            scores >= float(small_threshold),
        )
        boxes_xz = boxes_xz[keep]
        scores = scores[keep]
        slice_ys = slice_ys[keep]

    fused = np.concatenate(
        [boxes_xz, slice_ys.astype(np.float32).reshape(-1, 1)], axis=1
    ).astype(np.float32)
    return {"fused_boxes": fused, "fused_scores": scores.astype(np.float32)}


def volume_score_from_fused(fused: dict, top_k: int | None = None) -> float:
    """Per-volume aggregate. Default: ``max(scores)`` over surviving boxes; if
    ``top_k`` provided, ``mean`` of the top-K scores."""
    scores = fused.get("fused_scores")
    if scores is None or scores.size == 0:
        return 0.0
    if top_k is None:
        return float(np.max(scores))
    k = min(int(top_k), int(scores.size))
    if k <= 0:
        return float(np.max(scores))
    return float(np.mean(np.sort(scores)[-k:]))
