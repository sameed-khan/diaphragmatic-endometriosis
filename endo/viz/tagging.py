"""Per-slice TP/FP/FN event tagger for visualization (Component 8 §2.2).

Greedy IoU matching: predictions are sorted by score (descending) and matched
one-to-one to GT boxes if IoU >= threshold. The matching is symmetric — each
GT can be claimed by at most one prediction, and each prediction by at most
one GT. Unmatched preds become FPs; unmatched GTs become FNs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of axis-aligned boxes.

    Args:
        a: ``(N, 4)`` array of ``(x1, y1, x2, y2)``.
        b: ``(M, 4)`` array of ``(x1, y1, x2, y2)``.

    Returns:
        ``(N, M)`` array of IoUs in ``[0, 1]``.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    return iou.astype(np.float32)


def tag_slice_events(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> dict[str, list[Any]]:
    """Tag per-slice predictions and GTs as TP / FP / FN by greedy IoU match.

    Args:
        pred_boxes: ``(P, 4)`` array of predicted boxes.
        pred_scores: ``(P,)`` array of prediction scores (any range).
        gt_boxes: ``(G, 4)`` array of ground-truth boxes.
        iou_threshold: minimum IoU to count as a true positive.

    Returns:
        ``{'tp': [(box, score, gt_idx), ...],
           'fp': [(box, score), ...],
           'fn': [box_gt, ...]}``.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    P = pred_boxes.shape[0]
    G = gt_boxes.shape[0]

    tp: list[tuple[np.ndarray, float, int]] = []
    fp: list[tuple[np.ndarray, float]] = []
    fn: list[np.ndarray] = []

    if P == 0 and G == 0:
        return {"tp": tp, "fp": fp, "fn": fn}

    if P == 0:
        # Every GT is a missed detection.
        for g in range(G):
            fn.append(gt_boxes[g].copy())
        return {"tp": tp, "fp": fp, "fn": fn}

    if G == 0:
        # Every pred is a false positive.
        for p in range(P):
            fp.append((pred_boxes[p].copy(), float(pred_scores[p])))
        return {"tp": tp, "fp": fp, "fn": fn}

    iou = _box_iou(pred_boxes, gt_boxes)  # (P, G)

    # Sort preds by descending score for greedy assignment.
    order = np.argsort(-pred_scores, kind="stable")
    gt_taken = np.zeros(G, dtype=bool)
    pred_matched_to: dict[int, int] = {}

    for p in order:
        # Choose best IoU GT among those still unclaimed.
        candidates = np.where(~gt_taken)[0]
        if candidates.size == 0:
            continue
        ious_p = iou[p, candidates]
        best_local = int(np.argmax(ious_p))
        best_iou = float(ious_p[best_local])
        if best_iou >= iou_threshold:
            g = int(candidates[best_local])
            gt_taken[g] = True
            pred_matched_to[int(p)] = g

    for p in range(P):
        if p in pred_matched_to:
            tp.append((pred_boxes[p].copy(), float(pred_scores[p]), pred_matched_to[p]))
        else:
            fp.append((pred_boxes[p].copy(), float(pred_scores[p])))
    for g in range(G):
        if not gt_taken[g]:
            fn.append(gt_boxes[g].copy())

    return {"tp": tp, "fp": fp, "fn": fn}
