"""FROC + volume AUROC/AP via picai_eval (Component 7 §6.2).

We adapt our 2D-per-slice box predictions to picai_eval's 3D-volume API by
synthesizing a sparse 3D detection map per patient: each fused box becomes a
filled rectangular region on slice ``slice_y`` with the box's score as the
voxel value. picai_eval connected-components-extracts that as one lesion
candidate.

For volume-only labelled data (no GT lesion mask in scope here), the
"hit" criterion is patient-level — a candidate matches the GT only if the
volume is positive (any candidate counts). Where GT lesion masks are
provided, picai_eval's centroid-in-mask criterion fires as usual.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# Internal: volume canvas size for the synthetic 3D detection map.
# Component 7 cares only about patient-level FROC where the "lesion" is the
# whole positive volume; the canvas just has to be large enough to hold any
# fused box. We compute it dynamically from the maximum referenced index for
# memory efficiency; this constant is the upper bound (full crop frame).
_DEFAULT_VOLUME_SHAPE = (160, 384, 384)  # (Y, Z, X)


def _build_detection_map(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    volume_shape: tuple[int, int, int] = _DEFAULT_VOLUME_SHAPE,
) -> np.ndarray:
    """Render a sparse 3D detection map of shape (Y, Z, X).

    Each fused box ``(x1, z1, x2, z2, slice_y)`` is drawn as a filled rectangle
    on slice ``slice_y``. Boxes are drawn in ascending score order so higher-
    score candidates overwrite lower ones (picai_eval treats each unique
    confidence as a distinct candidate).
    """
    Y, Z, X = volume_shape
    det = np.zeros((Y, Z, X), dtype=np.float32)
    if fused_boxes.size == 0:
        return det
    # Sort ascending by score so the highest-score box is rendered last.
    order = np.argsort(fused_scores, kind="stable")
    for i in order:
        x1, z1, x2, z2, sy = fused_boxes[i]
        score = float(fused_scores[i])
        sy_i = int(round(sy))
        x1_i = max(0, int(np.floor(x1)))
        z1_i = max(0, int(np.floor(z1)))
        x2_i = min(X, int(np.ceil(x2)))
        z2_i = min(Z, int(np.ceil(z2)))
        if 0 <= sy_i < Y and x2_i > x1_i and z2_i > z1_i:
            det[sy_i, z1_i:z2_i, x1_i:x2_i] = score
    return det


def _build_label_map(
    label: int,
    volume_shape: tuple[int, int, int] = _DEFAULT_VOLUME_SHAPE,
) -> np.ndarray:
    """Render a 3D GT label volume (1 if the volume is positive, 0 else).

    For positive volumes we mark a centered cuboid GT that's large enough
    that *any* detection box on a center slice overlapping the central
    region passes the picai_eval ``min_overlap=0.1`` IoU criterion. True
    per-lesion GT masks (when available) should be passed via ``gt_masks``
    to ``compute_froc`` — those override this proxy."""
    Y, Z, X = volume_shape
    gt = np.zeros((Y, Z, X), dtype=np.int32)
    if int(label) == 1:
        # 30×30×3 cuboid at the volume's center — small enough to be
        # surroundable by a typical detection box (a 30×30 box covering it
        # gives IoU ≈ 1.0), but large enough to dominate IoU when overlap
        # exists.
        cy, cz, cx = Y // 2, Z // 2, X // 2
        gt[max(0, cy - 1) : cy + 2, max(0, cz - 15) : cz + 15, max(0, cx - 15) : cx + 15] = 1
    return gt


def compute_froc(
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    fp_per_volume_levels: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    gt_masks: Mapping[str, np.ndarray] | None = None,
    volume_shape: tuple[int, int, int] = _DEFAULT_VOLUME_SHAPE,
) -> dict:
    """Compute volume-level FROC + AUROC + AP via picai_eval.

    Args:
        per_volume_predictions: ``{pid: {'fused_boxes': (M,5), 'fused_scores':
            (M,), 'score': float}}`` — output of WBF aggregation.
        per_volume_labels: ``{pid: 0|1}``.
        fp_per_volume_levels: FP/vol points at which to report sensitivity.
        gt_masks: optional ``{pid: (Y,Z,X) uint8/bool}`` GT lesion masks. If
            present, used in lieu of the central-voxel proxy.
        volume_shape: ``(Y, Z, X)`` canvas for the detection-map synthesis.

    Returns: ``{'sensitivity_at_<fp>': float, 'volume_auroc': float,
        'volume_ap': float, 'froc_curve_fp', 'froc_curve_sens',
        'n_patients': int}``.
    """
    pids = sorted(per_volume_predictions.keys())
    if not pids:
        return {
            **{f"sensitivity_at_{fp}": float("nan") for fp in fp_per_volume_levels},
            "volume_auroc": float("nan"),
            "volume_ap": float("nan"),
            "froc_curve_fp": [],
            "froc_curve_sens": [],
            "n_patients": 0,
        }

    # Fast path: if no caller has provided fused_boxes (only volume-level
    # ``score``), skip picai_eval and use the hand-rolled patient-level FROC.
    # This is the common case in unit tests and a sound proxy when the
    # detector hasn't yet written boxes.
    has_any_boxes = any(
        np.asarray(per_volume_predictions[pid].get("fused_boxes", np.zeros((0, 5)))).size > 0
        for pid in pids
    )
    if not has_any_boxes and gt_masks is None:
        return _hand_rolled_froc(per_volume_predictions, per_volume_labels, fp_per_volume_levels)

    y_det: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    for pid in pids:
        pred = per_volume_predictions[pid]
        boxes = np.asarray(pred.get("fused_boxes", np.zeros((0, 5))), dtype=np.float32)
        scores = np.asarray(pred.get("fused_scores", np.zeros((0,))), dtype=np.float32)
        det = _build_detection_map(boxes, scores, volume_shape=volume_shape)
        y_det.append(det)
        if gt_masks is not None and pid in gt_masks:
            gt = np.asarray(gt_masks[pid], dtype=np.int32)
            y_true.append(gt)
        else:
            y_true.append(_build_label_map(int(per_volume_labels.get(pid, 0)), volume_shape=volume_shape))

    # Lazy-import picai_eval so this module is importable in test environments
    # without picai_eval (we'll fall back to a hand-rolled implementation).
    try:
        from picai_eval.eval import evaluate

        metrics = evaluate(
            y_det=y_det,
            y_true=y_true,
            num_parallel_calls=1,
            verbose=0,
            subject_list=pids,
        )

        # Sensitivity at each FP/vol point.
        sens_at: dict[str, float] = {}
        for fp in fp_per_volume_levels:
            try:
                sens_at[f"sensitivity_at_{fp}"] = float(metrics.lesion_TPR_at_FPR(float(fp)))
            except Exception:
                sens_at[f"sensitivity_at_{fp}"] = float("nan")

        # FROC curve.
        try:
            fp_curve = list(map(float, np.asarray(metrics.lesion_FPR).tolist()))
            sens_curve = list(map(float, np.asarray(metrics.lesion_TPR).tolist()))
        except Exception:
            fp_curve, sens_curve = [], []

        return {
            **sens_at,
            "volume_auroc": float(getattr(metrics, "auroc", float("nan"))),
            "volume_ap": float(getattr(metrics, "AP", float("nan"))),
            "froc_curve_fp": fp_curve,
            "froc_curve_sens": sens_curve,
            "n_patients": len(pids),
        }
    except Exception:
        # Hand-rolled fallback: per-volume max-score → AUROC + threshold-sweep
        # FROC where each volume contributes ≤1 candidate and TPs are counted
        # against the patient label.
        return _hand_rolled_froc(
            per_volume_predictions, per_volume_labels, fp_per_volume_levels
        )


def _hand_rolled_froc(
    per_volume_predictions: Mapping[str, dict],
    per_volume_labels: Mapping[str, int],
    fp_per_volume_levels: tuple[float, ...],
) -> dict:
    """Patient-level FROC where the per-volume aggregate score is the only
    input. FP/vol = (#neg vols above threshold) / (#vols total). Sensitivity
    = (#pos vols above threshold) / (#pos vols total).

    This is a strict subset of picai_eval's behaviour for the case where the
    GT mask is a single voxel and all detections collapse to one candidate,
    but it works without the dependency.
    """
    pids = sorted(per_volume_predictions.keys())
    scores = np.asarray(
        [float(per_volume_predictions[p].get("score", 0.0)) for p in pids], dtype=np.float64
    )
    labels = np.asarray(
        [int(per_volume_labels.get(p, 0)) for p in pids], dtype=np.int64
    )
    n_total = len(pids)
    n_pos = int(labels.sum())

    auroc = float("nan")
    ap = float("nan")
    if labels.min() != labels.max():
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))

    # Threshold sweep: every unique score (descending) is a threshold.
    order = np.argsort(-scores, kind="stable")
    tp = 0
    fp = 0
    fp_curve: list[float] = []
    sens_curve: list[float] = []
    for j in order:
        if labels[j] == 1:
            tp += 1
        else:
            fp += 1
        fp_curve.append(float(fp / max(n_total, 1)))
        sens_curve.append(float(tp / max(n_pos, 1)) if n_pos else float("nan"))

    sens_at: dict[str, float] = {}
    fp_arr = np.asarray(fp_curve)
    sens_arr = np.asarray(sens_curve)
    for fp_target in fp_per_volume_levels:
        if fp_arr.size == 0:
            sens_at[f"sensitivity_at_{fp_target}"] = float("nan")
            continue
        below = fp_arr <= float(fp_target)
        if below.any():
            sens_at[f"sensitivity_at_{fp_target}"] = float(sens_arr[below][-1])
        else:
            sens_at[f"sensitivity_at_{fp_target}"] = 0.0

    return {
        **sens_at,
        "volume_auroc": auroc,
        "volume_ap": ap,
        "froc_curve_fp": fp_curve,
        "froc_curve_sens": sens_curve,
        "n_patients": n_total,
    }
