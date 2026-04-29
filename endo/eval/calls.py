"""Per-call (TP/FP/FN) extraction + JSONL emission (Component 7 §6.5).

A *call* is a 3D connected component (26-connectivity) of the per-volume
detection map built from the raw fused WBF boxes. A *GT lesion* is a 3D
connected component of the cached lesion mask. Matching is one-to-one
greedy with the centroid-in-mask rule:

    A call matches a GT lesion if the call's centroid voxel is inside the
    GT lesion mask. For each GT lesion the highest-scoring matching call
    is TP; other matches are FP. GT lesions with no matching call are FN.

The detection map is rendered from raw fused boxes (no size-threshold
filter); ``passes_threshold`` records whether the call would survive the
fold's tuned thresholds.

Output paths:
    CV     — runs/<exp>/eval/per_call_<run_id>.jsonl
    Holdout — runs/<exp>/holdout/run_<ts>_<uuid>/per_call_<run_id>.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import label as cc_label

from endo.eval.froc import _DEFAULT_VOLUME_SHAPE, _build_detection_map

# 26-connectivity structuring element for 3D CC (matches lesion-bank).
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.int32)

# Resampled spacing (PRD §1.3): (0.82 mm in-plane, 1.5 mm slice thickness).
DEFAULT_INPLANE_MM = 0.82
DEFAULT_SLICE_MM = 1.5


def _voxel_volume_mm3(inplane_mm: float = DEFAULT_INPLANE_MM, slice_mm: float = DEFAULT_SLICE_MM) -> float:
    return float(inplane_mm * slice_mm * inplane_mm)


@dataclass
class PredCall:
    """A predicted 3D connected component."""

    component_id: int
    score: float
    voxel_count: int
    volume_mm3: float
    bbox_yz_x: tuple[int, int, int, int, int, int]  # (y0, y1, z0, z1, x0, x1)
    centroid_yz_x: tuple[float, float, float]
    max_dim_mm: float


@dataclass
class GTLesion:
    """A GT lesion 3D connected component."""

    component_id: int
    voxel_count: int
    volume_mm3: float
    bbox_yz_x: tuple[int, int, int, int, int, int]
    centroid_yz_x: tuple[float, float, float]
    mask_indices: tuple[np.ndarray, np.ndarray, np.ndarray] = field(repr=False)


def build_detection_map(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    volume_shape: tuple[int, int, int] = _DEFAULT_VOLUME_SHAPE,
) -> np.ndarray:
    """Public alias for the FROC detection-map renderer (raw boxes)."""
    return _build_detection_map(fused_boxes, fused_scores, volume_shape=volume_shape)


def extract_pred_calls(
    detection_map: np.ndarray,
    inplane_mm: float = DEFAULT_INPLANE_MM,
    slice_mm: float = DEFAULT_SLICE_MM,
) -> list[PredCall]:
    """Extract 3D 26-conn connected components from a detection map.

    The component score is ``max(voxel)`` over the component; volume is
    voxel count × voxel volume; ``max_dim_mm`` is the longer of the XZ
    bbox extents (in mm) — used to gate the size-dependent threshold.
    """
    binary = (detection_map > 0).astype(np.int32)
    if binary.sum() == 0:
        return []
    labeled, n_components = cc_label(binary, structure=_STRUCT_26)
    if n_components == 0:
        return []

    voxel_vol = inplane_mm * slice_mm * inplane_mm
    out: list[PredCall] = []
    for cid in range(1, int(n_components) + 1):
        ys, zs, xs = np.where(labeled == cid)
        if ys.size == 0:
            continue
        voxel_count = int(ys.size)
        score = float(detection_map[ys, zs, xs].max())
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        z0, z1 = int(zs.min()), int(zs.max() + 1)
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        # XZ extent in mm (Y is slice axis; size filter is on the XZ plane).
        dx_mm = float(x1 - x0) * inplane_mm
        dz_mm = float(z1 - z0) * inplane_mm
        max_dim_mm = max(dx_mm, dz_mm)
        out.append(
            PredCall(
                component_id=cid,
                score=score,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count) * voxel_vol,
                bbox_yz_x=(y0, y1, z0, z1, x0, x1),
                centroid_yz_x=(float(ys.mean()), float(zs.mean()), float(xs.mean())),
                max_dim_mm=max_dim_mm,
            )
        )
    return out


def extract_gt_lesions(
    gt_mask: np.ndarray,
    inplane_mm: float = DEFAULT_INPLANE_MM,
    slice_mm: float = DEFAULT_SLICE_MM,
) -> list[GTLesion]:
    """Extract 3D 26-conn connected components from a GT lesion mask
    (Y, Z, X) of integers/booleans."""
    binary = (np.asarray(gt_mask) > 0).astype(np.int32)
    if binary.sum() == 0:
        return []
    labeled, n_components = cc_label(binary, structure=_STRUCT_26)
    if n_components == 0:
        return []

    voxel_vol = inplane_mm * slice_mm * inplane_mm
    out: list[GTLesion] = []
    for cid in range(1, int(n_components) + 1):
        ys, zs, xs = np.where(labeled == cid)
        if ys.size == 0:
            continue
        voxel_count = int(ys.size)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        z0, z1 = int(zs.min()), int(zs.max() + 1)
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        out.append(
            GTLesion(
                component_id=cid,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count) * voxel_vol,
                bbox_yz_x=(y0, y1, z0, z1, x0, x1),
                centroid_yz_x=(float(ys.mean()), float(zs.mean()), float(xs.mean())),
                mask_indices=(ys, zs, xs),
            )
        )
    return out


def match_calls_to_gt(
    pred_calls: Sequence[PredCall],
    gt_lesions: Sequence[GTLesion],
    gt_mask: np.ndarray,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Greedy one-to-one matching of pred calls to GT lesions.

    Rule: a call matches a GT lesion iff its centroid voxel falls inside
    the GT lesion's mask. For each GT lesion, the highest-scoring matching
    call wins (TP); other matching calls become FP. GT lesions with no
    matching call are FN.

    Args:
        pred_calls: list of PredCall (any order).
        gt_lesions: list of GTLesion.
        gt_mask: original ``(Y,Z,X)`` mask used to extract ``gt_lesions``;
            we re-use it to look up which GT label a centroid lands in.

    Returns:
        ``(tp_match, fp_call_idxs, fn_lesion_ids)`` where ``tp_match`` maps
        gt_lesion.component_id -> index into ``pred_calls``.
    """
    if not pred_calls:
        return {}, [], [int(g.component_id) for g in gt_lesions]

    binary = (np.asarray(gt_mask) > 0).astype(np.int32)
    Y, Z, X = binary.shape
    labeled, _ = cc_label(binary, structure=_STRUCT_26)

    # For each call, find which (if any) GT label its centroid lies in.
    candidates_by_gt: dict[int, list[tuple[float, int]]] = {}
    matched_call_idxs: set[int] = set()
    for ci, call in enumerate(pred_calls):
        cy, cz, cx = call.centroid_yz_x
        iy = int(round(cy))
        iz = int(round(cz))
        ix = int(round(cx))
        if not (0 <= iy < Y and 0 <= iz < Z and 0 <= ix < X):
            continue
        gt_id = int(labeled[iy, iz, ix])
        if gt_id == 0:
            continue
        candidates_by_gt.setdefault(gt_id, []).append((float(call.score), ci))

    tp_match: dict[int, int] = {}
    for gt_id, cands in candidates_by_gt.items():
        # Highest score wins.
        cands.sort(key=lambda t: t[0], reverse=True)
        best_score, best_ci = cands[0]
        tp_match[gt_id] = best_ci
        matched_call_idxs.add(best_ci)

    fp_call_idxs: list[int] = [
        ci for ci in range(len(pred_calls)) if ci not in matched_call_idxs
    ]
    matched_gt_ids = set(tp_match.keys())
    fn_lesion_ids: list[int] = [
        int(g.component_id) for g in gt_lesions if int(g.component_id) not in matched_gt_ids
    ]
    return tp_match, fp_call_idxs, fn_lesion_ids


def passes_threshold_for_call(
    call: PredCall,
    large_thr: float,
    small_thr: float,
    box_size_split_mm: float,
) -> bool:
    """Apply the size-dependent threshold rule to a single call."""
    if call.max_dim_mm >= float(box_size_split_mm):
        return bool(call.score >= float(large_thr))
    return bool(call.score >= float(small_thr))


def build_call_records(
    *,
    run_id: str,
    entrypoint: str,
    fold: int | None,
    pid: str,
    pred_calls: Sequence[PredCall],
    gt_lesions: Sequence[GTLesion],
    tp_match: Mapping[int, int],
    fp_call_idxs: Sequence[int],
    fn_lesion_ids: Sequence[int],
    large_thr: float | None,
    small_thr: float | None,
    box_size_split_mm: float,
) -> list[dict]:
    """Materialize the JSONL records for one volume."""
    records: list[dict] = []

    def _maybe_passes(call: PredCall) -> bool | None:
        if large_thr is None or small_thr is None:
            return None
        return passes_threshold_for_call(
            call, float(large_thr), float(small_thr), float(box_size_split_mm)
        )

    # TP records: gt -> winning call.
    gt_by_id = {int(g.component_id): g for g in gt_lesions}
    for gt_id, ci in tp_match.items():
        call = pred_calls[ci]
        gt = gt_by_id.get(int(gt_id))
        records.append(
            {
                "run_id": run_id,
                "entrypoint": entrypoint,
                "fold": fold,
                "patient_id": pid,
                "call_id": f"{pid}_pred_{call.component_id}",
                "call_type": "tp",
                "score": float(call.score),
                "passes_threshold": _maybe_passes(call),
                "volume_mm3": float(call.volume_mm3),
                "voxel_count": int(call.voxel_count),
                "bbox_yz_x": list(call.bbox_yz_x),
                "centroid_yz_x": list(call.centroid_yz_x),
                "gt_lesion_id": f"{pid}_lesion_{int(gt_id)}",
            }
        )

    # FP records.
    for ci in fp_call_idxs:
        call = pred_calls[ci]
        records.append(
            {
                "run_id": run_id,
                "entrypoint": entrypoint,
                "fold": fold,
                "patient_id": pid,
                "call_id": f"{pid}_pred_{call.component_id}",
                "call_type": "fp",
                "score": float(call.score),
                "passes_threshold": _maybe_passes(call),
                "volume_mm3": float(call.volume_mm3),
                "voxel_count": int(call.voxel_count),
                "bbox_yz_x": list(call.bbox_yz_x),
                "centroid_yz_x": list(call.centroid_yz_x),
                "gt_lesion_id": None,
            }
        )

    # FN records.
    for gt_id in fn_lesion_ids:
        gt = gt_by_id.get(int(gt_id))
        if gt is None:
            continue
        records.append(
            {
                "run_id": run_id,
                "entrypoint": entrypoint,
                "fold": fold,
                "patient_id": pid,
                "call_id": f"{pid}_fn_{int(gt_id)}",
                "call_type": "fn",
                "score": None,
                "passes_threshold": None,
                "volume_mm3": float(gt.volume_mm3),
                "voxel_count": int(gt.voxel_count),
                "bbox_yz_x": list(gt.bbox_yz_x),
                "centroid_yz_x": list(gt.centroid_yz_x),
                "gt_lesion_id": f"{pid}_lesion_{int(gt_id)}",
            }
        )

    return records


def write_calls_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Append-then-truncate write of one record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
