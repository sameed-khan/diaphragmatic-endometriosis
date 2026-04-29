"""Component 7 orchestrator (Component 7 §5.1, §5.2; audit 2026-04-29).

Two entry points:

- ``run_cv_evaluation`` — pooled 5-fold CV evaluation. Per fold, runs
  **fresh inference** from the chosen ckpt (``best`` or ``last``), then
  WBF + cross-fold thresholding + metrics + bootstrap + stratified
  breakdowns. Writes ``runs/<exp>/eval/eval_report.csv`` (append-only),
  ``eval_thresholds.json``, and per-call JSONL.

- ``run_holdout_inference`` — one-shot 5-model ensemble inference on the
  holdout patients. The **only** legitimate caller setting
  ``DataModule.allow_holdout=True``.

Audit 2026-04-29 (correctness) changes:
  * Final eval ignores deep_eval npz caches — those remain training-time
    only (monitoring + HNM).
  * Threshold tuning for fold f uses the union of OTHER folds' val raw
    preds (no leakage). cv_pooled metrics use each volume's own fold's
    thresholds; ``ensemble_threshold`` is the mean of the 5 per-fold
    thresholds (used for holdout).
  * AUROC/AP from raw fused scores; FROC/sens@FP from thresholded boxes.
    Stratified breakdowns follow the same split.
  * FROC uses real GT lesion masks (not the central-cuboid proxy).
  * Per-call JSONL emits TP/FP/FN with 3D volumes via 26-conn CC.
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from endo.config.eval import EvalConfig
from endo.config.experiment import ExperimentConfig
from endo.data.manifest import (
    fold_split,
    manifest_by_pid,
    read_manifest_jsonl,
)
from endo.eval.calls import (
    _DEFAULT_VOLUME_SHAPE,
    build_call_records,
    build_detection_map,
    extract_gt_lesions,
    extract_pred_calls,
    match_calls_to_gt,
    write_calls_jsonl,
)
from endo.eval.metrics import compute_volume_metrics
from endo.eval.report import EvalReportRow, append_eval_report, write_eval_thresholds_json
from endo.eval.stratified import stratify_metrics
from endo.eval.threshold_search import grid_search_threshold
from endo.eval.wbf import _box_max_dim_mm, weighted_box_fusion_3d
from endo.inference_pass import SliceScore

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Helpers


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _make_run_id(prefix: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    return f"{prefix}_{ts}_{_git_sha()}"


def _aggregate_volume(
    slice_scores: list[SliceScore],
    image_size: tuple[int, int],
    eval_cfg: EvalConfig,
    apply_size_filter: bool = False,
    large_thr: float | None = None,
    small_thr: float | None = None,
) -> dict:
    """Run WBF on a single volume's slice list. Optionally apply the size-
    dependent threshold filter (used post-grid-search)."""
    fused = weighted_box_fusion_3d(
        slice_scores,
        image_size=image_size,
        iou_thr=eval_cfg.wbf_iou_threshold,
        skip_box_thr=eval_cfg.wbf_skip_box_threshold,
        large_threshold=large_thr if apply_size_filter else None,
        small_threshold=small_thr if apply_size_filter else None,
        box_size_threshold_mm=eval_cfg.box_size_split_mm,
    )
    fb = fused["fused_boxes"]
    fs = fused["fused_scores"]
    score = float(fs.max()) if fs.size > 0 else 0.0
    return {"fused_boxes": fb, "fused_scores": fs, "score": score}


def _apply_size_filter(
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    large_thr: float,
    small_thr: float,
    cfg: EvalConfig,
    inplane_mm: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    if fused_boxes.size == 0:
        return fused_boxes, fused_scores
    boxes_xz = fused_boxes[:, :4]
    max_dim_mm = _box_max_dim_mm(boxes_xz, inplane_mm=inplane_mm)
    is_large = max_dim_mm >= float(cfg.box_size_split_mm)
    keep = np.where(
        is_large, fused_scores >= float(large_thr), fused_scores >= float(small_thr)
    )
    return fused_boxes[keep], fused_scores[keep]


def _ckpt_path_for_fold(run_dir: Path, fold: int, choice: str) -> Path:
    """Resolve ``best.ckpt`` / ``last.ckpt`` under the fold's ckpts dir."""
    return run_dir / f"fold{fold}" / "ckpts" / f"{choice}.ckpt"


def _load_detector_for_fold(
    experiment: ExperimentConfig,
    ckpt_path: Path,
    device: str,
):
    """Load LesionDetectorLM from ``ckpt_path`` and overlay EMA weights when
    present. Mirrors :func:`endo.gru.feature_cache._load_detector_with_ema`.
    """
    import torch

    from endo.lightning_module import LesionDetectorLM

    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    lm = LesionDetectorLM(experiment)
    lm.load_state_dict(raw["state_dict"], strict=False)
    ema_sd = raw.get("ema_state_dict")
    if ema_sd is not None:
        try:
            lm.model.load_state_dict(ema_sd, strict=True)
            log.info("Overlaid EMA weights from %s", ckpt_path)
        except Exception as e:  # pragma: no cover — fallback
            log.warning("EMA weights present but could not be loaded (%s)", e)
    else:
        log.warning("No ema_state_dict in %s; using live weights", ckpt_path)
    lm.to(device)
    lm.eval()
    return lm


def _run_fresh_inference_for_fold(
    experiment: ExperimentConfig,
    fold: int,
    ckpt_choice: str,
    cfg: EvalConfig,
) -> tuple[dict[str, list[SliceScore]], list[str], "object"]:
    """Run fresh per-fold inference from a chosen checkpoint.

    Returns ``(slice_scores, val_pids, datamodule)``. The datamodule is
    returned because callers need its in-RAM lesion masks for FROC + per-
    call JSONL.
    """
    import torch

    from endo.data.datamodule import LesionDataModule
    from endo.inference_pass import inference_pass

    run_dir = experiment.run_dir()
    ckpt_path = _ckpt_path_for_fold(run_dir, fold, ckpt_choice)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"fold {fold}: ckpt {ckpt_path} missing (eval_ckpt={ckpt_choice})"
        )

    cohort_path = experiment.paths.data_root / "cohort.json"
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    dm = LesionDataModule(
        cache_root=experiment.paths.cache_root,
        manifest_path=manifest_path,
        cohort_path=cohort_path,
        fold=fold,
        batch_size=cfg.inference_batch_size,
        num_workers=0,
        allow_holdout=False,
    )
    dm.setup()
    val_pids = list(dm._val_pids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = _load_detector_for_fold(experiment, ckpt_path, device)
    slice_scores = inference_pass(
        model=lm,
        datamodule=dm,
        patient_ids=val_pids,
        split="val",
        batch_size=cfg.inference_batch_size,
    )
    return slice_scores, val_pids, dm


def _gt_masks_from_dm(
    dm: "object",
    pids: list[str],
    volume_shape: tuple[int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Pull the cached lesion masks out of a (set-up) DataModule and reshape
    to ``(Y, Z, X)`` matching the detection-map canvas.

    Cache shape is ``(X, Y, Z) = (408, 174, 408)``; the eval canvas is
    ``(Y, Z, X) = (160, 384, 384)``. We extract the same crop the dataset
    uses (centered, no jitter for val).
    """
    if volume_shape is None:
        volume_shape = _DEFAULT_VOLUME_SHAPE
    cache = getattr(dm, "_cache", {})
    target_shape = getattr(dm, "target_input_shape", (384, 160, 384))  # (X, Y, Z)
    cache_shape = getattr(dm, "cache_shape", (408, 174, 408))
    tx, ty, tz = target_shape
    cx, cy, cz = cache_shape
    px = (cx - tx) // 2
    py = (cy - ty) // 2
    pz = (cz - tz) // 2

    out: dict[str, np.ndarray] = {}
    Yv, Zv, Xv = volume_shape
    for pid in pids:
        entry = cache.get(pid)
        if entry is None:
            continue
        lesion_full = entry.get("lesion_mask")
        if lesion_full is None:
            # Negative volume — empty mask.
            out[pid] = np.zeros(volume_shape, dtype=np.uint8)
            continue
        # Crop (X, Y, Z) cached → target (X, Y, Z), then transpose to
        # (Y, Z, X).
        cropped = lesion_full[px : px + tx, py : py + ty, pz : pz + tz]
        cropped = np.ascontiguousarray(cropped, dtype=np.uint8)
        yzx = np.transpose(cropped, (1, 2, 0))  # (Y, Z, X)
        # Pad / trim to volume_shape.
        if yzx.shape == (Yv, Zv, Xv):
            out[pid] = yzx
            continue
        canvas = np.zeros(volume_shape, dtype=np.uint8)
        cy0, cz0, cx0 = (
            min(Yv, yzx.shape[0]),
            min(Zv, yzx.shape[1]),
            min(Xv, yzx.shape[2]),
        )
        canvas[:cy0, :cz0, :cx0] = yzx[:cy0, :cz0, :cx0]
        out[pid] = canvas
    return out


# ----------------------------------------------------------------------------
# Cross-fold thresholding


def _tune_thresholds_cross_fold(
    raw_preds_by_fold: dict[int, dict[str, dict]],
    labels_by_fold: dict[int, dict[str, int]],
    cfg: EvalConfig,
) -> dict[int, dict[str, float]]:
    """For each fold f, grid-search thresholds on the union of the OTHER
    folds' raw preds. Returns ``{f: {'large': lt, 'small': st}}``.
    """
    folds = sorted(raw_preds_by_fold.keys())
    out: dict[int, dict[str, float]] = {}
    for f in folds:
        pooled_preds: dict[str, dict] = {}
        pooled_labels: dict[str, int] = {}
        for g in folds:
            if g == f:
                continue
            pooled_preds.update(raw_preds_by_fold[g])
            pooled_labels.update(labels_by_fold[g])
        if not pooled_preds:
            log.warning(
                "fold %d: no other-fold tuning data available; falling back to "
                "this fold's raw preds (degenerate single-fold case).",
                f,
            )
            pooled_preds = dict(raw_preds_by_fold[f])
            pooled_labels = dict(labels_by_fold[f])
        gs = grid_search_threshold(pooled_preds, pooled_labels, eval_cfg=cfg)
        out[f] = {"large": gs["best_large_thr"], "small": gs["best_small_thr"]}
        log.info(
            "fold %d cross-fold thresholds: large=%.3f small=%.3f "
            "(tuned on %d other-fold volumes)",
            f,
            out[f]["large"],
            out[f]["small"],
            len(pooled_preds),
        )
    return out


def _ensemble_threshold(per_fold_thresholds: dict[int, dict[str, float]]) -> dict[str, float]:
    if not per_fold_thresholds:
        return {"large": 0.05, "small": 0.30}
    larges = [v["large"] for v in per_fold_thresholds.values()]
    smalls = [v["small"] for v in per_fold_thresholds.values()]
    return {"large": float(np.mean(larges)), "small": float(np.mean(smalls))}


def _try_gru_rescore_fresh(
    experiment: ExperimentConfig,
    fold: int,
    ckpt_path: Path,
    val_pids: list[str],
    eval_dir: Path,
    slice_scores: dict[str, list[SliceScore]],
) -> dict[str, list[SliceScore]] | None:
    """Rebuild the GRU feature cache for ``val_pids`` from ``ckpt_path``,
    then rescore. Returns ``None`` if the GRU module / artifacts are
    unavailable.
    """
    try:
        from endo.gru.feature_cache import extract_features_for_pids
        from endo.gru.rescorer import rescore_slice_scores
    except Exception as e:
        log.warning("GRU module unavailable (%s); skipping rescoring.", e)
        return None

    gru_ckpt = experiment.run_dir() / f"fold{fold}" / "gru" / "ckpt.pt"
    if not gru_ckpt.exists():
        log.warning("GRU ckpt missing for fold %d at %s; skipping rescoring.", fold, gru_ckpt)
        return None

    feature_dir = eval_dir / "feature_cache" / f"fold{fold}"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extract_features_for_pids(
            experiment,
            fold,
            pids=list(val_pids),
            output_dir=feature_dir,
            ckpt_path=ckpt_path,
            device=device,
        )
    except Exception as e:
        log.warning("Failed to rebuild GRU feature cache for fold %d (%s).", fold, e)
        return None

    try:
        return rescore_slice_scores(
            slice_scores, ckpt_path=gru_ckpt, feature_dir=feature_dir
        )
    except Exception as e:
        log.warning("GRU rescoring raised %s; falling back to non-rescored.", e)
        return None


def _emit_metric_rows(
    metrics: dict,
    *,
    run_id: str,
    entrypoint: str,
    scope: str,
    fold: int | None,
    stratum_kind: str | None,
    stratum_value: str | None,
    rescored: bool,
    n_patients: int,
    n_lesions: int,
    code_version: str,
) -> list[EvalReportRow]:
    rows: list[EvalReportRow] = []
    for metric_name, payload in metrics.items():
        if metric_name == "n_patients":
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(
            EvalReportRow(
                run_id=run_id,
                entrypoint=entrypoint,
                metric=metric_name,
                scope=scope,
                fold=fold,
                stratum_kind=stratum_kind,
                stratum_value=stratum_value,
                rescored=rescored,
                value=float(payload.get("value", float("nan"))),
                ci_lower_95=float(payload.get("ci_lower", float("nan"))),
                ci_upper_95=float(payload.get("ci_upper", float("nan"))),
                n_patients=n_patients,
                n_lesions=n_lesions,
                code_version=code_version,
            )
        )
    return rows


# ----------------------------------------------------------------------------
# CV evaluation


def run_cv_evaluation(
    experiment: ExperimentConfig,
    use_gru: bool = False,
    eval_dir: Path | None = None,
    image_size: tuple[int, int] = (384, 384),
) -> dict:
    """Run the CV-pooled evaluation across all 5 folds.

    Audit 2026-04-29 §3: per fold, run **fresh** inference from the
    configured ckpt (``best`` or ``last``), tune thresholds on the OTHER
    folds' raw preds, compute AUROC/AP from raw scores and FROC/sens@FP
    from thresholded preds, and emit per-call JSONL with TP/FP/FN.

    Folds with no checkpoint are logged-and-skipped.
    """
    cfg = experiment.eval
    run_dir = experiment.run_dir()
    if eval_dir is None:
        eval_dir = run_dir / "eval"
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id("cv")
    code_version = _git_sha()
    csv_path = eval_dir / "eval_report.csv"
    thresholds_path = eval_dir / "eval_thresholds.json"
    calls_path = eval_dir / f"per_call_{run_id}.jsonl"

    # Manifest for labels + stratification.
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    manifest_rows = read_manifest_jsonl(manifest_path)
    manifest_lookup = manifest_by_pid(manifest_rows)
    label_lookup = {pid: int(r.get("label") == "positive") for pid, r in manifest_lookup.items()}

    # Per-fold containers. Each fold contributes:
    #   - raw_preds[pid]   (post-WBF, pre-size-filter)
    #   - slice_scores[pid] (for the GRU rescorer / per-call JSONL)
    #   - val_pids
    #   - gt_masks[pid]    (Y, Z, X)
    fold_raw_preds: dict[int, dict[str, dict]] = {}
    fold_labels: dict[int, dict[str, int]] = {}
    fold_pids: dict[int, list[str]] = {}
    fold_gt_masks: dict[int, dict[str, np.ndarray]] = {}

    for fold in range(5):
        ckpt_path = _ckpt_path_for_fold(run_dir, fold, cfg.eval_ckpt)
        if not ckpt_path.exists():
            log.warning("fold %d: missing %s; skipping.", fold, ckpt_path)
            continue
        log.info("fold %d: running fresh inference from %s", fold, ckpt_path)
        try:
            slice_scores, val_pids, dm = _run_fresh_inference_for_fold(
                experiment, fold, cfg.eval_ckpt, cfg
            )
        except Exception as e:
            log.warning("fold %d: fresh inference failed (%s); skipping.", fold, e)
            continue

        if use_gru:
            rescored = _try_gru_rescore_fresh(
                experiment, fold, ckpt_path, val_pids, eval_dir, slice_scores
            )
            if rescored is not None:
                slice_scores = rescored

        # Aggregate to per-volume raw fused preds.
        raw_preds: dict[str, dict] = {}
        for pid in val_pids:
            raw_preds[pid] = _aggregate_volume(
                slice_scores.get(pid, []), image_size, cfg, apply_size_filter=False
            )
        labels = {pid: label_lookup.get(pid, 0) for pid in val_pids}
        gt_masks = _gt_masks_from_dm(dm, val_pids)

        fold_raw_preds[fold] = raw_preds
        fold_labels[fold] = labels
        fold_pids[fold] = list(val_pids)
        fold_gt_masks[fold] = gt_masks

    if not fold_raw_preds:
        log.warning("No folds produced predictions; eval_report.csv not updated.")
        return {"run_id": run_id, "rows": [], "thresholds": {}}

    # Cross-fold threshold tuning (audit §3.2).
    per_fold_thresholds = _tune_thresholds_cross_fold(fold_raw_preds, fold_labels, cfg)
    ensemble_thresholds = _ensemble_threshold(per_fold_thresholds)

    rows_to_write: list[EvalReportRow] = []

    # Pooled containers (per-volume own-fold thresholds applied).
    pooled_raw_preds: dict[str, dict] = {}
    pooled_thr_preds: dict[str, dict] = {}
    pooled_labels: dict[str, int] = {}
    pooled_gt_masks: dict[str, np.ndarray] = {}
    all_call_records: list[dict] = []

    for fold, raw_preds in fold_raw_preds.items():
        labels = fold_labels[fold]
        thresholds = per_fold_thresholds[fold]
        gt_masks = fold_gt_masks[fold]

        # Apply this fold's thresholds.
        thr_preds: dict[str, dict] = {}
        for pid, p in raw_preds.items():
            fb, fs = _apply_size_filter(
                p["fused_boxes"],
                p["fused_scores"],
                thresholds["large"],
                thresholds["small"],
                cfg,
            )
            thr_preds[pid] = {
                "fused_boxes": fb,
                "fused_scores": fs,
                "score": float(fs.max()) if fs.size > 0 else 0.0,
            }

        # Per-fold metrics (raw → AUROC/AP, thresholded → FROC/sens).
        metrics = compute_volume_metrics(
            thr_preds,
            labels,
            eval_cfg=cfg,
            raw_predictions=raw_preds,
            gt_masks=gt_masks,
        )
        rows_to_write.extend(
            _emit_metric_rows(
                metrics,
                run_id=run_id,
                entrypoint="cv",
                scope="per_fold",
                fold=fold,
                stratum_kind=None,
                stratum_value=None,
                rescored=use_gru,
                n_patients=len(thr_preds),
                n_lesions=int(sum(labels.values())),
                code_version=code_version,
            )
        )

        # Per-call JSONL emission for this fold (raw fused detection map).
        if cfg.emit_call_jsonl:
            for pid in raw_preds.keys():
                p = raw_preds[pid]
                det_map = build_detection_map(
                    np.asarray(p["fused_boxes"], dtype=np.float32),
                    np.asarray(p["fused_scores"], dtype=np.float32),
                    volume_shape=_DEFAULT_VOLUME_SHAPE,
                )
                pred_calls = extract_pred_calls(det_map)
                gt_mask = gt_masks.get(pid)
                if gt_mask is None:
                    gt_mask = np.zeros(_DEFAULT_VOLUME_SHAPE, dtype=np.uint8)
                gt_lesions = extract_gt_lesions(gt_mask) if int(labels.get(pid, 0)) == 1 else []
                tp_match, fp_idxs, fn_ids = match_calls_to_gt(
                    pred_calls, gt_lesions, gt_mask
                )
                all_call_records.extend(
                    build_call_records(
                        run_id=run_id,
                        entrypoint="cv",
                        fold=fold,
                        pid=pid,
                        pred_calls=pred_calls,
                        gt_lesions=gt_lesions,
                        tp_match=tp_match,
                        fp_call_idxs=fp_idxs,
                        fn_lesion_ids=fn_ids,
                        large_thr=thresholds["large"],
                        small_thr=thresholds["small"],
                        box_size_split_mm=cfg.box_size_split_mm,
                    )
                )

        # Accumulate into pool.
        for pid, p in raw_preds.items():
            pooled_raw_preds[pid] = p
            pooled_thr_preds[pid] = thr_preds[pid]
            pooled_labels[pid] = labels[pid]
            if pid in gt_masks:
                pooled_gt_masks[pid] = gt_masks[pid]

    # cv_pooled metrics use per-volume own-fold thresholds (no second tuning).
    pooled_metrics = compute_volume_metrics(
        pooled_thr_preds,
        pooled_labels,
        eval_cfg=cfg,
        raw_predictions=pooled_raw_preds,
        gt_masks=pooled_gt_masks,
    )
    rows_to_write.extend(
        _emit_metric_rows(
            pooled_metrics,
            run_id=run_id,
            entrypoint="cv",
            scope="cv_pooled",
            fold=None,
            stratum_kind=None,
            stratum_value=None,
            rescored=use_gru,
            n_patients=len(pooled_thr_preds),
            n_lesions=int(sum(pooled_labels.values())),
            code_version=code_version,
        )
    )

    # Stratified breakdowns — same raw/thresholded split.
    strat_results = stratify_metrics(
        pooled_thr_preds,
        pooled_labels,
        manifest_lookup,
        eval_cfg=cfg,
        raw_predictions=pooled_raw_preds,
        gt_masks=pooled_gt_masks,
    )
    for sr in strat_results:
        rows_to_write.extend(
            _emit_metric_rows(
                sr["metrics"],
                run_id=run_id,
                entrypoint="cv",
                scope="cv_pooled",
                fold=None,
                stratum_kind=sr["stratum_kind"],
                stratum_value=sr["stratum_value"],
                rescored=use_gru,
                n_patients=int(sr["n_patients"]),
                n_lesions=0,
                code_version=code_version,
            )
        )

    append_eval_report(csv_path, rows_to_write)

    write_eval_thresholds_json(
        thresholds_path,
        {
            "run_id": run_id,
            "per_fold_thresholds": {str(f): v for f, v in per_fold_thresholds.items()},
            "ensemble_threshold": ensemble_thresholds,
            "tuning_policy": "cross_fold_leave_one_out",
            "froc_ci_note": (
                "sens@FP CIs are computed from the per-volume max-score bootstrap "
                "(approximation); the point estimate uses the lesion-level "
                "detection-map sweep with GT masks."
            ),
        },
    )

    if cfg.emit_call_jsonl and all_call_records:
        write_calls_jsonl(calls_path, all_call_records)

    return {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "thresholds_path": str(thresholds_path),
        "calls_path": str(calls_path) if cfg.emit_call_jsonl else None,
        "ensemble_threshold": ensemble_thresholds,
        "per_fold_thresholds": {str(f): v for f, v in per_fold_thresholds.items()},
        "n_rows": len(rows_to_write),
    }


# ----------------------------------------------------------------------------
# Holdout one-shot ensemble


def _try_gru_rescore_holdout(
    experiment: ExperimentConfig,
    fold: int,
    ckpt_path: Path,
    holdout_pids: list[str],
    holdout_dir: Path,
    slice_scores: dict[str, list[SliceScore]],
) -> dict[str, list[SliceScore]] | None:
    try:
        from endo.gru.feature_cache import extract_features_for_pids
        from endo.gru.rescorer import rescore_slice_scores
    except Exception as e:
        log.warning("GRU module unavailable (%s); skipping holdout rescoring.", e)
        return None
    gru_ckpt = experiment.run_dir() / f"fold{fold}" / "gru" / "ckpt.pt"
    if not gru_ckpt.exists():
        log.warning("Holdout: GRU ckpt missing for fold %d; skipping rescoring.", fold)
        return None
    feature_dir = holdout_dir / "feature_cache" / f"fold{fold}"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extract_features_for_pids(
            experiment,
            fold,
            pids=list(holdout_pids),
            output_dir=feature_dir,
            ckpt_path=ckpt_path,
            device=device,
        )
        return rescore_slice_scores(
            slice_scores, ckpt_path=gru_ckpt, feature_dir=feature_dir
        )
    except Exception as e:
        log.warning("Holdout: GRU rescoring failed for fold %d (%s).", fold, e)
        return None


def run_holdout_inference(
    experiment: ExperimentConfig,
    ckpts: list[int] | str = "all",
    use_gru: bool = False,
    image_size: tuple[int, int] = (384, 384),
) -> Path:
    """One-shot 5-model ensemble inference on the holdout patients.

    THIS IS THE ONLY LEGITIMATE CALLER OF ``DataModule(allow_holdout=True)``.
    Per PRD I.9.3 / §13 amendment A.5.

    Audit 2026-04-29 changes: load the ``eval_ckpt`` (best/last) per fold,
    apply the **ensemble threshold** (mean of CV per-fold thresholds) to
    the fused boxes, emit per-call JSONL, and pass real GT lesion masks
    to FROC.
    """
    if isinstance(ckpts, str) and ckpts == "all":
        ckpt_indices = [0, 1, 2, 3, 4]
    elif isinstance(ckpts, list):
        ckpt_indices = list(ckpts)
    else:
        raise ValueError(f"ckpts must be 'all' or a list of fold indices, got {ckpts!r}")

    run_dir = experiment.run_dir()
    cfg = experiment.eval

    # Output subdir.
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    holdout_dir = run_dir / "holdout" / f"run_{ts}_{short_uuid}"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    csv_path = holdout_dir / "eval_report.csv"
    run_id = _make_run_id("holdout")
    code_version = _git_sha()
    calls_path = holdout_dir / f"per_call_{run_id}.jsonl"

    invocation_payload = {
        "run_id": run_id,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "ckpts_used": [int(f) for f in (ckpts if isinstance(ckpts, list) else [])] or "all",
        "use_gru": bool(use_gru),
        "code_version": code_version,
        "experiment_name": experiment.name,
        "experiment_uuid": experiment.uuid,
        "eval_ckpt": cfg.eval_ckpt,
    }
    (holdout_dir / "invocation.json").write_text(json.dumps(invocation_payload, indent=2))

    # Load thresholds from CV eval.
    thresholds_path = run_dir / "eval" / "eval_thresholds.json"
    if thresholds_path.exists():
        thr = json.loads(thresholds_path.read_text())
        ensemble_thr = thr.get("ensemble_threshold", {"large": 0.05, "small": 0.30})
    else:
        log.warning("No eval_thresholds.json at %s; using config defaults.", thresholds_path)
        ensemble_thr = {"large": 0.05, "small": 0.30}

    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    manifest_rows = read_manifest_jsonl(manifest_path)
    manifest_lookup = manifest_by_pid(manifest_rows)
    label_lookup = {pid: int(r.get("label") == "positive") for pid, r in manifest_lookup.items()}
    _, _, holdout_pids = fold_split(manifest_rows, fold=0)
    holdout_pids = list(holdout_pids)

    # ── DataModule build with allow_holdout=True (the *only* place). ──
    from endo.data.datamodule import LesionDataModule

    cohort_path = experiment.paths.data_root / "cohort.json"
    dm = LesionDataModule(
        cache_root=experiment.paths.cache_root,
        manifest_path=manifest_path,
        cohort_path=cohort_path,
        fold=0,
        batch_size=cfg.inference_batch_size,
        num_workers=0,
        allow_holdout=True,  # CRITICAL: only here.
    )
    dm.setup()

    from endo.inference_pass import inference_pass

    per_pid_slice_lists: dict[str, list[list[SliceScore]]] = {pid: [] for pid in holdout_pids}
    for fold in ckpt_indices:
        ckpt_path = _ckpt_path_for_fold(run_dir, fold, cfg.eval_ckpt)
        if not ckpt_path.exists():
            log.warning("fold %d: missing %s; skipping.", fold, ckpt_path)
            continue
        log.info("loading fold %d ckpt: %s", fold, ckpt_path)
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            module = _load_detector_for_fold(experiment, ckpt_path, device)
        except Exception as e:
            log.warning("fold %d: failed to load checkpoint (%s); skipping.", fold, e)
            continue

        scores = inference_pass(
            model=module,
            datamodule=dm,
            patient_ids=holdout_pids,
            split="holdout",
            batch_size=cfg.inference_batch_size,
        )

        if use_gru:
            rescored = _try_gru_rescore_holdout(
                experiment, fold, ckpt_path, holdout_pids, holdout_dir, scores
            )
            if rescored is not None:
                scores = rescored

        for pid, lst in scores.items():
            per_pid_slice_lists[pid].append(lst)

    # Mean-fusion across ckpts → raw fused preds → thresholded preds.
    raw_preds: dict[str, dict] = {}
    thr_preds: dict[str, dict] = {}
    for pid, lists in per_pid_slice_lists.items():
        if not lists:
            empty = {
                "fused_boxes": np.zeros((0, 5), dtype=np.float32),
                "fused_scores": np.zeros((0,), dtype=np.float32),
                "score": 0.0,
            }
            raw_preds[pid] = empty
            thr_preds[pid] = empty
            continue
        flat: list[SliceScore] = [s for lst in lists for s in lst]
        # Raw (no size filter).
        fused_raw = weighted_box_fusion_3d(
            flat,
            image_size=image_size,
            iou_thr=cfg.wbf_iou_threshold,
            skip_box_thr=cfg.wbf_skip_box_threshold,
            large_threshold=None,
            small_threshold=None,
            box_size_threshold_mm=cfg.box_size_split_mm,
        )
        raw_preds[pid] = {
            "fused_boxes": fused_raw["fused_boxes"],
            "fused_scores": fused_raw["fused_scores"],
            "score": float(fused_raw["fused_scores"].max())
            if fused_raw["fused_scores"].size > 0
            else 0.0,
        }
        fb, fs = _apply_size_filter(
            fused_raw["fused_boxes"],
            fused_raw["fused_scores"],
            ensemble_thr["large"],
            ensemble_thr["small"],
            cfg,
        )
        thr_preds[pid] = {
            "fused_boxes": fb,
            "fused_scores": fs,
            "score": float(fs.max()) if fs.size > 0 else 0.0,
        }

    holdout_labels = {pid: label_lookup.get(pid, 0) for pid in holdout_pids}
    gt_masks = _gt_masks_from_dm(dm, holdout_pids)

    metrics = compute_volume_metrics(
        thr_preds,
        holdout_labels,
        eval_cfg=cfg,
        raw_predictions=raw_preds,
        gt_masks=gt_masks,
    )
    rows = _emit_metric_rows(
        metrics,
        run_id=run_id,
        entrypoint="holdout",
        scope="holdout",
        fold=None,
        stratum_kind=None,
        stratum_value=None,
        rescored=use_gru,
        n_patients=len(thr_preds),
        n_lesions=int(sum(holdout_labels.values())),
        code_version=code_version,
    )

    strat = stratify_metrics(
        thr_preds,
        holdout_labels,
        manifest_lookup,
        eval_cfg=cfg,
        raw_predictions=raw_preds,
        gt_masks=gt_masks,
    )
    for sr in strat:
        rows.extend(
            _emit_metric_rows(
                sr["metrics"],
                run_id=run_id,
                entrypoint="holdout",
                scope="holdout",
                fold=None,
                stratum_kind=sr["stratum_kind"],
                stratum_value=sr["stratum_value"],
                rescored=use_gru,
                n_patients=int(sr["n_patients"]),
                n_lesions=0,
                code_version=code_version,
            )
        )

    append_eval_report(csv_path, rows)

    if cfg.emit_call_jsonl:
        all_records: list[dict] = []
        for pid in holdout_pids:
            p = raw_preds[pid]
            det_map = build_detection_map(
                np.asarray(p["fused_boxes"], dtype=np.float32),
                np.asarray(p["fused_scores"], dtype=np.float32),
                volume_shape=_DEFAULT_VOLUME_SHAPE,
            )
            pred_calls = extract_pred_calls(det_map)
            gt_mask = gt_masks.get(pid, np.zeros(_DEFAULT_VOLUME_SHAPE, dtype=np.uint8))
            gt_lesions = (
                extract_gt_lesions(gt_mask) if int(holdout_labels.get(pid, 0)) == 1 else []
            )
            tp_match, fp_idxs, fn_ids = match_calls_to_gt(pred_calls, gt_lesions, gt_mask)
            all_records.extend(
                build_call_records(
                    run_id=run_id,
                    entrypoint="holdout",
                    fold=None,
                    pid=pid,
                    pred_calls=pred_calls,
                    gt_lesions=gt_lesions,
                    tp_match=tp_match,
                    fp_call_idxs=fp_idxs,
                    fn_lesion_ids=fn_ids,
                    large_thr=ensemble_thr["large"],
                    small_thr=ensemble_thr["small"],
                    box_size_split_mm=cfg.box_size_split_mm,
                )
            )
        if all_records:
            write_calls_jsonl(calls_path, all_records)

    return holdout_dir
