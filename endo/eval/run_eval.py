"""Component 7 orchestrator (Component 7 §5.1, §5.2).

Two entry points:

- ``run_cv_evaluation`` — pooled 5-fold CV evaluation. Reads each fold's most
  recent ``runs/<exp>/fold{f}/runtime/deep_eval/epoch{n}_val.npz`` cache and
  runs WBF + threshold search + metrics + bootstrap + stratified breakdowns.
  Writes ``runs/<exp>/eval/eval_report.csv`` (append-only) and
  ``eval_thresholds.json``.

- ``run_holdout_inference`` — one-shot 5-model ensemble inference on the 122
  holdout patients. The **only** legitimate caller setting
  ``DataModule.allow_holdout=True``.
"""

from __future__ import annotations

import datetime
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


def _latest_deep_eval_npz(fold_dir: Path) -> Path | None:
    de = fold_dir / "runtime" / "deep_eval"
    if not de.exists():
        return None
    files = sorted(de.glob("epoch*_val.npz"))
    return files[-1] if files else None


def _load_deep_eval_npz(npz_path: Path) -> dict[str, list[SliceScore]]:
    """Reconstruct ``{pid: [SliceScore, ...]}`` from the §5.3.4 CSR-style npz."""
    data = np.load(npz_path, allow_pickle=True)
    pids = data["patient_ids"]
    slice_ys = data["slice_ys"]
    boxes_flat = data["boxes_flat"]
    scores_flat = data["scores_flat"]
    box_offsets = data["box_offsets"]
    aux_seg_max = data["aux_seg_max"]

    out: dict[str, list[SliceScore]] = {}
    n = len(pids)
    for i in range(n):
        pid = str(pids[i])
        s = int(box_offsets[i])
        e = int(box_offsets[i + 1])
        out.setdefault(pid, []).append(
            SliceScore(
                patient_id=pid,
                slice_y=int(slice_ys[i]),
                boxes=boxes_flat[s:e].astype(np.float32, copy=False),
                scores=scores_flat[s:e].astype(np.float32, copy=False),
                aux_seg_max=float(aux_seg_max[i]),
            )
        )
    for pid in list(out.keys()):
        out[pid].sort(key=lambda s: s.slice_y)
    return out


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


def _try_gru_rescore(
    experiment: ExperimentConfig,
    fold: int,
    slice_scores: dict[str, list[SliceScore]],
) -> dict[str, list[SliceScore]] | None:
    """Best-effort GRU rescoring. Returns ``None`` (and logs) if the GRU
    module or the fold's checkpoint/feature cache aren't available yet."""
    try:
        from endo.gru.rescorer import rescore_slice_scores  # type: ignore
    except Exception as e:  # pragma: no cover - GRU module may not exist yet
        log.warning("GRU rescorer unavailable (%s); falling back to non-rescored.", e)
        return None

    fold_dir = experiment.run_dir() / f"fold{fold}"
    ckpt_path = fold_dir / "gru" / "ckpt.pt"
    feature_dir = fold_dir / "gru" / "feature_cache"
    if not ckpt_path.exists() or not feature_dir.exists():
        log.warning(
            "GRU artifacts missing for fold %d (ckpt=%s, features=%s); skipping rescoring.",
            fold,
            ckpt_path.exists(),
            feature_dir.exists(),
        )
        return None
    try:
        return rescore_slice_scores(slice_scores, ckpt_path=ckpt_path, feature_dir=feature_dir)
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

    Reads each fold's latest ``deep_eval/epoch{n}_val.npz`` cache. Folds with
    no cache are logged-and-skipped. Writes ``eval_report.csv`` (append-only)
    and ``eval_thresholds.json`` to ``eval_dir`` (default
    ``runs/<exp>/eval/``).
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

    # Manifest for labels + stratification.
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    manifest_rows = read_manifest_jsonl(manifest_path)
    manifest_lookup = manifest_by_pid(manifest_rows)
    label_lookup = {pid: int(r.get("label") == "positive") for pid, r in manifest_lookup.items()}

    rows_to_write: list[EvalReportRow] = []
    per_fold_thresholds: dict[str, dict[str, float]] = {}

    # Pooled containers (raw fused = pre-filter outputs from WBF, used by the
    # CV-pooled grid search).
    pooled_raw_preds: dict[str, dict] = {}
    pooled_labels: dict[str, int] = {}

    for fold in range(5):
        fold_dir = run_dir / f"fold{fold}"
        npz = _latest_deep_eval_npz(fold_dir)
        if npz is None:
            log.warning("fold %d: no deep_eval npz at %s; skipping.", fold, fold_dir)
            continue
        log.info("fold %d: loading deep_eval cache from %s", fold, npz)
        slice_scores = _load_deep_eval_npz(npz)

        if use_gru:
            rescored_scores = _try_gru_rescore(experiment, fold, slice_scores)
            if rescored_scores is not None:
                slice_scores = rescored_scores
            else:
                log.info("fold %d: GRU rescoring not applied.", fold)

        # 1. Per-volume WBF (no size filter yet — grid search wants raw).
        raw_preds: dict[str, dict] = {}
        for pid, slices in slice_scores.items():
            raw_preds[pid] = _aggregate_volume(slices, image_size, cfg, apply_size_filter=False)
        # Per-fold label subset.
        fold_labels = {pid: label_lookup.get(pid, 0) for pid in raw_preds.keys()}

        # 2. Per-fold threshold grid search.
        gs = grid_search_threshold(raw_preds, fold_labels, eval_cfg=cfg)
        per_fold_thresholds[str(fold)] = {
            "large": gs["best_large_thr"],
            "small": gs["best_small_thr"],
        }
        log.info(
            "fold %d: best thresholds large=%.3f small=%.3f sens@2fp=%.3f",
            fold,
            gs["best_large_thr"],
            gs["best_small_thr"],
            gs.get("best_sens_at_2.0fp", float("nan")),
        )

        # 3. Apply per-fold thresholds and recompute final preds.
        final_preds = {}
        for pid, p in raw_preds.items():
            fb, fs = _apply_size_filter(
                p["fused_boxes"], p["fused_scores"], gs["best_large_thr"], gs["best_small_thr"], cfg
            )
            final_preds[pid] = {
                "fused_boxes": fb,
                "fused_scores": fs,
                "score": float(fs.max()) if fs.size > 0 else 0.0,
            }

        # 4. Per-fold metrics.
        metrics = compute_volume_metrics(final_preds, fold_labels, eval_cfg=cfg)
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
                n_patients=len(final_preds),
                n_lesions=int(sum(fold_labels.values())),
                code_version=code_version,
            )
        )

        # Pool with raw preds (CV-pooled grid search will re-derive thresholds).
        for pid, p in raw_preds.items():
            pooled_raw_preds[pid] = p
            pooled_labels[pid] = fold_labels[pid]

    if not pooled_raw_preds:
        log.warning("No folds produced predictions; eval_report.csv not updated.")
        return {"run_id": run_id, "rows": [], "thresholds": {}}

    # CV-pooled grid search.
    pooled_gs = grid_search_threshold(pooled_raw_preds, pooled_labels, eval_cfg=cfg)
    ensemble_thresholds = {
        "large": pooled_gs["best_large_thr"],
        "small": pooled_gs["best_small_thr"],
    }
    log.info(
        "cv_pooled: best thresholds large=%.3f small=%.3f",
        ensemble_thresholds["large"],
        ensemble_thresholds["small"],
    )

    # Apply CV-pooled thresholds.
    pooled_final: dict[str, dict] = {}
    for pid, p in pooled_raw_preds.items():
        fb, fs = _apply_size_filter(
            p["fused_boxes"], p["fused_scores"], ensemble_thresholds["large"], ensemble_thresholds["small"], cfg
        )
        pooled_final[pid] = {
            "fused_boxes": fb,
            "fused_scores": fs,
            "score": float(fs.max()) if fs.size > 0 else 0.0,
        }

    pooled_metrics = compute_volume_metrics(pooled_final, pooled_labels, eval_cfg=cfg)
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
            n_patients=len(pooled_final),
            n_lesions=int(sum(pooled_labels.values())),
            code_version=code_version,
        )
    )

    # Stratified.
    strat_results = stratify_metrics(
        pooled_final, pooled_labels, manifest_lookup, eval_cfg=cfg
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
            "per_fold_thresholds": per_fold_thresholds,
            "ensemble_threshold": ensemble_thresholds,
        },
    )

    return {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "thresholds_path": str(thresholds_path),
        "ensemble_threshold": ensemble_thresholds,
        "per_fold_thresholds": per_fold_thresholds,
        "n_rows": len(rows_to_write),
    }


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


# ----------------------------------------------------------------------------
# Holdout one-shot ensemble


def run_holdout_inference(
    experiment: ExperimentConfig,
    ckpts: list[int] | str = "all",
    use_gru: bool = False,
    image_size: tuple[int, int] = (384, 384),
) -> Path:
    """One-shot 5-model ensemble inference on the 122 holdout patients.

    THIS IS THE ONLY LEGITIMATE CALLER OF ``DataModule(allow_holdout=True)``.
    Per PRD I.9.3 / §13 amendment A.5: no other subcommand may toggle that
    flag. Each invocation produces a fresh
    ``runs/<exp>/holdout/run_<timestamp>_<uuid8>/`` subdir; "touch holdout
    once" is enforced by user discipline, not code.
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

    # Invocation record (per spec §5.3.9).
    import json as _json
    invocation_payload = {
        "run_id": run_id,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "ckpts_used": [int(f) for f in (ckpts if isinstance(ckpts, list) else [])] or "all",
        "use_gru": bool(use_gru),
        "code_version": code_version,
        "experiment_name": experiment.name,
        "experiment_uuid": experiment.uuid,
    }
    (holdout_dir / "invocation.json").write_text(_json.dumps(invocation_payload, indent=2))

    # Load thresholds from CV eval.
    thresholds_path = run_dir / "eval" / "eval_thresholds.json"
    if thresholds_path.exists():
        import json

        thr = json.loads(thresholds_path.read_text())
        ensemble_thr = thr.get("ensemble_threshold", {"large": 0.05, "small": 0.30})
    else:
        log.warning("No eval_thresholds.json at %s; using config defaults.", thresholds_path)
        ensemble_thr = {"large": 0.05, "small": 0.30}

    # Manifest + holdout pids.
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    manifest_rows = read_manifest_jsonl(manifest_path)
    manifest_lookup = manifest_by_pid(manifest_rows)
    label_lookup = {pid: int(r.get("label") == "positive") for pid, r in manifest_lookup.items()}
    _, _, holdout_pids = fold_split(manifest_rows, fold=0)
    holdout_pids = list(holdout_pids)

    # ── DataModule build with allow_holdout=True (the *only* place). ──
    # NOTE: This is the sole call site that legitimately sets
    # ``allow_holdout=True``. Per PRD I.9.3 + §13 amendment A.5, do NOT
    # replicate this anywhere else in the codebase.
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

    # Load detector ckpts and run inference.
    from endo.inference_pass import inference_pass
    from endo.lightning_module import LesionDetectorLM

    per_pid_slice_lists: dict[str, list[list[SliceScore]]] = {pid: [] for pid in holdout_pids}
    for fold in ckpt_indices:
        ckpt_path = run_dir / f"fold{fold}" / "ckpts" / "best.ckpt"
        if not ckpt_path.exists():
            log.warning("fold %d: missing best.ckpt at %s; skipping.", fold, ckpt_path)
            continue
        log.info("loading fold %d ckpt: %s", fold, ckpt_path)
        try:
            import torch as _torch

            raw = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            module = LesionDetectorLM(experiment)
            module.load_state_dict(raw["state_dict"], strict=False)
            ema_sd = raw.get("ema_state_dict")
            if ema_sd is not None:
                try:
                    module.model.load_state_dict(ema_sd, strict=True)
                except Exception:
                    pass
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            module.to(device)
            module.eval()
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
            rescored = _try_gru_rescore(experiment, fold, scores)
            if rescored is not None:
                scores = rescored

        for pid, lst in scores.items():
            per_pid_slice_lists[pid].append(lst)

    # Mean-fusion across ckpts: concatenate all per-ckpt slice lists into one
    # list (each model's boxes contribute equally to the WBF).
    final_preds: dict[str, dict] = {}
    for pid, lists in per_pid_slice_lists.items():
        if not lists:
            final_preds[pid] = {
                "fused_boxes": np.zeros((0, 5), dtype=np.float32),
                "fused_scores": np.zeros((0,), dtype=np.float32),
                "score": 0.0,
            }
            continue
        flat: list[SliceScore] = [s for lst in lists for s in lst]
        fused = weighted_box_fusion_3d(
            flat,
            image_size=image_size,
            iou_thr=cfg.wbf_iou_threshold,
            skip_box_thr=cfg.wbf_skip_box_threshold,
            large_threshold=ensemble_thr["large"],
            small_threshold=ensemble_thr["small"],
            box_size_threshold_mm=cfg.box_size_split_mm,
        )
        fb = fused["fused_boxes"]
        fs = fused["fused_scores"]
        final_preds[pid] = {
            "fused_boxes": fb,
            "fused_scores": fs,
            "score": float(fs.max()) if fs.size > 0 else 0.0,
        }

    holdout_labels = {pid: label_lookup.get(pid, 0) for pid in holdout_pids}
    metrics = compute_volume_metrics(final_preds, holdout_labels, eval_cfg=cfg)
    rows = _emit_metric_rows(
        metrics,
        run_id=run_id,
        entrypoint="holdout",
        scope="holdout",
        fold=None,
        stratum_kind=None,
        stratum_value=None,
        rescored=use_gru,
        n_patients=len(final_preds),
        n_lesions=int(sum(holdout_labels.values())),
        code_version=code_version,
    )

    strat = stratify_metrics(final_preds, holdout_labels, manifest_lookup, eval_cfg=cfg)
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
    return holdout_dir
