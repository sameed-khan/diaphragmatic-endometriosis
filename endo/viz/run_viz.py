"""Visualization orchestrator (Component 8 §2).

Loads ``best.ckpt`` for a given experiment / fold, runs inference on the
validation patients, applies per-slice NMS (or WBF if available), tags every
slice as TP / FP / FN, renders PNG overlays, and writes ``manifest.csv``.

The orchestrator is **idempotent**: a second run with an unchanged
``best.ckpt`` mtime is a no-op (returns the existing output directory).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from endo.config import ExperimentConfig
from endo.viz.render import render_slice_overlay, save_slice_png
from endo.viz.tagging import _box_iou, tag_slice_events

# Try the project's WBF module; fall back to torchvision NMS if not present.
try:  # pragma: no cover - exercised only when WBF lands.
    from endo.eval.wbf import per_slice_wbf as _per_slice_wbf  # type: ignore
except Exception:  # noqa: BLE001
    _per_slice_wbf = None


def _per_slice_nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy IoU NMS using torchvision; falls back to a pure-numpy variant."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        return boxes, scores
    try:
        from torchvision.ops import nms as tv_nms

        keep = tv_nms(
            torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold
        ).numpy()
    except Exception:  # noqa: BLE001 - keep this path robust.
        order = np.argsort(-scores, kind="stable")
        keep_list: list[int] = []
        suppressed = np.zeros(boxes.shape[0], dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep_list.append(int(i))
            ious = _box_iou(boxes[i:i + 1], boxes).flatten()
            suppressed |= ious >= iou_threshold
            suppressed[i] = False  # don't suppress self in the loop
        keep = np.array(keep_list, dtype=np.int64)
    return boxes[keep], scores[keep]


def _resolve_best_ckpt(run_dir: Path, fold: int) -> Path:
    """Find ``best.ckpt`` for a given fold under ``runs/<exp>/fold{f}/``."""
    fold_dir = run_dir / f"fold{fold}"
    candidates = [
        fold_dir / "best.ckpt",
        fold_dir / "ckpts" / "best.ckpt",
        fold_dir / "checkpoints" / "best.ckpt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No best.ckpt found under {fold_dir} (looked at: {[str(p) for p in candidates]})"
    )


def _gt_for_pid_slice(
    gt_lookup: dict[tuple[str, int], np.ndarray], pid: str, slice_y: int
) -> np.ndarray:
    arr = gt_lookup.get((pid, int(slice_y)))
    if arr is None:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32).reshape(-1, 4)


def _label_for_pid(
    cache: dict[str, dict[str, Any]], pid: str
) -> str:
    entry = cache.get(pid, {})
    row = entry.get("manifest_row", {}) if isinstance(entry, dict) else {}
    return "positive" if row.get("label") == "positive" else "negative"


def _compute_max_iou(pred_box: np.ndarray, gt_boxes: np.ndarray) -> float:
    if gt_boxes.size == 0:
        return 0.0
    ious = _box_iou(pred_box.reshape(1, 4), gt_boxes).flatten()
    return float(ious.max()) if ious.size else 0.0


def visualize_predictions_for_fold(
    experiment: ExperimentConfig,
    fold: int,
    output_dir: Path | None = None,
    device: str = "cuda",
    score_threshold: float = 0.05,
    max_pngs_per_event: int = 200,
    *,
    # Hooks for testing — both default to the production wiring.
    datamodule: Any | None = None,
    lightning_module: Any | None = None,
    sample_tp: int | None = None,
    sample_fp: int | None = None,
    sample_fn: int | None = None,
    rng_seed: int | None = None,
) -> Path:
    """Render per-slice prediction overlays for a fold's validation set.

    Args:
        experiment: Experiment config.
        fold: Fold index (0-based).
        output_dir: Where to write PNGs + manifest. Defaults to
            ``runs/<exp>/fold{fold}/viz/``.
        device: Torch device for inference.
        score_threshold: Drop predictions below this score before tagging.
        max_pngs_per_event: Hard cap on the number of PNGs for each event
            type (TP / FP / FN). Predictions are sorted by score; for FN we
            take the first ones encountered.
        datamodule: Pre-built DataModule (rare; mainly for tests).
        lightning_module: Pre-loaded LightningModule (rare; mainly for tests).

    Returns:
        Path to the output directory (which contains ``manifest.csv`` and
        the rendered PNGs).
    """
    run_dir = experiment.run_dir()
    fold_dir = run_dir / f"fold{fold}"
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        # Default to the post-train viz subdir (wandb-logging plan §3.1).
        out_dir = fold_dir / "viz" / "epoch_post-train"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    # Idempotency only applies when we're loading from disk; a live
    # LightningModule (training-time viz) always re-renders.
    if lightning_module is None:
        ckpt_path = _resolve_best_ckpt(run_dir, fold)
        sentinel = out_dir / ".ckpt_mtime"
        ckpt_mtime = ckpt_path.stat().st_mtime
        if (
            sentinel.exists()
            and manifest_path.exists()
            and abs(float(sentinel.read_text().strip() or "0") - ckpt_mtime) < 1e-6
        ):
            # Idempotent short-circuit: ckpt unchanged → existing output is current.
            return out_dir
    else:
        ckpt_path = None
        ckpt_mtime = None

    # Build the DataModule if not provided.
    if datamodule is None:
        from endo.data.datamodule import LesionDataModule

        datamodule = LesionDataModule(
            cache_root=experiment.paths.cache_root,
            manifest_path=experiment.paths.data_root / "manifest.jsonl",
            cohort_path=experiment.paths.data_root / "cohort.json",
            fold=fold,
            batch_size=getattr(experiment.training, "batch_size", 8),
            num_workers=2,
        )
        datamodule.setup()

    # Build / load the LightningModule.  ``LesionDetectorLM.__init__`` takes a
    # positional ``exp_cfg``, so we construct it manually and load state dict.
    if lightning_module is None:
        import torch as _torch
        from endo.lightning_module import LesionDetectorLM

        raw = _torch.load(str(ckpt_path), map_location=device, weights_only=False)
        lm = LesionDetectorLM(experiment)
        lm.load_state_dict(raw["state_dict"], strict=False)
        ema_sd = raw.get("ema_state_dict")
        if ema_sd is not None:
            try:
                lm.model.load_state_dict(ema_sd, strict=True)
            except Exception:
                pass
    else:
        lm = lightning_module
    lm.to(device)
    lm.eval()

    # Run inference.
    from endo.inference_pass import inference_pass

    val_pids = list(getattr(datamodule, "_val_pids", []))
    if not val_pids:
        # Some test DMs may not expose _val_pids; fall back to attribute access.
        val_pids = list(getattr(datamodule, "val_patient_ids", []))
    preds = inference_pass(lm, datamodule, val_pids, split="val")

    gt_lookup: dict[tuple[str, int], np.ndarray] = getattr(datamodule, "_gt_lookup", {})
    cache: dict[str, dict[str, Any]] = getattr(datamodule, "_cache", {})

    # Counters for caps.
    counts = {"tp": 0, "fp": 0, "fn": 0}

    rows: list[dict[str, Any]] = []

    for pid, slice_scores in preds.items():
        label_dir = _label_for_pid(cache, pid)
        vol_entry = cache.get(pid, {})
        volume = vol_entry.get("volume") if isinstance(vol_entry, dict) else None
        lesion_mask = vol_entry.get("lesion_mask") if isinstance(vol_entry, dict) else None

        for ss in slice_scores:
            sy = int(ss.slice_y)
            boxes = np.asarray(ss.boxes, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(ss.scores, dtype=np.float32).reshape(-1)
            mask = scores >= float(score_threshold)
            boxes = boxes[mask]
            scores = scores[mask]
            boxes, scores = _per_slice_nms(boxes, scores, iou_threshold=0.5)

            gt_boxes = _gt_for_pid_slice(gt_lookup, pid, sy)
            tagged = tag_slice_events(boxes, scores, gt_boxes, iou_threshold=0.3)

            for event_type in ("tp", "fp", "fn"):
                items = tagged[event_type]
                if not items or counts[event_type] >= max_pngs_per_event:
                    continue
                # Render once per (slice, event_type).
                fname = f"{label_dir}_{pid}_{event_type}_slice{sy}.png"
                png_path = out_dir / fname
                if volume is not None:
                    img = render_slice_overlay(
                        volume=volume,
                        slice_y=sy,
                        lesion_mask_center=(
                            lesion_mask[:, sy, :] if lesion_mask is not None else None
                        ),
                        pred_boxes=boxes,
                        pred_scores=scores,
                        gt_boxes=gt_boxes,
                        event_type=event_type,
                        patient_id=pid,
                    )
                    save_slice_png(img, png_path)
                    counts[event_type] += 1

                # Emit one CSV row per highlighted event entity in this slice.
                if event_type == "tp":
                    for box, score, gt_idx in items:
                        rows.append(
                            {
                                "patient_id": pid,
                                "slice_y": sy,
                                "event_type": event_type,
                                "score": float(score),
                                "gt_iou": _compute_max_iou(box, gt_boxes),
                                "png_path": str(png_path),
                            }
                        )
                elif event_type == "fp":
                    for box, score in items:
                        rows.append(
                            {
                                "patient_id": pid,
                                "slice_y": sy,
                                "event_type": event_type,
                                "score": float(score),
                                "gt_iou": _compute_max_iou(box, gt_boxes),
                                "png_path": str(png_path),
                            }
                        )
                else:  # fn
                    for gt_box in items:
                        rows.append(
                            {
                                "patient_id": pid,
                                "slice_y": sy,
                                "event_type": event_type,
                                "score": float("nan"),
                                "gt_iou": float("nan"),
                                "png_path": str(png_path),
                            }
                        )

    # Write manifest.csv (overwrite — idempotent).
    fieldnames = ["patient_id", "slice_y", "event_type", "score", "gt_iou", "png_path"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if ckpt_mtime is not None:
        sentinel = out_dir / ".ckpt_mtime"
        sentinel.write_text(f"{ckpt_mtime}")
    return out_dir


def sample_tp_fp_fn(
    viz_dir: Path,
    *,
    n_tp: int = 20,
    n_fp: int = 20,
    n_fn: int = 20,
    seed: int = 0,
    sample_subdir_name: str = "wandb_sample",
) -> Path:
    """Reproducibly select up to ``n_tp + n_fp + n_fn`` PNGs from ``viz_dir``.

    Reads ``viz_dir/manifest.csv``, groups rows by ``event_type``, samples
    deterministically (seeded), and copies the chosen PNGs into
    ``viz_dir/<sample_subdir_name>/``. Returns that subdir's path.

    The directory is overwritten on each call so the contents are always
    consistent with the seed + manifest.
    """
    import random
    import shutil

    viz_dir = Path(viz_dir)
    manifest_path = viz_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"viz manifest not found: {manifest_path}")

    rows: dict[str, list[dict[str, str]]] = {"tp": [], "fp": [], "fn": []}
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = (row.get("event_type") or "").lower()
            if ev in rows:
                rows[ev].append(row)

    # Dedup by png_path (rows can repeat per highlighted entity).
    by_event: dict[str, list[str]] = {}
    for ev, rs in rows.items():
        seen: set[str] = set()
        ordered: list[str] = []
        for r in rs:
            p = r.get("png_path") or ""
            if p and p not in seen:
                seen.add(p)
                ordered.append(p)
        by_event[ev] = ordered

    rng = random.Random(int(seed))
    counts = {"tp": int(n_tp), "fp": int(n_fp), "fn": int(n_fn)}
    sample_dir = viz_dir / sample_subdir_name
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ev, k in counts.items():
        candidates = by_event.get(ev, [])
        if not candidates or k <= 0:
            continue
        pick = rng.sample(candidates, k=min(k, len(candidates)))
        for src in pick:
            src_path = Path(src)
            if not src_path.exists():
                # Manifest may have an absolute path that is now relative; try
                # resolving against viz_dir.
                src_path = viz_dir / src_path.name
            if not src_path.exists():
                continue
            try:
                shutil.copy2(src_path, sample_dir / f"{ev}_{src_path.name}")
            except Exception:  # noqa: BLE001
                continue
    return sample_dir
