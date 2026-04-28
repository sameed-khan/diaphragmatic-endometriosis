"""Periodic deep-eval callback.

Fires every ``deep_eval_refresh_every_epochs`` epochs starting at
``deep_eval_start_epoch``. Runs two passes of :func:`inference_pass`:

  1. Validation set → ``runs/<exp>/fold{f}/runtime/deep_eval/epoch{n}_val.npz``
     (consumed by Component 7's eval).
  2. Training negatives → top-K hard pool → ``hard_negatives.json`` (consumed
     by :class:`WeightedScheduledSampler` at next epoch boundary).

Coarse volume-level metrics (AUROC, sensitivity at 2 FP per volume) are also
logged via Lightning ``log_dict`` for periodic monitoring.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl

from endo.config.sampler import SamplerConfig
from endo.inference_pass import SliceScore, inference_pass
from endo.sampler.weighted import WeightedScheduledSampler

log = logging.getLogger(__name__)


def _slice_max_score(s: SliceScore) -> float:
    if s.scores.size == 0:
        return float(s.aux_seg_max)
    return float(max(s.scores.max(), s.aux_seg_max))


def _flatten_for_npz(
    scores: dict[str, list[SliceScore]],
) -> dict[str, np.ndarray]:
    """Pack per-patient SliceScore lists into the §5.3.4 CSR-style schema."""
    pids: list[str] = []
    slice_ys: list[int] = []
    aux_seg_max: list[float] = []
    boxes_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    box_offsets: list[int] = [0]

    cur = 0
    for pid in sorted(scores.keys()):
        for s in scores[pid]:
            pids.append(pid)
            slice_ys.append(int(s.slice_y))
            aux_seg_max.append(float(s.aux_seg_max))
            n = int(s.boxes.shape[0]) if s.boxes.ndim == 2 else 0
            cur += n
            box_offsets.append(cur)
            if n > 0:
                boxes_list.append(s.boxes.astype(np.float32, copy=False))
                scores_list.append(s.scores.astype(np.float32, copy=False))

    boxes_flat = (
        np.concatenate(boxes_list, axis=0)
        if boxes_list
        else np.zeros((0, 4), dtype=np.float32)
    )
    scores_flat = (
        np.concatenate(scores_list, axis=0)
        if scores_list
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "patient_ids": np.asarray(pids, dtype=object),
        "slice_ys": np.asarray(slice_ys, dtype=np.int32),
        "boxes_flat": boxes_flat.astype(np.float32, copy=False),
        "scores_flat": scores_flat.astype(np.float32, copy=False),
        "box_offsets": np.asarray(box_offsets, dtype=np.int32),
        "aux_seg_max": np.asarray(aux_seg_max, dtype=np.float32),
    }


def _coarse_volume_metrics(
    val_scores: dict[str, list[SliceScore]],
    val_volume_labels: dict[str, int] | None,
) -> dict[str, float]:
    """Cheap proxies for monitoring: max-score-per-volume AUROC and the
    sensitivity at FP=2/volume on negatives.

    Returns NaN for AUROC if labels are unavailable or single-class.
    """
    if not val_scores:
        return {"volume_auroc": float("nan"), "sens_at_2fp": float("nan")}

    vol_max: dict[str, float] = {
        pid: max((_slice_max_score(s) for s in slices), default=0.0)
        for pid, slices in val_scores.items()
    }
    pids = list(vol_max.keys())
    scores = np.asarray([vol_max[p] for p in pids], dtype=np.float64)

    auroc = float("nan")
    if val_volume_labels is not None:
        labels = np.asarray(
            [int(val_volume_labels.get(p, 0)) for p in pids], dtype=np.int64
        )
        if labels.min() != labels.max():
            try:
                from sklearn.metrics import roc_auc_score

                auroc = float(roc_auc_score(labels, scores))
            except Exception as e:  # pragma: no cover - sklearn always available
                log.warning("coarse AUROC failed: %s", e)

    # Sensitivity-at-2FP: per-volume — find threshold where mean negatives kept
    # equals 2 (here we treat each "negative slice" as one FP candidate). With
    # only volume-level info, this is a coarse proxy: take the max score per
    # volume and find the threshold giving FP rate 2/n_negatives.
    sens = float("nan")
    if val_volume_labels is not None and labels.min() != labels.max():
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if neg_scores.size > 0 and pos_scores.size > 0:
            target_fp_rate = 2.0 / max(neg_scores.size, 1)
            thr = np.quantile(neg_scores, max(0.0, 1.0 - target_fp_rate))
            sens = float((pos_scores >= thr).mean())
    return {"volume_auroc": auroc, "sens_at_2fp": sens}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


class PeriodicDeepEvalCallback(pl.Callback):
    """Lightning callback driving Component 5's deep-eval refresh."""

    def __init__(
        self,
        sampler_cfg: SamplerConfig,
        run_dir: Path,
        train_neg_pids: list[str],
        val_pids: list[str],
        ema_callback: Optional["object"] = None,
        score_threshold: float = 0.05,
        val_volume_labels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = sampler_cfg
        self.run_dir = Path(run_dir)
        self.train_neg_pids = list(train_neg_pids)
        self.val_pids = list(val_pids)
        self.ema_callback = ema_callback
        self.score_threshold = float(score_threshold)
        self.val_volume_labels = val_volume_labels

        self.runtime_dir = self.run_dir / "runtime"
        self.deep_eval_dir = self.runtime_dir / "deep_eval"

    # ─── schedule ────────────────────────────────────────────────────

    def _should_run(self, epoch: int) -> bool:
        cfg = self.cfg
        if epoch < cfg.deep_eval_start_epoch:
            return False
        every = max(1, int(cfg.deep_eval_refresh_every_epochs))
        return ((epoch - cfg.deep_eval_start_epoch) % every) == 0

    # ─── EMA helpers ─────────────────────────────────────────────────

    def _maybe_swap_to_ema(self) -> bool:
        """Return True if we performed the swap (caller must restore)."""
        cb = self.ema_callback
        if cb is None:
            return False
        already_swapped = bool(getattr(cb, "_is_swapped", False))
        if already_swapped:
            return False
        # Try common method names; tolerate absence in tests/mocks.
        for name in ("swap_to_ema", "_swap_to_ema", "apply_shadow"):
            fn = getattr(cb, name, None)
            if callable(fn):
                fn()
                return True
        log.warning("ema_callback has no swap method; running on live weights.")
        return False

    def _restore_live(self, did_swap: bool) -> None:
        if not did_swap:
            return
        cb = self.ema_callback
        for name in ("restore_live", "_restore_live", "restore"):
            fn = getattr(cb, name, None)
            if callable(fn):
                fn()
                return
        log.warning("ema_callback swap had no matching restore; live weights NOT restored.")

    # ─── main hook ───────────────────────────────────────────────────

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0))
        if not self._should_run(epoch):
            return

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.deep_eval_dir.mkdir(parents=True, exist_ok=True)

        did_swap = self._maybe_swap_to_ema()
        try:
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is None:
                log.warning("PeriodicDeepEvalCallback: trainer has no datamodule; skipping.")
                return

            # Pass 1: val.
            t0 = time.perf_counter()
            val_scores = inference_pass(
                model=pl_module,
                datamodule=datamodule,
                patient_ids=self.val_pids,
                split="val",
            )
            val_secs = time.perf_counter() - t0

            npz_payload = _flatten_for_npz(val_scores)
            npz_path = self.deep_eval_dir / f"epoch{epoch}_val.npz"
            np.savez_compressed(npz_path, **npz_payload)

            # Pass 2: train negatives → top-K → hard pool.
            t0 = time.perf_counter()
            neg_scores = inference_pass(
                model=pl_module,
                datamodule=datamodule,
                patient_ids=self.train_neg_pids,
                split="train_negatives",
            )
            neg_secs = time.perf_counter() - t0

            # Build slice-index lookup so we can map (pid, slice_y) → dataset idx.
            slice_ix = self._slice_index_lookup(trainer)

            ranked: list[tuple[float, int]] = []
            for pid, slices in neg_scores.items():
                for s in slices:
                    if s.scores.size > 0:
                        m = float(s.scores.max())
                    else:
                        m = float(s.aux_seg_max)
                    if m < self.score_threshold:
                        continue
                    key = (pid, int(s.slice_y))
                    ds_idx = slice_ix.get(key) if slice_ix is not None else None
                    if ds_idx is None:
                        continue
                    ranked.append((m, ds_idx))

            ranked.sort(key=lambda kv: kv[0], reverse=True)
            top_k = int(self.cfg.hard_pool_top_k)
            top = ranked[:top_k]
            slice_indices = [int(idx) for _, idx in top]

            payload = {
                "epoch_written": int(epoch),
                "model_checkpoint_epoch": int(epoch),
                "slice_indices": slice_indices,
                "n_slices": int(len(slice_indices)),
                "score_threshold": float(self.score_threshold),
            }
            _atomic_write_json(self.runtime_dir / "hard_negatives.json", payload)

            # Plumb into sampler.
            self._set_sampler_hard_pool(trainer, slice_indices)

            # Coarse metrics.
            coarse = _coarse_volume_metrics(val_scores, self.val_volume_labels)
            try:
                pl_module.log_dict(
                    {
                        "deep_eval/val_volume_auroc_coarse": float(coarse["volume_auroc"]),
                        "deep_eval/val_froc_at_2fp_coarse": float(coarse["sens_at_2fp"]),
                    },
                    sync_dist=False,
                )
            except Exception as e:  # pragma: no cover - log_dict shape varies in mocks
                log.debug("log_dict skipped (%s)", e)

            log.info(
                "deep_eval epoch=%d val_secs=%.2f neg_secs=%.2f hard_pool_size=%d",
                epoch,
                val_secs,
                neg_secs,
                len(slice_indices),
            )
        finally:
            self._restore_live(did_swap)

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _slice_index_lookup(trainer: pl.Trainer) -> dict[tuple[str, int], int] | None:
        """Best-effort lookup from (patient_id, slice_y) to dataset index.

        Looks for a ``slice_index`` attribute on the train dataset / sampler.
        Returns ``None`` if no usable mapping is found (callback then writes
        an empty hard pool but still emits the file).
        """
        # Try the train dataloader's dataset.
        for attr in ("train_dataloader",):
            try:
                dl = getattr(trainer, attr)
                dl = dl() if callable(dl) else dl
            except Exception:
                dl = None
            if dl is None:
                continue
            ds = getattr(dl, "dataset", None)
            cand = getattr(ds, "slice_index", None) if ds is not None else None
            if cand is not None:
                return {(pid, int(sy)): i for i, (pid, sy, _kind) in enumerate(cand)}
            sampler = getattr(dl, "sampler", None)
            cand = getattr(sampler, "_slice_index", None)
            if cand is not None:
                return {(pid, int(sy)): i for i, (pid, sy, _kind) in enumerate(cand)}
        return None

    @staticmethod
    def _set_sampler_hard_pool(trainer: pl.Trainer, indices: list[int]) -> None:
        try:
            dl = trainer.train_dataloader
            dl = dl() if callable(dl) else dl
        except Exception as e:
            log.warning("Could not access train_dataloader: %s", e)
            return
        sampler = getattr(dl, "sampler", None) if dl is not None else None
        if sampler is None:
            log.warning("train_dataloader has no sampler; hard pool not set.")
            return
        if not hasattr(sampler, "set_hard_pool"):
            log.warning(
                "Sampler %s has no set_hard_pool; hard pool not set.",
                type(sampler).__name__,
            )
            return
        try:
            sampler.set_hard_pool(indices)
        except Exception as e:
            log.warning("sampler.set_hard_pool failed: %s", e)
            return
        if not isinstance(sampler, WeightedScheduledSampler):
            log.warning(
                "sampler is %s, not WeightedScheduledSampler; hard pool set defensively.",
                type(sampler).__name__,
            )
