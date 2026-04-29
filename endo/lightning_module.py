"""Lightning wrapper around ``LesionDetector`` for training + validation.

Component 6 §7 + PRD §6.8/§6.9, §8 invariants, §13 amendment A.8 (best ckpt
by ``val/slice_auroc``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from endo.config import ExperimentConfig
from endo.data.samples import Batch
from endo.model.detector import LesionDetector
from endo.model.losses import compute_total_loss


def _split_decay_params(model: torch.nn.Module) -> tuple[list, list]:
    """Split parameters into (weight-decay, no-decay) groups.

    Per common practice and the spec's §7: norm and bias params get
    weight_decay=0; everything else gets the configured decay.
    """
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Bias or 1-D parameter (LN/GN/BN scale or bias) -> no decay.
        if p.ndim <= 1 or name.endswith(".bias"):
            nodecay.append(p)
        else:
            decay.append(p)
    return decay, nodecay


class LesionDetectorLM(pl.LightningModule):
    """LightningModule for the 2.5D MR lesion detector."""

    def __init__(self, exp_cfg: ExperimentConfig) -> None:
        super().__init__()
        self.exp_cfg = exp_cfg
        # ``save_hyperparameters`` w/ a Pydantic model is fragile; persist the
        # serialized form so ckpts remain self-describing without a typed dep.
        try:
            self.save_hyperparameters({"experiment": exp_cfg.model_dump(mode="json")})
        except Exception:
            pass

        self.model = LesionDetector(exp_cfg.model)
        # Wired in by the training entrypoint after the sampler exists.
        self.score_ema_tracker = None

        # Per-epoch validation buffers (populated in ``validation_step``).
        self._val_max_scores: list[float] = []
        self._val_labels: list[int] = []

    # ------------------------------------------------------------------
    # Forward / inference.
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        return self.model(x)

    # ------------------------------------------------------------------
    # Train / val steps.
    # ------------------------------------------------------------------
    def training_step(self, batch: Batch, batch_idx: int = 0) -> Tensor:
        cls_scores, bbox_preds, aux_seg_logits = self.model(batch.volume_5ch)
        det_losses = self.model.head.loss(
            cls_scores,
            bbox_preds,
            gt_boxes_per_image=list(batch.boxes),
            gt_labels_per_image=list(batch.labels),
            image_size=(384, 384),
        )
        total, components = compute_total_loss(
            det_losses,
            aux_seg_logits,
            batch.lesion_mask_center,
            aux_seg_weight=self.exp_cfg.training.aux_seg_weight,
        )
        # NaN guard: bf16-mixed precision can produce non-finite losses on
        # pathological mini-batches. Skip the step rather than poisoning the
        # weights — produce a zero-loss tensor that still has a grad path
        # back to the model parameters (so Lightning's backward pass works)
        # and contributes zero gradient.
        if not torch.isfinite(total):
            import logging as _logging
            comp_summary = {k: float(v) if torch.isfinite(v) else "nan/inf" for k, v in components.items()}
            _logging.getLogger("endo.lightning_module").warning(
                "non-finite loss at batch %d (%s) — skipping step",
                batch_idx, comp_summary,
            )
            # Zero loss with grad path. Use `nan_to_num` so even if
            # aux_seg_logits is inf, the result is finite and zero-valued.
            safe = torch.nan_to_num(aux_seg_logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
            total = safe.sum() * 0.0
            components = {k: total.detach() for k in components}

        # Per-step logging (Lightning aggregates by ``log_every_n_steps``).
        log_kw = {
            "on_step": True,
            "on_epoch": True,
            "batch_size": batch.volume_5ch.shape[0],
        }
        self.log("train/loss_cls", components["loss_cls"], **log_kw)
        self.log("train/loss_bbox", components["loss_bbox"], **log_kw)
        self.log("train/loss_aux_seg", components["loss_aux_seg"], **log_kw)
        self.log("train/loss_total", components["loss_total"], prog_bar=True, **log_kw)

        # Update score EMA tracker for negative slices only (I.8.3).
        tracker = getattr(self, "score_ema_tracker", None)
        if tracker is not None:
            self._update_score_ema(batch, cls_scores, bbox_preds, tracker)

        return total

    @torch.no_grad()
    def _update_score_ema(
        self,
        batch: Batch,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        tracker: Any,
    ) -> None:
        preds = self.model.head.predict(
            cls_scores,
            bbox_preds,
            image_size=(384, 384),
        )
        is_pos = batch.is_positive_slice.detach().cpu().tolist()
        slice_ys = batch.slice_ys.detach().cpu().tolist()
        for i, (pid, sy, pos) in enumerate(zip(batch.patient_ids, slice_ys, is_pos)):
            if bool(pos):
                continue
            scores = preds[i]["scores"]
            max_score = float(scores.max().item()) if scores.numel() > 0 else 0.0
            tracker.update((pid, int(sy)), max_score, is_positive_slice=False)

    def validation_step(self, batch: Batch, batch_idx: int = 0) -> dict[str, Tensor]:
        cls_scores, bbox_preds, aux_seg_logits = self.model(batch.volume_5ch)
        det_losses = self.model.head.loss(
            cls_scores,
            bbox_preds,
            gt_boxes_per_image=list(batch.boxes),
            gt_labels_per_image=list(batch.labels),
            image_size=(384, 384),
        )
        total, components = compute_total_loss(
            det_losses,
            aux_seg_logits,
            batch.lesion_mask_center,
            aux_seg_weight=self.exp_cfg.training.aux_seg_weight,
        )

        log_kw = {"on_step": False, "on_epoch": True, "batch_size": batch.volume_5ch.shape[0]}
        self.log("val/loss_cls", components["loss_cls"], **log_kw)
        self.log("val/loss_bbox", components["loss_bbox"], **log_kw)
        self.log("val/loss_aux_seg", components["loss_aux_seg"], **log_kw)
        self.log("val/loss_total", components["loss_total"], **log_kw)

        # Slice-level scores for AUROC.
        preds = self.model.head.predict(cls_scores, bbox_preds, image_size=(384, 384))
        is_pos = batch.is_positive_slice.detach().cpu().tolist()
        for i in range(batch.volume_5ch.shape[0]):
            scores = preds[i]["scores"]
            max_score = float(scores.max().item()) if scores.numel() > 0 else 0.0
            self._val_max_scores.append(max_score)
            self._val_labels.append(int(bool(is_pos[i])))

        return {"loss_total": components["loss_total"]}

    def on_validation_epoch_start(self) -> None:
        self._val_max_scores.clear()
        self._val_labels.clear()

    def on_validation_epoch_end(self) -> None:
        labels = np.asarray(self._val_labels, dtype=np.int64)
        scores = np.asarray(self._val_max_scores, dtype=np.float64)
        if labels.size == 0 or len(set(labels.tolist())) < 2:
            auroc = 0.0
        else:
            auroc = float(roc_auc_score(labels, scores))
        # ModelCheckpoint monitor key (PRD §13 amendment A.8).
        self.log("val/slice_auroc", auroc, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimizer + LR schedule.
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        train_cfg = self.exp_cfg.training
        decay, nodecay = _split_decay_params(self.model)
        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_cfg.weight_decay},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=train_cfg.base_lr,
            betas=(0.9, 0.999),
        )

        # Total / warmup steps. ``estimated_stepping_batches`` is the total
        # over the whole fit; derive per-epoch from max_epochs.
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 0
        max_epochs = max(int(train_cfg.max_epochs), 1)
        steps_per_epoch = max(total_steps // max_epochs, 1) if total_steps else 1
        warmup_steps = max(steps_per_epoch * int(train_cfg.warmup_epochs), 1)
        cosine_steps = max(total_steps - warmup_steps, 1)

        min_ratio = train_cfg.min_lr / max(train_cfg.base_lr, 1e-12)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear from 0 -> 1 over warmup. Step 0 -> 0.
                return step / float(max(warmup_steps, 1))
            # Cosine decay from 1 -> min_ratio.
            progress = (step - warmup_steps) / float(max(cosine_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_ratio + (1.0 - min_ratio) * cos)

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
