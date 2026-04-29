"""Single-implementation inference primitive (PRD §6.7).

Used by:
  - :class:`endo.sampler.periodic_eval.PeriodicDeepEvalCallback`
  - the ``eval`` and ``predict_holdout`` subcommands (Component 7)
  - the GRU feature cache builder (a backbone-only sibling lives there).

Returns a ``{patient_id: [SliceScore, ...]}`` mapping. The caller is
responsible for any cross-slice aggregation (WBF, FROC, etc.).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch


@dataclass
class SliceScore:
    """Per-slice detector output (PRD §6.5)."""

    patient_id: str
    slice_y: int
    boxes: np.ndarray  # (N, 4) float32, (x1, z1, x2, z2)
    scores: np.ndarray  # (N,) float32
    aux_seg_max: float


def _autocast_ctx(device: torch.device):
    """bf16 autocast on CUDA; no-op elsewhere (e.g. CPU-only test runs)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32).cpu().numpy()


def inference_pass(
    model,  # pl.LightningModule with `.model` attribute (LesionDetector)
    datamodule,  # pl.LightningDataModule with `inference_dataloader(patient_ids)`
    patient_ids: list[str],
    split: Literal["val", "train_negatives", "holdout"],
    batch_size: int = 16,
) -> dict[str, list[SliceScore]]:
    """Run model in eval mode over every valid slice of every patient.

    The model is expected to expose ``model.model`` returning a tuple
    ``(cls_scores, bbox_preds, aux_seg_logits)`` where:

      - ``cls_scores`` / ``bbox_preds`` are per-FPN-level lists of tensors
        consumable by ``model.model.head.predict`` (NMS lives on the head,
        per the cross-component contract in `endo/CLAUDE.md`);
      - ``aux_seg_logits`` is ``(B, 1, H, W)`` (or ``(B, H, W)``) of the
        slice-level presence head's logits.

    Throughput target: ≥ 50 slices/sec on L40S. Inference uses bf16 autocast
    when on CUDA. ``batch_size`` is forwarded to
    ``datamodule.inference_dataloader`` so callers (deep-eval, holdout, GRU
    feature cache) can tune it without touching the DataModule's training
    batch size.
    """
    detector = model.model
    device = next(detector.parameters()).device if any(p is not None for p in detector.parameters()) else torch.device("cpu")  # type: ignore[arg-type]

    # Some models may not expose a fixed image_size; pull from a config attr if present.
    image_size = getattr(detector, "image_size", None)
    if image_size is None:
        image_size = (384, 384)

    # Move into eval mode for the duration of this pass.
    was_training = model.training
    model.eval()

    out: dict[str, list[SliceScore]] = {pid: [] for pid in patient_ids}

    try:
        loader = datamodule.inference_dataloader(patient_ids, batch_size=batch_size)
    except TypeError:
        # Older DMs may not accept batch_size — fall back to the contract
        # that just takes patient_ids.
        try:
            loader = datamodule.inference_dataloader(patient_ids)
        except TypeError:
            loader = datamodule.inference_dataloader(patient_ids=patient_ids)

    try:
        with torch.no_grad(), _autocast_ctx(device):
            for batch in loader:
                vol = batch.volume_5ch.to(device, non_blocking=True)
                cls_scores, bbox_preds, aux_seg = detector(vol)

                # Per-image post-NMS predictions. The head, not the detector,
                # owns the (cls_scores, bbox_preds) -> NMS predictions API.
                preds = detector.head.predict(cls_scores, bbox_preds, image_size=image_size)

                # aux_seg may be (B, 1, H, W) or (B, H, W). Handle both.
                aux = aux_seg
                if aux.dim() == 4:
                    aux_max = aux.sigmoid().amax(dim=(1, 2, 3))  # (B,)
                elif aux.dim() == 3:
                    aux_max = aux.sigmoid().amax(dim=(1, 2))
                else:
                    raise ValueError(
                        f"Unexpected aux_seg shape {tuple(aux.shape)}; expected 3D or 4D."
                    )

                pids: Iterable[str] = batch.patient_ids
                slice_ys = batch.slice_ys.detach().cpu().numpy().astype(np.int64)
                aux_max_np = _to_numpy(aux_max)

                for i, pid in enumerate(pids):
                    pred = preds[i]
                    boxes = _to_numpy(pred["boxes"]).astype(np.float32, copy=False)
                    scores = _to_numpy(pred["scores"]).astype(np.float32, copy=False)
                    if boxes.ndim == 1:
                        boxes = boxes.reshape(0, 4)
                    out.setdefault(pid, []).append(
                        SliceScore(
                            patient_id=pid,
                            slice_y=int(slice_ys[i]),
                            boxes=boxes,
                            scores=scores,
                            aux_seg_max=float(aux_max_np[i]),
                        )
                    )
    finally:
        if was_training:
            model.train()

    # Order each patient's slices by slice_y ascending (loader is allowed to
    # interleave patients across batches, e.g. with batch_size > 1).
    for pid in list(out.keys()):
        out[pid].sort(key=lambda s: s.slice_y)
    return out
