"""EMA callback wrapping timm's ``ModelEmaV3``.

Component 6 §8 + PRD I.8.5 (swap to EMA for validation/deep-eval and restore)
+ I.8.9 (fp32 shadow buffer).
"""

from __future__ import annotations

import copy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3


class EmaCallback(pl.Callback):
    """Maintain a fp32 EMA shadow of the live model and swap during validation."""

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = float(decay)
        self.ema: ModelEmaV3 | None = None
        self._saved_live_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.ema is None:
            self._init_ema(pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema is None:
            self._init_ema(pl_module)

    def _init_ema(self, pl_module: pl.LightningModule) -> None:
        live: nn.Module = pl_module.model
        # fp32 shadow on the live device; detached deepcopy under the hood.
        self.ema = ModelEmaV3(
            live,
            decay=self.decay,
            device=pl_module.device if pl_module.device is not None else None,
        )
        # Force shadow params/buffers to fp32 (per PRD I.8.9).
        for p in self.ema.module.parameters():
            p.data = p.data.float()
        for b in self.ema.module.buffers():
            if b.is_floating_point():
                b.data = b.data.float()

    # ------------------------------------------------------------------
    # Update on every train batch.
    # ------------------------------------------------------------------
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema is None:
            self._init_ema(pl_module)
        assert self.ema is not None
        self.ema.update(pl_module.model)

    # ------------------------------------------------------------------
    # Swap live <-> EMA across validation.
    # ------------------------------------------------------------------
    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            return
        self._saved_live_state = {
            k: v.detach().clone() for k, v in pl_module.model.state_dict().items()
        }
        ema_state = {
            k: v.to(dtype=self._saved_live_state[k].dtype) if k in self._saved_live_state else v
            for k, v in self.ema.module.state_dict().items()
        }
        pl_module.model.load_state_dict(ema_state, strict=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._saved_live_state is None:
            return
        pl_module.model.load_state_dict(self._saved_live_state, strict=True)
        self._saved_live_state = None

    # ------------------------------------------------------------------
    # Checkpoint persistence.
    # ------------------------------------------------------------------
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema is not None:
            checkpoint["ema_state_dict"] = copy.deepcopy(self.ema.module.state_dict())
            checkpoint["ema_decay"] = self.decay

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        ema_sd = checkpoint.get("ema_state_dict")
        if ema_sd is None:
            return
        if self.ema is None:
            self._init_ema(pl_module)
        assert self.ema is not None
        self.ema.module.load_state_dict(ema_sd, strict=True)
        self.decay = float(checkpoint.get("ema_decay", self.decay))
