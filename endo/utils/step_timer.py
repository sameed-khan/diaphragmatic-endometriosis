"""Lightning callback that logs ``train/throughput_samples_per_sec`` per step."""

from __future__ import annotations

import time

import pytorch_lightning as pl


class StepTimerCallback(pl.Callback):
    """Logs per-step throughput (samples / second) and seconds-per-epoch.

    The throughput is computed as ``batch_size / elapsed_seconds`` for the
    just-finished batch. The epoch wall time is logged at
    ``train/seconds_per_epoch`` on ``on_train_epoch_end``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._step_t0: float | None = None
        self._epoch_t0: float | None = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_t0 = time.perf_counter()

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int
    ) -> None:
        self._step_t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._step_t0 is None:
            return
        elapsed = max(time.perf_counter() - self._step_t0, 1e-6)
        try:
            bs = int(batch.volume_5ch.shape[0])
        except Exception:
            bs = 1
        try:
            pl_module.log(
                "train/throughput_samples_per_sec",
                float(bs) / elapsed,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        except Exception:  # noqa: BLE001
            pass

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._epoch_t0 is None:
            return
        secs = float(time.perf_counter() - self._epoch_t0)
        try:
            pl_module.log(
                "train/seconds_per_epoch",
                secs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        except Exception:  # noqa: BLE001
            pass
        self._epoch_t0 = None
