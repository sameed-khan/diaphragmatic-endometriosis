"""Aug pipeline counter callback.

Drains ``TrainAugmentation.drain_stats()`` (when present) at each
``on_train_epoch_end`` and surfaces the rolling means as
``aug/paste_attempts_mean``, ``aug/paste_oob_clip_frac_mean``,
``aug/elastic_skip_rate``.

The augmentation pipeline currently does not maintain these counters; the
callback gracefully no-ops when ``drain_stats`` is absent so existing tests
keep passing.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl

log = logging.getLogger(__name__)


class AugStatsCallback(pl.Callback):
    def __init__(self, augmenter) -> None:
        super().__init__()
        self._augmenter = augmenter

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._augmenter is None:
            return
        drain = getattr(self._augmenter, "drain_stats", None)
        if drain is None or not callable(drain):
            return
        try:
            stats = drain() or {}
        except Exception as e:  # noqa: BLE001
            log.warning("aug drain_stats raised %s", e)
            return
        for k, v in stats.items():
            try:
                pl_module.log(
                    f"aug/{k}",
                    float(v),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
            except Exception:  # noqa: BLE001
                pass
