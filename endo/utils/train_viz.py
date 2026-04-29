"""Training-time viz callback.

When ``LoggingConfig.viz.log_during_training=True``, every
``log_every_n_epochs`` validation epochs this callback runs
``visualize_predictions_for_fold`` with the live LightningModule and writes
PNGs to ``<fold_dir>/viz/epoch_{n}/``. Capped at
``n_train_predictions_logged`` events per type to keep the cost bounded.

The cost is non-trivial (full inference over the val set), so the default
is OFF; the e2e config opts in at ``log_every_n_epochs=1`` to validate
the integration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl

from endo.config import ExperimentConfig

log = logging.getLogger(__name__)


class TrainTimeVizCallback(pl.Callback):
    def __init__(
        self,
        experiment: ExperimentConfig,
        fold: int,
        fold_dir: Path,
    ) -> None:
        super().__init__()
        self.experiment = experiment
        self.fold = int(fold)
        self.fold_dir = Path(fold_dir)
        viz_cfg = experiment.logging.viz
        self.every_n = max(1, int(viz_cfg.log_every_n_epochs))
        self.cap = int(viz_cfg.n_train_predictions_logged)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Skip Lightning's pre-fit sanity validation (no current_epoch progress).
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        if (epoch % self.every_n) != 0:
            return
        try:
            from endo.viz.run_viz import visualize_predictions_for_fold

            out_dir = self.fold_dir / "viz" / f"epoch_{epoch}"
            log.info("training-time viz @ epoch=%d → %s", epoch, out_dir)
            visualize_predictions_for_fold(
                experiment=self.experiment,
                fold=self.fold,
                output_dir=out_dir,
                lightning_module=pl_module,
                datamodule=getattr(trainer, "datamodule", None),
                max_pngs_per_event=self.cap,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("training-time viz failed at epoch=%d: %s", epoch, e)
