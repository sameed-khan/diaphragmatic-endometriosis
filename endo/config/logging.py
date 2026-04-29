"""Logging + W&B configuration tree.

Wired into ``ExperimentConfig.logging`` and consumed by:
  * ``endo.utils.logging_setup`` — file/console handlers + rotating per-fold logs.
  * ``endo.utils.wandb_init``    — W&B run construction (project/group/name/tags).
  * ``endo.cli.run_experiment``  — flag plumbing (`--wandb / --no-wandb / -v`).

The whole subtree is **drift-exempt** in ``ExperimentConfig.diff(...)`` — toggling
logging settings between resumes never trips the experiment.yaml drift guard.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "diaphragmatic-endometriosis"
    entity: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "online"
    log_system_metrics: bool = True
    upload_checkpoints: bool = True
    upload_eval_reports: bool = True
    upload_viz_artifacts: bool = True
    upload_hard_pool_snapshots: bool = False

    model_config = {"extra": "forbid"}


class FileLoggingConfig(BaseModel):
    level_console: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    level_file: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    rotate_max_bytes: int = 50_000_000
    rotate_backups: int = 3

    model_config = {"extra": "forbid"}


class VizLoggingConfig(BaseModel):
    log_during_training: bool = False
    log_every_n_epochs: int = 0
    n_train_predictions_logged: int = 8
    sample_tp_per_fold: int = 20
    sample_fp_per_fold: int = 20
    sample_fn_per_fold: int = 20

    model_config = {"extra": "forbid"}


class AugLoggingConfig(BaseModel):
    log_samples: Literal["never", "epoch0", "always"] = "epoch0"
    n_samples: int = 8

    model_config = {"extra": "forbid"}


class LoggingConfig(BaseModel):
    file: FileLoggingConfig = Field(default_factory=FileLoggingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    viz: VizLoggingConfig = Field(default_factory=VizLoggingConfig)
    aug: AugLoggingConfig = Field(default_factory=AugLoggingConfig)
    log_every_n_steps: int = 10

    model_config = {"extra": "forbid"}
