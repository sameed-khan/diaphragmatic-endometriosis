"""End-to-end test config for the logging + W&B integration.

Goal: 2 epochs × 1000 samples on fold 0, then holdout eval, all logged to W&B
under experiment "e2e-testing", run "run1" / "run1-holdout". Success criteria
defined in agent/logging_wandb_plan.md §9.3. NOT for production.
"""

from __future__ import annotations

from pathlib import Path

from endo.config import (
    AugLoggingConfig,
    AugmentationConfig,
    EvalConfig,
    ExperimentConfig,
    FileLoggingConfig,
    GRUConfig,
    GeometricConfig,
    IntensityConfig,
    LoggingConfig,
    ModelConfig,
    PasteConfig,
    PathsConfig,
    SamplerConfig,
    TrainingConfig,
    VizLoggingConfig,
    WandbConfig,
)

experiment = ExperimentConfig(
    uuid="00000000-0000-4000-8000-00000000e2e7",
    name="e2e-testing",
    description="2-epoch × 1000-sample end-to-end gate for logging + W&B + holdout.",
    tags={"phase": "e2e-test"},
    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(),
    training=TrainingConfig(
        max_epochs=2,
        batch_size=4,
        num_workers=4,
        base_lr=2e-4,
        warmup_epochs=0,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=1000,
        deep_eval_start_epoch=99,  # disabled — only 2 epochs
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_max=4),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(epochs=1),
    eval=EvalConfig(use_gru=False, bootstrap_n=50),
    logging=LoggingConfig(
        file=FileLoggingConfig(level_console="INFO", level_file="DEBUG"),
        wandb=WandbConfig(
            enabled=True,
            mode="online",
            experiment_name="e2e-testing",
            run_name="run1",
            upload_checkpoints=False,
            upload_eval_reports=True,
            upload_viz_artifacts=True,
            upload_hard_pool_snapshots=False,
        ),
        viz=VizLoggingConfig(
            log_during_training=True,
            log_every_n_epochs=1,
            n_train_predictions_logged=4,
            sample_tp_per_fold=20,
            sample_fp_per_fold=20,
            sample_fn_per_fold=20,
        ),
        aug=AugLoggingConfig(log_samples="epoch0", n_samples=4),
    ),
    seed=42,
)
