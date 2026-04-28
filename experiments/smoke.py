"""Tiny config used by the smoke training script (5 volumes, 2 epochs, ~5 min)."""

from __future__ import annotations

from pathlib import Path

from endo.config import (
    AugmentationConfig,
    EvalConfig,
    ExperimentConfig,
    GRUConfig,
    GeometricConfig,
    IntensityConfig,
    ModelConfig,
    PasteConfig,
    PathsConfig,
    SamplerConfig,
    TrainingConfig,
)

experiment = ExperimentConfig(
    uuid="00000000-0000-4000-8000-000000000001",
    name="smoke",
    description="Smoke test for the integration gate. 5-volume subset, 2 epochs.",
    tags={"phase": "smoke"},

    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(),  # full default RTMDet head
    training=TrainingConfig(
        max_epochs=2,
        batch_size=4,
        num_workers=2,
        base_lr=2e-4,
        warmup_epochs=0,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=100,
        deep_eval_start_epoch=99,  # disable for smoke
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.3, n_paste_max=2),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(epochs=2),
    eval=EvalConfig(use_gru=False, bootstrap_n=10),
    seed=42,
)
