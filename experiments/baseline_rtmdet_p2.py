"""Week-1 production baseline.

ConvNeXt-tiny backbone + custom 4-level FPN with P2 + vendored RTMDet head + aux seg head.
Lesion copy-paste augmentation (p=0.5, multi-paste). Stage-1 detector + Stage-2 GRU.
Target: volume AUROC ≥ 0.80, sens@2FP ≥ 0.70 on patient-level 5-fold CV.
"""

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
    uuid="b3a7f1e9-4c8a-4d2b-9f1c-0e6a8b9c1d2e",
    name="baseline-rtmdet-p2",
    description=(
        "Week-1 production baseline. RTMDet-S head + ConvNeXt-tiny backbone + "
        "4-level FPN with P2 + aux seg head. Lesion copy-paste augmentation "
        "(p=0.5). 5-fold CV. GRU rescorer."
    ),
    tags={"phase": "1", "head": "rtmdet", "backbone": "convnext_tiny", "p2": "true"},

    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(
        backbone_name="convnext_tiny.fb_in22k",
        in_channels=5,
        fpn_channels=256,
        fpn_strides=(4, 8, 16, 32),
        head_n_classes=1,
        head_stacked_convs=2,
        aux_seg_channels=64,
    ),
    training=TrainingConfig(
        max_epochs=60,
        batch_size=8,
        num_workers=8,
        base_lr=2e-4,
        min_lr=1e-6,
        weight_decay=0.05,
        warmup_epochs=1,
        aux_seg_weight=0.3,
        ema_decay=0.999,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=6000,
        pos_frac_start=0.50,
        pos_frac_end=0.25,
        decay_epochs=30,
        neg_in_pos_vol_share=0.50,
        hard_pool_substitution_rate=0.30,
        hard_pool_start_epoch=5,
        deep_eval_refresh_every_epochs=10,
        deep_eval_start_epoch=10,
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_sigma=1.0, n_paste_max=7),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(
        input_dim=768, hidden_dim=128, bidirectional=True, dropout_input=0.3,
        epochs=20, lr=1e-3, weight_decay=0.01,
    ),
    eval=EvalConfig(
        use_gru=True,
        bootstrap_n=1000,
        bootstrap_seed=42,
        large_threshold_grid=[0.01, 0.03, 0.05, 0.10],
        small_threshold_grid=[0.10, 0.20, 0.30, 0.40, 0.50],
    ),
    seed=42,
)
