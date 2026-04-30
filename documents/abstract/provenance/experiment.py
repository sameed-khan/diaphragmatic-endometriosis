"""Retry of `baseline_rtmdet_p2` with fp32 + 5-epoch budget.

Why this exists: the original `baseline_rtmdet_p2` run (run_dir
`runs/baseline-rtmdet-p2_b3a7f1e9/`) used `precision="16-mixed"` (after
`bf16-mixed` proved totally unstable on launch). Two of five folds (3, 4)
diverged from fp16 forward-overflow in the cls / aux-seg heads, and the
three surviving folds (0, 1, 2) overfit hard — peak val/slice_auroc
hit at epoch 3-6 (0.927-0.944) and declined ~0.13-0.18 AUROC by epoch 36.

This retry config:
  * `precision="32-true"` — eliminates the fp16 forward-overflow failure mode
    permanently.
  * `max_epochs=5` — best-ckpt of the original run was saved at epochs 3-6;
    going past epoch ~7 only burns compute.
  * `deep_eval_start_epoch=2`, `deep_eval_refresh_every_epochs=1` — produce
    deep-eval npz cache every epoch from epoch 2 onward (skip epoch 0/1 which
    are during warmup).
  * `hard_pool_start_epoch=1` — start hard-negative-mining substitution as
    early as possible given the truncated budget.
  * Logging W&B group `baseline-cv-retry` so these runs show up alongside
    the original `baseline-cv` group on the dashboard for comparison.
  * All other hyperparameters identical to `baseline_rtmdet_p2.py` so any
    folds we re-train here are drop-in replacements for that experiment's
    fold checkpoints.
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
    LoggingConfig,
    ModelConfig,
    PasteConfig,
    PathsConfig,
    SamplerConfig,
    TrainingConfig,
    WandbConfig,
)

experiment = ExperimentConfig(
    uuid="d25975e4-a1b1-484e-8b38-dec75048f283",
    name="baseline-rtmdet-p2-retry",
    description=(
        "Retry of baseline_rtmdet_p2 with precision=32-true and max_epochs=5. "
        "Used to (a) re-train folds 3 and 4 cleanly after fp16 divergence in "
        "the original run, and (b) host a 5-fold ensemble whose other 3 fold "
        "ckpts are copied in from the original run. deep_eval and HNM run "
        "every epoch (after warmup) for visibility."
    ),
    tags={"phase": "1", "head": "rtmdet", "backbone": "convnext_tiny",
          "p2": "true", "precision": "fp32", "purpose": "retry"},

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
        max_epochs=5,
        batch_size=8,
        num_workers=8,
        base_lr=2e-4,
        min_lr=1e-6,
        weight_decay=0.05,
        warmup_epochs=1,
        aux_seg_weight=0.3,
        ema_decay=0.999,
        precision="32-true",
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
        hard_pool_start_epoch=1,
        deep_eval_refresh_every_epochs=1,
        deep_eval_start_epoch=2,
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
    logging=LoggingConfig(
        wandb=WandbConfig(enabled=True, group="baseline-cv-retry"),
    ),
    seed=42,
)
