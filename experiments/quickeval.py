"""Short partial training run for evaluation-pipeline testing.

5 epochs × 1000 samples/epoch on the full fold-0 split. Produces a real
``best.ckpt`` and at least one ``deep_eval/epoch{n}_val.npz`` so Components
6.5 (GRU rescorer), 7 (eval), 8 (viz) can be validated end-to-end with real
detector outputs. NOT a production training run — final 5-fold runs use
``baseline_rtmdet_p2.py``.
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
    uuid="00000000-0000-4000-8000-000000000002",
    name="quickeval-rtmdet-p2",
    description=(
        "## Quick Eval-Test Run\n"
        "Real fold 0, 5 epochs × 1000 samples/epoch. Produces a checkpoint "
        "and deep_eval cache files for testing Components 6.5/7/8.\n\n"
        "NOT for production. Use baseline_rtmdet_p2.py for final runs."
    ),
    tags={"phase": "quickeval", "head": "rtmdet", "backbone": "convnext_tiny", "p2": "true"},

    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(),  # default RTMDet head
    training=TrainingConfig(
        max_epochs=5,
        batch_size=4,                  # A10 24 GB; matches smoke-validated config
        num_workers=4,
        base_lr=2e-4,
        min_lr=1e-6,
        weight_decay=0.05,
        warmup_epochs=1,
        aux_seg_weight=0.3,
        ema_decay=0.99,                # short decay for short run
        precision="32-true",  # fp32 to dodge bf16 NaN sensitivity in this short run
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=1000,
        pos_frac_start=0.50,
        pos_frac_end=0.30,
        decay_epochs=4,
        neg_in_pos_vol_share=0.50,
        hard_pool_substitution_rate=0.30,
        hard_pool_start_epoch=2,
        deep_eval_refresh_every_epochs=2,
        deep_eval_start_epoch=2,
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_sigma=1.0, n_paste_max=4),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(epochs=5, lr=1e-3, weight_decay=0.01),
    eval=EvalConfig(use_gru=False, bootstrap_n=200, bootstrap_seed=42),
    seed=42,
)
