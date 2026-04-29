"""Evaluation configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    use_gru: bool = True

    bootstrap_n: int = 1000
    bootstrap_seed: int = 42

    # Box-size-dependent post-WBF threshold grids.
    large_threshold_grid: list[float] = Field(default_factory=lambda: [0.01, 0.03, 0.05, 0.10])
    small_threshold_grid: list[float] = Field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50])

    # Box-size split (max dim in mm).
    box_size_split_mm: float = 10.0

    # WBF parameters.
    wbf_iou_threshold: float = 0.4
    wbf_skip_box_threshold: float = 0.001

    # FROC sensitivity points.
    froc_fp_points: list[float] = Field(default_factory=lambda: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

    # Eval batch size for inference passes.
    inference_batch_size: int = 16

    # Stratification keys (always evaluated when present in manifest).
    stratify_keys: list[str] = Field(
        default_factory=lambda: ["scanner_model", "variant", "slice_thickness_bin"]
    )

    # Final eval reads a fresh ckpt per fold (best or last). The CV path no
    # longer consumes the deep_eval npz cache — that cache is training-time
    # only.
    eval_ckpt: Literal["best", "last"] = "best"

    # Emit per-call (TP/FP/FN) JSONL alongside metrics.
    emit_call_jsonl: bool = True
