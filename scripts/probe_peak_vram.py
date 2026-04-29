"""Measure peak training-step VRAM for the production config.

One-off probe: instantiate `LesionDetector` at the baseline_rtmdet_p2 settings,
run a forward+backward at the configured batch_size with bf16-mixed autocast,
and report `torch.cuda.max_memory_allocated/reserved`. Also probes the EMA
shadow copy and validation-mode forward pass to capture peaks across the full
training cycle.

Run:
    uv run -m scripts.probe_peak_vram

Why a probe rather than analytic estimate: ConvNeXt-tiny activations + 4-level
FPN with P2 + RTMDet head + aux seg head + bf16 autocast intermediates are
hard to estimate within ~30%; PyTorch caching allocator and cudnn workspace
add another fudge factor. A direct forward+backward gives the real peak.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from timm.utils import ModelEmaV3

from endo.config import ExperimentConfig
from endo.config.loader import load_experiment
from endo.model.detector import LesionDetector
from endo.model.losses import compute_total_loss


def _summarize() -> dict:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def _build_synthetic_batch(batch_size: int, in_channels: int, hw: int, device: torch.device):
    x = torch.randn(batch_size, in_channels, hw, hw, device=device)
    boxes = [
        torch.tensor([[80.0, 80.0, 200.0, 200.0]], device=device)
        for _ in range(batch_size)
    ]
    labels = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(batch_size)]
    mask_center = torch.zeros(batch_size, 1, hw, hw, device=device)
    mask_center[:, :, 80:200, 80:200] = 1.0
    return x, boxes, labels, mask_center


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/baseline_rtmdet_p2.py")
    ap.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    ap.add_argument("--precision", default=None, help="Override training.precision (e.g. bf16-mixed, 32-true)")
    ap.add_argument("--steps", type=int, default=3, help="Forward/backward steps to run for steady-state peak")
    ap.add_argument("--output", default="agent/vram_probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VRAM probe")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    cfg: ExperimentConfig = load_experiment(Path(args.config))
    bs = args.batch_size or cfg.training.batch_size
    precision = args.precision or cfg.training.precision
    hw = 384  # network input is 2D 384x384; Z=160 is volume depth, not network input

    print(f"[probe] config={args.config} bs={bs} precision={precision} hw={hw}")
    print(f"[probe] in_channels={cfg.model.in_channels} fpn_channels={cfg.model.fpn_channels}")
    print(f"[probe] aux_seg_target_size={cfg.model.aux_seg_target_size}")
    print(f"[probe] ema_decay={cfg.training.ema_decay}")

    # 1. Pre-model baseline.
    baseline = _summarize()
    print(f"[probe] baseline (pre-model): {baseline}")

    # 2. Build model on GPU.
    model = LesionDetector(cfg.model).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[probe] params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M")
    after_model = _summarize()
    print(f"[probe] after model load: {after_model}")

    # 3. Optimizer (AdamW like the real LM uses; AdamW = ~2x param memory for moments).
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.base_lr,
        weight_decay=cfg.training.weight_decay,
    )
    after_optim = _summarize()
    print(f"[probe] after optimizer ctor: {after_optim}")

    # 4. EMA shadow (timm ModelEmaV3, fp32) — same as ema_callback.
    ema = ModelEmaV3(model, decay=cfg.training.ema_decay)
    for p in ema.module.parameters():
        p.requires_grad_(False)
    after_ema = _summarize()
    print(f"[probe] after EMA shadow init: {after_ema}")

    # 5. Training step(s) with autocast.
    use_autocast = precision in ("bf16-mixed", "16-mixed")
    autocast_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if precision == "16-mixed" else None

    train_peaks = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        x, boxes, labels, mask_center = _build_synthetic_batch(bs, cfg.model.in_channels, hw, device)
        optim.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                cls_scores, bbox_preds, aux_seg_logits = model(x)
                det_losses = model.head.loss(
                    cls_scores, bbox_preds,
                    gt_boxes_per_image=boxes,
                    gt_labels_per_image=labels,
                    image_size=(hw, hw),
                )
                total, _ = compute_total_loss(
                    det_losses, aux_seg_logits, mask_center,
                    aux_seg_weight=cfg.training.aux_seg_weight,
                )
        else:
            cls_scores, bbox_preds, aux_seg_logits = model(x)
            det_losses = model.head.loss(
                cls_scores, bbox_preds,
                gt_boxes_per_image=boxes,
                gt_labels_per_image=labels,
                image_size=(hw, hw),
            )
            total, _ = compute_total_loss(
                det_losses, aux_seg_logits, mask_center,
                aux_seg_weight=cfg.training.aux_seg_weight,
            )

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            scaler.step(optim)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            optim.step()

        # EMA update.
        ema.update(model)

        peak = _summarize()
        train_peaks.append(peak)
        print(f"[probe] train step {step}: loss={float(total):.4f} peak={peak}")

    # 6. Validation: model.eval(), no_grad, forward only — measure separately.
    del x, boxes, labels, mask_center, total
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()

    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                x = torch.randn(bs, cfg.model.in_channels, hw, hw, device=device)
                _ = model(x)
        else:
            x = torch.randn(bs, cfg.model.in_channels, hw, hw, device=device)
            _ = model(x)
    val_peak = _summarize()
    print(f"[probe] val peak: {val_peak}")

    # Concurrent train+EMA peak is what matters for sizing — pick max across train steps.
    train_peak_max = max((p["max_reserved_mib"] for p in train_peaks), default=0.0)

    out = {
        "config": args.config,
        "batch_size": bs,
        "precision": precision,
        "in_channels": cfg.model.in_channels,
        "input_hw": hw,
        "params_M": n_params / 1e6,
        "trainable_M": n_trainable / 1e6,
        "baseline": baseline,
        "after_model": after_model,
        "after_optimizer": after_optim,
        "after_ema": after_ema,
        "train_peaks": train_peaks,
        "train_peak_max_reserved_mib": train_peak_max,
        "val_peak": val_peak,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[probe] wrote {out_path}")
    print(f"[probe] PEAK (train, reserved) = {train_peak_max:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
