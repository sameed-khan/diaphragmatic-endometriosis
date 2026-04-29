"""Component 6 unit tests (M1-M15, minus M8/M.INT).

These run on synthetic 384x384 inputs; no cohort access. Where the spec
calls for a "small synthetic 64x64", we still use 384x384 because the
detector head + aux seg + FPN all assume strides 4..32 producing power-of-2
feature maps; 64x64 doesn't divide by 32 cleanly into the head's prior
grid. The forward cost is dominated by ConvNeXt-tiny which is fine on CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from endo.config import ModelConfig
from endo.model.assigner import DynamicSoftLabelAssigner
from endo.model.aux_seg_head import AuxSegHead
from endo.model.detector import LesionDetector
from endo.model.fpn import FPN
from endo.model.losses import compute_total_loss, dice_bce_loss
from endo.model.rtmdet_head import RTMDetHead

from .conftest import make_synthetic_batch


# ---------------------------------------------------------------------------
# M1 - backbone accepts 5-channel input and produces 4 stages at correct strides
# ---------------------------------------------------------------------------
def test_M1_backbone_5ch_input():
    backbone = timm.create_model(
        "convnext_tiny.fb_in22k",
        pretrained=False,
        in_chans=5,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    x = torch.randn(1, 5, 384, 384)
    feats = backbone(x)
    assert len(feats) == 4
    expected_strides = [4, 8, 16, 32]
    expected_chs = [96, 192, 384, 768]
    for f, s, c in zip(feats, expected_strides, expected_chs):
        assert f.shape[0] == 1
        assert f.shape[1] == c, f"got {f.shape[1]}, want {c}"
        assert f.shape[2] == 384 // s and f.shape[3] == 384 // s


# ---------------------------------------------------------------------------
# M2 - conv1 5-channel renormalization is sane
# ---------------------------------------------------------------------------
def test_M2_conv1_renormalization():
    cfg = ModelConfig()  # 5-channel default.
    model = LesionDetector(cfg)
    in_conv = LesionDetector._find_input_conv(model.backbone)
    assert in_conv is not None
    assert in_conv.weight.shape[1] == 5
    # Build a 3-channel reference and check the magnitude relationship.
    ref = timm.create_model(
        cfg.backbone_name, pretrained=True, in_chans=3,
        features_only=True, out_indices=(0, 1, 2, 3),
    )
    ref_conv = LesionDetector._find_input_conv(ref)
    expected = float(ref_conv.weight.abs().mean()) * (3.0 / 5.0)
    actual = float(in_conv.weight.abs().mean())
    # Allow up to 25% drift (covers timm's per-channel scaling vs the doc spec).
    assert 0.75 <= actual / expected <= 1.25, (actual, expected)


# ---------------------------------------------------------------------------
# M3 - FPN preserves spatial sizes per level and hits target channels
# ---------------------------------------------------------------------------
def test_M3_fpn_output_shapes():
    fpn = FPN(in_channels=[96, 192, 384, 768], out_channels=256)
    feats = [
        torch.randn(2, 96, 96, 96),
        torch.randn(2, 192, 48, 48),
        torch.randn(2, 384, 24, 24),
        torch.randn(2, 768, 12, 12),
    ]
    outs = fpn(feats)
    assert len(outs) == 4
    expected_hw = [96, 48, 24, 12]
    for o, hw in zip(outs, expected_hw):
        assert o.shape == (2, 256, hw, hw)


# ---------------------------------------------------------------------------
# M4 - AuxSegHead produces stride-1 output
# ---------------------------------------------------------------------------
def test_M4_aux_seg_head_output_stride1():
    head = AuxSegHead(in_channels=256, mid_channels=64)
    p2 = torch.randn(2, 256, 96, 96)
    y = head(p2)
    assert y.shape == (2, 1, 384, 384)


# ---------------------------------------------------------------------------
# M5 - RTMDet head forward shapes
# ---------------------------------------------------------------------------
def test_M5_rtmdet_head_forward_shapes():
    head = RTMDetHead(
        num_classes=1, in_channels=256, feat_channels=256,
        stacked_convs=2, strides=(4, 8, 16, 32), share_conv=False,
    )
    feats = [
        torch.randn(2, 256, 96, 96),
        torch.randn(2, 256, 48, 48),
        torch.randn(2, 256, 24, 24),
        torch.randn(2, 256, 12, 12),
    ]
    cls_scores, bbox_preds = head(feats)
    expected_hw = [96, 48, 24, 12]
    for c, b, hw in zip(cls_scores, bbox_preds, expected_hw):
        assert c.shape == (2, 1, hw, hw)
        assert b.shape == (2, 4, hw, hw)


# ---------------------------------------------------------------------------
# M6 - RTMDet head loss returns finite numbers on synthetic GT
# ---------------------------------------------------------------------------
def test_M6_rtmdet_head_loss_smoke():
    head = RTMDetHead(
        num_classes=1, in_channels=256, feat_channels=256,
        stacked_convs=2, strides=(4, 8, 16, 32),
    )
    feats = [
        torch.randn(2, 256, 96, 96),
        torch.randn(2, 256, 48, 48),
        torch.randn(2, 256, 24, 24),
        torch.randn(2, 256, 12, 12),
    ]
    cls_scores, bbox_preds = head(feats)
    gt_boxes = [
        torch.tensor([[100.0, 120.0, 200.0, 220.0]]),
        torch.zeros((0, 4)),
    ]
    gt_labels = [torch.zeros(1, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)]
    losses = head.loss(cls_scores, bbox_preds, gt_boxes, gt_labels, image_size=(384, 384))
    assert "loss_cls" in losses and "loss_bbox" in losses
    for v in losses.values():
        assert torch.isfinite(v).all()


# ---------------------------------------------------------------------------
# M7 - RTMDet head predict returns a sensible structure
# ---------------------------------------------------------------------------
def test_M7_rtmdet_head_predict_smoke():
    head = RTMDetHead(
        num_classes=1, in_channels=256, feat_channels=256,
        stacked_convs=2, strides=(4, 8, 16, 32),
    )
    feats = [
        torch.randn(2, 256, 96, 96),
        torch.randn(2, 256, 48, 48),
        torch.randn(2, 256, 24, 24),
        torch.randn(2, 256, 12, 12),
    ]
    cls_scores, bbox_preds = head(feats)
    # Force a few high-confidence cells so we exit the empty branch.
    cls_scores[0][:, :, :3, :3] = 5.0
    preds = head.predict(cls_scores, bbox_preds, image_size=(384, 384), score_threshold=0.05)
    assert len(preds) == 2
    for p in preds:
        assert "boxes" in p and "scores" in p and "labels" in p
        assert p["boxes"].shape[1] == 4


# ---------------------------------------------------------------------------
# M8 - Assigner smoke (no mmdet parity available)
# ---------------------------------------------------------------------------
def test_M8_assigner_smoke():
    """Without mmdet installed, ensure the vendored assigner runs and returns
    sane-shape outputs on a fixed synthetic input.
    """
    assigner = DynamicSoftLabelAssigner()
    P = 50
    pred_scores = torch.full((P, 1), -3.0)
    priors = torch.zeros((P, 4))
    priors[:, 0] = torch.linspace(10, 370, P)  # x
    priors[:, 1] = torch.linspace(10, 370, P)  # y
    priors[:, 2] = 8.0
    priors[:, 3] = 8.0
    decoded = torch.zeros((P, 4))
    decoded[:, 0] = priors[:, 0] - 16
    decoded[:, 1] = priors[:, 1] - 16
    decoded[:, 2] = priors[:, 0] + 16
    decoded[:, 3] = priors[:, 1] + 16
    gt_bboxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    gt_labels = torch.tensor([0], dtype=torch.int64)
    out = assigner(pred_scores=pred_scores, priors=priors,
                   decoded_bboxes=decoded, gt_bboxes=gt_bboxes,
                   gt_labels=gt_labels)
    assert out.gt_inds.shape == (P,)
    assert out.labels.shape == (P,)


# ---------------------------------------------------------------------------
# M9 - dice_bce_loss is near-zero for a perfect prediction
# ---------------------------------------------------------------------------
def test_M9_dice_bce_loss_zero_for_perfect():
    target = torch.zeros((2, 16, 16), dtype=torch.uint8)
    target[:, 4:12, 4:12] = 1
    # Use saturated logits as a stand-in for "perfect": +20 where target=1,
    # -20 elsewhere. Sigmoid(+/-20) ~= 1/0 to numerical precision.
    logits = torch.full_like(target, fill_value=-20, dtype=torch.float32)
    logits[target.bool()] = 20.0
    loss = dice_bce_loss(logits, target)
    assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# M10 - total = cls + bbox + 0.3 * aux_seg
# ---------------------------------------------------------------------------
def test_M10_total_loss_aggregates_correctly():
    det = {
        "loss_cls": torch.tensor(2.0, requires_grad=True),
        "loss_bbox": torch.tensor(1.5, requires_grad=True),
    }
    target = torch.zeros((1, 16, 16), dtype=torch.uint8)
    target[:, 4:12, 4:12] = 1
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    total, components = compute_total_loss(det, logits, target, aux_seg_weight=0.3)
    expected = 2.0 + 1.5 + 0.3 * float(components["loss_aux_seg"])
    assert math.isclose(total.item(), expected, rel_tol=1e-5)
    for k in ("loss_cls", "loss_bbox", "loss_aux_seg", "loss_total"):
        assert k in components


# ---------------------------------------------------------------------------
# M11 - LightningModule training_step end-to-end
# ---------------------------------------------------------------------------
def test_M11_lightning_module_training_step_smoke(exp_cfg):
    from endo.lightning_module import LesionDetectorLM
    lm = LesionDetectorLM(exp_cfg)
    batch = make_synthetic_batch(B=2, n_pos=1)
    out = lm.training_step(batch, batch_idx=0)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# M12 - validation_step + on_validation_epoch_end logs slice_auroc
# ---------------------------------------------------------------------------
def test_M12_lightning_module_validation_step_smoke(exp_cfg):
    from endo.lightning_module import LesionDetectorLM
    lm = LesionDetectorLM(exp_cfg)
    # Stub Lightning's logging so .log() is a no-op recorder.
    logged: dict[str, float] = {}

    def fake_log(name, value, *a, **kw):
        logged[name] = float(value.item()) if isinstance(value, torch.Tensor) else float(value)

    lm.log = fake_log  # type: ignore[assignment]
    lm.on_validation_epoch_start()
    batch = make_synthetic_batch(B=4, n_pos=2)
    lm.validation_step(batch, batch_idx=0)
    lm.on_validation_epoch_end()
    assert "val/slice_auroc" in logged


# ---------------------------------------------------------------------------
# M14 - EMA callback swap-and-restore round-trip preserves live weights
# ---------------------------------------------------------------------------
def test_M14_ema_callback_swap_swap_back(exp_cfg):
    from endo.ema_callback import EmaCallback
    from endo.lightning_module import LesionDetectorLM

    lm = LesionDetectorLM(exp_cfg)
    cb = EmaCallback(decay=0.999)
    cb._init_ema(lm)
    # Take a snapshot of one parameter pre-swap.
    name = next(iter(lm.model.state_dict()))
    pre = lm.model.state_dict()[name].detach().clone()

    # Mutate EMA buffer so the swap is observable.
    with torch.no_grad():
        for p in cb.ema.module.parameters():
            p.data.add_(1.0)

    cb.on_validation_epoch_start(trainer=None, pl_module=lm)  # type: ignore[arg-type]
    # Inside validation, params should differ from pre-swap.
    mid = lm.model.state_dict()[name].detach().clone()
    cb.on_validation_epoch_end(trainer=None, pl_module=lm)  # type: ignore[arg-type]
    post = lm.model.state_dict()[name].detach().clone()

    if pre.is_floating_point():
        # Check at least *some* parameter changed during swap.
        any_diff = False
        for k in lm.model.state_dict():
            v_pre = pre if k == name else None
            v_mid = mid if k == name else None
            if v_pre is not None and v_mid is not None and v_pre.is_floating_point():
                any_diff = bool(not torch.allclose(v_pre, v_mid))
                break
        assert any_diff or pre.numel() == 0
    assert torch.equal(pre, post), "live state not restored after on_validation_epoch_end"


# ---------------------------------------------------------------------------
# Audit 2026-04-29 — external swap_to_ema / restore_live round-trip
# ---------------------------------------------------------------------------
def test_ema_callback_external_swap_round_trip(exp_cfg):
    from endo.ema_callback import EmaCallback
    from endo.lightning_module import LesionDetectorLM

    lm = LesionDetectorLM(exp_cfg)
    cb = EmaCallback(decay=0.999)
    cb._init_ema(lm)

    # Mutate EMA shadow so swap is observable.
    with torch.no_grad():
        for p in cb.ema.module.parameters():
            p.data.add_(2.0)

    name = next(iter(lm.model.state_dict()))
    pre = lm.model.state_dict()[name].detach().clone()

    assert cb.swap_to_ema(lm) is True
    assert cb._is_swapped is True
    # Idempotent — second swap is a no-op.
    assert cb.swap_to_ema(lm) is False

    mid = lm.model.state_dict()[name].detach().clone()
    if pre.is_floating_point() and pre.numel() > 0:
        assert not torch.allclose(pre, mid), "external swap did not modify live state"

    assert cb.restore_live() is True
    assert cb._is_swapped is False
    post = lm.model.state_dict()[name].detach().clone()
    assert torch.equal(pre, post), "live state not restored after restore_live"


# ---------------------------------------------------------------------------
# M15 - Warmup linear -> cosine LR schedule shape
# ---------------------------------------------------------------------------
def test_M15_warmup_cosine_lr_schedule(exp_cfg):
    from endo.lightning_module import LesionDetectorLM

    lm = LesionDetectorLM(exp_cfg)

    # Simulate a Trainer by injecting estimated_stepping_batches.
    class _StubTrainer:
        estimated_stepping_batches = 100

    lm.trainer = _StubTrainer()  # type: ignore[assignment]

    cfg = lm.configure_optimizers()
    optim = cfg["optimizer"]
    sched = cfg["lr_scheduler"]["scheduler"]
    base_lr = exp_cfg.training.base_lr
    min_lr = exp_cfg.training.min_lr

    # max_epochs = 2, warmup_epochs = 1 -> warmup_steps = 50, total = 100.
    # Snapshot the LR at the start of each step, then advance the scheduler.
    lrs: list[float] = []
    for step in range(101):
        lrs.append(optim.param_groups[0]["lr"])
        optim.step = lambda *a, **k: None  # avoid touching state_dict
        sched.step()

    # Step 0 -> 0 (linear warmup at step 0).
    assert lrs[0] < base_lr * 0.05
    # End of warmup is the step *after* warmup_steps - 1 increments, i.e.
    # lrs[warmup_steps] should be ~= base_lr (LR before step ``warmup_steps``).
    end_warmup_lr = lrs[51]
    assert abs(end_warmup_lr - base_lr) < base_lr * 0.05, (end_warmup_lr, base_lr)
    # End of training near min_lr.
    assert abs(lrs[-1] - min_lr) < base_lr * 0.05
