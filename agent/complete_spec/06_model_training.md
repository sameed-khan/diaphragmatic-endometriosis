# Component 6 — Model + LightningModule + Training Loop

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/model.py`, `src/rtmdet_head.py`, `src/assigner.py`, `src/aux_seg_head.py`, `src/losses.py`, `src/lightning_module.py`, `src/ema_callback.py`, `train.py`
**Date:** 2026-04-27
**Companion:** Implements §3, §4, §7 of `agent/training_pipeline_decisions_phase1.md`. Consumes Components 3–5 (DataModule, Augmentation, Sampler/HNM); produces inference cache via the Component 5 callback that Component 7 consumes.

---

## 1. Purpose

Define the model (ConvNeXt-tiny + custom 4-level FPN with P2 + vendored RTMDet head + aux UNet seg head), the training loop (Lightning module + AdamW/cosine/EMA/bf16), and the entrypoint (`train.py`) for a single fold. Consume the data layer below and produce the deep-eval cache + checkpoints that Component 7 reads.

---

## 2. Scope

**In scope:**

- Backbone construction with conv1 5-channel surgery (timm built-in + verification override).
- Custom 4-level FPN producing features at strides {4, 8, 16, 32}.
- Vendored RTMDet head + DynamicSoftLabelAssigner — copied from MMDet, dependencies stripped.
- Auxiliary UNet decoder up to stride 1 with Dice+BCE supervision.
- LightningModule: training_step, validation_step, configure_optimizers, EMA wiring, ScoreEMATracker hook.
- EMA via timm's `ModelEmaV3` with fp32 shadow; weight swap at validation/inference.
- `train.py` — fold-aware entrypoint with precheck + Lightning Trainer construction.
- Slice-level train-time metrics: slice-binary AUROC, mean per-slice IoU, per-loss component logging.
- Parity test for the vendored assigner against installed MMDet (sanity check on porting).

**Out of scope:**

- Volume-level metrics, FROC, AP, bootstrap CIs — Component 7.
- Post-training final eval — Component 7.
- GRU rescorer — Component 6.5.
- Smoke test + viz — Component 8.

---

## 3. Coordinate convention (locked at model boundary)

Per Component 4 §9 update:

- Tensor input: `(B, 5, H=Z=384, W=X=384)` — anatomical Z (I-S) is the vertical PyTorch H axis; X (R-L) is W.
- 5-channel axis is dim 1 (channel position).
- Box format throughout: `(x1, z1, x2, z2)` ≡ `(W_min, H_min, W_max, H_max)`. **No permutation between dataloader and detector head.**
- Single class: `n_classes = 1` (lesion). Class label always `0`.

---

## 4. Model assembly

```python
# src/model.py

@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str = "convnext_tiny.fb_in22k"
    in_channels: int = 5
    fpn_channels: int = 256
    fpn_strides: tuple[int, ...] = (4, 8, 16, 32)   # P2, P3, P4, P5
    head_n_classes: int = 1
    head_share_conv: bool = False
    head_stacked_convs: int = 2
    head_feat_channels: int = 256
    aux_seg_channels: int = 64

class LesionDetector(nn.Module):
    """Composed: backbone → FPN → (RTMDet head, aux seg head)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = self._build_backbone(cfg)
        self.fpn = FPN(in_channels_list=self._backbone_channels(), out_channels=cfg.fpn_channels)
        self.head = RTMDetHead(
            num_classes=cfg.head_n_classes,
            in_channels=cfg.fpn_channels,
            feat_channels=cfg.head_feat_channels,
            stacked_convs=cfg.head_stacked_convs,
            strides=cfg.fpn_strides,
            share_conv=cfg.head_share_conv,
        )
        self.aux_seg_head = AuxSegHead(
            in_channels=cfg.fpn_channels,
            mid_channels=cfg.aux_seg_channels,
        )

    def _build_backbone(self, cfg) -> nn.Module:
        m = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            in_chans=cfg.in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),   # strides 4, 8, 16, 32
        )
        # Verify timm's conv1 5-channel surgery matches doc spec
        self._verify_conv1_renormalization(m, cfg.in_channels)
        return m

    def _verify_conv1_renormalization(self, model, in_chans):
        """Doc spec: new_w = pretrained.repeat(1, 2, 1, 1)[:, :5] * (3/5).
           timm default: replicates and scales by 3/in_chans. Should match for in_chans=5.
           If verification fails, override with the doc-specified surgery."""
        # Implementation: load fresh 3ch model, compare conv1 weight ratios.
        ...

    def forward_features(self, x):
        """Returns FPN feature pyramid, used by both heads."""
        feats = self.backbone(x)
        feats_pyramid = self.fpn(feats)
        return feats_pyramid

    def forward(self, x):
        feats_pyramid = self.forward_features(x)
        return feats_pyramid   # heads called separately by LightningModule
```

### 4.1 FPN

```python
# src/model.py (or src/fpn.py)

class FPN(nn.Module):
    """Top-down 4-level FPN with lateral 1×1 + 3×3 smoothing.
       Strides {4, 8, 16, 32} = ConvNeXt-tiny's out_indices (0,1,2,3)."""

    def __init__(self, in_channels_list: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        # feats[0] = stride 4, feats[3] = stride 32
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        # Top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
        outs = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        return outs   # [P2, P3, P4, P5]
```

### 4.2 Aux seg head (stride 1, per Q5 (b))

```python
# src/aux_seg_head.py

class AuxSegHead(nn.Module):
    """Lightweight UNet-style decoder.
       Inputs: P2 (stride 4) + P3, P4 (skip-connected through upsample).
       Output: (B, 1, 384, 384) sigmoid logits at stride 1.
       ~5 transposed-conv stages: stride 4 → 2 → 1, plus 2 lateral integrations."""

    def __init__(self, in_channels: int = 256, mid_channels: int = 64):
        super().__init__()
        # Decoder takes P2 (stride 4) and upsamples 4× to stride 1 via 2 transpose convs.
        # Optional skip integration from earlier-stride feats omitted for simplicity (P2 is already finest).
        self.up1 = nn.ConvTranspose2d(in_channels, mid_channels, 4, stride=2, padding=1)  # 4 → 2
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, 4, stride=2, padding=1) # 2 → 1
        self.norm1 = nn.GroupNorm(8, mid_channels)
        self.norm2 = nn.GroupNorm(8, mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, fpn_outs: list[torch.Tensor]) -> torch.Tensor:
        x = fpn_outs[0]   # P2, stride 4
        x = self.act(self.norm1(self.up1(x)))   # → stride 2
        x = self.act(self.norm2(self.up2(x)))   # → stride 1
        x = self.out_conv(x)                     # (B, 1, 384, 384) logits
        return x
```

---

## 5. Vendored RTMDet head + assigner

### 5.1 Vendoring procedure

1. Copy `mmdet/models/dense_heads/rtmdet_head.py` → `src/rtmdet_head.py`.
2. Copy `mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py` → `src/assigner.py`.
3. Strip imports: `mmcv.cnn`, `mmengine.model`, `mmdet.registry`, `mmdet.utils`, `ConfigDict`. Replace with plain `torch.nn` equivalents.
4. Replace `BaseDenseHead` inheritance with `nn.Module`. Implement only `forward`, `loss_by_feat`, `predict_by_feat` directly.
5. Replace `BBoxOverlaps2D` with `torchvision.ops.box_iou` or hand-rolled CIoU helper.
6. Strip the `with_objectness` branch — we don't use it.
7. Strip integration with `mmcv` config — pass plain Python args.

Estimated post-strip LOC: ~600 across the two files.

### 5.2 Public API after vendoring

```python
class RTMDetHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int,
        strides: tuple[int, ...],
        share_conv: bool = False,
    ): ...

    def forward(self, feats: list[torch.Tensor]) -> tuple[list, list]:
        """Returns (cls_scores, bbox_preds) for each FPN level."""
        ...

    def loss(
        self,
        cls_scores: list[torch.Tensor],
        bbox_preds: list[torch.Tensor],
        gt_boxes_per_image: list[torch.Tensor],   # length B; each (N_i, 4) in (x1, z1, x2, z2)
        gt_labels_per_image: list[torch.Tensor],  # length B; each (N_i,)
        image_size: tuple[int, int],              # (H, W)
    ) -> dict[str, torch.Tensor]:
        """Returns {'loss_cls': ..., 'loss_bbox': ...}."""
        ...

    def predict(
        self,
        cls_scores: list[torch.Tensor],
        bbox_preds: list[torch.Tensor],
        image_size: tuple[int, int],
        score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.5,
        max_per_image: int = 100,
    ) -> list[dict]:
        """Returns per-image {'boxes': (N, 4), 'scores': (N,), 'labels': (N,)}."""
        ...
```

### 5.3 Assigner parity test (critical)

Before training begins:

```python
# tests/model/test_assigner_parity.py
def test_assigner_matches_mmdet():
    """Vendored DynamicSoftLabelAssigner must produce byte-identical outputs to mmdet.
       Catches porting bugs before any training cost is paid."""
    import mmdet
    from src.assigner import DynamicSoftLabelAssigner as VendoredAssigner

    # Construct identical fixed inputs (priors, gt_boxes, gt_labels, decoded_pred_boxes, cls_scores).
    ours = VendoredAssigner(...).assign(inputs)
    theirs = mmdet.models.task_modules.DynamicSoftLabelAssigner(...).assign(inputs)

    assert torch.equal(ours.gt_inds, theirs.gt_inds)
    assert torch.equal(ours.labels, theirs.labels)
    assert torch.allclose(ours.max_overlaps, theirs.max_overlaps, atol=1e-6)
```

If parity fails, do not train. Investigate the porting diff. This is the single highest-value test in Component 6 because assigner bugs are silent and devastating.

`mmdet` is added as a **dev dependency only** for this parity test; it is not imported in production code.

---

## 6. Loss composition

```python
# src/losses.py

def compute_total_loss(
    det_losses: dict[str, torch.Tensor],  # from RTMDetHead.loss
    aux_seg_logits: torch.Tensor,         # (B, 1, H, W)
    aux_seg_target: torch.Tensor,         # (B, H, W) uint8
    aux_seg_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    aux_seg_loss = dice_bce_loss(aux_seg_logits.squeeze(1), aux_seg_target.float())
    total = det_losses["loss_cls"] + det_losses["loss_bbox"] + aux_seg_weight * aux_seg_loss
    components = {
        "total_loss": total,
        "loss_cls": det_losses["loss_cls"].detach(),
        "loss_bbox": det_losses["loss_bbox"].detach(),
        "loss_aux_seg": aux_seg_loss.detach(),
    }
    return total, components

def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    probs = torch.sigmoid(logits)
    # Soft Dice
    intersection = (probs * target).sum(dim=(-2, -1))
    union = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    return bce + dice.mean()
```

`focal γ=1.5` is configured inside the vendored `RTMDetHead` (replaces MMDet's default). CIoU is the default RTMDet box regression loss.

---

## 7. LightningModule

```python
# src/lightning_module.py

@dataclass
class TrainingConfig:
    base_lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    max_epochs: int = 60
    aux_seg_weight: float = 0.3
    ema_decay: float = 0.999
    log_every_n_steps: int = 10

class LesionDetectorLM(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        score_ema_tracker: ScoreEMATracker,   # from Component 5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["score_ema_tracker"])
        self.model = LesionDetector(model_cfg)
        self.train_cfg = train_cfg
        self.score_ema_tracker = score_ema_tracker
        # Slice-level metrics (initialized in on_validation_start)
        self._val_slice_scores = []
        self._val_slice_labels = []
        self._val_slice_ious = []

    def training_step(self, batch, batch_idx):
        x = batch.volume_5ch                    # (B, 5, 384, 384)
        feats_pyramid = self.model.forward_features(x)
        cls_scores, bbox_preds = self.model.head(feats_pyramid)
        det_losses = self.model.head.loss(
            cls_scores, bbox_preds,
            gt_boxes_per_image=batch.boxes,
            gt_labels_per_image=batch.labels,
            image_size=(384, 384),
        )
        aux_seg_logits = self.model.aux_seg_head(feats_pyramid)
        total, components = compute_total_loss(
            det_losses, aux_seg_logits, batch.lesion_mask_center,
            self.train_cfg.aux_seg_weight,
        )

        # Log per-step
        for k, v in components.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "total_loss"))

        # Update ScoreEMATracker for HNM (Component 5)
        with torch.no_grad():
            preds = self.model.head.predict(cls_scores, bbox_preds, image_size=(384, 384))
            for i, (pid, sy, is_pos_slice) in enumerate(
                zip(batch.patient_ids, batch.slice_ys.tolist(), batch.is_positive_slice.tolist())
            ):
                if not is_pos_slice:
                    max_score = float(preds[i]["scores"].max()) if len(preds[i]["scores"]) > 0 else 0.0
                    self.score_ema_tracker.update((pid, sy), max_score, is_negative_slice=True)

        return total

    def on_validation_start(self):
        self._val_slice_scores.clear()
        self._val_slice_labels.clear()
        self._val_slice_ious.clear()

    def validation_step(self, batch, batch_idx):
        x = batch.volume_5ch
        feats_pyramid = self.model.forward_features(x)
        cls_scores, bbox_preds = self.model.head(feats_pyramid)
        det_losses = self.model.head.loss(cls_scores, bbox_preds, batch.boxes, batch.labels, (384, 384))
        aux_seg_logits = self.model.aux_seg_head(feats_pyramid)
        total, components = compute_total_loss(
            det_losses, aux_seg_logits, batch.lesion_mask_center, self.train_cfg.aux_seg_weight,
        )
        for k, v in components.items():
            self.log(f"val/{k}", v, on_epoch=True)

        # Slice-level metrics
        preds = self.model.head.predict(cls_scores, bbox_preds, image_size=(384, 384))
        for i in range(x.shape[0]):
            pred_boxes, pred_scores = preds[i]["boxes"], preds[i]["scores"]
            gt_boxes = batch.boxes[i]
            self._val_slice_scores.append(float(pred_scores.max()) if len(pred_scores) > 0 else 0.0)
            self._val_slice_labels.append(int(batch.is_positive_slice[i].item()))
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = box_iou(gt_boxes, pred_boxes)            # (N_gt, N_pred)
                self._val_slice_ious.append(float(ious.max(dim=1).values.mean()))
            elif len(gt_boxes) > 0:
                self._val_slice_ious.append(0.0)

    def on_validation_epoch_end(self):
        # Slice-binary AUROC
        if len(set(self._val_slice_labels)) > 1:
            auroc = roc_auc_score(self._val_slice_labels, self._val_slice_scores)
        else:
            auroc = 0.0
        self.log("val/slice_auroc", auroc, on_epoch=True, prog_bar=True)
        # Mean per-slice IoU (only over positive slices with predictions)
        if self._val_slice_ious:
            self.log("val/mean_per_slice_iou", float(np.mean(self._val_slice_ious)), on_epoch=True)

    def configure_optimizers(self):
        # Filter out norm/bias from weight decay (standard practice)
        decay_params, nodecay_params = self._split_weight_decay_params()
        optim = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.train_cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.train_cfg.base_lr,
        )
        # Warmup linear → cosine
        sched = WarmupCosineLR(
            optim,
            warmup_epochs=self.train_cfg.warmup_epochs,
            max_epochs=self.train_cfg.max_epochs,
            base_lr=self.train_cfg.base_lr,
            min_lr=self.train_cfg.min_lr,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.train_cfg.max_epochs,
        )
        return [optim], [{"scheduler": sched, "interval": "step"}]
```

---

## 8. EMA via callback

```python
# src/ema_callback.py

from timm.utils import ModelEmaV3

class EmaCallback(pl.Callback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema = None
        self._original_state = None

    def on_fit_start(self, trainer, pl_module):
        # fp32 shadow buffer (per research — bf16 EMA drifts after ~10k steps)
        self.ema = ModelEmaV3(
            pl_module.model,
            decay=self.decay,
            device=pl_module.device,
        )

    def on_train_batch_end(self, trainer, pl_module, *_args, **_kw):
        self.ema.update(pl_module.model)

    def on_validation_epoch_start(self, trainer, pl_module):
        # Swap to EMA weights for validation
        self._original_state = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.ema.module.state_dict())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Restore live weights
        if self._original_state is not None:
            pl_module.model.load_state_dict(self._original_state)
            self._original_state = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Persist EMA state alongside live weights
        checkpoint["ema_state_dict"] = self.ema.module.state_dict()
```

The deep-eval callback (Component 5) inherits the EMA-swap behavior because it runs inside `on_validation_epoch_end`, which fires after this callback's swap is reverted. To make deep eval also use EMA weights, wire the `PeriodicDeepEvalCallback` to *also* swap before its `inference_pass()` call. Implementation note in Component 5; will adjust spec there.

---

## 9. `train.py` — fold-aware entrypoint

```python
# train.py

@dataclass
class RunConfig:
    fold: int
    seed: int = 42
    output_dir: Path = Path("runs/baseline")
    cache_root: Path = Path("/scratch/.../cache/v1")
    splits_path: Path = Path("data/splits.json")
    wandb_project: str = "dia-endo"
    wandb_run_name: str | None = None
    # Sampler epoch-length mode (forwarded to SamplerConfig)
    epoch_mode: Literal["fixed_count", "full_pass"] = "fixed_count"
    samples_per_epoch: int = 6000   # only used when epoch_mode == "fixed_count"

def main(cfg: RunConfig):
    pl.seed_everything(cfg.seed, workers=True)

    _precheck(cfg)   # see §10

    # Build DataModule (Component 3) — wires Component 4 augmentation + Component 5 sampler
    score_ema = ScoreEMATracker()
    train_aug = TrainAugmentation(
        lesion_bank=load_lesion_bank(cfg.cache_root),
        paste_cfg=PasteConfig(),
        geom_cfg=GeometricConfig(),
        intensity_cfg=IntensityConfig(),
    )
    sampler = WeightedScheduledSampler.from_dataset_partitions(
        cfg=SamplerConfig(),
        loss_ema_tracker=score_ema,
        hard_pool_path=cfg.output_dir / "runtime/hard_negatives.json",
    )
    dm = LesionDataModule(
        cache_root=cfg.cache_root, splits_path=cfg.splits_path, fold=cfg.fold,
        batch_size=8, num_workers=8,
        augment_train=train_aug, sampler_train=sampler,
        allow_holdout=False,
    )

    # Build LightningModule
    lm = LesionDetectorLM(
        model_cfg=ModelConfig(),
        train_cfg=TrainingConfig(max_epochs=60),
        score_ema_tracker=score_ema,
    )

    # Build callbacks
    callbacks = [
        EmaCallback(decay=0.999),
        PeriodicDeepEvalCallback(
            cfg=PeriodicDeepEvalConfig(),
            datamodule=dm,
            train_negative_patient_ids=dm.train_negative_patient_ids,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=cfg.output_dir / "ckpts",
            filename="epoch{epoch:03d}-auroc{val/slice_auroc:.4f}",
            monitor="val/slice_auroc",
            mode="max",
            save_top_k=1,
            save_last=True,
            every_n_epochs=5,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Build Trainer
    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger(project=cfg.wandb_project, name=cfg.wandb_run_name, save_dir=cfg.output_dir),
        log_every_n_steps=10,
        deterministic=False,   # bf16 + EMA make full determinism impractical
        gradient_clip_val=1.0,
    )

    trainer.fit(lm, datamodule=dm)
```

CLI:

```bash
# Default: fixed_count, 6000 samples/epoch (~3 GPU-h per fold)
uv run python train.py --fold 0 --output-dir runs/baseline_fold0 --wandb-run-name fold0

# Override to full-pass mode (~40 min/epoch, ~40 GPU-h per fold) — for ablations or capacity tests
uv run python train.py --fold 0 --epoch-mode full_pass --output-dir runs/full_pass_fold0

# Override fixed-count to a different value
uv run python train.py --fold 0 --epoch-mode fixed_count --samples-per-epoch 8000
```

`--epoch-mode` and `--samples-per-epoch` are wired through `RunConfig` → `SamplerConfig`. When `epoch_mode == "full_pass"`, `--samples-per-epoch` is ignored. The default (`fixed_count`, 6000) is the recommended Stage-1 baseline.

---

## 10. Pre-flight checks (`_precheck`)

Before `trainer.fit`, verify:

1. **Cache integrity**: `cache/v1/preprocessed_manifest.csv`, `cache/v1/gt_boxes.parquet`, `cache/v1/lesion_banks/lesion_bank_*.pkl`, `cache/v1/border_bands/` all exist and non-empty.
2. **Splits consistency**: every `cohort='cross-validation'` patient in `splits.json` has a row in `preprocessed_manifest.csv` and vice versa; fold assignments match.
3. **Cache version match**: `cache/v1/code_version.txt` exists; warn if it differs from current `git rev-parse HEAD`.
4. **GPU + RAM available**: `torch.cuda.is_available()`, `torch.cuda.get_device_properties(0).total_memory >= 40 GB`, `psutil.virtual_memory().available >= 48 GB`.
5. **Assigner parity**: run the §5.3 parity test once at startup; refuse to start if it fails.

**No QC-signoff check.** Per user round 9, the QC review is recommended but not gated by `train.py`.

If any check fails, raise `PrecheckError` with a clear message pointing to which step to run.

---

## 11. Test plan

Tests in `tests/model/`. Run via `uv run pytest tests/model/`.

### 11.1 Unit tests (synthetic)

| # | Test | Assertion |
|---|---|---|
| M1 | `test_backbone_5ch_input` | Forward (1, 5, 384, 384) through ConvNeXt-tiny — outputs 4 levels at strides (4,8,16,32) |
| M2 | `test_conv1_renormalization_matches_doc` | Compare timm's conv1 weight to doc spec `pretrained.repeat * 3/5`; assert allclose |
| M3 | `test_fpn_output_shapes` | FPN over 4 backbone feats → 4 outputs each at correct stride and channels |
| M4 | `test_aux_seg_head_output_stride1` | Output is (B, 1, 384, 384) for input strides as expected |
| M5 | `test_rtmdet_head_forward_shapes` | head(P2..P5) → (cls_scores, bbox_preds), each list of length 4 with correct per-level shapes |
| M6 | `test_rtmdet_head_loss_smoke` | Forward + loss returns finite, non-NaN losses for synthetic GT |
| M7 | `test_rtmdet_head_predict_smoke` | Forward + predict returns valid boxes/scores/labels |
| M8 | `test_assigner_parity_with_mmdet` | **Critical**: vendored assigner output byte-equals MMDet's on fixed input |
| M9 | `test_dice_bce_loss_zero_for_perfect` | Dice+BCE on identical (logits=∞ where target=1, -∞ where target=0) → ≈ 0 |
| M10 | `test_total_loss_aggregates_correctly` | total_loss = cls + bbox + 0.3 * aux_seg; component dict has all keys |
| M11 | `test_lightning_module_training_step_smoke` | Single training_step on a batch returns scalar loss tensor with grad |
| M12 | `test_lightning_module_validation_step_smoke` | Single val step + on_validation_epoch_end logs slice_auroc |
| M13 | `test_score_ema_tracker_updated_on_train_negatives` | After 1 step with mixed batch, tracker has entries only for negative slices |
| M14 | `test_ema_callback_swap_swap_back` | Live weights restored after validation_epoch_end |
| M15 | `test_warmup_cosine_lr_schedule` | LR at step 0 = 1/10 base; at end = min_lr; smooth in between |

### 11.2 Integration tests (real data, single batch)

| # | Test | Assertion |
|---|---|---|
| M16 | `test_real_one_train_batch` | Build model + DataModule fold 0; one training_step on real batch returns finite loss |
| M17 | `test_real_one_val_batch` | One validation_step on real batch logs val/slice_auroc, val/mean_per_slice_iou |
| M18 | `test_real_two_epoch_loss_decreases` | Run 2 epochs on a 5-volume subset; final epoch loss < first epoch loss |
| M19 | `test_real_checkpoint_save_load` | Save best checkpoint; reload into fresh LM; validation reproduces original numbers |
| M20 | `test_real_wandb_metrics_logged` | After 1 epoch: WandB run contains expected metric keys (offline-mode test) |

### 11.3 Cohort-level smoke (1 fold, 2 epochs, 5-volume subset — same as Component 8 §8)

Acceptance gate before moving to Component 7:

1. All §11.1 unit tests pass.
2. All §11.2 integration tests pass.
3. Smoke run completes 2 epochs on a 5-volume subset with monotonically decreasing total loss.
4. GPU peak VRAM < 40 GB (leaves 6 GB headroom on L40S).
5. Assigner parity test (§M8) passes.
6. Pre-flight check refuses to start if cache files missing.

---

## 12. Logging

W&B logged keys per training step (sampled to `log_every_n_steps=10`):

- `train/total_loss`, `train/loss_cls`, `train/loss_bbox`, `train/loss_aux_seg`
- `lr`

Per validation epoch:

- `val/total_loss`, `val/loss_cls`, `val/loss_bbox`, `val/loss_aux_seg`
- `val/slice_auroc`, `val/mean_per_slice_iou`

Per deep-eval refresh (every 10 epochs, via Component 5 callback):

- `deep_eval/val_volume_auroc_coarse`
- `deep_eval/val_froc_at_2fp_coarse`
- `deep_eval/val_inference_seconds`
- `deep_eval/train_neg_inference_seconds`
- `deep_eval/hard_pool_size`
- `sampler/p_pos`, `sampler/hard_pool_substitution_active`

Per epoch (sampler/HNM):

- `loss_ema/n_tracked`, `loss_ema/top1_score`, `loss_ema/median_score`

---

## 13. Failure modes

| Failure | Detection | Action |
|---|---|---|
| Assigner parity mismatch | precheck | Refuse to start; investigate vendored assigner diff |
| OOM on first batch | trainer raises | Drop batch_size to 6 (add to RunConfig); if still OOM, reduce input to 320×320 (this is a CACHE change — invalidates the cache, requires full preprocessing rerun) |
| Loss NaN at step >100 | Lightning callback | Hard-fail; investigate intensity aug ranges, AMP settings |
| EMA drift symptoms (val auroc oscillates) | val trace | Reduce decay to 0.998 or move EMA buffer to fp32 explicitly |
| WandB connection fails | logger | Fall back to TensorBoard logger; warn |

---

## 14. Wall-clock budget

- Per training step (batch=8, bf16): ~250 ms target on L40S.
- **Default mode (`fixed_count`, samples_per_epoch=6000)**:
  - Per epoch: 6000 / 8 × 0.25 s ≈ **3 min wall-clock**.
  - 60 epochs: ~3 GPU-h per fold (training only).
  - Plus 6 deep-eval refreshes × 5 min each = ~30 min per fold.
  - Total per fold: **~3.5 GPU-h**. 5-fold CV: **~17.5 GPU-h**.
  - Matches doc §12's projection of ~5 GPU-h per fold (the doc was conservative).
- **Override mode (`full_pass`, samples_per_epoch ≈ 75K)**:
  - Per epoch: ~40 min wall-clock.
  - 60 epochs: ~40 GPU-h per fold (training only).
  - 5-fold CV: ~200 GPU-h — **exceeds the 168 GPU-h L40S allocation**. Use only for single-fold ablations.
- **Cohort-level smoke (5-volume × 2-epoch)**: < 5 min in either mode.

**Operational note:** the deep-eval cadence is `every_n_epochs=10` regardless of mode. In default mode this is "every 30 min wall-clock" — a useful cadence. In full-pass mode this is "every ~7 hours" — too slow; if running full-pass for an ablation, lower `refresh_every_epochs` to 3 or 5.

---

## 15. Acceptance checklist (Component 6 done)

- [ ] All `src/*.py` files exist with the APIs in §4–§9.
- [ ] All §11.1 unit tests pass.
- [ ] All §11.2 integration tests pass on real data.
- [ ] Cohort smoke (§11.3) passes including assigner parity.
- [ ] One full fold trains end-to-end, producing a `best.ckpt` and at least one `deep_eval_epoch{n}_val.npz`.
- [ ] `train.py --fold 0` runs to completion under wall-clock budget.
- [ ] WandB run shows all expected metric keys.
- [ ] Pre-flight check refuses to start with missing cache files (verified in test).

When this checklist is green, Component 6.5 (GRU rescorer) can begin, then Component 7 (post-training eval).
