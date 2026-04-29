# `endo/model/` — backbone + FPN + RTMDet head + aux seg head

Implements Component 6 (`agent/complete_spec/06_model_training.md`). Vendoring details: RTMDet head + DynamicSoftLabelAssigner copied from `mmdet @ cfd5d3a9...` on 2026-04-28 with `mmcv` / `mmengine` / `mmdet.registry` imports stripped (per Phase 0d). `pyproject.toml` does NOT depend on `mmdet`.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `LesionDetector` and the head/loss public API. |
| `detector.py` | `LesionDetector` — composes backbone, FPN, RTMDet head, aux seg head. `forward(x) -> (cls_scores, bbox_preds, aux_seg_logits)` for training; `predict(x, image_size, ...)` for end-to-end NMS prediction (used rarely; production callers go through `head.predict(cls_scores, bbox_preds, image_size)`). Builds the timm backbone in `features_only=True` mode and runs the doc-spec 5-channel conv1 surgery if timm's automatic 5-ch surgery deviates by >25%. |
| `fpn.py` | 4-level FPN with P2 (strides 4, 8, 16, 32). |
| `aux_seg_head.py` | Aux segmentation head that takes the P2 (stride-4) FPN feature and upsamples to `(B, 1, 384, 384)`. Trained via `dice_bce_loss` against `lesion_mask_center`. |
| `losses.py` | `dice_bce_loss(logits, target_uint8)`; `compute_total_loss(det_losses, aux_seg_logits, aux_seg_target, aux_seg_weight) -> (total, components)`. Components are keyed `loss_cls`, `loss_bbox`, `loss_aux_seg`, `loss_total` (note: `loss_total`, not `total_loss`). |
| `rtmdet_head.py` | Vendored `RTMDetHead`. `forward(feats) -> (cls_scores, bbox_preds)`. `loss(...)` runs the assigner + sigmoid focal loss on classification + CIoU on bbox. **CIoU is computed under `torch.amp.autocast(enabled=False)` with inputs promoted to fp32; if the result is still non-finite (rare degenerate-box case), falls back to a normalized L1 with per-coord clamp to 1.0.** `predict(cls_scores, bbox_preds, image_size, ...)` runs per-image score thresholding + per-class NMS via `torchvision.ops.batched_nms` and returns `[{boxes, scores, labels}, ...]`. |
| `assigner.py` | Vendored `DynamicSoftLabelAssigner`. Replaces `BBoxOverlaps2D` with `torchvision.ops.box_iou`. The byte-exact assigner-parity test (PRD M8) was downgraded to a smoke-shape test because `mmdet` isn't installable on the production env. |

## Contracts

- **Forward signature**: `LesionDetector.forward(x) -> (cls_scores: list[Tensor], bbox_preds: list[Tensor], aux_seg_logits: Tensor)`. List length is `len(strides) = 4` (P2-P5). Aux seg logits are `(B, 1, H=384, W=384)`.
- **NMS API**: production callers (LightningModule, inference_pass, etc.) MUST call `model.head.predict(cls_scores, bbox_preds, image_size=(384, 384), ...)`. The `LesionDetector.predict(x, image_size, ...)` shim takes raw input and re-runs the backbone — only used in test fixtures.
- **Loss component keys**: `{loss_cls, loss_bbox, loss_aux_seg, loss_total}`. Don't rename — the LightningModule logs by these keys and the smoke test asserts `loss_total` decreases.
- **Box format**: `(x1, z1, x2, z2)` half-open. The head's loss + predict treat these as `(x1, y1, x2, y2)` standard coords; the X / Z mapping comes from the dataset's transpose contract (PRD I.8.8).
- **bf16 NaN policy**: see the CIoU fallback above. Don't remove the `enabled=False` autocast wrap or the `isfinite` fallback without restoring an equivalent guard.

## Invariants tracked

- I.8.8 — `(B, 5, H=Z=384, W=X=384)` shape contract.
- I.8.9 — EMA shadow weights are fp32 (enforced in `endo.ema_callback`, not here).
- I.8.10 — vendored `DynamicSoftLabelAssigner` byte-equals MMDet on a fixed input. (Currently a shape-smoke test; restore byte-parity if mmdet ever becomes installable.)

## Don't

- Don't reintroduce `mmdet` / `mmcv` / `mmengine` imports — they break `uv sync` on Python 3.12.
- Don't change the FPN strides without updating `RTMDetHead.strides` AND the prior-construction logic in `_build_priors`.
- Don't fold the aux seg head into the same convolutional path as the detection head — they share the FPN P2 input but separate heads is what makes the `aux_seg_weight=0.3` safe to drop.
