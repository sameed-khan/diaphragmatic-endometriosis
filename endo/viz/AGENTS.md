# `endo/viz/` — per-slice TP/FP/FN visualization

Implements Component 8's visualization half (`agent/complete_spec/08_smoke_and_viz.md` §2). Smoke-train lives in `scripts/smoke_train.py` — out of scope here.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `tag_slice_events`, `render_slice_overlay`, `save_slice_png`, `visualize_predictions_for_fold`. |
| `tagging.py` | `tag_slice_events(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.3) -> {"tp": [(box, score, gt_idx)...], "fp": [(box, score)...], "fn": [box_gt...]}`. Greedy IoU matching, sorted by score desc. A predicted box matches at most one GT (the one with highest IoU above threshold). |
| `render.py` | `render_slice_overlay(volume, slice_y, lesion_mask_center, pred_boxes, pred_scores, gt_boxes, event_type, patient_id, fig_size_px=512, apply_anat_orientation=True) -> (H, W, 3) uint8`. Native slice frame is `(X, Z)` (rows=X, cols=Z). For display, applies rot90 CCW + fliplr to put the image in **radiology coronal orientation**: `S` at top, patient's `R` on viewer's left. The transform is applied in lockstep to the image, mask, and box coords:  `new_row = Z_dim - z`, `new_col = X_dim - x`. Boxes `(x1, z1, x2, z2)` map to `(X_dim - x2, Z_dim - z2, X_dim - x1, Z_dim - z1)`. Colors: predictions red solid, GT boxes green dashed, lesion mask green semi-transparent (alpha 0.40). `save_slice_png(image, output_path)` writes via matplotlib's PNG encoder. |
| `run_viz.py` | `visualize_predictions_for_fold(experiment, fold, output_dir, device, score_threshold=0.05, max_pngs_per_event=200)`. Loads `runs/<exp>/fold{f}/ckpts/best.ckpt` with EMA overlay, builds the DataModule (default fold-{f} val), runs full inference over every val pid × valid slice via `inference_pass`. Per slice: drop predictions below `score_threshold`, per-slice NMS at IoU 0.5 (WBF integration deferred — currently `torchvision.ops.nms` fallback when `endo.eval.wbf.per_slice_wbf` isn't yet defined), tag TP/FP/FN, save up to `max_pngs_per_event` PNGs per event type with one row per highlighted entity in `manifest.csv`. **Idempotent**: tracks `best.ckpt` mtime in `viz/.ckpt_mtime`; re-running with the same ckpt is a no-op. |

## Contracts

- **Anatomic orientation transform** is applied in lockstep to image, mask, AND boxes. Don't transform any one of them in isolation — that re-creates the original "TP boxes don't overlap the lesion mask" bug.
- **Box frame**: input `pred_boxes` and `gt_boxes` MUST be in cached `(384, 160, 384)` voxel coords with `(x1, z1, x2, z2)` order. The renderer reflects them through `_anat_transform_box`.
- **Coloring** is fixed (red preds, green GT + mask). If you add a new event type or score-gradient, add it as an extra layer; don't repurpose the existing channels.
- **Idempotency**: re-running viz against the same ckpt mtime returns immediately. Bump the mtime (re-train, `touch best.ckpt`) to force regeneration.
- **Cap discipline**: `max_pngs_per_event=200` is per event type **across the whole fold** — first-come, first-served. The orchestrator does not currently sample 20-positive + 20-negative *patients* per the spec's flavor text; the cap is over event entities, not patient subsamples. If you change this, document it here.

## Invariants checked by tests

V1, V3, V4, V5 (event tagging) plus a render smoke that asserts a non-empty PNG is created.

## Don't

- Don't apply the rot90/fliplr to only the image — boxes and mask MUST follow.
- Don't render in (X, Z) without the anatomic transform unless `apply_anat_orientation=False` (the test stub uses this; production calls don't).
- Don't write to `runs/<exp>/fold{f}/checkpoints/...` for ckpts — production uses `ckpts/`. The viz `_resolve_best_ckpt` searches `ckpts/best.ckpt` first, then `checkpoints/best.ckpt` for backward compat.
