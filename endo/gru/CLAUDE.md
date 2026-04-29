# `endo/gru/` — Stage-2 BiGRU rescorer

Implements Component 6.5 (`agent/complete_spec/06_5_gru_rescorer.md`). Stage-2 sits on top of a frozen Stage-1 detector — it consumes GAP-pooled stage-3 backbone features and emits a per-slice probability that gets multiplied into each detector box's score.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `GRURescorer`, `extract_features_for_fold`, `train_gru_for_fold`, `rescore_detector_outputs`, `rescore_slice_scores`. |
| `feature_cache.py` | `extract_features_for_fold(experiment, fold, output_dir=None, device, force, pids, ckpt_path)`. Loads `runs/<exp>/fold{f}/ckpts/best.ckpt`, overlays `ema_state_dict` if present, runs backbone-only forward over each fold-{f} validation patient via `inference_dataloader([pid])`, GAP-pools the last backbone stage with `F.adaptive_avg_pool2d(...).flatten(1)` (768-d). Writes one `<pid>.npz` per patient with `feats: (N_valid_slices, 768) fp16`, `slice_ys: (N,) int32`, `patient_label: () int8`. Idempotent: skips when all expected files exist unless `force=True`. |
| `rescorer.py` | `GRURescorer(GRUConfig)` — `Dropout → BiGRU → Linear → sigmoid`. `forward(feats, mask) -> (B, T)` probabilities (mask zeros padded positions). `volume_score(probs, mask, agg, k)` — `'max'` or `'topk'` with mask-aware reduction. `rescore_detector_outputs(gru_ckpt_path, feature_cache_path, detector_boxes_per_slice)` per PRD §6.10. `rescore_slice_scores(slice_scores, ckpt_path, feature_dir)` adapter for `endo.eval.run_eval`'s `dict[pid, list[SliceScore]]` shape. |
| `train.py` | `train_gru_for_fold(experiment, fold, output_dir=None, device)`. Discovers train npz from the **other 4 folds**' `feature_cache/` dirs, val npz from this fold's. Pads sequences to fold-max-length, trains AdamW with `BCE(volume_score(max)) + aux_loss_weight * BCE(volume_score(topk))`. Tracks best val AUROC, writes `ckpt.pt` (`{state_dict, config dump, epoch, val_auroc}`) + `gru_provenance.json`. |

## Contracts

- **Feature shape**: `(N_valid_slices, 768)` per patient. The 768 is fixed by `convnext_tiny.fb_in22k`'s stage-3 output; if the backbone changes, update `endo.config.gru.GRUConfig.input_dim` AND the feature-extraction call site.
- **Label semantics**: `patient_label` is volume-level (`1` if any positive slice, `0` otherwise). Slice-level labels live in the detector's `gt_boxes.parquet` — the GRU never sees per-slice labels.
- **Cross-fold structure**: `train_gru_for_fold(0)` reads features from `runs/<exp>/fold{1..4}/gru/feature_cache/` — i.e. each detector fold's val features are used as training data for every OTHER fold's GRU. This avoids leakage and matches Component 6.5 §3.
- **Idempotency**: `extract_features_for_fold` skips if every expected `<pid>.npz` exists unless `force=True`.
- **Rescorer load contract**: `GRURescorer` is built from the saved `config` field in the checkpoint, filtered to the current `GRUConfig` schema so older / extended dumps still load.

## Invariants checked by tests

G1 (forward shape), G3 (volume score max + topk), G4 (mask handling), G6 (rescore multiplies scores), G7 (missing-slice passthrough), G.INT.2 (synthetic correlated dataset → val AUROC > 0.7 in 5 epochs).

## Don't

- Don't train the GRU on its own fold's features — that leaks val-set signal into the rescorer.
- Don't pass the live LightningModule's `model` directly. Reload it from `best.ckpt` (with EMA overlay) so the GRU sees the deployment weights.
- Don't reshape `(N, 768)` features into spatial maps — the rescorer is a sequence model, not a 2D conv.
