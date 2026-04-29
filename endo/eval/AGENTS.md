# `endo/eval/` — post-training evaluation (CV + holdout)

Implements Component 7 (`agent/complete_spec/07_post_training_eval.md`) and the I.9.* invariants. CV pooling reads each fold's `runtime/deep_eval/epoch{n}_val.npz` to avoid re-running inference; holdout actually re-runs inference with `allow_holdout=True`.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the public API surface. |
| `wbf.py` | `weighted_box_fusion_3d(slice_scores, image_size, ...)` — concatenates per-slice 2D boxes across the volume, fuses overlapping boxes per slice via `ensemble_boxes.weighted_boxes_fusion`. Applies the size-dependent threshold filter: large boxes pass at `score ≥ large_thr`, small at `≥ small_thr`. Returns `{"fused_boxes": (M, 5)=(x1,z1,x2,z2,slice_y), "fused_scores": (M,)}`. |
| `froc.py` | `compute_froc(per_volume_predictions, per_volume_labels, fp_per_volume_levels)`. Wraps `picai_eval` by rasterizing fused boxes onto a synthetic 3D detection map (Y=160, Z=384, X=384). Has a fast-path hand-rolled fallback for unit-test inputs without fused boxes. Returns sensitivity at each FP level and the full FROC curve. |
| `metrics.py` | `bootstrap_ci(values, statistic_fn, n=1000, seed=42, alpha=0.05)` — patient-level resampling. `compute_volume_metrics` returns AUROC, AP, sens@FP with bootstrap CIs in a single pass. |
| `threshold_search.py` | `grid_search_threshold` over `(large_thr, small_thr)` pairs, maximizing sens@2FP. Returns the best pair and the full grid table. |
| `stratified.py` | `stratify_metrics(per_volume_metrics, manifest_rows, strata)` — per-stratum AUROC / AP with bootstrap restricted to the stratum (I.9.7). |
| `report.py` | `EvalReportRow` dataclass (PRD §4.1 schema), atomic append-only CSV writer (`append_eval_report` — NEVER overwrites), `write_eval_thresholds_json`. |
| `run_eval.py` | `run_cv_evaluation(experiment, use_gru, eval_dir)` — loops folds, loads each fold's deep_eval npz cache, aggregates per-volume via WBF, computes metrics, emits cv_pooled + per_fold + stratified rows. Folds without a deep_eval npz are skipped with a warning. `run_holdout_inference(experiment, ckpts, use_gru)` — the SOLE caller of `LesionDataModule(allow_holdout=True)`. Loads each requested ckpt, runs inference on the 122-pid holdout, ensembles across ckpts (mean), optionally GRU-rescores, writes `holdout/run_<ts>_<uuid8>/{eval_report.csv, invocation.json}`. |

## Contracts

- **Append-only CSV** (I.9.1): `report.append_eval_report` writes the header on first invocation and only ever appends rows. Re-running `eval` keeps the previous run's rows under their original `run_id`.
- **GRU rescoring** is opt-in via `--use-gru`; if `endo.gru.rescorer.rescore_slice_scores` isn't importable (or the per-fold ckpt / feature-cache files are missing), the orchestrator falls back to non-rescored with a clear warning. The non-rescored row set is always emitted alongside the rescored set when `--use-gru` is on.
- **Holdout discipline (I.9.3)**: only `run_holdout_inference` sets `allow_holdout=True`. Don't replicate that anywhere else. Each call produces a fresh `holdout/run_<ts>_<uuid8>/` subdir.
- **Ckpt loader contract**: `LesionDetectorLM.__init__` requires a positional `exp_cfg`, so `Lightning.load_from_checkpoint` can't reconstruct it. `run_eval.run_holdout_inference` builds the LightningModule manually (`LesionDetectorLM(experiment); load_state_dict(...)`) and overlays `ema_state_dict` when present, then calls `.to(device)`.
- **Bootstrap** is patient-level, default `n=1000`, seed 42 (I.9.6).

## Invariants checked by tests

E1, E3, E5, E6, E9, E10, E11, E12, E15. E11 is the one-fold synthetic deep-eval npz round-trip; E12 verifies both the rescored and non-rescored row sets are produced; E15 verifies CSV append-only.

## Don't

- Don't truncate `eval_report.csv` between runs — that breaks I.9.1.
- Don't re-implement WBF inside `run_eval.py` — go through `endo.eval.wbf` so the size-threshold filter and slice-y carry-through stay correct.
- Don't call `LesionDetectorLM.load_from_checkpoint(...)` directly. Use the manual-load idiom (load raw torch ckpt, build LM with `experiment`, `load_state_dict(strict=False)`, overlay EMA, `.to(device)`).
