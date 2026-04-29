# Eval Correctness Addendum — 2026-04-29

This addendum documents the post-training evaluation changes implemented on
branch `codex-audit/correctness` per the plan in
`agent/audit_plan_correctness_2026-04-29.md` and the audit findings in
`agent/audit_2026-04-29.md`.

It supplements (does not replace) `agent/complete_spec/07_post_training_eval.md`.

## Summary of behavior changes

### 1. Fresh inference (replaces deep-eval cache reads)

`run_cv_evaluation` no longer reads `runs/<exp>/fold{f}/runtime/deep_eval/epoch*.npz`.
Instead, for each fold it loads the configured checkpoint
(`EvalConfig.eval_ckpt ∈ {"best","last"}`, default `"best"`), overlays the
`ema_state_dict` if present, and runs `inference_pass` over the fold's
validation patients.

Deep-eval npz caches remain training-time-only artifacts consumed by the
hard-negative mining loop and Lightning monitoring metrics. They are
never used for the final CV report.

### 2. Cross-fold threshold tuning (option a)

For fold `f`, the post-WBF size-dependent thresholds are tuned on the
**union of the OTHER 4 folds' raw fused predictions**. Each volume is
then evaluated with its own fold's tuned thresholds. There is no second
`cv_pooled` grid search — that path created an obvious leakage.

`ensemble_threshold` (used for holdout) is the **mean** of the 5 per-fold
thresholds and is written to `eval_thresholds.json` along with
`tuning_policy: "cross_fold_leave_one_out"`.

### 3. Raw vs thresholded metric split

- **AUROC / AP** are computed from raw fused scores (no size filter).
- **FROC / sens@FP** are computed from thresholded predictions.
- **Stratified breakdowns** follow the same split.

`compute_volume_metrics` and `stratify_metrics` accept a new
`raw_predictions=` kwarg; when omitted, both fall back to the thresholded
preds (back-compat).

### 4. Lesion-level FROC with real GT masks

`compute_volume_metrics(..., gt_masks=...)` plumbs masks through to
`compute_froc`. Masks are extracted from the (already-loaded) cache via
the new helper `_gt_masks_from_dm`, which centers the cached
`(X, Y, Z) → (Y, Z, X)` mask onto the eval canvas (`(160, 384, 384)`).
For negatives we pass an all-zero mask. The fallback central-cuboid proxy
in `endo/eval/froc.py` is preserved for callers that don't pass masks.

### 5. Per-call (TP/FP/FN) JSONL output

New module `endo/eval/calls.py` extracts 3D 26-connectivity components
from the **raw** fused detection map per volume (component score = max
voxel score; volume = voxels × `0.82·1.5·0.82 mm³`). Calls are matched to
GT lesions via the **centroid-in-mask** rule (each GT lesion's
highest-scoring matching call is TP; other matching calls become FP; GT
lesions with no matching call are FN).

Each call records its `passes_threshold` flag against the fold's tuned
thresholds (or the holdout ensemble thresholds), but the threshold is
not used to suppress emission.

#### Schema

One JSON object per line:

```
{
  "run_id":            "<run_id>",
  "entrypoint":        "cv" | "holdout",
  "fold":              <int>|null,
  "patient_id":        "<pid>",
  "call_id":           "<pid>_pred_<k>" | "<pid>_fn_<k>",
  "call_type":         "tp" | "fp" | "fn",
  "score":             <float>|null,           # null for FN
  "passes_threshold":  <bool>|null,            # null for FN / no thresholds
  "volume_mm3":        <float>,
  "voxel_count":       <int>,
  "bbox_yz_x":         [y0,y1, z0,z1, x0,x1],
  "centroid_yz_x":     [y, z, x],
  "gt_lesion_id":      "<pid>_lesion_<k>"|null # set for TP and FN
}
```

#### Output paths

- CV: `runs/<exp>/eval/per_call_<run_id>.jsonl`
- Holdout: `runs/<exp>/holdout/run_<ts>_<uuid>/per_call_<run_id>.jsonl`

### 6. GRU rescoring with fresh feature cache

`extract_features_for_pids(experiment, fold, *, pids, output_dir,
ckpt_path, device)` rebuilds the per-patient `<pid>.npz` cache from the
chosen detector checkpoint. CV eval writes to
`runs/<exp>/eval/feature_cache/fold{f}/`; holdout to
`runs/<exp>/holdout/run_<ts>_<uuid>/feature_cache/fold{f}/`. The original
training-time per-fold caches under `runs/<exp>/fold{f}/gru/feature_cache/`
are not modified.

### 7. EMA swap API on `EmaCallback`

`EmaCallback.swap_to_ema(pl_module)` and `EmaCallback.restore_live()` are
public methods. `PeriodicDeepEvalCallback` now passes `pl_module` into
the swap so deep-eval (val + train-negative passes during training) runs
on EMA shadow weights. This aligns the hard-negative miner with the
deployment weights and removes the prior silent "no swap method" warning.

### 8. `ScoreEMATracker` decay from config

`endo/cli/run_experiment.py` now constructs `ScoreEMATracker(decay=
experiment.sampler.score_ema_decay)` instead of using the dataclass
default of 0.9.

### 9. FROC sens@FP CIs labelled as approximate

The bootstrap CIs for sens@FP are computed from the per-volume max-score
sweep (a fast proxy), not the lesion-level detection-map sweep used for
the point estimate. `eval_thresholds.json` now carries a `froc_ci_note`
making this explicit.

## Items intentionally **not** addressed

Per the plan and the user's go-ahead:

- No deep-eval cache pid-validation guard.
- No holdout "touch guard" / repeat-touch lockfile.
- No mmdet parity restoration / synthetic parity scaffolding.
- No augmentation / dataloader throughput changes (handled separately).
- No bf16-vs-fp16 stability changes.
