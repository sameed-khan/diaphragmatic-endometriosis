# `endo/sampler/` — class-mix scheduling, score EMA, deep-eval refresh

Implements Component 5 (`agent/complete_spec/05_sampler_hnm.md`). Three pieces wired together:

1. `WeightedScheduledSampler` decides which slice indices the dataloader emits per step.
2. `ScoreEMATracker` keeps a per-`(pid, slice_y)` rolling score over the negative training pool.
3. `PeriodicDeepEvalCallback` periodically reruns inference on val + train-negatives, refreshes the hard-negative pool the sampler reads, and writes the deep-eval npz that Component 7 consumes.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `weighted.py` | `WeightedScheduledSampler(slice_index, cfg, seed)`. Partitions `slice_index` into pos / neg-in-pos-vol / neg-in-neg-vol pools at construction. `__iter__` mixes them with linearly-decaying `pos_frac` from `pos_frac_start` (epoch 0) to `pos_frac_end` (epoch `decay_epochs`). When `epoch ≥ hard_pool_start_epoch` and a non-empty hard pool has been set via `set_hard_pool(indices)`, a fraction `hard_pool_substitution_rate` of neg-in-neg-vol draws are replaced with hard-pool draws. `set_epoch(n)` is called by Lightning's automatic sampler hook. |
| `score_ema.py` | `ScoreEMATracker(decay)`. `update((pid, sy), score, *, is_positive_slice)` — keyword-only on `is_positive_slice` to make it explicit (positive slices are no-op, I.8.3). `top_k(k) -> list[int]` returns dataset indices of the top-K negative slices by current EMA. The "dataset index" mapping is owned by the `PeriodicDeepEvalCallback` because the tracker doesn't see the dataset. |
| `periodic_eval.py` | `PeriodicDeepEvalCallback(sampler_cfg, run_dir, train_neg_pids, val_pids, ema_callback=None, score_threshold, val_volume_labels)`. Fires on `_should_run(epoch)` (epoch ≥ `deep_eval_start_epoch` AND `(epoch - start) % refresh_every == 0`). Two passes via `inference_pass`: (1) val → `runs/<exp>/fold{f}/runtime/deep_eval/epoch{n}_val.npz` (CSR-style array per PRD §5.3.4); (2) train negatives → top-K hard pool → `hard_negatives.json` written atomically (tmp + replace). Logs coarse `val_volume_auroc` + `val_sens_at_2fp` proxies. |

## Contracts

- **Sampler `slice_index`**: 3-tuples `(pid, sy, kind)` where `kind ∈ {"pos_slice", "neg_slice_pos_vol", "neg_slice_neg_vol"}`. The dataset emits 4-tuples; the CLI strips the `is_pos_slice` field before passing to the sampler. The PeriodicDeepEvalCallback's lookup uses positional indexing (`entry[0]`, `entry[1]`) so it works on either shape.
- **Score-EMA invariant** (I.8.3): `tracker.update` MUST receive `is_positive_slice=False` (or `True` to no-op). `LesionDetectorLM._update_score_ema` always passes `False` because it only calls update for negative slices.
- **Atomic hard-pool replacement**: `_atomic_write_json` writes `<path>.tmp` then `os.replace` so the sampler's `__iter__` never sees a half-written JSON. If the file is missing or corrupt, the sampler treats the hard pool as empty (warn + continue).
- **EMA-during-deep-eval**: `PeriodicDeepEvalCallback` accepts an `ema_callback` and swaps to EMA shadow weights before the val/train-neg passes (matches I.8.5). If the callback's `swap` API isn't present, runs on live weights with a warning.
- **Path discipline (A.4)**: deep-eval npz + hard_negatives.json live under `runs/<exp>/fold{f}/runtime/`, NOT under `cache/`. The cache is shared across experiments; these are model-dependent.

## Invariants checked by tests

S1-S6 (sampler decay, mix probabilities, hard-pool substitution gating), S8-S10 (score EMA), S12-S14 (deep-eval callback gating, hard-negatives JSON, deep-eval npz roundtrip), S.INT.* (real-cache integration).

## Don't

- Don't read or write the deep-eval npz path directly from any code outside `endo.sampler.periodic_eval` and `endo.eval.run_eval`. The schema is private to those two modules.
- Don't call `tracker.update` positionally — `is_positive_slice` is keyword-only on purpose. (We had a regression where this silently passed once and crashed at the SECOND call after the kw was added.)
- Don't change `slice_index` to a different tuple shape without auditing every consumer (`weighted.py`, `periodic_eval.py`, the CLI's strip-shim, and the dataset/datamodule).
