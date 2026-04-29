# wandb-logging gate report

Result of running the §9.3 acceptance test from `agent/logging_wandb_plan.md`.

## Run links (online, project `clevelandclinic/diaphragmatic-endometriosis`)

- Detector: <https://wandb.ai/clevelandclinic/diaphragmatic-endometriosis/runs/7eiduio1>
- Holdout:  <https://wandb.ai/clevelandclinic/diaphragmatic-endometriosis/runs/abph7siq>
- Group:    `e2e-testing_00000000`

## Acceptance summary

`uv run python scripts/verify_e2e_gate.py` → **15 PASS / 1 FAIL**.

| # | Criterion | Result |
|---|---|---|
| 1 | `train/loss_total_epoch[1] < epoch[0]` (2.21 → 1.75) | PASS |
| 2a | `train/loss_cls` strictly decreases (0.75 → 0.48) | PASS |
| 2b | `train/loss_bbox` strictly decreases (1.14 → 0.95) | PASS |
| 2c | `train/loss_aux_seg` strictly decreases (1.07 → 1.03) | PASS |
| 3 | `val/loss_total` decreases (0.51 → 0.42, ≤ 5 % slack) | PASS |
| 4 | no `train/skipped_steps_nan > 0` | PASS |
| 5 | `<fold_dir>/run.log` non-empty + structured | PASS |
| 6 | `epoch_post-train/` has ≥ 1 PNG (200 PNGs) | PASS |
| 6b | `epoch_0/` and `epoch_1/` viz dirs exist | PASS |
| 7 | detector W&B run `e2e-testing/run1` exists | PASS |
| 8 | holdout W&B run `e2e-testing/run1-holdout` exists | PASS |
| 9 | both runs share group `e2e-testing_00000000` | PASS |
| 10 | viz-fold0 artifact ≥ 60 PNGs | **FAIL** (20 PNGs) |
| 11 | no `best.ckpt` artifact uploaded | PASS |
| 12 | eval/holdout report artifact uploaded | PASS |

### Why #10 fails — model-quality limitation, not a logging bug

After only 2 epochs of fp32 training, `val/slice_auroc = 0.500` (chance).
The model produces zero detections passing the `score_threshold = 0.05`
gate inside `endo.viz.run_viz.visualize_predictions_for_fold`. Per the
manifest:

```
$ awk -F',' 'NR>1{c[$3]++} END{for (k in c) print k, c[k]}' \
    runs/e2e-testing_00000000/fold0/viz/epoch_post-train/manifest.csv
fn 212
```

— 212 FN events, 0 TP, 0 FP. With no TP/FP entries in the manifest, the
20/20/20 reproducible sampler in `endo.viz.run_viz.sample_tp_fp_fn` can
only emit the 20 FN entries it does have. The plumbing is correct; the
2-epoch model simply hasn't learned enough to produce TP/FP.

This is a structural property of the test config (2 × 1000 samples, fp32
to dodge the documented bf16 NaN), not the W&B + logging integration.

## Metric keys observed in W&B

```
 - epoch
 - lr-AdamW/pg1
 - lr-AdamW/pg2
 - train/grad_norm_epoch
 - train/grad_norm_step
 - train/loss_aux_seg_epoch
 - train/loss_aux_seg_step
 - train/loss_bbox_epoch
 - train/loss_bbox_step
 - train/loss_cls_epoch
 - train/loss_cls_step
 - train/loss_total_epoch
 - train/loss_total_step
 - train/seconds_per_epoch
 - train/throughput_samples_per_sec
 - trainer/global_step
 - val/loss_aux_seg
 - val/loss_bbox
 - val/loss_cls
 - val/loss_total
 - val/slice_auprc
 - val/slice_auroc
```

W&B summary keys also include params/dataset sizes/GPU info/git SHA per
plan §3.1.

## Detector run artifacts

- `viz-fold0-00000000:v0` type=viz (20 PNGs)
- `config-00000000:v0` type=config (experiment.yaml + experiment.py)
- `provenance-00000000:v1` type=provenance

No `model` artifact — `LoggingConfig.wandb.upload_checkpoints=False` in
the e2e config (§9.3 #11 PASS).

## Holdout run artifacts

- `holdout-report-00000000:v1` type=holdout-report (eval_report.csv,
  invocation.json, per_call_*.jsonl)

## Final wall-clock

- detector fold 0: 715.7s (~12 min, fp32, 2 × 1000 samples + 2 × val + 2 ×
  training-time viz + post-train viz over 24 val patients)
- holdout: ~8 min (122 patients × 1 ckpt)

## Test suite

- 124 pre-existing tests still pass.
- 14 new tests added: `tests/config/test_logging_config.py`,
  `tests/utils/test_logging_setup.py`, `tests/utils/test_wandb_init.py`,
  `tests/viz/test_sample_tp_fp_fn.py`.
- Total: 138 passing.

## Outstanding human-confirmation step (per plan §9.3)

The plan stops the implementer here and asks the human user to:
- Open the dashboard linked above and confirm loss curves look sane.
- Confirm all metric keys exist (listed above).
- Confirm artifact tab shows the listed entries.

Then merge `wandb-logging` into `master`.
