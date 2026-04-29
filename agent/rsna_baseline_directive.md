# RSNA baseline — supervisor agent directive

**Drafted:** 2026-04-29 by the planning pass
**Target reader:** the Opus 4.7 supervising agent that takes baseline 5-fold
training + holdout evaluation end-to-end while the user is asleep
**Working directory:** `/home/ubuntu/diaphragmatic-endometriosis` on the Lambda
Labs GPU machine (single A100 40 GB SXM)
**Hand-off contract:** when the user wakes up, they should be able to read
`runs/baseline-rtmdet-p2_*/eval/abstract_figure.md` and
`runs/baseline-rtmdet-p2_*/eval/abstract_numbers.json` and paste the headline
result + figure into the RSNA abstract submission with no further data work.

---

## 0. Pre-flight (do this first, in order)

Before launching any training, verify in this order. Any failure → STOP and
write a diagnostic note to `agent/rsna_baseline_run_log.md`; don't try to
self-heal silently.

1. **Repo state**: `git status` clean on `master`, `git rev-parse HEAD` matches
   the commit that introduced this directive (or a descendant). If a different
   branch is checked out, ask the user — don't switch.
2. **GPU state**: `nvidia-smi` should show **MIG enabled with 4× `1g.10gb`
   instances** already configured by the planning pass. UUIDs are listed in
   `agent/rsna_baseline_mig_uuids.txt` (created in §6 below). If MIG is
   disabled or partitioned differently, re-run the partitioning sequence in §6
   before proceeding.
3. **Wandb** (optional but preferred): `.env` has `WANDB_API_KEY`. The CLI
   defaults to `wandb.enabled` from the experiment file
   (`baseline_rtmdet_p2.py` has it on by default — verify
   `experiment.logging.wandb.enabled`). If a run fails to start because of W&B
   auth, fall back to `--no-wandb` for that fold and note it.

## 1. What's being trained

The experiment under test is **`experiments/baseline_rtmdet_p2.py`** (DO NOT
edit it; per the experiment-file contract, edits trigger a drift-guard at
resume). Headline knobs:

- ConvNeXt-tiny backbone + 4-level FPN with P2 + vendored RTMDet head + aux
  seg head
- Lesion copy-paste augmentation (`p_any_paste=0.5`, up to 7 pastes/sample)
- 5-fold CV over 486 patients (positives stratified by `manufacturer_model_name`,
  negatives by model × `slice_thickness_bin`)
- 60 epochs × 6000 samples/epoch / `bs=8`, `bf16-mixed`, AdamW
- Stage-2 BiGRU rescorer per fold (`gru.epochs=20`)
- Volume-level evaluation with patient-level bootstrap CIs (n=1000),
  cross-fold threshold tuning, FROC + AUROC + AP via `picai_eval`,
  per-call JSONL with TP/FP/FN
- **Lesion-volume-binned sensitivity, Brier, ECE, reliability curve, and the
  multi-panel abstract figure** — added by the planning pass; see
  `endo/eval/lesion_strata.py`, `endo/eval/calibration.py`, and
  `scripts/make_abstract_figure.py`. The CV + holdout eval entry points
  already call them — you do not need to wire them in.

**Decision locked by the user (planning pass):**
The holdout result presented in the abstract is the **5-model ensemble** of
the 5 CV-fold checkpoints (the existing `run_holdout_inference` path,
`ensemble_threshold = mean(per_fold_thresholds)`). Do **not** train a
"6th model on all CV data". If something pushes you toward that, surface it
to the user — don't act on it.

## 2. Launch sequence — 5-fold parallel-then-sequential training

A100-40GB MIG cannot host 5×~8 GiB slices, so we fan out to **4 parallel
folds** (`1g.10gb` each ≈ 9.75 GiB usable) and run the **5th fold sequentially**
on the first slice that frees up. This is the wall-time-optimal layout under
the MIG constraint.

For each fold-slice mapping use one tmux/screen pane (or one `setsid` background
process) with logs tee'd to `runs/baseline-rtmdet-p2_*/fold{N}/runtime/launch.log`.

```bash
# Resolve MIG UUIDs once, persist for the supervisor's reference.
nvidia-smi -L | grep MIG > agent/rsna_baseline_mig_uuids.txt

# Folds 0–3 in parallel. Each command lands on its own MIG slice.
SLICE0=$(awk 'NR==1{print $6}' agent/rsna_baseline_mig_uuids.txt | tr -d ')')
SLICE1=$(awk 'NR==2{print $6}' agent/rsna_baseline_mig_uuids.txt | tr -d ')')
SLICE2=$(awk 'NR==3{print $6}' agent/rsna_baseline_mig_uuids.txt | tr -d ')')
SLICE3=$(awk 'NR==4{print $6}' agent/rsna_baseline_mig_uuids.txt | tr -d ')')

for f in 0 1 2 3; do
  slice_var="SLICE$f"
  setsid bash -c "CUDA_VISIBLE_DEVICES=${!slice_var} \
    uv run -m endo.cli.run_experiment train \
      --exp experiments/baseline_rtmdet_p2.py \
      --fold $f \
      > runs/baseline-rtmdet-p2_*/fold$f/runtime/launch.log 2>&1" &
done
wait  # do NOT actually wait — instead supervise and launch fold 4 when one finishes (see §3)
```

(The `for` loop above is illustrative — you should **not** `wait` for all four
because that blocks fold 4. See §3 for the supervised launch.)

Once one of folds 0–3 finishes (signal: `runs/.../fold{N}/runtime/done` exists,
or the process exits cleanly with a `best.ckpt` written), launch **fold 4** on
that slice's UUID:

```bash
CUDA_VISIBLE_DEVICES=$FREED_SLICE \
  uv run -m endo.cli.run_experiment train \
    --exp experiments/baseline_rtmdet_p2.py \
    --fold 4 \
    > runs/.../fold4/runtime/launch.log 2>&1 &
```

**Wall-time budget:** Per-fold time on `1g.10gb` is uncertain — measure during
the first 2 epochs of fold 0 and project. The user has accepted ">8 h" wake-up
windows; if your projection exceeds 14 h, message the user before continuing
(they may want to abort and re-plan with a smaller sampler budget).

## 3. Monitoring cadence

Wake up roughly every **20–30 minutes** (don't oversample — burns cache, costs
tokens, doesn't move work forward). On each wake:

1. **Liveness**: are 4 (then 5) python processes still on the GPU? `nvidia-smi
   --query-compute-apps=pid,used_memory,gpu_uuid --format=csv`. Cross-check
   against the launch PIDs.
2. **Progress**: tail each `fold{N}/run.log` and `fold{N}/runtime/launch.log`.
   Confirm:
   - `epoch X/60` is advancing (epoch latency × remaining epochs ≤ budget).
   - `val/slice_auroc` is being logged and is non-degenerate (≠ 0.5 ± 0.02).
   - The NaN guard (`endo/lightning_module.py`) hasn't fired more than ~5
     times — that's logged at WARNING. Higher rates indicate something has
     destabilized; alert the user.
3. **VRAM headroom**: each slice should use ≤ 8 GiB out of 9.75 GiB. If a fold
   is consistently above 9 GiB it may OOM on a peak augmentation batch —
   record and continue, but flag if any fold OOMs.
4. **Wandb dashboard**: if W&B is on, the run is at `wandb.ai/<entity>/diaphragmatic-endo/<run-id>`.
   Don't stare at it — the file logs are authoritative — but a quick glance
   confirms metrics are flowing.
5. **Done detection**: a fold is done when `runs/<exp>/fold{N}/ckpts/best.ckpt`
   and `last.ckpt` both exist AND the python process for that fold has exited
   with code 0. The CLI also writes `runs/<exp>/fold{N}/runtime/done` on
   success — easier to grep.

## 4. Stage 2 — GRU rescorer per fold

After **each** detector fold finishes, you can run that fold's GRU rescorer
either inline (immediately on the freed slice if no other detector training
is queued) or in a deferred batch after all 5 detector folds are done.
Inline is cheaper because it overlaps with other folds; deferred is simpler
to reason about. **Default to deferred**: after all 5 detectors finish,
launch all 5 GRU rescorers in parallel on the 4 MIG slices (4 parallel + 1
sequential, same pattern):

```bash
for f in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$FREE_SLICE_FOR_FOLD_$f \
    uv run -m endo.cli.run_experiment train_gru \
      --exp experiments/baseline_rtmdet_p2.py --fold $f &
done
wait
CUDA_VISIBLE_DEVICES=$ANY_SLICE \
  uv run -m endo.cli.run_experiment train_gru \
    --exp experiments/baseline_rtmdet_p2.py --fold 4
```

The GRU stage is small (BiGRU on 768-D features, 20 epochs) and should finish
in tens of minutes per fold, not hours.

## 5. Final evaluation passes

### 5a. CV evaluation (across all 5 folds, pooled + per-fold + stratified)

```bash
CUDA_VISIBLE_DEVICES=$ANY_MIG_SLICE \
  uv run -m endo.cli.run_experiment eval \
    --exp experiments/baseline_rtmdet_p2.py \
    --use-gru
```

What this writes:

- `runs/<exp>/eval/eval_report.csv` (append-only) — all metrics × scopes ×
  strata × rescored ∈ {True, False}.
- `runs/<exp>/eval/eval_thresholds.json` — per-fold + ensemble thresholds.
- `runs/<exp>/eval/per_call_<run_id>.jsonl` — per-lesion TP/FP/FN with
  `volume_mm3`. **Drives the lesion-volume-binned sensitivity rows** that the
  planning pass added.
- `runs/<exp>/eval/raw_preds_fold{0..4}.json` and
  `raw_preds_cv_pooled.json` — per-volume max scores used by the figure.
- `runs/<exp>/eval/reliability_cv_pooled.json` — bin-level reliability data.

If any fold has no ckpt, `run_cv_evaluation` will log-and-skip that fold but
still pool the rest. Decide: if 4 folds finished and 1 didn't, proceed with
4-fold eval and note the degraded sample size in the abstract notes; if ≤ 3
folds finished, abort and notify the user.

### 5b. Holdout inference (5-model ensemble on 122 patients)

```bash
CUDA_VISIBLE_DEVICES=$ANY_MIG_SLICE \
  uv run -m endo.cli.run_experiment predict_holdout \
    --exp experiments/baseline_rtmdet_p2.py \
    --use-gru
```

What this writes (under `runs/<exp>/holdout/run_<ts>_<sha>/`):

- `eval_report.csv` with `entrypoint=holdout`
- `per_call_<run_id>.jsonl`, `raw_preds_holdout.json`,
  `reliability_holdout.json` (figure-ready)
- `invocation.json` recording which checkpoints were used

**Holdout is the only place that sets `allow_holdout=True` on the
DataModule.** Don't replicate this anywhere else.

### 5c. Multi-panel abstract figure

```bash
uv run python scripts/make_abstract_figure.py --exp baseline-rtmdet-p2
```

Outputs (in `runs/<exp>/eval/`):
- `abstract_figure.pdf` and `.png` — six panels: ROC, FROC,
  sens@2FP × scanner, sens@2FP × thickness, lesion-sens × volume bin,
  reliability.
- `abstract_numbers.json` — headline values + 95 % CIs ready to paste into
  the abstract text. NaN/Inf are sanitized to `null` so it's valid JSON.

## 6. MIG configuration (reference — already done by planning pass)

```bash
sudo nvidia-smi -i 0 -mig 0     # tear down (only if currently set up wrong)
sudo nvidia-smi -i 0 -mig 1     # enable MIG mode
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb -C   # 4 GPU+compute instances
nvidia-smi -L                    # MIG UUIDs to use as CUDA_VISIBLE_DEVICES
```

Post-condition (verify with `nvidia-smi -L`): four MIG devices each named
`MIG 1g.10gb` with distinct UUIDs.

## 7. Failure modes & responses

| Symptom | Likely cause | Response |
|---|---|---|
| Drift-guard error at fold-launch | someone edited `experiments/baseline_rtmdet_p2.py` mid-run | STOP. Diff against the materialized `runs/<exp>/experiment.yaml`. Notify user — do not pass `--force-resync` without explicit user OK |
| OOM mid-training on one fold | augmentation batch peak | Reduce that fold's `n_paste_max` to 4 in `experiment.augmentation.paste` AND restart from `last.ckpt`. Do this **only** with user OK; record the change |
| CUDA NaN guard fires repeatedly (>10/epoch) | bf16 instability or bad data | Stop the fold. Try `precision="32-true"` for that fold only — but this needs user OK first |
| One fold's process dies with no ckpt | likely OOM on a peak step or driver hiccup | Re-launch the fold on the same slice. If it dies again with same error, skip the fold and continue (CV eval can pool the remaining 4) |
| `eval_report.csv` already has rows from a prior `run_id` | append-only contract working as designed | Do nothing. Each `run_id` is unique; the figure reads the latest |
| Disk full | cache + ckpts | First, GC `runs/<exp>/fold{N}/runtime/deep_eval/epoch*.npz` for finished folds (these are training-time only and ignored by final eval per the 2026-04-29 audit). If still tight, message user |
| Holdout inference can't find a ckpt | a fold didn't finish | Holdout will skip that fold and ensemble the remaining ones. Note the reduced ensemble size in the abstract notes |

**Strict prohibitions** (these will lose work or invalidate the run):

- Do NOT delete `runs/<exp>/eval/eval_report.csv` to "start fresh" — it's
  append-only by design (I.9.1).
- Do NOT call `LesionDetectorLM.load_from_checkpoint` directly. Use the
  manual-load idiom in `endo/eval/run_eval.py:_load_detector_for_fold`.
- Do NOT pass `allow_holdout=True` to the DataModule outside
  `run_holdout_inference` — that's a hard invariant (I.8.1).
- Do NOT amend a finished fold's checkpoint after eval has read it.
- Do NOT run `--force-resync` without explicit user approval; it bypasses the
  drift guard which exists to prevent silent config divergence.

## 8. Final deliverables checklist

When you wake the user up, the following must exist and the user must be able
to find them in <30 seconds:

- [ ] `runs/baseline-rtmdet-p2_*/eval/abstract_figure.pdf` (multi-panel) and
      its `.png` companion
- [ ] `runs/baseline-rtmdet-p2_*/eval/abstract_numbers.json` (headline values
      + 95 % CIs for the abstract text body)
- [ ] `runs/baseline-rtmdet-p2_*/eval/eval_report.csv` with both
      `entrypoint=cv` and `entrypoint=holdout` rows present, `rescored ∈
      {true, false}` both populated
- [ ] `runs/baseline-rtmdet-p2_*/eval/eval_thresholds.json` with all 5 fold
      thresholds + `ensemble_threshold`
- [ ] `agent/rsna_baseline_run_log.md` — your operational diary: what you
      launched, what you observed at each wake-up, what (if anything) you
      changed, what failed and how you handled it. The user will skim this
      on wake-up to gut-check the result before pasting numbers into the
      abstract.

## 9. Headline metrics the user will paste into the abstract

These are the numbers the user wants visible in `abstract_numbers.json`:

- **Volume AUROC**: cv_pooled value + 95 % CI; holdout value + 95 % CI; mean
  ± std across the 5 per-fold values.
- **Sensitivity at 2.0 FP/volume** (sens@2FP): cv_pooled, holdout.
- **Sensitivity at 0.5 FP/volume**: cv_pooled, holdout (the lower-FP regime
  is what radiologists actually want; reviewers will look for it).
- **Lesion-level sensitivity by volume bin**: `<=200`, `200–1000`,
  `1000–5000`, `>5000` mm³ (the bin edges live in
  `endo/eval/lesion_strata.py:DEFAULT_VOLUME_EDGES_MM3`; verify the bin
  populations are non-degenerate before reporting — if any bin has fewer
  than ~5 lesions in the cv_pooled set, drop it from the abstract figure).
- **Sensitivity by scanner_model** (`SIGNA Artist` n≈369 vs `SIGNA Explorer`
  n≈239): the high-vs-low-end-scanner comparison reviewers care about.
- **Sensitivity by `slice_thickness_bin`** (`≤4mm` n≈583 vs `>4mm` n≈25 — note
  the imbalance; the >4mm cell is small, frame the comparison cautiously).
- **Brier + ECE** at cv_pooled and holdout. Only include in the abstract if
  Brier ≤ 0.20 and ECE ≤ 0.10 — otherwise the model is poorly calibrated and
  including the numbers invites reviewer pushback. Reliability curve in the
  figure regardless (it's diagnostic).

## 10. Things the user will NOT want in the abstract

- Per-fold raw numbers (CV mean ± std is enough)
- Threshold values (those go in supplementary, not the abstract body)
- Bootstrap method details (one sentence: "patient-level bootstrap, n=1000")
- Anything from training-time `deep_eval/*.npz` — those are training-monitor
  only per the 2026-04-29 audit; final eval ignores them

## 11. After you finish

1. Write `agent/rsna_baseline_run_log.md` with the timeline, deviations,
   sample-size caveats, and explicit "ready to paste" status.
2. Don't auto-commit. Leave the working tree clean of unintended changes;
   the user will review and commit.
3. Don't push to remote.
4. Don't tear down MIG — the user may want to inspect the live state.
