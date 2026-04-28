# Handoff — Context Rotation for Contracts Overview

**To:** the next agent picking up this planning thread.
**From:** the planning agent who wrote Components 1–8.
**Date:** 2026-04-27
**Status:** All 8 component specs locked. Cross-component contracts overview is the final remaining task.

---

## 1. What this project is

The user (Sameed Khan, CWRU, RSNA abstract submission) is building a 2.5D MR detector for diaphragmatic endometriosis lesions. 608 volumes (108 positive, 500 negative). Single L40S GPU. One-week wall-clock budget.

Authoritative scope, decisions, targets, risk register: **`agent/training_pipeline_decisions_phase1.md`**. Read it first — it locks the architecture (ConvNeXt-tiny + custom FPN with P2 + RTMDet head + aux seg head), the augmentation philosophy (lesion copy-paste with donor bank), the 5-fold CV protocol, and the GRU rescorer. The phase-1 doc is non-negotiable; downstream specs implement it.

---

## 2. What's already done

Eight component specs in `agent/complete_spec/`:

| # | File | Component | Owner files (production) |
|---|---|---|---|
| 1 | `01_preprocessing.md` | Preprocessing pipeline (NIfTI → resampled, normalized, cropped+padded `.npy` cache + GT boxes + border bands) | `scripts/preprocess.py`, `scripts/analyze_inplane_spacing.py` |
| 2 | `02_lesion_bank.md` | Single global lesion bank (donor side of copy-paste) | `scripts/build_lesion_bank.py`, `src/lesion_bank.py` |
| 3 | `03_dataset_datamodule.md` | Slice-level Dataset + Lightning DataModule, RAM-resident, fold-aware, holdout-protected | `src/dataset.py`, `src/datamodule.py` |
| 4 | `04_augmentation.md` | Online augmentation (paste, geometric, intensity, box re-derivation, 5-channel slice extraction) + 4-tier QC gate | `src/augmentation.py`, `scripts/qc_paste_agentic_review.py` |
| 5 | `05_sampler_hnm.md` | Weighted/scheduled sampler + per-batch ScoreEMATracker + every-10-epoch deep-eval callback (writes hard pool JSON + deep-eval npz cache) | `src/sampler.py`, `src/periodic_eval_callback.py`, `src/inference_pass.py` |
| 6 | `06_model_training.md` | Model + LightningModule + training loop + EMA + train.py entrypoint with precheck | `src/model.py`, `src/rtmdet_head.py`, `src/assigner.py`, `src/aux_seg_head.py`, `src/losses.py`, `src/lightning_module.py`, `src/ema_callback.py`, `train.py` |
| 6.5 | `06_5_gru_rescorer.md` | Stage-2 GRU rescorer trained on frozen-detector backbone features | `src/gru_feature_cache.py`, `src/gru_rescorer.py`, `train_gru.py` |
| 7 | `07_post_training_eval.md` | CV evaluation + ensemble holdout one-shot (FROC, AUROC, AP, bootstrap CIs, stratified breakdowns) — CSV-only output | `src/eval/*.py`, `eval.py`, `eval_holdout.py` |
| 8 | `08_smoke_and_viz.md` | Smoke training script + per-slice prediction viz with TP/FP/FN tagging | `scripts/smoke_train.py`, `scripts/visualize_predictions.py` |

Each spec is self-contained: purpose, scope, inputs, outputs, pipeline, CLI, test plan, acceptance gate. The specs assume an engineering agent with zero context can implement them.

---

## 3. The final remaining task

**Write a comprehensive PRD-level synthesis with explicit cross-component contracts.** The user flagged this as the deliberate "second pass" at the start of the planning session:

> "we will progress through this entire planning session and then look at a second round run through at a high level overview that will involve **contracts** and will explicitly outline the contracts that each component has with the other components and state upfront which invariants our code and our data need to reinforce *after preprocessing completion*"

Concretely, the synthesis should produce **`agent/complete_spec/00_PRD.md`** (or similar — pick a clear name) that contains:

1. **Top-level architecture diagram** (ASCII or mermaid) showing all components and the data/control flow between them. Should fit on one screen.

2. **The data contracts.** For each artifact-on-disk that crosses component boundaries:
   - Filename pattern + path
   - Schema (columns / array shapes / dtypes)
   - Producer component (which spec writes it)
   - Consumer component(s) (which specs read it)
   - Invariants that must hold (e.g., "every patient_id in `gt_boxes.parquet` exists in `preprocessed_manifest.csv`", "every box is inside `[0, 384) × [0, 384)` cropped frame")

   Files to include at minimum: `preprocessed_manifest.csv`, `gt_boxes.parquet`, `volumes/<pid>/volume.npy`, `volumes/<pid>/lesion_mask.npy`, `border_bands/<pid>.npy`, `lesion_banks/lesion_bank_<sha>.pkl`, `runtime/hard_negatives.json`, `runtime/deep_eval/epoch{n}_val.npz`, `gru_features/fold{f}/<pid>.npz`, `gru_ckpts/fold{f}.pt`, `runs/baseline_fold{f}/ckpts/best.ckpt`, `eval_report.csv`, `eval_thresholds.json`, `outputs/<run>/viz/*.png`.

3. **The runtime contracts.** Cross-component Python-level interfaces:
   - `Sample` dataclass (Component 3 §4.1)
   - `Batch` dataclass (Component 3 §4.2)
   - `LesionBankEntry` dataclass (Component 2 §4.1)
   - `SliceScore` dataclass (Component 5 §7)
   - `LesionDataModule.inference_dataloader()` signature and its holdout guard
   - `inference_pass()` signature (Component 5 §7) — shared by Components 5 and 7
   - `RTMDetHead` public API (Component 6 §5.2)
   - `compute_total_loss` signature (Component 6 §6)
   - `rescore_detector_outputs` signature (Component 6.5 §8)

4. **Invariants — what must be true after preprocessing completion** (the user explicitly asked for this). Examples:
   - All 608 cohort volumes resampled to `(0.82, 1.5, 0.82) mm` voxels
   - All cached arrays at shape `(408, 174, 408)`
   - `gt_boxes.parquet` has exactly 197 CCs across the cohort matching §1.3 of the phase-1 doc
   - `lesion_vs_ring_z >= 0.121` for every positive (regression check vs §1.4)
   - Holdout patients have no `border_bands/<pid>.npy` file
   - `splits.json` and `preprocessed_manifest.csv` have consistent fold assignments
   - `lesion_bank_<sha>.pkl` contains only CCs from `cohort='cross-validation' AND label='positive'` patients (no holdout leak)

5. **Invariants — what must be true at training time** (e.g., DataModule with `allow_holdout=False` cannot load holdout patients; sampler `set_epoch(n)` is called once per epoch; ScoreEMATracker is updated by every `training_step`; periodic deep eval fires only at `epoch >= 10 AND epoch % 10 == 0`).

6. **Invariants — what must be true at evaluation time** (e.g., `eval_holdout.py` requires `--i-mean-it` flag and prior `cv_pooled` rows; ensemble inference uses all 5 folds; CV-pooled threshold from `eval_thresholds.json` applies to holdout).

7. **The full execution sequence** end-to-end:
   - Step 0: `analyze_inplane_spacing.py` (one-time, hardcoded result into `preprocess.py`)
   - Step 1: `preprocess.py`
   - Step 2: `build_lesion_bank.py`
   - Step 3: Component 4 QC gate (Tier 1 unit tests, Tier 2 metric tests, Tier 3 agentic review, Tier 4 human signoff)
   - Step 4: `smoke_train.py` (gates everything; ~5 min)
   - Step 5: `train.py --fold {0,1,2,3,4}` (sequential, ~3.5 GPU-h each)
   - Step 6: `gru_feature_cache.py + train_gru.py` per fold
   - Step 7: `eval.py` (CV) — with and without `--use-gru`
   - Step 8: `eval_holdout.py --i-mean-it` (one-time)
   - Step 9: `visualize_predictions.py` per fold (optional, for QC)

8. **Resource accounting.** Total GPU-h, total disk, total RAM peak — sanity-check that the plan fits the L40S budget (target: ≤ 25 GPU-h Stage-1 + Stage-2 per phase-1 doc §12).

9. **Open issues / known limitations.** Things flagged during planning that aren't yet resolved (e.g., the assigner-parity test for vendored RTMDet, the in-plane resample target spacing pending the analysis script's actual output, etc).

---

## 4. Before writing the PRD: investigate the rsi repo

The user has an existing repo at `/home/sak185/rsi` (a needle-detection model with online compositing) that is structurally similar to our planned augmentation pipeline. **Do a thorough architectural review of the rsi codebase before writing the PRD.** The earlier subagent investigation identified key files at:

- `/home/sak185/rsi/packages/needle/rsi_needle/train.py`
- `/home/sak185/rsi/packages/needle/rsi_needle/trainer/needle_trainer.py` (the canonical Ultralytics-subclass example, ~1200 LOC)
- `/home/sak185/rsi/packages/needle/rsi_needle/dataset/fullimage_dataset.py` (online paste pattern)
- `/home/sak185/rsi/packages/needle/rsi_needle/engine/compositor.py` (paste mechanics, physics-specific — not directly applicable but the structural pattern is)

**Important:** the user previously decided to use **RTMDet + Lightning** for THIS project, *not* the Ultralytics-subclass pattern that rsi uses. So the rsi review is for **lifting structural patterns** (per-worker stateful compositor, epoch-aware mixing schedule, bbox-from-mask post-paste, donor bank organization) — NOT for copying the trainer architecture. Do not "correct" the spec back to Ultralytics. The decision is locked.

Read the rsi training stack to inform contract design (especially around augmentation, sampling, and dataset patterns). Anything you learn that *strengthens* a contract or invariant in the PRD should be incorporated.

---

## 5. How the user works (collaboration style)

Read the prior conversation if available; if not, here's the gist:

- **Surfaces tradeoffs explicitly.** Don't make decisions silently — present options A/B/C with pros and cons, ask for direction. The user pushes back when they disagree (e.g., normalization scope, validation metrics during training, RTMDet vs YOLO).
- **Direct and concise.** Short responses preferred. Bullet points + tables. No marketing language. No filler.
- **Engineering-aware but high-level.** The user is an experienced ML engineer. They want technical depth but not pedagogy. They'll ask for clarification when needed (e.g., "explain to me the flow of the data through the model architecture and functionally what are the tradeoffs of A vs B").
- **Pushes back on overengineering.** If a feature isn't earning its keep, drop it. The phase-1 doc §14 ("Out of Scope for Week 1") is taken seriously.
- **Trusts but verifies.** If you propose something with backing, cite it. If you guess, say "guess." Evidence-based recommendations carry weight; vibes-based ones do not.
- **Iterative rounds of Q&A.** The planning followed a "sketch → ask 3-6 questions → write spec" cadence. This worked. Continue it for the PRD.

User-specific notes:
- Email: `ctippareddy2@gmail.com` / `sameed.khan@case.edu`
- Affiliation: CWRU
- Disk quota check: use `quotagrp`, not `du -sh` or `quota -s` (per `MEMORY.md`)
- Always use `uv` for Python (`uv run`, `uv add`)
- Always use `polars` not `pandas` for tabular data
- Working directory: `/scratch/pioneer/users/sak185/diaphragmatic-endometriosis/`
- Compute node: 250 GB RAM, 20 cores, single L40S GPU, Linux, bash

---

## 6. Key documents to read in order

1. **`agent/training_pipeline_decisions_phase1.md`** — the authoritative source-of-truth document. ~550 lines.
2. **`agent/complete_spec/01_preprocessing.md` through `08_smoke_and_viz.md`** — the eight component specs you're synthesizing.
3. **`CLAUDE.md`** (project root) — short operational notes (uv, polars, quotagrp).
4. **`/home/sak185/rsi/packages/needle/rsi_needle/`** — for architectural patterns; specifically `trainer/needle_trainer.py` and `dataset/fullimage_dataset.py`.
5. **`agent/eda_synthesis.md`, `agent/research_2026_modeling.md`, `agent/research_medical_imaging_approaches.md`** — context but not authoritative; the phase-1 doc supersedes where they conflict.

Do not regenerate or overwrite the eight spec files. They are locked. The PRD is a *synthesis document* on top of them, not a replacement.

---

## 7. Decisions locked during the prior planning session

These are commitments — do not relitigate without explicit user permission:

- **In-plane resample target: `(0.82, 1.5, 0.82) mm`** (pending one-time analysis script confirmation; default per cohort median)
- **Cache shape: `(408, 174, 408)`** with ±5 mm jitter margin
- **fp16 volume cache, uint8 masks**
- **Liver mask NOT in runtime cache** (consumed only inside Component 1 to derive border_band, then discarded)
- **Volume-wide z-score normalization** (Option A in earlier discussion); ROI provides stats, not application mask
- **Single global lesion bank** (not per-fold); deliberate val-leak accepted for donor diversity; holdout never enters
- **Paste-first augmentation ordering** (paste → geometric → intensity → re-derive boxes → extract 5ch)
- **Multi-paste schedule**: `Bernoulli(0.5)` outer × `HalfGaussian(σ=1.0)` inner clipped to [1, 7]
- **Train-time validation = slice-level proxies only** (slice AUROC, mean per-slice IoU, val losses); volume metrics deferred to post-training (Component 7) plus a 6-point coarse trace via Component 5's deep-eval callback every 10 epochs
- **Hard-negative pool = union of (ScoreEMATracker top-K, deep-eval top-K)**, refresh every 10 epochs
- **Best checkpoint selected by `val/slice_auroc`**
- **Detector: hand-vendored RTMDet head + DynamicSoftLabelAssigner** from MMDet, plus a parity test against installed MMDet
- **Aux seg head output at stride 1** (per-pixel supervision)
- **Sampler epoch length: `fixed_count`, default `samples_per_epoch=6000`** (~3 GPU-h per fold); `full_pass` available as override
- **No QC-signoff precheck in `train.py`** (review is recommended but not gated by training entrypoint)
- **Holdout boundary**: enforced at DataModule level (`allow_holdout=False` default); also at `eval_holdout.py` (`--i-mean-it` flag + audit JSON)
- **CSV-only eval output** (`eval_report.csv`); presentation layer deferred
- **Tensor convention at model boundary: `(B, 5, H=Z=384, W=X=384)`** with boxes in `(x1, z1, x2, z2) ≡ (W_min, H_min, W_max, H_max)` — no permutation
- **Coordinate frames**: NIfTI `(512, N, 512)` with axis 1 = through-plane (Y, A-P); cache `(X, Y, Z) = (408, 174, 408)`; tensor at model `(5, Z, X)` per the transpose in Component 4 §9
- **5-fold ensemble holdout inference**, CV-pooled WBF threshold, patient-level bootstrap CIs (1000 resamples)

---

## 8. Suggested approach for the PRD

1. **Read** the phase-1 doc and all 8 spec docs end-to-end. Take ~30 minutes.
2. **Investigate** the rsi repo. Specifically: how `needle_trainer.py` and `fullimage_dataset.py` structure their interfaces. Look for cross-component contracts they enforce that we're missing.
3. **Sketch** the architecture diagram + the data-contract table; share with user; iterate.
4. **Write** `agent/complete_spec/00_PRD.md`. Aim for thorough but focused — every section carries weight; no filler.
5. **Surface** any contradictions or gaps you find between the eight specs while writing the PRD. The user will want to know.
6. **Verify** the resource accounting math against the L40S budget.

The PRD's job is to be the **single document a brand-new engineering agent can read and understand the entire system before opening any of the eight component specs.** Test that mental model as you write.

Good luck. The user is a responsive collaborator — ask questions when uncertain rather than guessing.
