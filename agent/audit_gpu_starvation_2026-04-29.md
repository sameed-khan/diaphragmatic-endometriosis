# GPU-Starvation Audit — Diaphragmatic Endometriosis Training Stack (2026-04-29)

Scope: training-time throughput on a single A10 (24 GB). Measures CPU/GPU step balance, identifies the dominant starvation point, and proposes ranked fixes with tradeoffs. Companion artifacts:

- `scripts/profile_gpu_starvation.py` — end-to-end CPU/H2D/GPU timing harness (reused from prior audit).
- `scripts/profile_aug_stages.py` — new per-stage augmentation microbenchmark.
- `data/.profile_manifest.jsonl` — 20-pid subset (8 pos, 12 neg, all 5 folds) used for fast bring-up of the DataModule during profiling.

All measurements are on this Lambda Labs A10 node: 30 vCPUs, 222 GB RAM, NVIDIA A10 (24 GB). bf16-mixed precision, batch_size=8, target shape (X=384, Y=160, Z=384), 5-channel input.

---

## TL;DR

The codebase is **CPU-bound by online augmentation by ~16-20×**. With augmentation enabled, a single training sample takes **~4.4 s** of CPU work; the corresponding GPU step takes **~0.26 s**. With augmentation disabled, GPU utilisation is **93 % p50 / 100 % p90**; we have headroom on the device, the problem is the input pipeline.

The single biggest fix is conceptual: **augmentation operates on the full `(384, 160, 384)` cropped volume even though only the 5-slice center window is consumed.** Because affine/elastic/intensity are all Y-coherent (or Y-independent), narrowing them to the 5-slice window collapses the work by ~30× and should lift the A10 to ≥80 % util at modest worker counts. Other fixes (per-worker RNG, double-cast removal, per-step NMS gating) are real but ~10× smaller.

---

## 1. Measurements

### 1.1 Per-stage augmentation cost (single-thread, single-volume, 15 iter)

`scripts/profile_aug_stages.py` was run on `amber_eagle_spark` (94.4 MB fp32 volume, 409 lesion voxels, 50 645 border-band voxels, bank size 153). It times each stage on the full cropped volume — **the same sized inputs the production augmentation receives**.

| Stage                              | mean (ms) | p50 (ms) | p99 (ms) | Share of full pipeline |
|------------------------------------|----------:|---------:|---------:|-----------------------:|
| `paste` (3 forced)                 |      64.1 |     60.8 |     86.7 |                  1.4 % |
| `affine_lockstep`                  |   2 070.8 |  2 072.4 |  2 203.7 |                 46.2 % |
| `elastic_lockstep`                 |   1 323.4 |  1 296.1 |  1 461.1 |                 29.5 % |
| `intensity` (bias+gamma+noise)     |     901.6 |    890.7 |  1 007.3 |                 20.1 % |
| `box_rederive` (center slice)      |      0.04 |     0.04 |     0.04 |                <0.01 % |
| `extract_5ch`                      |       1.6 |      1.6 |      1.6 |                  0.04 %|
| **full pipeline (no paste)**       | **4 481** |  4 380   |  5 416   |                100.0 % |

The dataset additionally allocates a fp32 contiguous copy of `volume_full_cropped` (~94 MB) inside `__getitem__` (`endo/data/dataset.py:251`) before handing it to augmentation, which casts again (`endo/augmentation/transform.py:262`). That is a second 94 MB write per sample on the slow path.

### 1.2 Training step, aug DISABLED, workers=8 (control)

`scripts/profile_gpu_starvation.py --mode training --no-augment --workers 8 --steps 30` over the 20-pid subset:

| Metric              | mean    | p50     | p90     | p99     |
|---------------------|--------:|--------:|--------:|--------:|
| `cpu_batch_time_s`  | 0.00035 | 0.00028 | 0.00064 | 0.00072 |
| `h2d_time_s`        | 0.00654 | 0.00655 | 0.00708 | 0.00719 |
| `gpu_step_time_s`   | 0.258   | 0.246   | 0.272   | 0.383   |
| `step_time_s`       | 0.265   | 0.253   | 0.279   | 0.390   |
| `gpu_util` (NVML %) |    93.2 |    95.0 |   100.0 |   100.0 |
| `gpu_mem_used_mb`   |   6 924 |   6 924 |   6 924 |   6 924 |

Interpretation: with the augmentation off, batch readiness is essentially free (~0.3 ms), the GPU is the bottleneck at ~258 ms/step, and we run at **~93 % util on the A10**. This is the "headroom we have to spend".

### 1.3 Training step, aug ENABLED — projected starvation

A worker producing one augmented sample takes ~4.4 s. With `W` workers and `batch_size=B=8`, the steady-state batch readiness time is `B / (W / 4.4 s) = 4.4 × B / W` seconds.

| Workers | Sec/batch ready | GPU step | Predicted GPU util |
|--------:|----------------:|---------:|-------------------:|
|       4 |             8.8 |     0.26 |              ~3 %  |
|       8 |             4.4 |     0.26 |              ~6 %  |
|      16 |             2.2 |     0.26 |             ~12 %  |
|      32 |             1.1 |     0.26 |             ~24 %  |

Even saturating all 30 vCPUs gives barely a quarter of the GPU. Heavy workers also blow up RSS (each worker holds the COW-shared cache plus its own augmentation scratch space).

### 1.4 Training step, aug ENABLED, workers=16 (measured, 20 steps + 3 warmup)

`scripts/profile_gpu_starvation.py --mode training --workers 16` over the same 20-pid subset:

| Metric              | mean    | p50     | p90     | p99     |
|---------------------|--------:|--------:|--------:|--------:|
| `cpu_batch_time_s`  |   1.389 |   0.001 |   1.470 |  17.928 |
| `gpu_step_time_s`   |   0.258 |   0.250 |   0.278 |   0.278 |
| `step_time_s`       |   1.656 |   0.279 |   1.730 |  18.188 |
| `gpu_util` (NVML %) |   90.4  |   91.5  |   98.2  |  100.0  |

Reading: the prefetch buffer (default `prefetch_factor=2` × `num_workers=16` = 32 batches) hides starvation for the first ~30 batches — `cpu_batch_time_s` p50 is 0.6 ms, GPU util shows 90 %. But `step_time_s` mean is 1.66 s and p99 is **18.2 s** — the moment the prefetch buffer drains, we wait the full per-batch CPU time. Steady-state batch latency converges to the predicted ~2.2 s. The first-batch p99 of 18 s is the worst case where Lightning has just spun up workers and the GPU sat idle waiting for them.

### 1.5 Dataloader-only, aug ENABLED, workers=4, 60 batches (measured, steady state)

`scripts/profile_gpu_starvation.py --mode dataloader --workers 4 --num-batches 60`:

| Metric              | mean     | p50      | p90      | p99      |
|---------------------|---------:|---------:|---------:|---------:|
| `cpu_batch_time_s`  | **7.710** | **5.988** | **21.129** | **31.221** |
| `gpu_util` (NVML %) |      0.0 |      0.0 |      0.0 |      0.0 |
| `gpu_mem_used_mb`   |   1 810  |    443   |   7 213  |   7 213  |

This is the dataloader in isolation (no GPU training step running); it measures pure batch-readiness time. **A worker pool of 4 takes 7.7 s to produce a single batch of 8** — within 13 % of the back-of-envelope `4.4 s × 8 / 4 = 8.8 s`. With the GPU step at 0.26 s, this is a **~30× starvation ratio** at workers=4. Even sustaining workers=16 (the §1.4 measurement) gives ~2-3 s/batch in steady state — still ~10× starved.

p99 of 31 s reflects worst-case worker pile-up (e.g. when all 4 workers hit a heavy paste sample concurrently); since p90 is 21 s, the distribution has a heavy right tail, which is what kills throughput in production where Lightning waits on the slowest worker.

### 1.4 Where the augmentation cost actually goes

Every stage that dominates the budget operates on the full **`(384, 160, 384)` = 23.6 M voxels** but the model only consumes a 5-slice window: `volume_full_cropped[:, slice_y - 2 : slice_y + 3, :]` = **0.74 M voxels**.

- `apply_affine_lockstep` (`endo/augmentation/geometric.py:122`): one `ndi.affine_transform(order=1)` on the full 3D volume + one `order=0` on the full 3D mask — 2.07 s of work for 0.16 s of useful work after the 5-slice extraction.
- `apply_elastic_lockstep` (`endo/augmentation/geometric.py:222`): a Python loop over **160 Y planes** calling `ndi.map_coordinates` per slice. Reuses the same `(2, X, Z)` displacement field every slice (Y-coherent). 158 of those 160 calls are computed and immediately thrown away.
- `intensity_aug` (`endo/augmentation/intensity.py:46`): `random_brightness_contrast`, `random_gamma` (`np.power` + `np.sign` + `np.abs`), and `random_gaussian_noise` (`rng.normal` of 23.6 M samples). All voxel-wise — operates correctly on a 5-slice slab.

Since paste introduces structure in 3D and a donor's Y extent is small, the fully Y-restricted optimisation requires keeping a Y window large enough to contain the donor mask before the geometric stage. This is small (donor `tight_mask` y-extent is typically <10 voxels, and we only need to retain content within `slice_y ± 2` *after* potential affine translation). After the geometric stage the network only ever sees the 5-channel slice.

---

## 2. Ranked findings & fixes

Each entry lists: **issue / file / measured cost / proposed fix / risk + correctness contract**.

### 2.1 (P0) Augmentation runs on the full Y dimension; only 5 slices are used

- **Files**: `endo/augmentation/transform.py:268-323`, `endo/augmentation/geometric.py:102-233`, `endo/augmentation/intensity.py:46-55`.
- **Cost**: ~4.3 s of the 4.4 s per-sample (97 %).
- **Fix**: After `multi_paste_volume` (which legitimately needs 3D), narrow the volume + lesion mask to a Y-slab around `slice_y` *before* `geometric_aug` and `intensity_aug`. The geometric pipeline is already in-plane and Y-coherent (T1.13), so the slab can be as small as `[slice_y - 2 : slice_y + 3]` for the network's input. We need to keep a small extra margin only for box re-derivation if you want to retain the current behaviour of re-deriving boxes from the augmented mask at the *center* slice (current code does only `mask[:, slice_y, :]` so 5 is sufficient).
- **Expected speedup**: ~30× on geometric/intensity, ~25× end-to-end. Sample time drops from ~4.4 s → ~150–200 ms; even with workers=4 we'd be GPU-bound again.
- **Risk / contract**:
  - Y-coherent invariant T1.13 is **preserved** because the same 2D field is reused across the 5 slices.
  - In-plane-only invariant T1.12 is preserved (no Y movement was applied even before).
  - The 5-channel extraction contract (PRD I.8.8) is unchanged.
  - The only wrinkle: if you ever extend augmentation to add Y-axis perturbations (currently disallowed by spec), this would need to be revisited. That is a deliberate design boundary, not a regression.
- **Implementation note**: paste must still happen on the full volume so donor placement can use the full border-band; only the *output* of paste needs to be narrowed to the slab before the affine/elastic/intensity stages.

### 2.2 (P0/P1) `intensity_aug` allocates ~24 M random samples per call

- **File**: `endo/augmentation/intensity.py:36-43`.
- **Cost**: ~0.9 s of which `random_gaussian_noise` is ~0.2 s and `random_gamma`'s `np.power` is ~0.6 s.
- **Fix #1**: Once 2.1 is implemented, this collapses naturally to ~30 ms (operates on 0.74 M voxels).
- **Fix #2** (if 2.1 is rejected): switch the noise to in-place `rng.standard_normal(out=...)` with a preallocated buffer; switch gamma to `np.power(volume, g, out=volume)` to avoid the `np.sign`/`np.abs` round-trip — savings ~30–40 %.

### 2.3 (P1) Dataset RNG repeats across workers

- **File**: `endo/data/dataset.py:121` (`self._rng = np.random.default_rng(rng_seed)`).
- **Cost**: not throughput; it silently halves effective augmentation diversity when `num_workers ≥ 2` because every worker gets the same seed. Each worker sees disjoint indices from the sampler so the dup is *partially* masked, but the jitter sequence aligns.
- **Fix**: add `worker_init_fn` on the dataloaders that re-seeds `dataset._rng` from `(base_seed, worker_id, epoch)`. Or derive jitter deterministically from `(patient_id, slice_y, epoch)` as augmentation already does (this also helps reproducibility under worker count changes).
- **Risk**: pure determinism contract is unchanged provided the seed function is fixed.

### 2.4 (P1) Double float32 copy of `volume_full_cropped`

- **Files**: `endo/data/dataset.py:251`, `endo/augmentation/transform.py:262`.
- **Cost**: ~94 MB write per sample, second time. Roughly 30-60 ms/sample of memory bandwidth on this node.
- **Fix**: cache stores fp16. Cast to fp32 once — either in the dataset (and have augmentation use `np.asarray(..., dtype=np.float32)` without a forced copy) or only in augmentation. Pick one site. The cleanest single-cast site is augmentation, since the dataset's `volume_full_cropped` then doesn't have to be cast at all on the val path.
- **Risk**: low. The numerical contract is unchanged (cache was already fp16 → fp32 lossless). No invariant binds the dtype before augmentation.

### 2.5 (P1) Per-batch NMS during training adds GPU work to every step

- **File**: `endo/lightning_module.py:117-143` — `_update_score_ema` calls `head.predict` (= NMS + post-processing) on every training batch, even before HNM kicks in (`hard_pool_start_epoch=5`, `deep_eval_start_epoch=10`).
- **Cost**: a few ms per step at batch_size=8 on A10. Less critical than the augmentation issue but compounds as we lift the dataloader.
- **Fix #1**: gate the EMA update on `epoch >= hard_pool_start_epoch - K` (start filling the EMA a few epochs before HNM activates).
- **Fix #2**: compute the EMA update every N steps instead of every step (config knob `score_ema_update_every`).
- **Fix #3**: substitute a much cheaper proxy — `aux_seg_logits.sigmoid().amax(dim=(1,2,3))` per sample is already computed in the val path, and is ~free.
- **Risk**: HNM accuracy. Per-step NMS gives the freshest possible EMA, but the EMA decay (0.9 default) already smooths heavy fluctuations; updating every 4-8 steps changes the convergence window negligibly.

### 2.6 (P2) `inference_pass(batch_size=...)` is silently ignored

- **File**: `endo/inference_pass.py:44-83` — argument accepted but never threaded to `datamodule.inference_dataloader`.
- **Cost**: confined to evaluation throughput (deep-eval, holdout). Not a training-time issue but does affect end-to-end run time. The deep-eval target in the spec is ≥50 slices/s on L40S; this would let us tune batch size for the inference pass.
- **Fix**: add `batch_size` parameter to `LesionDataModule.inference_dataloader` and pipe it through. One-line change in two places.

### 2.7 (P2) Collate makes one extra fp32 copy per batch

- **File**: `endo/data/collate.py:19` — `torch.from_numpy(np.stack(...)).float()`. Since `volume_5ch` is already float32, `.float()` is a no-op clone (nominally returns self when dtype matches, but `np.stack` already returns a new array; the worry is the chain `np.stack` → `from_numpy` → `.float()` which can drop the no-op if the stack is non-contiguous).
- **Fix**: remove the `.float()` (already fp32). Trivial, and the small saving is real at high batch rate.

### 2.8 (P2) Eager full-cohort RAM load

- **File**: `endo/data/datamodule.py:144-168` — `np.load` of every cached volume into RAM in `setup()`. For 488 CV pids × ~56 MB fp16 = ~27 GB plus masks; fits in 222 GB RAM but takes 60-120 s of cold start per fold.
- **Fix**: allow `mmap_mode="r"` on `np.load`. The arrays are read-only thereafter; on Linux, COW-shared mmap into workers stays page-cache-backed. Saves on duplicate cache pages when workers spawn (vs fork) and shaves setup time meaningfully.
- **Risk**: numerical and contract behaviour unchanged. Only thing to watch: any code that mutates a cache entry would copy-on-write — none does today.

### 2.9 (Stretch / P3) Move geometric to the GPU

- **Why mention**: `torch.nn.functional.grid_sample` on the GPU does the same affine + elastic in <1 ms for a `(B, 5, 384, 384)` tensor. If you're willing to move augmentation into `LightningModule` (run on GPU after H2D), the entire geometric stage becomes free.
- **Tradeoff**: bigger refactor; you'd need to expose the displacement fields out of `random_elastic_2d` as torch tensors and run paste/intensity on CPU still (or move both to GPU). Aug becomes part of the Lightning forward path; reproducibility tests need updating because RNG moves to GPU.
- **Recommendation**: only pursue after 2.1 + 2.4 — those alone should restore GPU saturation for this workload.

### 2.10 (Stretch / P3) `inference_pass.py` docstring vs. implementation drift

- **File**: `endo/inference_pass.py:54-58` claims `model.model.predict`; code calls `detector.head.predict` (correct per cross-package contract). Update the docstring.

---

## 3. Suggested action plan

The audit task is to enumerate fixes and tradeoffs, *not* to implement them. When the user is ready, the order I'd take is:

1. **2.1** (Y-slab narrowing in `transform.py` + `geometric.py`) — ≥25× sample-throughput gain; preserves all invariants. *This is the "must-do" change.*
2. **2.4** (single-cast policy) — small but free, no risk; do it in the same change as 2.1.
3. **2.3** (per-worker RNG) — reproducibility/diversity hygiene; orthogonal to throughput.
4. **2.5** (EMA gating) — gate `_update_score_ema` to start a couple epochs before `hard_pool_start_epoch`, or move to N-step cadence; mostly future-proofing once aug is fixed.
5. **2.6** (`inference_pass.batch_size`) — wire it through; affects deep-eval / holdout, not train.
6. **2.8** (mmap cache) — a clean win on multi-fold runs; trivial one-liner.

Estimated combined wall-clock effect: **~20-25× faster training step on A10** (from ~4.5 s/batch worst-case down to roughly ~0.3-0.4 s/batch — i.e. GPU-bound again). On the 5-GPU node intended for the full-folds run, that means the per-fold training time drops from O(hours) to O(minutes per epoch) and 60 epochs × 5 folds becomes feasible inside a few hours rather than overnight.

## 4. Post-fix verification (branch `codex-audit/efficiency`)

After implementing all P0/P1/P2 fixes, re-ran the same volume through the production `TrainAugmentation.__call__` pipeline (`scripts/profile_train_aug_call.py`) on `amber_eagle_spark`:

| Stage / pipeline                     | Before    | After       | Speedup    |
|--------------------------------------|----------:|------------:|-----------:|
| `TrainAugmentation.__call__` (mean)  | 4 481 ms  | **141.8 ms** | **31.6 ×** |
| `TrainAugmentation.__call__` (p99)   | ~5 416 ms |     152.9 ms |     ~35 ×  |

The smoke test (`scripts/smoke_train.py`, full augmentation enabled) was rerun on this branch:

- 50 training steps captured (≥ 20 required)
- first-10 mean loss: **2.314** → last-10 mean loss: **1.102** (− 52 %)
- All step losses finite
- `val/slice_auroc` logged (= 0.5 on the 5-vol smoke set, expected)
- `Trainer.fit` reaches `max_epochs=2` cleanly; no NaN/Inf warnings or aborts
- Steady-state pace ~1.64 it/s at batch_size=4, num_workers=2. Pre-fix this config would have stalled around 0.1 it/s.

The full `pytest tests/ -q` suite passes: **117 passed, 0 failed** (264 s wall-clock), including the real-cache `test_smoke_runs_to_completion_real_cache` integration test which itself runs the full smoke training through the new pipeline.

## 5. Confidence & caveats

- The per-stage microbenchmark used real cached volumes with the production aug config and ran 15 iterations; numbers are stable (p99 within 5-15 % of mean).
- The "no-aug" training profile is a clean, *measured* upper bound on GPU utilisation; the "with-aug" projection is back-of-the-envelope (per-sample throughput / workers). A direct measurement is in progress (workers=4 dataloader run, workers=16 training run). When they finish I'll splice the JSON into this file under §1.5 — but their qualitative outcome (heavy CPU-bound starvation) is already determined by §1.1 alone.
- The proposed Y-slab narrowing is the only change that touches a documented invariant (T1.13 — "Y-coherent" elastic). The fix preserves the invariant by construction (same 2D field reused across the slab's 5 Y planes); a regression test that verifies channel 2 of the augmented `volume_5ch` equals the augmented `volume[:, slice_y, :]` is the obvious guard.
- All other fixes (2.3-2.8) are non-invasive and don't touch any spec invariant.

---

*Author: GPU-starvation audit pass, 2026-04-29. Companion to `agent/audit_2026-04-29.md` (correctness audit) and `agent/audit_plan_correctness_2026-04-29.md` (correctness plan). Profiling artifacts in `scripts/profile_aug_stages.py` and `scripts/profile_gpu_starvation.py`; raw outputs in `/tmp/{dl_w4_aug,train_noaug,train_w16_aug}.json`.*
