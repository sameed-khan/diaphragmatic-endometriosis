# Component 5 — Sampler + Periodic Deep Eval

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/sampler.py`, `src/periodic_eval_callback.py`, `src/inference_pass.py`
**Date:** 2026-04-27
**Companion:** Implements §5.1 / §5.3 of `agent/training_pipeline_decisions_phase1.md`. Plugged into Component 3 (DataModule) via the `sampler_train` argument and into Component 6 (LightningModule) via Lightning callbacks. Writes the inference cache that Component 7 consumes.

---

## 1. Purpose

Three coupled responsibilities:

1. **Weighted, epoch-aware slice sampling** — implements the §5.1 50/25/25 → 25/37.5/37.5 mix decay, the §5.3 center-slice index policy, and the hard-negative substitution.
2. **Per-batch hard-negative signal** — maintains a per-slice training-loss EMA for negative slices as a continuous, free signal of FP-proneness.
3. **Periodic deep-eval pass** (every 10 epochs starting epoch 10) — runs full inference on **(a)** training negatives → top-K hard pool refresh, AND **(b)** validation set → volume-level metrics for Component 7. This single inference pass serves both purposes; Component 7 only re-computes from the cached scores.

The sampler reads from a hard pool that is the **union** of (top of loss-EMA) and (top of deep-eval scores). The two signals complement: EMA gives recent-but-noisy ranking from in-training samples; deep-eval gives stable-but-stale ranking from a frozen-snapshot inference pass.

---

## 2. Scope

**In scope:**

- `WeightedScheduledSampler(Sampler)` — weighted sampling with epoch-aware mix, hard pool integration, fixed epoch length.
- `LossEMATracker` — per-slice EMA of negative-slice loss, updated by Component 6 LightningModule on each train batch.
- `PeriodicDeepEvalCallback(pl.Callback)` — runs the dedicated 10-epoch inference pass over (training negatives + val set), refreshes hard pool, writes deep-eval cache for Component 7.
- `inference_pass()` — shared utility that runs model in eval mode over an arbitrary patient list, returns per-slice scores. Reused by Component 7.

**Out of scope:**

- The model itself, the training loop, loss computation — Component 6.
- Volume-level metric computation (AUROC, FROC, AP) — Component 7 (consumes the deep-eval cache produced here).
- Post-training final eval — Component 7.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ DataModule (Component 3)                                        │
│   train_dataloader uses → WeightedScheduledSampler              │
└──────────────────┬──────────────────────────────────────────────┘
                   │ set_epoch(n) per epoch
                   │ reads hard_negatives.json
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ WeightedScheduledSampler                                        │
│   - epoch-aware mix p_pos[epoch]                                │
│   - reads hard pool (loss_ema + deep_eval) at __iter__          │
└─────────────────────────────────────────────────────────────────┘

LightningModule (Component 6) per train batch:
┌──────────────────┐
│ training_step    │ → LossEMATracker.update(slice_id, loss)
└──────────────────┘

LightningModule (Component 6) on validation_epoch_end:
┌──────────────────┐
│ Lightning val    │ → cheap slice-level proxy metrics (Component 6 owns)
└──────────────────┘
       │
       ▼ (Lightning fires callbacks)
┌─────────────────────────────────────────────────────────────────┐
│ PeriodicDeepEvalCallback.on_validation_epoch_end                │
│   if epoch >= 10 and epoch % 10 == 0:                           │
│     scores_val = inference_pass(model, val_patient_ids)         │
│     scores_neg = inference_pass(model, train_negative_pids)     │
│     write deep_eval_cache_epoch{n}.npz   [for Component 7]      │
│     write hard_negatives.json            [for Sampler]          │
│     log val volume AUROC, FROC@2FP to W&B [coarse periodic]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. WeightedScheduledSampler

```python
# src/sampler.py

@dataclass
class SamplerConfig:
    pos_frac_start: float = 0.50
    pos_frac_end: float = 0.25
    decay_epochs: int = 30                     # linear decay
    neg_in_pos_vol_share: float = 0.50         # fraction of negative budget from positive vols
    hard_pool_substitution_rate: float = 0.30  # fraction of negative-vol draws that come from hard pool
    hard_pool_start_epoch: int = 5
    epoch_mode: Literal["fixed_count", "full_pass"] = "fixed_count"
    samples_per_epoch: int = 6000              # used when epoch_mode == "fixed_count"
                                               # default ≈ 2× n_positive_slices for ~3 GPU-h/fold on L40S
                                               # set to len(train_dataset) (~75K) when epoch_mode == "full_pass"
    seed: int = 42

class WeightedScheduledSampler(Sampler):
    def __init__(
        self,
        positive_slices: list[int],                  # indices into Dataset
        negative_slices_in_positive_vols: list[int],
        negative_slices_in_negative_vols: list[int],
        hard_pool_path: Path,                        # cache/v1/runtime/hard_negatives.json
        loss_ema_tracker: LossEMATracker,
        cfg: SamplerConfig,
    ): ...

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # Compute current p_pos via linear decay
        t = min(epoch / self.cfg.decay_epochs, 1.0)
        self.p_pos = self.cfg.pos_frac_start + t * (self.cfg.pos_frac_end - self.cfg.pos_frac_start)

    def _build_hard_pool(self) -> list[int]:
        """Union of top-K from deep-eval and top-K from loss EMA."""
        deep_eval_pool = []
        if self.hard_pool_path.exists():
            deep_eval_pool = json.loads(self.hard_pool_path.read_text())["slice_indices"]
        ema_pool = self.loss_ema_tracker.top_k(k=1000)
        return list(set(deep_eval_pool + ema_pool))   # dedup; up to 2K total

    def __iter__(self):
        rng = np.random.default_rng(self.cfg.seed + self.epoch)
        if self.cfg.epoch_mode == "full_pass":
            n = (len(self.positive_slices)
                 + len(self.negative_slices_in_positive_vols)
                 + len(self.negative_slices_in_negative_vols))
        else:
            n = self.cfg.samples_per_epoch

        hard_pool = self._build_hard_pool() if self.epoch >= self.cfg.hard_pool_start_epoch else []
        use_hard_pool = len(hard_pool) > 0

        for _ in range(n):
            r = rng.random()
            if r < self.p_pos:
                yield rng.choice(self.positive_slices)
            elif r < self.p_pos + (1 - self.p_pos) * self.cfg.neg_in_pos_vol_share:
                yield rng.choice(self.negative_slices_in_positive_vols)
            else:
                # Negative-vol slice: maybe substitute from hard pool
                if use_hard_pool and rng.random() < self.cfg.hard_pool_substitution_rate:
                    yield rng.choice(hard_pool)
                else:
                    yield rng.choice(self.negative_slices_in_negative_vols)

    def __len__(self):
        if self.cfg.epoch_mode == "full_pass":
            return (len(self.positive_slices)
                    + len(self.negative_slices_in_positive_vols)
                    + len(self.negative_slices_in_negative_vols))
        return self.cfg.samples_per_epoch
```

### 4.1 Epoch-mix table (point checks)

| Epoch | p_pos | p_neg_in_pos_vol | p_neg_in_neg_vol |
|---|---|---|---|
| 0  | 0.500 | 0.250 | 0.250 |
| 10 | 0.417 | 0.292 | 0.292 |
| 20 | 0.333 | 0.333 | 0.333 |
| 30+ | 0.250 | 0.375 | 0.375 |

### 4.2 Sampler parent-process semantics

Per Q6: `__iter__` runs in the parent process. `np.random.default_rng(seed + epoch)` ensures reproducibility for a given fold + epoch. Workers fork after `__iter__` has yielded indices, so all workers see the same per-epoch sequence. Calling `set_epoch(n)` between epochs is the responsibility of Lightning (it does this automatically via `DistributedSampler.set_epoch`-style hooks; we mirror the API).

---

## 5. LossEMATracker

```python
# src/sampler.py (or src/loss_ema.py)

class LossEMATracker:
    """Per-slice EMA of training loss for negative slices.
       Updated by LightningModule.training_step on each batch.
       Read by WeightedScheduledSampler at epoch boundary."""

    def __init__(self, ema_decay: float = 0.9, k_top: int = 1000):
        self.ema = {}      # {(patient_id, slice_y): ema_loss}
        self.decay = ema_decay
        self.k_top = k_top

    def update(self, slice_id: tuple[str, int], loss: float, is_negative_slice: bool):
        if not is_negative_slice:
            return
        if slice_id in self.ema:
            self.ema[slice_id] = self.decay * self.ema[slice_id] + (1 - self.decay) * loss
        else:
            self.ema[slice_id] = loss

    def top_k(self, k: int | None = None) -> list[int]:
        """Return the k slice indices (Dataset indices, not slice_ids) with highest EMA loss."""
        k = k or self.k_top
        # Returns Dataset indices; tracker holds a slice_id → dataset_idx map populated at construction
        ...
```

The tracker lives on the LightningModule (in `__init__`). Component 6 calls `tracker.update(slice_id, loss, is_negative_slice)` per training sample (in `training_step` after loss compute, before backward — easy to plumb since `Batch` carries `patient_ids`, `slice_ys`, `is_positive_slice`).

The sampler holds a reference to the tracker (passed in constructor). Sampler reads `tracker.top_k()` at each epoch boundary.

**Memory:** at most ~75K negative slices × ~50 bytes each ≈ 4 MB. Trivial.

---

## 6. PeriodicDeepEvalCallback

```python
# src/periodic_eval_callback.py

@dataclass
class PeriodicDeepEvalConfig:
    refresh_every_epochs: int = 10
    start_epoch: int = 10                # first deep eval at end of epoch 10
    hard_pool_size: int = 1000
    output_dir: Path = Path("cache/v1/runtime/deep_eval")

class PeriodicDeepEvalCallback(pl.Callback):
    def __init__(
        self,
        cfg: PeriodicDeepEvalConfig,
        datamodule: LesionDataModule,
        train_negative_patient_ids: list[str],
    ): ...

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._should_run(trainer.current_epoch):
            return

        # Pass A: val set (used by Component 7 for volume metrics + by W&B for coarse logging)
        val_scores = inference_pass(
            model=pl_module,
            datamodule=self.datamodule,
            patient_ids=self.datamodule.val_patient_ids,
            split="val",
        )

        # Pass B: training negatives (used for HNM hard pool)
        train_neg_scores = inference_pass(
            model=pl_module,
            datamodule=self.datamodule,
            patient_ids=self.train_negative_patient_ids,
            split="train_negatives",
        )

        # Save deep eval cache for Component 7
        epoch = trainer.current_epoch
        np.savez_compressed(
            self.cfg.output_dir / f"epoch{epoch}_val.npz",
            **val_scores,
        )

        # Refresh hard pool
        hard_pool_indices = self._top_k_negative_slices(train_neg_scores, k=self.cfg.hard_pool_size)
        (self.cfg.output_dir.parent / "hard_negatives.json").write_text(
            json.dumps({"epoch": epoch, "slice_indices": hard_pool_indices}, indent=2)
        )

        # Cheap coarse volume metrics → W&B
        coarse = compute_coarse_volume_metrics(val_scores, self.datamodule.val_gt)
        pl_module.log_dict({
            "deep_eval/val_volume_auroc_coarse": coarse["volume_auroc"],
            "deep_eval/val_froc_at_2fp_coarse": coarse["sens_at_2fp"],
        }, sync_dist=False)
```

`compute_coarse_volume_metrics` is a thin convenience function that lives in `src/inference_pass.py` and delegates to Component 7's metric primitives. The "coarse" qualifier reflects: no bootstrap CIs, no stratified breakdowns, no AP — just the two headline numbers for periodic monitoring. Component 7's full eval at end-of-training computes the rest from the same cached scores.

---

## 7. Shared inference primitive

```python
# src/inference_pass.py

@dataclass
class SliceScore:
    patient_id: str
    slice_y: int
    boxes: np.ndarray       # (N, 4)
    scores: np.ndarray      # (N,) — RTMDet decoded box confidences post per-slice NMS
    aux_seg_max: float      # max sigmoid value of aux seg head (slice-level presence proxy)

def inference_pass(
    model: pl.LightningModule,
    datamodule: LesionDataModule,
    patient_ids: list[str],
    split: str,                    # "val" | "train_negatives" | "holdout"
    batch_size: int = 16,
) -> dict[str, list[SliceScore]]:
    """Run model in eval mode on every valid slice of every patient_id.
       Returns {patient_id: [SliceScore for each slice_y in valid range]}.
       Caller is responsible for grouping/aggregation (WBF, FROC, etc.)."""
    ...
```

This primitive is reused by:

- `PeriodicDeepEvalCallback` (during training).
- Component 7's final post-training eval script.
- Component 7's holdout inference script.

Single implementation, single contract.

---

## 8. Output contracts

### 8.1 `cache/v1/runtime/hard_negatives.json`

```json
{
  "epoch_written": 20,
  "model_checkpoint_epoch": 20,
  "slice_indices": [12345, 67890, ...],
  "n_slices": 1000,
  "score_threshold": 0.42
}
```

Replaced atomically (write to `.tmp`, `os.replace`).

### 8.2 `cache/v1/runtime/deep_eval/epoch{n}_val.npz`

Compressed npz with arrays:

- `patient_ids` — `(N_slices,)` str
- `slice_ys` — `(N_slices,)` int32
- `boxes_flat` — `(M, 4)` float32 (concatenated across slices)
- `scores_flat` — `(M,)` float32
- `box_offsets` — `(N_slices + 1,)` int32 (CSR-style indexing into `boxes_flat`/`scores_flat`)
- `aux_seg_max` — `(N_slices,)` float32

Component 7 reads these at end-of-training (or any time) and runs WBF + FROC + AUROC on top.

---

## 9. Test plan

Tests in `tests/sampler/`. Run via `uv run pytest tests/sampler/`.

### 9.1 Unit tests (synthetic indices)

| # | Test | Assertion |
|---|---|---|
| S1 | `test_sampler_p_pos_decay_schedule` | At epochs 0, 10, 20, 30, 60: p_pos matches §4.1 table |
| S2 | `test_sampler_mix_at_epoch_0` | 100K samples; pos ≈ 50%, neg-in-pos-vol ≈ 25%, neg-in-neg-vol ≈ 25% (±1%) |
| S3 | `test_sampler_mix_at_epoch_30` | 100K samples; pos ≈ 25%, neg-in-pos-vol ≈ 37.5%, neg-in-neg-vol ≈ 37.5% (±1%) |
| S4 | `test_sampler_seeded_reproducible` | Two passes same seed + epoch → identical sequences |
| S5 | `test_sampler_hard_pool_substitution_off_pre_epoch_5` | Epoch=4: zero substitution from hard pool even when JSON exists |
| S6 | `test_sampler_hard_pool_substitution_on_post_epoch_5` | Epoch=10 with 1000-element hard pool: ~30% of neg-in-neg-vol draws come from pool |
| S7 | `test_sampler_fixed_epoch_length` | `len(sampler) == samples_per_epoch` |
| S8 | `test_loss_ema_initialization` | First update sets value; subsequent updates use EMA decay |
| S9 | `test_loss_ema_skips_positive_slices` | Update with `is_negative_slice=False` → no entry created |
| S10 | `test_loss_ema_top_k` | Construct EMA with known values; `top_k(5)` returns correct indices |
| S11 | `test_inference_pass_returns_correct_shape` | Mock model, 3 patients, 10 slices each → 30 SliceScore entries |
| S12 | `test_periodic_callback_skips_pre_start_epoch` | `current_epoch=5`, start_epoch=10 → callback no-op |
| S13 | `test_periodic_callback_writes_hard_negatives_json` | At epoch 10: file written with correct schema |
| S14 | `test_periodic_callback_writes_deep_eval_cache` | At epoch 10: npz written with all expected arrays |

### 9.2 Integration tests (real cache + real model stub)

| # | Test | Assertion |
|---|---|---|
| S15 | `test_real_sampler_reads_real_dataset` | Build sampler over real fold-0; iterate one epoch → indices all valid |
| S16 | `test_real_callback_runs_with_lightning` | Mini Lightning trainer, 11 epochs, mock model; callback fires at epoch 10 and writes both files |
| S17 | `test_real_inference_pass_throughput` | Run inference_pass on 100 real patients; ≥ 50 slices/s on L40S with mock model |
| S18 | `test_real_deep_eval_npz_roundtrip` | Write deep-eval cache; load it; reconstruct per-patient SliceScore lists |

### 9.3 Acceptance gate

Before Component 6 begins:

1. All unit + integration tests pass.
2. `WeightedScheduledSampler` swappable for `UniformSliceSampler` in DataModule with no DataModule changes (interface compat).
3. `PeriodicDeepEvalCallback` registers with Lightning trainer and fires only at configured epochs.
4. `inference_pass()` is the single inference primitive — Components 7 and the holdout script will consume it directly.

---

## 10. Logging

Per epoch (info-level):
- `sampler/p_pos`, `sampler/hard_pool_size`, `sampler/hard_pool_substitution_active`
- `loss_ema/n_tracked`, `loss_ema/top1_loss`, `loss_ema/median_loss`

On deep-eval refresh (info-level):
- `deep_eval/epoch`, `deep_eval/val_inference_seconds`, `deep_eval/train_neg_inference_seconds`
- `deep_eval/val_volume_auroc_coarse`, `deep_eval/val_froc_at_2fp_coarse`
- `deep_eval/hard_pool_size`, `deep_eval/score_threshold`

W&B logs the coarse volume metrics so we have a 6-point trace over a 60-epoch training (epochs 10, 20, 30, 40, 50, 60).

---

## 11. Failure modes

| Failure | Detection | Action |
|---|---|---|
| `hard_negatives.json` corrupted/missing mid-run | sampler `__iter__` | Treat as empty pool; log warning; continue |
| Callback inference OOM | callback exception | Log + skip this refresh; sampler falls back to loss-EMA-only pool; alert |
| Loss EMA tracker keys grow unbounded | RSS monitor | Hard cap at 100K entries; LRU eviction (informational — should never happen at our scale) |
| Deep eval cache disk fill | `np.savez` raises | Hard fail; user must clean `cache/v1/runtime/deep_eval/` between runs (or implement retention policy) |
| Coarse metrics regress vs slice proxies | W&B trace | Sentinel only; investigate before next training epoch — a lagging volume signal can confirm a slice-proxy regression is real |

---

## 12. Wall-clock budget

- `inference_pass` over val (~12K slices) on L40S: ~1 min.
- `inference_pass` over training negatives (~72K slices): ~4 min.
- Per refresh total: ~5 min.
- 6 refreshes over 60-epoch training: 30 min total → ~10% of Stage-1 budget. Acceptable.

---

## 13. Acceptance checklist (Component 5 done)

- [ ] `src/sampler.py`, `src/periodic_eval_callback.py`, `src/inference_pass.py` exist with the APIs in §4–§7.
- [ ] All §9.1 unit tests pass.
- [ ] All §9.2 integration tests pass.
- [ ] `WeightedScheduledSampler` interface-compatible with `UniformSliceSampler` (Dataset accepts either).
- [ ] `PeriodicDeepEvalCallback` integrates with Lightning trainer and fires at configured epochs.
- [ ] `inference_pass()` documented as the shared primitive consumed by Component 7.
- [ ] Hard pool JSON + deep-eval npz schemas match §8 exactly.

When this checklist is green, Component 6 (Model + Training Loop) can begin.
