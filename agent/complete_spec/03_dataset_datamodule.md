# Component 3 — Dataset + Lightning DataModule

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/dataset.py`, `src/datamodule.py`
**Date:** 2026-04-27
**Companion:** Implements §3.2, §3.3, §5 of `agent/training_pipeline_decisions_phase1.md`. Provides the I/O surface that Components 4 (augmentation) and 5 (sampler/HNM) plug into.

---

## 1. Purpose

Provide the read-side data layer for training, validation, and post-training inference. Component 3 owns:

- Eager-loading the entire preprocessed cache into RAM at `setup()`.
- Slice-level `Dataset.__getitem__` that returns one 5-channel slice triplet per call.
- Fold-aware patient selection from `splits.json`.
- Plug-in points for augmentation (Component 4) and sampling policy (Component 5) — both injected as constructor arguments. Default behavior is no augmentation + uniform sampling, so Component 3 is fully testable in isolation.

**Component 3 is intentionally agnostic about augmentation, sampling weights, hard-negative mining, and inference orchestration.** Those are Components 4, 5, 7.

---

## 2. Scope

**In scope:**

- `LesionDataset` class — slice-level `__getitem__`, fold-aware patient list, RAM-resident volume cache.
- `LesionDataModule` class — Lightning DataModule wrapping train/val/test datasets, dataloader construction, RAM allocation in `setup()`.
- Default uniform sampler and identity augmentation so the DataModule yields valid batches with no extra plumbing.
- Validation iteration: deterministic, no augmentation, slice-level (per Q6 — train-time eval is slice proxies only).
- Test-set / inference iteration: deterministic, no augmentation, full-volume sweep (one slice at a time, all valid `k`), ordered for downstream WBF aggregation.
- A separate `setup_inference()` mode for running over holdout (gated by an explicit `allow_holdout=True` flag — guards against accidental holdout leakage).

**Out of scope:**

- Augmentation transforms — Component 4.
- Positive oversampling, mix scheduling, hard-negative mining — Component 5.
- Volume-level metrics, FROC, AUROC aggregation — Component 7.

---

## 3. Inputs

| Input | Path | Used for |
|---|---|---|
| Volume cache | `cache/v1/volumes/<patient>/volume.npy` | Image data |
| Lesion mask cache | `cache/v1/volumes/<patient>/lesion_mask.npy` (positives only) | Per-slice GT labels |
| Border-band cache | `cache/v1/border_bands/<patient>.npy` (CV cohort) | Forwarded to augmentation (paste-site selection) |
| Preprocessed manifest | `cache/v1/preprocessed_manifest.csv` | Patient list, fold, cohort, label, pad offsets |
| GT boxes | `cache/v1/gt_boxes.parquet` | Per-slice 2D box list |
| Splits | `data/splits.json` | Fold assignment (mirrors manifest; cross-check) |

---

## 4. Outputs (downstream contract)

### 4.1 Sample format returned by `Dataset.__getitem__`

```python
@dataclass
class Sample:
    # Image data — pre-augmentation, pre-jitter
    volume_5ch: np.ndarray         # float32 (5, 384, 384) — already sliced to the 5-channel center triplet
    lesion_mask_center: np.ndarray # uint8   (384, 384)    — center-slice lesion mask
    boxes: np.ndarray              # float32 (N, 4)        — (x1, z1, x2, z2) in slice coords; N may be 0
    labels: np.ndarray             # int64   (N,)          — all zeros (single-class detection); shape matches boxes
    # Metadata
    patient_id: str
    slice_y: int                   # center-slice index in the cropped+padded (384, 160, 384) frame
    is_positive_volume: bool       # patient-level label
    is_positive_slice: bool        # this specific slice contains lesion voxels
    pad_offset: tuple[int, int, int]  # forwarded for any back-projection downstream
    # Forwarded for Component 4 (augmentation)
    volume_full_cropped: np.ndarray | None   # float16 (384, 160, 384) — populated when aug is active; None at val/inference for memory
    lesion_mask_full_cropped: np.ndarray | None  # uint8  (384, 160, 384)
    border_band_coords: np.ndarray | None    # int16  (M, 3) in cropped (384, 160, 384) frame; None for holdout
```

**Note on Sample shape:** at training time, the augmentation hook (Component 4) needs the full cropped volume to perform geometric augs and lesion paste *before* the 5-channel slice is finally extracted. At validation/inference, no augmentation is applied, so we can short-circuit and only emit the 5-channel triplet. The Dataset accepts an `augment: callable | None` argument:

- If `augment is None`: emit only `volume_5ch` etc.; full arrays are `None`. (Validation/inference path.)
- If `augment is not None`: emit full cropped arrays + center-slice references; the augmentation callable is responsible for slicing the 5-channel triplet at the end of its pipeline.

This split keeps validation memory low and avoids redundant work.

### 4.2 Batch format (after `default_collate`)

```python
@dataclass
class Batch:
    volume_5ch: torch.Tensor       # float32 (B, 5, 384, 384)
    lesion_mask_center: torch.Tensor  # uint8 (B, 384, 384)
    boxes: list[torch.Tensor]      # list of length B; per-image tensor (N_i, 4)
    labels: list[torch.Tensor]     # list of length B; per-image tensor (N_i,)
    patient_ids: list[str]         # length B
    slice_ys: torch.Tensor         # int64 (B,)
    is_positive_volume: torch.Tensor  # bool (B,)
    is_positive_slice: torch.Tensor   # bool (B,)
```

`boxes` and `labels` are lists (not stacked tensors) because `N_i` varies per slice. Standard detection-head input format.

A custom `collate_fn` handles this. RTMDet head accepts `list[Tensor]` for boxes/labels.

### 4.3 DataModule public surface

```python
class LesionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cache_root: Path,
        splits_path: Path,
        fold: int,                          # 0..4
        batch_size: int = 8,
        num_workers: int = 8,
        augment_train: Callable | None = None,    # Component 4 hook
        sampler_train: Sampler | None = None,     # Component 5 hook
        slice_window: int = 5,                    # 5-channel triplet per §3.2
        target_input_shape: tuple[int, int, int] = (384, 160, 384),
        allow_holdout: bool = False,              # MUST be True to access holdout
    ): ...

    def setup(self, stage: str): ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...

    # Used only by the post-training Component 7 (and the inference script).
    # Yields one slice at a time, all valid k, in patient-grouped order.
    def inference_dataloader(self, patient_ids: list[str]) -> DataLoader: ...
```

---

## 5. Coordinate frames and sub-cropping

The cache holds `(408, 174, 408)` arrays. The model input is `(384, 160, 384)`. The sub-crop is the **paste-first ordering** described in Component 1's design note:

```
RAM-resident cache (408, 174, 408)
        │
        ▼
[1] Sub-crop (384, 160, 384) at jitter offset (jx, jy, jz)
        │   train: jitter offset uniform in [-12, +12] x [-7, +7] x [-12, +12]
        │   val/inference: jitter offset = (12, 7, 12)  — exact center
        ▼
[2] Translate border_band coords by -(jx, jy, jz), filter to valid range
        │
        ▼
[3] Apply paste augmentation (Component 4) — modifies volume + lesion_mask in-place
        │
        ▼
[4] Apply geometric aug (Component 4) — rotation/scale/elastic on volume + lesion_mask
        │
        ▼
[5] Apply intensity aug (Component 4) — γ/bias/noise on volume only
        │
        ▼
[6] Re-derive boxes from final lesion_mask via scipy.ndimage.label (per slice_y range)
        │
        ▼
[7] Extract 5-channel slice [k-2..k+2] at center-slice k
        │
        ▼
Sample
```

Steps 1–2 belong to Component 3. Steps 3–6 belong to Component 4. Step 7 is shared (Component 3 owns the slice-extraction primitive; Component 4 calls it).

For validation/inference: skip steps 3–5; step 6 is unnecessary because we use cached `gt_boxes.parquet` directly (val volumes are not augmented, so cached boxes apply); step 1 uses the centered offset `(12, 7, 12)`.

---

## 6. Class definitions (skeleton)

```python
# src/dataset.py
class LesionDataset(Dataset):
    """Slice-level dataset over a fold's worth of cached patients."""

    def __init__(
        self,
        patient_ids: list[str],
        cache: dict[str, dict],     # {pid: {volume, lesion_mask | None, border_band | None}}
        gt_boxes_by_pid_slice: dict[tuple[str, int], np.ndarray],
        manifest_lookup: dict[str, dict],   # rows from preprocessed_manifest.csv
        slice_index: list[tuple[str, int, bool]],  # (patient_id, slice_y, is_positive_slice)
        target_input_shape: tuple[int, int, int],
        slice_window: int,
        augment: Callable | None,
    ):
        self.patient_ids = patient_ids
        self.cache = cache
        self.gt_lookup = gt_boxes_by_pid_slice
        self.manifest = manifest_lookup
        self.slice_index = slice_index
        self.target_shape = target_input_shape
        self.slice_window = slice_window
        self.augment = augment

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Sample:
        patient_id, slice_y_cached, is_positive_slice = self.slice_index[idx]
        entry = self.cache[patient_id]

        if self.augment is None:
            return self._build_inference_sample(patient_id, slice_y_cached, entry, is_positive_slice)
        return self._build_training_sample(patient_id, slice_y_cached, entry, is_positive_slice)

    def _build_inference_sample(...): ...
    def _build_training_sample(...):
        # Steps 1-2 of §5: sub-crop and border-band translation
        # Hands off to self.augment(...) which executes steps 3-7
        ...
```

```python
# src/datamodule.py
class LesionDataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        # 1. Read preprocessed_manifest.csv, splits.json
        # 2. Resolve patient lists for fold:
        #    - train: cohort='cross-validation' AND fold != self.fold
        #    - val:   cohort='cross-validation' AND fold == self.fold
        #    - holdout: cohort='holdout' (only loaded if allow_holdout=True)
        # 3. Eager-load every needed patient into self.cache (~38 GB)
        # 4. Build gt_boxes_by_pid_slice from cache/v1/gt_boxes.parquet
        # 5. Build slice_index for train and val:
        #    - train: every valid (patient_id, slice_y) where slice_y in [slice_window//2, target_shape[1] - slice_window//2)
        #    - val:   same range, but only patients in val fold
        # 6. Instantiate self.train_dataset, self.val_dataset
        ...
```

---

## 7. RAM strategy

- **Eager load.** `setup()` loads every needed patient via `np.load(path)` (no mmap). Volumes converted to a stable in-memory dict keyed by `patient_id`.
- **Total budget:** 38 GB cohort cache + ~5 MB lesion bank (loaded by Component 4) + ~50 MB border bands. Comfortable on 250 GB node (verified `free -g`).
- **Worker copies:** PyTorch DataLoader uses `fork` on Linux; child workers see the parent's loaded arrays as copy-on-write. As long as the dataset is read-only, RSS stays at ~38 GB regardless of `num_workers`. Reads do not trigger CoW (numpy refcount lives in the array object header, not the data buffer).
- **`persistent_workers=True`** so workers persist across epochs and don't re-fork.
- **No mmap.** First-epoch performance is at full speed; no page-fault stalls.

---

## 8. Sampler defaults

Component 3 ships a default `UniformSliceSampler` so it's testable without Component 5:

```python
class UniformSliceSampler(Sampler):
    """Yields random integers in [0, len(dataset)). Replacement, fixed-length epoch."""
    def __init__(self, dataset_len: int, num_samples_per_epoch: int, seed: int = 42):
        ...
    def __iter__(self): ...
    def __len__(self): return self.num_samples_per_epoch
```

`num_samples_per_epoch` defaults to `dataset_len` so the default behavior matches "see every slice once on average."

Component 5 will replace this with `WeightedScheduledSampler` (positive oversampling + epoch-aware mix) and integrate the hard-negative mining.

---

## 9. Validation dataloader

- **Always uses the centered jitter offset** `(12, 7, 12)` so val is deterministic.
- **No augmentation** (`augment=None`).
- **Iterates every valid (patient_id, slice_y)** in val fold. Order is `(patient_id ASC, slice_y ASC)` so per-patient slices stay grouped (helpful if val is ever extended to volume-level proxies).
- **Batch size:** same as train (8). Shuffle: False.
- **Per-fold val slice count:** ~80 patients × ~150 valid slices ≈ 12K slices. At ~80 ms/batch (model fwd, no aug, no aux loss heavy lifting), full pass ≈ 2 min on L40S.

---

## 10. Inference dataloader

Used by the post-training Component 7 + the holdout inference script.

```python
def inference_dataloader(self, patient_ids: list[str]) -> DataLoader:
    """Yields slices in (patient_id, slice_y) order, batch_size=8, no aug, no shuffle.
       Caller is responsible for grouping slice outputs by patient_id for WBF."""
```

- Refuses to include any holdout patient unless `self.allow_holdout` is True.
- Otherwise behaves identically to `val_dataloader` but with caller-supplied patient list.

---

## 11. Holdout protection

Two layers:

1. **Construction guard:** `LesionDataModule(..., allow_holdout=False)` raises if `setup()` ever loads a `cohort='holdout'` patient into `self.cache`. Default is `False`. The training `train.py` script sets `allow_holdout=False` permanently. The holdout inference script (Component 7's holdout entrypoint) is the only caller that sets `allow_holdout=True`.
2. **Runtime guard:** `inference_dataloader(patient_ids)` cross-checks against `self.allow_holdout` and raises if any requested patient is in the holdout cohort and the flag is False.

Both guards must trip before any holdout data can enter the dataloader.

---

## 12. Test plan

Tests live in `tests/dataset/`. Run via `uv run pytest tests/dataset/`.

### 12.1 Unit tests (synthetic cache fixtures)

Use a tiny synthetic cache built in `conftest.py` with 4 patients (2 positive CV, 1 negative CV, 1 holdout-positive), each `(40, 20, 40)` arrays for fast tests.

| Test | Setup | Assertion |
|---|---|---|
| `test_dataset_len_matches_slice_index` | Synthetic dataset over 3 patients, slice_window=5 | `len(ds) == sum(valid_slice_count_per_patient)` where valid range is `[2, 18)` |
| `test_dataset_returns_5ch_correct_shape` | Default `augment=None`, single sample | `sample.volume_5ch.shape == (5, 40, 40)` |
| `test_dataset_5ch_center_alignment` | Sample at slice_y=10 | Channel 2 of `volume_5ch` equals `volume[:, 10, :]` after centered crop |
| `test_dataset_boxes_match_lookup` | Positive sample with known GT box | `sample.boxes` matches `gt_boxes.parquet` for that (pid, slice_y) |
| `test_dataset_no_boxes_for_negative_slice` | Negative slice from a positive volume | `sample.boxes.shape == (0, 4)` |
| `test_dataset_metadata_correct` | Various samples | `is_positive_volume`, `is_positive_slice`, `patient_id`, `slice_y` correct |
| `test_inference_path_no_full_arrays` | `augment=None` sample | `sample.volume_full_cropped is None` |
| `test_training_path_includes_full_arrays` | `augment=identity` sample | `sample.volume_full_cropped.shape == (384, 160, 384)` (or synth shape equivalent) |
| `test_jitter_centered_at_validation` | val sample | `pad_offset` reflects centered crop, not jittered |
| `test_border_band_translated_correctly` | training sample with known jitter offset | All `border_band_coords` are in `[0, target_shape)` and reflect the -jitter shift |
| `test_collate_fn_lists_for_boxes` | Batch of 4 samples with mixed N | `batch.boxes` is `list[Tensor]` of length 4 |
| `test_holdout_blocked_by_default` | DataModule with `allow_holdout=False`, attempt to setup with holdout patient in fold | Raises `HoldoutAccessError` |
| `test_holdout_inference_dataloader_refuses` | `allow_holdout=False`, call `inference_dataloader([holdout_pid])` | Raises |
| `test_holdout_inference_dataloader_allows` | `allow_holdout=True`, call same | Returns valid DataLoader |
| `test_uniform_sampler_seeded_reproducible` | Two passes with same seed | Yield identical index sequences |

### 12.2 Integration tests (real cache, fold 0)

Requires Components 1 + 2 to have run on the real cohort.

| Test | Assertion |
|---|---|
| `test_real_setup_loads_correct_patient_count` | After `setup()`, `len(self.cache)` == train_count + val_count for fold 0 (~480 + ~120 patients) |
| `test_real_ram_within_budget` | Process RSS after `setup()` < 50 GB (38 GB cache + Python overhead) |
| `test_real_train_dataloader_yields` | First batch of `train_dataloader()` returns valid `Batch` shapes |
| `test_real_val_dataloader_full_pass` | Iterating `val_dataloader()` to exhaustion succeeds and yields ~12K samples |
| `test_real_no_holdout_in_train_or_val` | No `cohort=='holdout'` patient appears in any sample's `patient_id` |
| `test_real_dataloader_throughput` | At `num_workers=8`, train_dataloader yields ≥ 30 batches/sec (no model, no aug) |
| `test_real_box_validity_in_val_pass` | Every box in val pass is inside `[0, 384) × [0, 384)` |
| `test_real_positive_slice_fraction_correct` | Fraction of `is_positive_slice=True` in train pass matches positive-slice prevalence (~6%) |

### 12.3 Smoke gate (full DataModule, real data)

Acceptance gate before moving to Component 4:

1. `setup(stage='fit')` completes in < 90 s on the cohort.
2. RSS after setup < 50 GB.
3. One full epoch of train_dataloader (with default uniform sampler, no aug, no model) completes in < 8 min wall-clock — establishes the baseline data throughput.
4. Holdout protection trips on every prohibited code path (§11 tests pass).
5. Validation pass yields slice-level GT boxes that round-trip to `gt_boxes.parquet` exactly (sanity check on coordinate frames).

---

## 13. Logging

DataModule logs to `logs/datamodule_<run_id>.log`:

- `setup()` patient counts per cohort/fold/label
- RAM after load
- Slice index size per dataset
- Per-epoch first-batch latency (sentinel for regression)

---

## 14. Failure modes

| Failure | Detection | Action |
|---|---|---|
| Holdout patient leaks into train/val | `setup()` guard | Raises `HoldoutAccessError`; do not mask, do not continue |
| Cache file missing for a manifest patient | `np.load` raises | Hard-fail; preprocessing didn't complete |
| `gt_boxes.parquet` row references slice outside `[0, 174)` | setup-time validation | Hard-fail; Component 1 bug |
| Batch with all-empty boxes (no slice in batch is positive) | sentinel log | OK; expected occasionally with default uniform sampler. Component 5 will balance this. |
| Sample with negative box coords | `_build_*_sample` assertion | Hard-fail; coordinate-frame bug |

---

## 15. Estimated wall-clock

- `setup()` real cohort: ~60 s (38 GB sequential `np.load` from `/scratch`).
- One epoch train (no model, no aug): ~5 min wall-clock with 8 workers.
- Test suite: ~30 s unit + ~3 min integration on real cache.

---

## 16. Acceptance checklist (Component 3 done)

- [ ] `src/dataset.py` and `src/datamodule.py` exist with the public surface in §4.3.
- [ ] All §12.1 unit tests pass.
- [ ] All §12.2 integration tests pass on real cache.
- [ ] Smoke gate (§12.3) passes.
- [ ] Holdout protection (§11) verified across all code paths.
- [ ] Default uniform sampler ships and works.
- [ ] DataModule importable and instantiable via `LesionDataModule(...)`.

When this checklist is green, Component 4 (Augmentation) can begin.
