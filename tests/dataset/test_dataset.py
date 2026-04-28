"""Component 3 unit tests (synthetic mini-cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from endo.data.collate import collate_fn
from endo.data.dataset import LesionDataset
from endo.data.datamodule import HoldoutAccessError, LesionDataModule
from endo.data.samples import Sample

from tests.dataset.conftest import (
    CACHE_SHAPE,
    PAD_OFFSET,
    TARGET_SHAPE,
    build_gt_lookup,
    build_slice_index,
    load_in_memory_cache,
)


SLICE_WINDOW = 5
HALF = SLICE_WINDOW // 2

# Cache pad offset on Y; valid slice_y_cached range is [py + half, py + ty - half).
PY = PAD_OFFSET[1]
TY = TARGET_SHAPE[1]
SLICE_Y_LO = PY + HALF                # 4
SLICE_Y_HI = PY + TY - HALF           # 16


def _make_dataset(synth_cache, pids, augment=None):
    cache = load_in_memory_cache(synth_cache)
    gt = build_gt_lookup(synth_cache)
    idx = build_slice_index(cache, pids, gt, SLICE_Y_LO, SLICE_Y_HI)
    ds = LesionDataset(
        patient_ids=pids,
        cache=cache,
        gt_boxes_by_pid_slice=gt,
        slice_index=idx,
        target_input_shape=TARGET_SHAPE,
        slice_window=SLICE_WINDOW,
        augment=augment,
        rng_seed=0,
        cache_shape=CACHE_SHAPE,
    )
    return ds, cache, gt, idx


# ----------------------------------------------------------------------
# D1
# ----------------------------------------------------------------------

def test_dataset_len_matches_slice_index(synth_cache):
    pids = ["p_pos_cv_0", "p_pos_cv_1", "p_neg_cv_0"]
    ds, _, _, idx = _make_dataset(synth_cache, pids)
    assert len(ds) == len(idx)
    expected_per_pid = SLICE_Y_HI - SLICE_Y_LO  # 12
    assert len(ds) == len(pids) * expected_per_pid


# ----------------------------------------------------------------------
# D2
# ----------------------------------------------------------------------

def test_dataset_returns_5ch_correct_shape(synth_cache):
    pids = ["p_pos_cv_0"]
    ds, _, _, _ = _make_dataset(synth_cache, pids)
    sample = ds[0]
    assert isinstance(sample, Sample)
    # (C=5, H=Z=36, W=X=36)
    assert sample.volume_5ch.shape == (SLICE_WINDOW, TARGET_SHAPE[2], TARGET_SHAPE[0])
    assert sample.volume_5ch.dtype == np.float32
    assert sample.lesion_mask_center.shape == (TARGET_SHAPE[2], TARGET_SHAPE[0])
    assert sample.lesion_mask_center.dtype == np.uint8


# ----------------------------------------------------------------------
# D3
# ----------------------------------------------------------------------

def test_dataset_5ch_center_alignment(synth_cache):
    """At validation (centered jitter), channel HALF of volume_5ch must equal
    the center-slice of the cropped volume after the (Z, X) transpose.
    """
    pids = ["p_pos_cv_0"]
    ds, cache, _, idx = _make_dataset(synth_cache, pids, augment=None)
    # Find an entry with a known slice_y_cached so we can recompute the slice
    # in the cache frame.
    target_idx = next(
        i for i, (pid, sy, _ip, _k) in enumerate(idx) if pid == "p_pos_cv_0" and sy == 8
    )
    sample = ds[target_idx]
    pid, sy_cached, _ip, _k = idx[target_idx]
    # centered crop start = pad_offset
    px, py, pz = PAD_OFFSET
    tx, ty, tz = TARGET_SHAPE
    full = cache[pid]["volume"][px : px + tx, py : py + ty, pz : pz + tz].astype(np.float32)
    sy_target = sy_cached - py
    expected_center = full[:, sy_target, :].T  # (Z, X)
    np.testing.assert_allclose(sample.volume_5ch[HALF], expected_center, rtol=0, atol=0)


# ----------------------------------------------------------------------
# D4
# ----------------------------------------------------------------------

def test_dataset_boxes_match_lookup(synth_cache):
    """Validation path with centered jitter: returned boxes equal the cached
    boxes minus the centered pad-offset (since we crop by pad_offset).
    """
    pids = ["p_pos_cv_0"]
    ds, _, gt, idx = _make_dataset(synth_cache, pids, augment=None)
    # Pick a slice with boxes (sy_cached=8 has 1 box per the fixture plan).
    target_idx = next(
        i for i, (pid, sy, _ip, _k) in enumerate(idx) if pid == "p_pos_cv_0" and sy == 8
    )
    sample = ds[target_idx]
    pid, sy_cached, _ip, _k = idx[target_idx]
    px, _py, pz = PAD_OFFSET
    raw = gt[(pid, sy_cached)]
    expected = raw.copy()
    expected[:, 0] -= px
    expected[:, 2] -= px
    expected[:, 1] -= pz
    expected[:, 3] -= pz
    # Clip to target.
    expected = np.clip(expected, 0, max(TARGET_SHAPE[0], TARGET_SHAPE[2]))
    np.testing.assert_allclose(np.sort(sample.boxes, axis=0), np.sort(expected, axis=0))


# ----------------------------------------------------------------------
# D5
# ----------------------------------------------------------------------

def test_dataset_no_boxes_for_negative_slice(synth_cache):
    """A positive volume slice that has no GT box → empty boxes array."""
    pids = ["p_pos_cv_0"]
    ds, _, _, idx = _make_dataset(synth_cache, pids, augment=None)
    # Pick a slice within range that has no plan box (e.g., sy=5).
    target_idx = next(
        i for i, (pid, sy, ip, _k) in enumerate(idx) if pid == "p_pos_cv_0" and sy == 5 and not ip
    )
    sample = ds[target_idx]
    assert sample.boxes.shape == (0, 4)
    assert sample.labels.shape == (0,)
    assert sample.is_positive_volume is True
    assert sample.is_positive_slice is False


# ----------------------------------------------------------------------
# D6
# ----------------------------------------------------------------------

def test_dataset_inference_path_no_full_arrays(synth_cache):
    pids = ["p_pos_cv_0"]
    ds, _, _, _ = _make_dataset(synth_cache, pids, augment=None)
    sample = ds[0]
    assert sample.volume_full_cropped is None
    assert sample.lesion_mask_full_cropped is None
    assert sample.border_band_coords is None


# ----------------------------------------------------------------------
# D7
# ----------------------------------------------------------------------

def test_dataset_training_path_includes_full_arrays(synth_cache):
    pids = ["p_pos_cv_0"]
    ds, _, _, _ = _make_dataset(synth_cache, pids, augment=lambda s: s)
    sample = ds[0]
    assert sample.volume_full_cropped is not None
    assert sample.volume_full_cropped.shape == TARGET_SHAPE
    assert sample.volume_full_cropped.dtype == np.float32
    assert sample.lesion_mask_full_cropped is not None
    assert sample.lesion_mask_full_cropped.shape == TARGET_SHAPE
    assert sample.border_band_coords is not None
    # All band coords inside cropped frame.
    coords = sample.border_band_coords
    if coords.size:
        assert (coords >= 0).all()
        assert (coords[:, 0] < TARGET_SHAPE[0]).all()
        assert (coords[:, 1] < TARGET_SHAPE[1]).all()
        assert (coords[:, 2] < TARGET_SHAPE[2]).all()


# ----------------------------------------------------------------------
# D8
# ----------------------------------------------------------------------

def test_dataset_jitter_centered_at_validation(synth_cache):
    """At augment=None the crop is exactly centered: sample.slice_y == sy_cached - py."""
    pids = ["p_pos_cv_0"]
    ds, _, _, idx = _make_dataset(synth_cache, pids, augment=None)
    py = PAD_OFFSET[1]
    for i in range(len(ds)):
        pid, sy_cached, _ip, _k = idx[i]
        sample = ds[i]
        assert sample.slice_y == sy_cached - py
        assert sample.pad_offset == PAD_OFFSET


# ----------------------------------------------------------------------
# D10
# ----------------------------------------------------------------------

def test_collate_fn_lists_for_boxes(synth_cache):
    pids = ["p_pos_cv_0", "p_neg_cv_0"]
    ds, _, _, idx = _make_dataset(synth_cache, pids, augment=None)
    # Pick 4 indices: 2 with boxes, 2 without if possible.
    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_fn(samples)
    assert isinstance(batch.boxes, list)
    assert len(batch.boxes) == len(samples)
    for b in batch.boxes:
        assert isinstance(b, torch.Tensor)
        assert b.dtype == torch.float32
        assert b.ndim == 2 and b.shape[-1] == 4
    assert batch.volume_5ch.shape == (
        len(samples), SLICE_WINDOW, TARGET_SHAPE[2], TARGET_SHAPE[0]
    )
    assert batch.volume_5ch.dtype == torch.float32
    assert batch.lesion_mask_center.dtype == torch.uint8
    assert batch.slice_ys.dtype == torch.long


# ----------------------------------------------------------------------
# D11
# ----------------------------------------------------------------------

def test_datamodule_holdout_blocked_by_default(synth_cache):
    # The fixture only creates 5 patients across folds 0/1; the holdout pid
    # is not in any fold. Default DataModule with allow_holdout=False should
    # _not_ load holdout — verify by inspecting cache, and verify the guard
    # raises if we somehow get a holdout pid into the load list (we trigger
    # this by asking for the holdout pid via inference_dataloader after a
    # clean setup).
    dm = LesionDataModule(
        cache_root=synth_cache.cache_root,
        manifest_path=synth_cache.manifest_path,
        cohort_path=synth_cache.cohort_path,
        fold=0,
        batch_size=2,
        num_workers=0,
        slice_window=SLICE_WINDOW,
        target_input_shape=TARGET_SHAPE,
        cache_shape=CACHE_SHAPE,
        allow_holdout=False,
        persistent_workers=False,
        pin_memory=False,
    )
    dm.setup()
    # Guarantee: no holdout pid in cache.
    assert synth_cache.holdout_pid not in dm._cache
    # And the holdout pid is tracked in _holdout_pids.
    assert synth_cache.holdout_pid in dm._holdout_pids


# ----------------------------------------------------------------------
# D12
# ----------------------------------------------------------------------

def test_inference_dataloader_refuses_when_allow_holdout_false(synth_cache):
    dm = LesionDataModule(
        cache_root=synth_cache.cache_root,
        manifest_path=synth_cache.manifest_path,
        cohort_path=synth_cache.cohort_path,
        fold=0,
        batch_size=2,
        num_workers=0,
        slice_window=SLICE_WINDOW,
        target_input_shape=TARGET_SHAPE,
        cache_shape=CACHE_SHAPE,
        allow_holdout=False,
        persistent_workers=False,
        pin_memory=False,
    )
    dm.setup()
    with pytest.raises(HoldoutAccessError):
        dm.inference_dataloader([synth_cache.holdout_pid])


# ----------------------------------------------------------------------
# D13
# ----------------------------------------------------------------------

def test_inference_dataloader_allows_when_allow_holdout_true(synth_cache):
    dm = LesionDataModule(
        cache_root=synth_cache.cache_root,
        manifest_path=synth_cache.manifest_path,
        cohort_path=synth_cache.cohort_path,
        fold=0,
        batch_size=2,
        num_workers=0,
        slice_window=SLICE_WINDOW,
        target_input_shape=TARGET_SHAPE,
        cache_shape=CACHE_SHAPE,
        allow_holdout=True,
        persistent_workers=False,
        pin_memory=False,
    )
    dm.setup()
    assert synth_cache.holdout_pid in dm._cache
    dl = dm.inference_dataloader([synth_cache.holdout_pid])
    batch = next(iter(dl))
    # batch should contain only holdout pid samples
    for pid in batch.patient_ids:
        assert pid == synth_cache.holdout_pid


# ----------------------------------------------------------------------
# Additional: DataModule train/val dataloader smoke
# ----------------------------------------------------------------------

def test_datamodule_train_val_dataloaders_yield_valid_batches(synth_cache):
    dm = LesionDataModule(
        cache_root=synth_cache.cache_root,
        manifest_path=synth_cache.manifest_path,
        cohort_path=synth_cache.cohort_path,
        fold=0,
        batch_size=2,
        num_workers=0,
        slice_window=SLICE_WINDOW,
        target_input_shape=TARGET_SHAPE,
        cache_shape=CACHE_SHAPE,
        allow_holdout=False,
        persistent_workers=False,
        pin_memory=False,
    )
    dm.setup()
    train_batch = next(iter(dm.train_dataloader()))
    val_batch = next(iter(dm.val_dataloader()))
    for b in (train_batch, val_batch):
        assert b.volume_5ch.shape == (2, SLICE_WINDOW, TARGET_SHAPE[2], TARGET_SHAPE[0])
        assert b.lesion_mask_center.shape == (2, TARGET_SHAPE[2], TARGET_SHAPE[0])
        assert isinstance(b.boxes, list) and len(b.boxes) == 2
        assert isinstance(b.labels, list) and len(b.labels) == 2
