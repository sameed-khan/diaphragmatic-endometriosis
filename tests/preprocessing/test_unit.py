"""Unit tests for Component 1 — synthetic-data only (P1.1–P1.11)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preprocess as pp  # noqa: E402


# ---------- P1.1
def test_resample_isotropic():
    arr = np.ones((8, 8, 8), dtype=np.float32)
    out = pp.resample_to_grid(arr, source_spacing=(1.0, 1.0, 1.0),
                              target_spacing=(0.5, 0.5, 0.5), mask=False)
    assert out.shape == (16, 16, 16)


# ---------- P1.2
def test_resample_mask_nn():
    arr = np.zeros((8, 8, 8), dtype=np.uint8)
    arr[4, 4, 4] = 1
    out = pp.resample_to_grid(arr, source_spacing=(1.0, 1.0, 1.0),
                              target_spacing=(0.5, 0.5, 0.5), mask=True)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 1})


# ---------- P1.3
def test_norm_stats_inside_roi():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[2:5, 2:5, 2:5] = 1
    vol[roi == 1] = 100.0
    vol[roi == 0] = 0.0  # outside should NOT influence stats
    stats = pp.roi_normalization_stats(vol, roi)
    assert stats["mean"] == pytest.approx(100.0)
    assert stats["std"] == pytest.approx(0.0, abs=1e-6)
    assert stats["p1"] == pytest.approx(100.0)
    assert stats["p99"] == pytest.approx(100.0)


# ---------- P1.4
def test_clip_zscore_roundtrip():
    rng = np.random.default_rng(0)
    vol = rng.normal(loc=50.0, scale=10.0, size=(20, 20, 20)).astype(np.float32)
    roi = np.ones_like(vol, dtype=np.uint8)
    stats = pp.roi_normalization_stats(vol, roi)
    out = pp.apply_normalization(vol, stats)
    expected = (np.clip(vol, stats["p1"], stats["p99"]) - stats["mean"]) / stats["std"]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


# ---------- P1.5
def test_post_resample_bbox():
    roi = np.zeros((20, 20, 20), dtype=np.uint8)
    roi[5:15, 3:7, 8:18] = 1
    bbox = pp.post_resample_bbox(roi)
    assert bbox == ((5, 15), (3, 7), (8, 18))


# ---------- P1.6
def test_crop_and_pad_centering():
    vol = np.zeros((30, 30, 30), dtype=np.float32)
    bbox = ((10, 20), (10, 15), (10, 20))
    vol[10:20, 10:15, 10:20] = 7.0
    target = (20, 11, 20)
    out, pad_off = pp.crop_and_pad(vol, bbox, target_shape=target, pad_value=0.0)
    assert out.shape == target
    # foreground starts at center-pad offsets
    px, py, pz = pad_off
    assert (px, py, pz) == (5, 3, 5)
    assert out[px, py, pz] == pytest.approx(7.0)
    assert out[px - 1, py, pz] == 0.0  # padded zone


# ---------- P1.7
def test_crop_and_pad_oversized_bbox():
    vol = np.zeros((40, 40, 40), dtype=np.float32)
    bbox = ((0, 25), (10, 15), (10, 15))  # x-extent 25 > training input 384? no — must exceed target
    target = (20, 11, 20)
    with pytest.raises(RuntimeError, match="exceeds"):
        pp.crop_and_pad(vol, bbox, target_shape=target)


# ---------- P1.8
def test_derive_2d_boxes_single_cc():
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    mask[10:13, 5:7, 10:13] = 1
    rows, n_cc = pp.derive_2d_boxes(mask, "pid", connectivity=26,
                                    spacing_xz_mm=(0.82, 0.82))
    assert n_cc == 1
    # 2 y-slices touched (y=5 and y=6); each row spans x[10..13), z[10..13)
    assert len(rows) == 2
    for r in rows:
        assert r["x1"] == 10 and r["x2"] == 13
        assert r["z1"] == 10 and r["z2"] == 13
        assert r["cc_id"] == 1


# ---------- P1.9
def test_derive_2d_boxes_disjoint_ccs():
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[5:7, 5:7, 5:7] = 1     # CC A
    mask[20:22, 5:7, 20:22] = 1  # CC B (disjoint)
    rows, n_cc = pp.derive_2d_boxes(mask, "pid", connectivity=26,
                                    spacing_xz_mm=(0.82, 0.82))
    assert n_cc == 2
    cc_ids = {r["cc_id"] for r in rows}
    assert cc_ids == {1, 2}
    # 2 slices per CC × 2 CCs = 4 rows
    assert len(rows) == 4


# ---------- P1.10
def test_border_band_right_side_only():
    liver = np.zeros((50, 50, 50), dtype=np.uint8)
    liver[15:25, 15:25, 15:25] = 1
    coords = pp.compute_border_band(liver, spacing=(1.0, 1.0, 1.0))
    assert coords.size > 0
    centroid_x = float(np.argwhere(liver > 0)[:, 0].mean())
    assert (coords[:, 0] > centroid_x).all()


# ---------- P1.11
def _make_synth_nifti(arr: np.ndarray, spacing: tuple[float, float, float], path: Path):
    aff = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float64)
    img = nib.Nifti1Image(arr, aff)
    nib.save(img, str(path))


def test_idempotency_skip(tmp_path: Path):
    raw = np.zeros((512, 30, 512), dtype=np.float32)
    raw[100:200, 5:25, 100:300] = 50.0
    liver = np.zeros((512, 30, 512), dtype=np.uint8)
    liver[120:180, 8:22, 150:250] = 1
    liver_roi = np.zeros((512, 30, 512), dtype=np.uint8)
    liver_roi[110:200, 5:25, 130:280] = 1
    lesion = np.zeros((512, 30, 512), dtype=np.uint8)
    lesion[150:155, 12:14, 200:205] = 1

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_p = raw_dir / "raw.nii.gz"
    liver_p = raw_dir / "liver.nii.gz"
    liver_roi_p = raw_dir / "liver_roi.nii.gz"
    lesion_p = raw_dir / "lesion.nii.gz"
    spacing = (0.82, 1.5, 0.82)
    _make_synth_nifti(raw, spacing, raw_p)
    _make_synth_nifti(liver, spacing, liver_p)
    _make_synth_nifti(liver_roi, spacing, liver_roi_p)
    _make_synth_nifti(lesion, spacing, lesion_p)

    # raw_sha256 — match what manifest would carry
    import hashlib
    raw_sha = hashlib.sha256(raw_p.read_bytes()).hexdigest()

    cache_root = tmp_path / "cache" / "v1"
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_row = {
        "patient_id": "synth_pos",
        "cohort": "cross-validation",
        "label": "positive",
        "fold": 0,
        "scanner": {"model": "SIGNA Artist", "variant": "A"},
        "paths": {
            "raw": "raw/raw.nii.gz",
            "lesion_mask": "raw/lesion.nii.gz",
            "liver_mask": "raw/liver.nii.gz",
            "liver_roi": "raw/liver_roi.nii.gz",
        },
        "hashes": {"raw_sha256": raw_sha},
    }

    cfg = pp.PreprocessConfig(
        manifest_path=tmp_path / "fake_manifest.jsonl",
        cohort_path=tmp_path / "fake_cohort.json",
        raw_root=tmp_path,
        cache_root=cache_root,
        workers=1,
        force=False,
        dry_run=False,
        code_version="testver",
    )

    res1 = pp.preprocess_one(manifest_row, cfg, existing_row=None)
    assert res1.success, res1.error
    assert not res1.skipped

    # Re-run with the previously written manifest row as "existing"
    res2 = pp.preprocess_one(manifest_row, cfg, existing_row=res1.manifest_row)
    assert res2.success
    assert res2.skipped
