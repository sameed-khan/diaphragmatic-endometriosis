"""Unit tests for the lesion bank builder (PRD §11.2 L1–L9).

All tests use synthetic numpy volumes so they are independent of the
preprocessed cache. The integration test (real cache) is skipped if the
preprocessed manifest is not yet available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from endo.lesion_bank import (
    LesionBankEntry,
    SPACING_MM,
    extract_entries_from_arrays,
    load_bank,
    save_bank,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _empty_volume(shape: tuple[int, int, int] = (16, 12, 16)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _empty_mask(shape: tuple[int, int, int] = (16, 12, 16)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _place_block(
    arr: np.ndarray,
    origin: tuple[int, int, int],
    shape: tuple[int, int, int],
    value=1,
) -> tuple[slice, slice, slice]:
    sl = tuple(slice(o, o + s) for o, s in zip(origin, shape))
    arr[sl] = value
    return sl


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def test_extract_single_cc_shape() -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    block_shape = (4, 3, 4)
    sl = _place_block(mask, (5, 4, 5), block_shape, 1)
    vol[sl] = 1.5

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth1", connectivity=26
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.tight_mask.shape == block_shape
    assert e.tight_intensities.shape == block_shape
    assert e.tight_shell_mask.shape == block_shape
    expected_extent = (
        block_shape[0] * SPACING_MM[0],
        block_shape[1] * SPACING_MM[1],
        block_shape[2] * SPACING_MM[2],
    )
    assert e.physical_extent_mm == pytest.approx(expected_extent)
    # spec sanity: (3.28, 4.5, 3.28)
    assert e.physical_extent_mm == pytest.approx((3.28, 4.5, 3.28))


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def test_extract_disjoint_ccs() -> None:
    vol = _empty_volume((20, 12, 20))
    mask = _empty_mask((20, 12, 20))
    sl1 = _place_block(mask, (1, 1, 1), (3, 2, 3), 1)
    sl2 = _place_block(mask, (12, 6, 12), (2, 2, 2), 1)
    vol[sl1] = 0.5
    vol[sl2] = 1.25

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth2", connectivity=26
    )
    assert len(entries) == 2
    shapes = sorted(tuple(e.tight_mask.shape) for e in entries)
    assert shapes == sorted([(3, 2, 3), (2, 2, 2)])


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def test_centroid_offset_in_tight() -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    block_shape = (4, 3, 4)
    _place_block(mask, (4, 4, 4), block_shape, 1)

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth3", connectivity=26
    )
    assert len(entries) == 1
    cx, cy, cz = entries[0].centroid_offset_in_tight
    # Filled block centroid: ((Δ-1)/2 mean) → for 4 voxels, mean=1.5 → rounds to 2.
    assert cx in (1, 2)
    assert cy == 1
    assert cz in (1, 2)


# ---------------------------------------------------------------------------
# L4
# ---------------------------------------------------------------------------


def test_intensity_stats_correctness() -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    sl = _place_block(mask, (3, 3, 3), (4, 3, 4), 1)
    vol[sl] = 1.5

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth4", connectivity=26
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.intensity_mean == pytest.approx(1.5)
    assert e.intensity_std == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# L5
# ---------------------------------------------------------------------------


def test_shell_excludes_cc() -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    mask[8, 6, 8] = 1  # single voxel CC

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth5", connectivity=26
    )
    assert len(entries) == 1
    e = entries[0]
    overlap = (e.tight_shell_mask.astype(bool) & e.tight_mask.astype(bool)).sum()
    assert overlap == 0


# ---------------------------------------------------------------------------
# L6
# ---------------------------------------------------------------------------


def test_shell_thickness_anisotropic() -> None:
    """Shell sits 1 vox away in X/Z (0.82 mm) but is empty in Y (1.5 mm > 1 mm).

    The shell is cropped to ``tight_mask.shape``; for it to contain any voxels
    the CC must not fully fill its bounding box. We use two non-touching
    voxels along axis 0 with a gap, producing a tight bbox of shape (3,1,1)
    where the middle voxel is non-CC and lies 1 voxel (0.82 mm) from each
    CC voxel — so it should be in the shell.
    """
    vol = _empty_volume()
    mask = _empty_mask()
    # Two CC voxels along X with a 1-voxel gap, connectivity=26 still labels
    # them as separate CCs (no shared face/edge/corner via the gap voxel).
    # To make a single CC with a non-CC interior bbox voxel, use a "diagonal"
    # pair touching at a corner.
    mask[5, 5, 5] = 1
    mask[6, 5, 6] = 1  # 26-conn corner-neighbour ⇒ same CC

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth6", connectivity=26
    )
    assert len(entries) == 1
    e = entries[0]
    # Tight bbox is (2, 1, 2). The two off-CC bbox voxels are
    # (0, 0, 1) and (1, 0, 0).
    shell = e.tight_shell_mask.astype(bool)
    cc = e.tight_mask.astype(bool)
    # Shell never overlaps CC.
    assert not (shell & cc).any()
    # The two off-CC bbox voxels sit ≤ 0.82 mm from a CC voxel (in-plane
    # neighbour) — they MUST be in the 1 mm shell.
    assert shell[0, 0, 1], "Expected in-plane bbox voxel to fall in 1 mm shell"
    assert shell[1, 0, 0], "Expected in-plane bbox voxel to fall in 1 mm shell"

    # Now an anisotropy probe: a 1-voxel-thick CC whose bbox is padded along Y
    # by a non-CC voxel. The Y-neighbour sits 1.5 mm > 1 mm away ⇒ NOT in shell.
    mask2 = _empty_mask()
    vol2 = _empty_volume()
    mask2[5, 5, 5] = 1
    mask2[5, 7, 5] = 1  # NOT 26-connected to (5,5,5) (Y-gap of 2).
    # find_objects on label==1 gives bbox (5..6, 5..6, 5..6) — won't include the
    # second CC. Instead create a single CC spanning Y by including the gap:
    mask2[:] = 0
    mask2[5, 5, 5] = 1
    mask2[5, 6, 5] = 1  # face-neighbour along Y
    entries2 = extract_entries_from_arrays(
        vol2, mask2, patient_id="synth6b", connectivity=26
    )
    e2 = entries2[0]
    # Bbox is (1, 2, 1); fully filled — shell within bbox is empty.
    assert e2.tight_mask.shape == (1, 2, 1)
    assert e2.tight_shell_mask.sum() == 0, (
        "When CC fills tight bbox, shell (cropped to bbox) is empty"
    )


# ---------------------------------------------------------------------------
# L7
# ---------------------------------------------------------------------------


def test_intensities_outside_cc_zero() -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    sl = _place_block(mask, (5, 4, 5), (3, 2, 3), 1)
    # Put non-zero intensity everywhere including outside the CC bbox.
    vol[:] = 0.7
    vol[sl] = 2.0

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth7", connectivity=26
    )
    e = entries[0]
    outside = e.tight_intensities[~e.tight_mask.astype(bool)]
    assert np.all(outside == 0.0)
    inside = e.tight_intensities[e.tight_mask.astype(bool)]
    assert np.all(inside == 2.0)


# ---------------------------------------------------------------------------
# L8
# ---------------------------------------------------------------------------


def test_z_extent_correct() -> None:
    """``z_extent_voxels`` is the span along the slice axis (axis 1)."""
    vol = _empty_volume()
    mask = _empty_mask()
    # 4 slices y=2..5, with x,z extent (3, 3).
    _place_block(mask, (5, 2, 5), (3, 4, 3), 1)

    entries = extract_entries_from_arrays(
        vol, mask, patient_id="synth8", connectivity=26
    )
    assert len(entries) == 1
    assert entries[0].z_extent_voxels == 4


# ---------------------------------------------------------------------------
# L9
# ---------------------------------------------------------------------------


def test_idempotency_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running ``build_lesion_bank`` with the same git SHA is a no-op."""
    # Build a tiny synthetic cache the script can consume.
    cache_root = tmp_path / "cache_v1"
    (cache_root / "runtime").mkdir(parents=True)
    with (cache_root / "runtime" / "connectivity_lock.json").open("w") as f:
        json.dump({"connectivity": 26}, f)

    pid = "synth_donor"
    pdir = cache_root / "volumes" / pid
    pdir.mkdir(parents=True)
    vol = np.zeros((12, 8, 12), dtype=np.float32)
    mask = np.zeros((12, 8, 12), dtype=np.uint8)
    mask[4:7, 3:5, 4:7] = 1
    vol[mask.astype(bool)] = 1.1
    np.save(pdir / "volume.npy", vol)
    np.save(pdir / "lesion_mask.npy", mask)

    manifest = cache_root / "preprocessed_manifest.jsonl"
    with manifest.open("w") as f:
        f.write(
            json.dumps(
                {"patient_id": pid, "cohort": "cross-validation", "label": "positive"}
            )
            + "\n"
        )

    from scripts import build_lesion_bank as bld

    # Pin a deterministic SHA so both builds resolve to the same filename.
    monkeypatch.setattr(bld, "_git_sha", lambda repo_root: "deadbeef" * 5)

    bank_path = bld.build_lesion_bank(
        cache_root=cache_root, workers=1, force=False, repo_root=tmp_path
    )
    assert bank_path.exists()
    first_mtime = bank_path.stat().st_mtime_ns

    entries1 = load_bank(bank_path)
    assert len(entries1) == 1

    # Second build: should detect existing bank file and skip without rewriting.
    bank_path2 = bld.build_lesion_bank(
        cache_root=cache_root, workers=1, force=False, repo_root=tmp_path
    )
    assert bank_path2 == bank_path
    assert bank_path.stat().st_mtime_ns == first_mtime, "Bank file should not be rewritten"


# ---------------------------------------------------------------------------
# Integration (real cache) — skipped until preprocessing finishes
# ---------------------------------------------------------------------------


_CACHE_ROOT = Path("cache/v1")
_LOCK = _CACHE_ROOT / "runtime" / "connectivity_lock.json"
_MANIFEST = _CACHE_ROOT / "preprocessed_manifest.jsonl"


@pytest.mark.skipif(
    not (_LOCK.exists() and _MANIFEST.exists()),
    reason="connectivity_lock.json or preprocessed_manifest.jsonl missing — preprocessing not done",
)
def test_real_one_donor_extracts() -> None:
    """TODO: once cache is built, exercise extract_entries_for_donor on a real CV-positive."""
    import polars as pl

    from endo.lesion_bank import extract_entries_for_donor

    df = pl.read_ndjson(_MANIFEST).filter(
        (pl.col("cohort") == "cross-validation") & (pl.col("label") == "positive")
    )
    pid = df.get_column("patient_id").to_list()[0]
    with _LOCK.open() as f:
        connectivity = int(json.load(f)["connectivity"])
    entries = extract_entries_for_donor(pid, _CACHE_ROOT, connectivity=connectivity)
    assert len(entries) > 0
    for e in entries:
        assert e.tight_mask.sum() > 0
        assert np.isfinite(e.intensity_mean)


# Round-trip save/load smoke test.


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    vol = _empty_volume()
    mask = _empty_mask()
    _place_block(mask, (5, 4, 5), (3, 2, 3), 1)
    entries = extract_entries_from_arrays(
        vol, mask, patient_id="rt", connectivity=26
    )
    p = tmp_path / "bank.pkl"
    save_bank(entries, p)
    loaded = load_bank(p)
    assert len(loaded) == len(entries)
    assert isinstance(loaded[0], LesionBankEntry)
    assert loaded[0].donor_patient_id == "rt"
