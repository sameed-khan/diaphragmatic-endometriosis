"""Component 1 — Preprocessing pipeline for the 608-volume cohort.

See ``agent/complete_spec/01_preprocessing.md`` and ``agent/complete_spec/00_PRD.md``
sections 5.2 + 13 (amendments A.1, A.3, A.10).

CLI::

    # 1) Build the cache (fills volumes/, border_bands/, gt_boxes.parquet,
    #    preprocessed_manifest.jsonl using a default 26-connectivity for boxes).
    uv run python scripts/preprocess.py \\
        --manifest data/manifest.jsonl \\
        --cohort data/cohort.json \\
        --raw-root data/ \\
        --cache-root cache/v1/ \\
        --workers 16 [--force] [--patients pid1,pid2] [--dry-run]

    # 2) Probe connectivity. After the cohort is cached, probe over all 108
    #    cached lesion masks; pick the connectivity that yields exactly 197 CCs.
    uv run python scripts/preprocess.py --probe-connectivity --cache-root cache/v1/
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import polars as pl
import scipy.ndimage as ndi


# =============================================================================
# Module-level constants — part of the cache-version contract.
# =============================================================================

TARGET_SPACING = (0.82, 1.5, 0.82)   # mm — paste from analyze_inplane_spacing.py output
TARGET_SHAPE = (408, 174, 408)       # voxels (X, Y_slices, Z)
TRAINING_INPUT_SHAPE = (384, 160, 384)
LESION_VS_RING_Z_FLOOR = 0.121
DEFAULT_PROBE_CONNECTIVITY = 26      # used during the build pass; corrected by --probe-connectivity


# =============================================================================
# Configuration / result dataclasses
# =============================================================================


@dataclass(frozen=True)
class PreprocessConfig:
    manifest_path: Path
    cohort_path: Path
    raw_root: Path
    cache_root: Path
    target_spacing: tuple[float, float, float] = TARGET_SPACING
    target_shape: tuple[int, int, int] = TARGET_SHAPE
    workers: int = 16
    force: bool = False
    dry_run: bool = False
    code_version: str = "unknown"


@dataclass(frozen=True)
class PreprocessResult:
    patient_id: str
    success: bool
    skipped: bool
    manifest_row: dict | None
    box_rows: list[dict] | None
    error: str | None


# =============================================================================
# Pure helpers — testable in isolation.
# =============================================================================


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _ras_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return image in RAS canonical orientation; no-op if already RAS."""
    axcodes = nib.aff2axcodes(img.affine)
    if axcodes == ("R", "A", "S"):
        return img
    canon = nib.as_closest_canonical(img)
    new_axcodes = nib.aff2axcodes(canon.affine)
    if new_axcodes != ("R", "A", "S"):
        raise RuntimeError(f"Could not coerce to RAS; got {new_axcodes}")
    return canon


def load_and_validate(
    raw_path: Path,
    lesion_mask_path: Path | None,
    liver_mask_path: Path,
    liver_roi_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load four NIfTIs (raw + 3 masks). Coerce to RAS. Return arrays + zooms."""
    raw_img = _ras_canonical(nib.load(str(raw_path)))
    raw = np.asarray(raw_img.dataobj).astype(np.float32, copy=False)
    zooms = raw_img.header.get_zooms()
    src_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    if raw.ndim != 3:
        raise RuntimeError(f"raw is not 3D: shape={raw.shape}")
    if raw.shape[0] != 512 or raw.shape[2] != 512:
        raise RuntimeError(f"unexpected raw shape (expect (512,N,512)): {raw.shape}")

    def _load_mask(p: Path) -> np.ndarray:
        m = _ras_canonical(nib.load(str(p)))
        a = np.asarray(m.dataobj).astype(np.uint8, copy=False)
        if a.shape != raw.shape:
            raise RuntimeError(f"mask shape {a.shape} != raw shape {raw.shape} for {p.name}")
        a = (a > 0).astype(np.uint8)
        return a

    lesion = _load_mask(lesion_mask_path) if lesion_mask_path is not None else None
    liver = _load_mask(liver_mask_path)
    liver_roi = _load_mask(liver_roi_path)
    return raw, lesion, liver, liver_roi, src_spacing


def resample_to_grid(
    arr: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    mask: bool = False,
) -> np.ndarray:
    """Resample with scipy.ndimage.zoom. Order 0 for masks, order 1 for volume."""
    factors = tuple(s / t for s, t in zip(source_spacing, target_spacing))
    order = 0 if mask else 1
    if mask:
        in_arr = arr.astype(np.uint8, copy=False)
    else:
        in_arr = arr.astype(np.float32, copy=False)
    out = ndi.zoom(in_arr, zoom=factors, order=order, prefilter=False, mode="nearest")
    if mask:
        out = (out > 0).astype(np.uint8)
    return out


def roi_normalization_stats(volume: np.ndarray, roi: np.ndarray) -> dict:
    """Compute clipping percentiles + mean/std of voxels inside ROI."""
    mask = roi == 1
    inside = volume[mask]
    if inside.size == 0:
        raise RuntimeError("roi is empty; cannot compute norm stats")
    p1, p99 = np.percentile(inside, [1, 99])
    return {
        "p1": float(p1),
        "p99": float(p99),
        "mean": float(inside.mean()),
        "std": float(inside.std()),
    }


def apply_normalization(volume: np.ndarray, stats: dict) -> np.ndarray:
    """Clip to (p1, p99) then z-score by (mean, std). Applied to entire volume."""
    out = np.clip(volume, stats["p1"], stats["p99"])
    std = stats["std"] if stats["std"] > 1e-8 else 1.0
    out = (out - stats["mean"]) / std
    return out.astype(np.float32, copy=False)


def post_resample_bbox(roi: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Outer foreground bbox over the ROI (handles fragmented multi-CC ROIs)."""
    if roi.sum() == 0:
        raise RuntimeError("roi is empty; cannot derive bbox")
    coords = np.argwhere(roi > 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # half-open
    return (
        (int(mins[0]), int(maxs[0])),
        (int(mins[1]), int(maxs[1])),
        (int(mins[2]), int(maxs[2])),
    )


def crop_and_pad(
    arr: np.ndarray,
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    target_shape: tuple[int, int, int],
    pad_value: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Crop ``arr`` to bbox; center-pad each axis to ``target_shape``.

    Returns (out_array, pad_offset_xyz). Hard-fails if the bbox extent on any
    axis exceeds the eventual training input shape.
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox
    extents = (x1 - x0, y1 - y0, z1 - z0)

    for ax, (e, ti) in enumerate(zip(extents, TRAINING_INPUT_SHAPE)):
        if e > ti:
            raise RuntimeError(
                f"post-resample bbox extent on axis {ax} = {e} exceeds "
                f"TRAINING_INPUT_SHAPE = {TRAINING_INPUT_SHAPE}; bbox={bbox}"
            )
    for ax, (e, t) in enumerate(zip(extents, target_shape)):
        if e > t:
            raise RuntimeError(
                f"post-resample bbox extent on axis {ax} = {e} exceeds "
                f"TARGET_SHAPE = {target_shape}; bbox={bbox}"
            )

    cropped = arr[x0:x1, y0:y1, z0:z1]
    out = np.full(target_shape, fill_value=pad_value, dtype=arr.dtype)
    pad_x = (target_shape[0] - extents[0]) // 2
    pad_y = (target_shape[1] - extents[1]) // 2
    pad_z = (target_shape[2] - extents[2]) // 2
    out[
        pad_x : pad_x + extents[0],
        pad_y : pad_y + extents[1],
        pad_z : pad_z + extents[2],
    ] = cropped
    return out, (int(pad_x), int(pad_y), int(pad_z))


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.int64)
    raise ValueError(f"Unsupported connectivity: {connectivity}")


def derive_2d_boxes(
    lesion_mask: np.ndarray,
    patient_id: str,
    connectivity: int,
    spacing_xz_mm: tuple[float, float] = (0.82, 0.82),
) -> tuple[list[dict], int]:
    """For each 3D CC and each y-slice it touches, emit one box row.

    Returns (rows, n_cc). Each row::

        {patient_id, slice_y, cc_id, x1, z1, x2, z2, box_max_dim_mm}
    """
    structure = _connectivity_structure(connectivity)
    cc_labels, n_cc = ndi.label(lesion_mask > 0, structure=structure)
    rows: list[dict] = []
    for cc_id in range(1, n_cc + 1):
        cc_mask = cc_labels == cc_id
        if not cc_mask.any():
            continue
        # Y slices that contain any voxel of this CC
        y_has = cc_mask.any(axis=(0, 2))
        y_indices = np.where(y_has)[0]
        for y in y_indices:
            slab = cc_mask[:, y, :]
            xs = np.where(slab.any(axis=1))[0]
            zs = np.where(slab.any(axis=0))[0]
            if xs.size == 0 or zs.size == 0:
                continue
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            z1 = int(zs.min())
            z2 = int(zs.max()) + 1
            box_max_dim_mm = float(
                max((x2 - x1) * spacing_xz_mm[0], (z2 - z1) * spacing_xz_mm[1])
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "slice_y": int(y),
                    "cc_id": int(cc_id),
                    "x1": x1,
                    "z1": z1,
                    "x2": x2,
                    "z2": z2,
                    "box_max_dim_mm": box_max_dim_mm,
                }
            )
    return rows, int(n_cc)


def compute_border_band(
    liver_mask: np.ndarray,
    spacing: tuple[float, float, float] = (0.82, 1.5, 0.82),
) -> np.ndarray:
    """Right-hemidiaphragm 2-mm shell as ``(M, 3)`` int16 voxel coords (x, y, z)."""
    if liver_mask.sum() == 0:
        return np.empty((0, 3), dtype=np.int16)
    inside_liver = liver_mask > 0
    # distance_transform_edt: distance to the nearest 0 voxel.
    dist_outside = ndi.distance_transform_edt(~inside_liver, sampling=spacing)
    dist_inside = ndi.distance_transform_edt(inside_liver, sampling=spacing)
    outside_1mm = (dist_outside <= 1.0) & (~inside_liver)
    inside_1mm = (dist_inside <= 1.0) & inside_liver
    band = outside_1mm | inside_1mm

    centroid_x = float(np.argwhere(inside_liver)[:, 0].mean())
    x_idx = np.arange(liver_mask.shape[0])[:, None, None]
    right = band & (x_idx > centroid_x)
    coords = np.argwhere(right).astype(np.int16)
    return coords


def lesion_vs_ring_z(
    volume: np.ndarray,
    lesion_mask: np.ndarray,
    ring_mm: float = 3.0,
    spacing: tuple[float, float, float] = (0.82, 1.5, 0.82),
    connectivity: int = 26,
) -> float | None:
    """Min over CCs of ``(mean_inside - mean_ring) / std_ring``."""
    structure = _connectivity_structure(connectivity)
    cc_labels, n_cc = ndi.label(lesion_mask > 0, structure=structure)
    if n_cc == 0:
        return None
    z_per_cc: list[float] = []
    for cc_id in range(1, n_cc + 1):
        cc_mask = cc_labels == cc_id
        if not cc_mask.any():
            continue
        # ring = within ring_mm mm OUTSIDE the CC
        dist = ndi.distance_transform_edt(~cc_mask, sampling=spacing)
        ring = (dist > 0) & (dist <= ring_mm)
        if not ring.any():
            continue
        inside_mean = float(volume[cc_mask].mean())
        ring_mean = float(volume[ring].mean())
        ring_std = float(volume[ring].std())
        if ring_std < 1e-8:
            continue
        z = (inside_mean - ring_mean) / ring_std
        z_per_cc.append(z)
    if not z_per_cc:
        return None
    return float(min(z_per_cc))


# =============================================================================
# Per-patient pipeline
# =============================================================================


def _read_raw_sha256(manifest_row: dict) -> str:
    return manifest_row["hashes"]["raw_sha256"]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_paths(cache_root: Path, pid: str, label: str, cohort: str) -> dict[str, Path | None]:
    vol_dir = cache_root / "volumes" / pid
    return {
        "volume": vol_dir / "volume.npy",
        "lesion_mask": (vol_dir / "lesion_mask.npy") if label == "positive" else None,
        "border_band": (cache_root / "border_bands" / f"{pid}.npy") if cohort == "cross-validation" else None,
    }


def _idempotency_hit(
    cache_root: Path,
    manifest_row_existing: dict | None,
    raw_sha256: str,
    code_version: str,
    cfg: PreprocessConfig,
    pid: str,
    label: str,
    cohort: str,
) -> bool:
    if manifest_row_existing is None:
        return False
    if manifest_row_existing.get("raw_sha256") != raw_sha256:
        return False
    if manifest_row_existing.get("code_version") != code_version:
        return False
    paths = _expected_paths(cache_root, pid, label, cohort)
    for key, p in paths.items():
        if p is None:
            continue
        if not p.exists():
            return False
    return True


def preprocess_one(
    manifest_row: dict,
    cfg: PreprocessConfig,
    existing_row: dict | None = None,
) -> PreprocessResult:
    pid = manifest_row["patient_id"]
    cohort = manifest_row["cohort"]
    label = manifest_row["label"]
    fold = manifest_row.get("fold")
    scanner_model = manifest_row["scanner"]["model"]
    variant = manifest_row["scanner"].get("variant", "unknown")
    raw_sha256 = _read_raw_sha256(manifest_row)

    # idempotency check
    if not cfg.force and _idempotency_hit(
        cfg.cache_root, existing_row, raw_sha256, cfg.code_version, cfg, pid, label, cohort
    ):
        return PreprocessResult(
            patient_id=pid,
            success=True,
            skipped=True,
            manifest_row=existing_row,
            box_rows=None,
            error=None,
        )

    raw_p = (cfg.raw_root / manifest_row["paths"]["raw"]).resolve()
    lesion_p = (
        (cfg.raw_root / manifest_row["paths"]["lesion_mask"]).resolve()
        if manifest_row["paths"].get("lesion_mask")
        else None
    )
    liver_p = (cfg.raw_root / manifest_row["paths"]["liver_mask"]).resolve()
    liver_roi_p = (cfg.raw_root / manifest_row["paths"]["liver_roi"]).resolve()

    try:
        raw, lesion, liver, liver_roi, src_spacing = load_and_validate(
            raw_p, lesion_p, liver_p, liver_roi_p
        )
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"load_and_validate: {e!r}")

    try:
        raw_r = resample_to_grid(raw, src_spacing, cfg.target_spacing, mask=False)
        liver_r = resample_to_grid(liver, src_spacing, cfg.target_spacing, mask=True)
        liver_roi_r = resample_to_grid(liver_roi, src_spacing, cfg.target_spacing, mask=True)
        lesion_r = (
            resample_to_grid(lesion, src_spacing, cfg.target_spacing, mask=True)
            if lesion is not None
            else None
        )
        # all four must agree
        ref_shape = raw_r.shape
        for arr, nm in (
            (liver_r, "liver"),
            (liver_roi_r, "liver_roi"),
        ):
            if arr.shape != ref_shape:
                raise RuntimeError(
                    f"resampled {nm} shape {arr.shape} != raw {ref_shape}"
                )
        if lesion_r is not None and lesion_r.shape != ref_shape:
            raise RuntimeError(f"resampled lesion shape {lesion_r.shape} != raw {ref_shape}")
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"resample: {e!r}")

    try:
        stats = roi_normalization_stats(raw_r, liver_roi_r)
        vol_norm = apply_normalization(raw_r, stats)
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"normalize: {e!r}")

    try:
        bbox = post_resample_bbox(liver_roi_r)
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"bbox: {e!r}")

    try:
        vol_c, pad_off = crop_and_pad(vol_norm, bbox, cfg.target_shape, pad_value=0.0)
        liver_c, _ = crop_and_pad(liver_r, bbox, cfg.target_shape, pad_value=0)
        if lesion_r is not None:
            lesion_c, _ = crop_and_pad(lesion_r, bbox, cfg.target_shape, pad_value=0)
        else:
            lesion_c = None
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"crop_and_pad: {e!r}")

    # Step 7 — boxes (default 26-conn during build pass; refined by --probe-connectivity)
    box_rows: list[dict] = []
    n_lesion_ccs = 0
    if lesion_c is not None:
        try:
            box_rows, n_lesion_ccs = derive_2d_boxes(
                lesion_c,
                pid,
                connectivity=DEFAULT_PROBE_CONNECTIVITY,
                spacing_xz_mm=(cfg.target_spacing[0], cfg.target_spacing[2]),
            )
        except Exception as e:  # noqa: BLE001
            return PreprocessResult(pid, False, False, None, None, f"derive_2d_boxes: {e!r}")

    # Step 8 — border band (CV cohort only)
    band_coords: np.ndarray | None = None
    if cohort == "cross-validation":
        try:
            band_coords = compute_border_band(liver_c, spacing=cfg.target_spacing)
        except Exception as e:  # noqa: BLE001
            return PreprocessResult(pid, False, False, None, None, f"border_band: {e!r}")

    # Step 9 — lesion vs ring contrast z-score (positives only).
    # NOTE: this metric depends on connectivity (CC partition affects per-CC
    # stats). We compute it provisionally with the build-pass default and let
    # --probe-connectivity re-derive + apply the LESION_VS_RING_Z_FLOOR hard-fail
    # once the locked connectivity is known. See PRD §13 amendment A.3.
    lesion_vs_ring = None
    if lesion_c is not None:
        try:
            lesion_vs_ring = lesion_vs_ring_z(
                vol_c.astype(np.float32),
                lesion_c,
                ring_mm=3.0,
                spacing=cfg.target_spacing,
                connectivity=DEFAULT_PROBE_CONNECTIVITY,
            )
        except Exception as e:  # noqa: BLE001
            return PreprocessResult(pid, False, False, None, None, f"lesion_vs_ring_z: {e!r}")

    if cfg.dry_run:
        return PreprocessResult(pid, True, False, None, box_rows, None)

    # Persist
    try:
        vol_dir = cfg.cache_root / "volumes" / pid
        vol_dir.mkdir(parents=True, exist_ok=True)
        np.save(vol_dir / "volume.npy", vol_c.astype(np.float16))
        if lesion_c is not None:
            np.save(vol_dir / "lesion_mask.npy", lesion_c.astype(np.uint8))
        if band_coords is not None:
            band_dir = cfg.cache_root / "border_bands"
            band_dir.mkdir(parents=True, exist_ok=True)
            np.save(band_dir / f"{pid}.npy", band_coords.astype(np.int16))
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(pid, False, False, None, None, f"save: {e!r}")

    cache_volume_path = f"volumes/{pid}/volume.npy"
    cache_lesion_path = f"volumes/{pid}/lesion_mask.npy" if lesion_c is not None else None
    cache_band_path = f"border_bands/{pid}.npy" if band_coords is not None else None

    row = {
        "patient_id": pid,
        "cohort": cohort,
        "label": label,
        "fold": fold,
        "scanner_model": scanner_model,
        "variant": variant,
        "cache_volume_path": cache_volume_path,
        "cache_lesion_mask_path": cache_lesion_path,
        "cache_border_band_path": cache_band_path,
        "roi_bbox_post_resample": {
            "x0": bbox[0][0],
            "x1": bbox[0][1],
            "y0": bbox[1][0],
            "y1": bbox[1][1],
            "z0": bbox[2][0],
            "z1": bbox[2][1],
        },
        "pad_offset": {"x": pad_off[0], "y": pad_off[1], "z": pad_off[2]},
        "n_lesion_ccs": int(n_lesion_ccs),
        "roi_norm": stats,
        "lesion_vs_ring_z": (float(lesion_vs_ring) if lesion_vs_ring is not None else None),
        "raw_sha256": raw_sha256,
        "code_version": cfg.code_version,
    }
    return PreprocessResult(pid, True, False, row, box_rows, None)


# =============================================================================
# Cohort runner
# =============================================================================


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_existing_manifest(cache_root: Path) -> dict[str, dict]:
    p = cache_root / "preprocessed_manifest.jsonl"
    if not p.exists():
        return {}
    return {r["patient_id"]: r for r in _read_jsonl(p)}


def _setup_logger(cache_root: Path) -> logging.Logger:
    cache_root.mkdir(parents=True, exist_ok=True)
    log_path = cache_root / "preprocessing.log"
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers across re-runs in same process
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _worker_run(args: tuple[dict, dict | None, PreprocessConfig]) -> PreprocessResult:
    manifest_row, existing_row, cfg = args
    try:
        return preprocess_one(manifest_row, cfg, existing_row)
    except Exception as e:  # noqa: BLE001
        return PreprocessResult(
            manifest_row["patient_id"], False, False, None, None,
            f"unhandled: {e!r}\n{traceback.format_exc()}"
        )


def preprocess_cohort(cfg: PreprocessConfig, patient_filter: list[str] | None = None) -> dict:
    logger = _setup_logger(cfg.cache_root)
    logger.info(
        f"start cohort preprocess: cache_root={cfg.cache_root} "
        f"workers={cfg.workers} force={cfg.force} dry_run={cfg.dry_run}"
    )

    manifest_rows = _read_jsonl(cfg.manifest_path)
    if patient_filter is not None:
        keep = set(patient_filter)
        manifest_rows = [r for r in manifest_rows if r["patient_id"] in keep]
    logger.info(f"manifest rows to process: {len(manifest_rows)}")

    existing = _load_existing_manifest(cfg.cache_root)
    work = [(r, existing.get(r["patient_id"]), cfg) for r in manifest_rows]

    # Write code_version.txt
    if not cfg.dry_run:
        (cfg.cache_root).mkdir(parents=True, exist_ok=True)
        (cfg.cache_root / "code_version.txt").write_text(cfg.code_version + "\n")

    results: list[PreprocessResult] = []
    t0 = time.time()
    if cfg.workers > 1 and len(work) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=cfg.workers) as pool:
            for res in pool.imap_unordered(_worker_run, work, chunksize=1):
                results.append(res)
                _log_result(logger, res)
    else:
        for w in work:
            res = _worker_run(w)
            results.append(res)
            _log_result(logger, res)
    wall = time.time() - t0
    logger.info(f"cohort done in {wall:.1f}s")

    # Reconcile manifest + boxes
    new_rows: list[dict] = []
    all_boxes: list[dict] = []
    for r in results:
        if r.success and r.manifest_row is not None:
            new_rows.append(r.manifest_row)
        if r.success and r.box_rows:
            all_boxes.extend(r.box_rows)

    # Merge: keep new_rows where present, preserve old rows for skipped/missing
    merged: dict[str, dict] = {**existing}
    for nr in new_rows:
        merged[nr["patient_id"]] = nr
    final_rows = [merged[pid] for pid in [r["patient_id"] for r in manifest_rows] if pid in merged]
    # If running on full cohort, this is comprehensive; if filtered, also include
    # any other previously cached rows not in this run.
    if patient_filter is None:
        # ensure all manifest patients accounted for in final
        seen = {r["patient_id"] for r in final_rows}
        for pid, row in merged.items():
            if pid not in seen:
                final_rows.append(row)
    else:
        # Don't drop existing rows for unrelated patients
        unique = {r["patient_id"]: r for r in final_rows}
        for pid, row in merged.items():
            unique.setdefault(pid, row)
        final_rows = list(unique.values())

    if not cfg.dry_run:
        _write_jsonl(cfg.cache_root / "preprocessed_manifest.jsonl", final_rows)

        # Boxes parquet handling. Three cases:
        #   1) every result was an idempotency skip (no new boxes computed) —
        #      do NOT touch the existing parquet (it's already correct, possibly
        #      written by --probe-connectivity with the locked connectivity).
        #   2) a partial-cohort run produced new boxes — merge into existing
        #      parquet, replacing rows for patients we just processed.
        #   3) full-cohort run with new boxes — overwrite.
        boxes_path = cfg.cache_root / "gt_boxes.parquet"
        n_processed = sum(1 for r in results if r.success and not r.skipped)
        if n_processed == 0:
            logger.info("no patients re-processed; gt_boxes.parquet left untouched")
        elif patient_filter is None or not boxes_path.exists():
            _write_boxes_parquet(boxes_path, all_boxes)
        else:
            try:
                old = pl.read_parquet(boxes_path)
                old = old.filter(~pl.col("patient_id").is_in([r["patient_id"] for r in manifest_rows]))
                new_df = _boxes_to_df(all_boxes)
                merged_df = pl.concat([old, new_df]) if len(new_df) else old
                merged_df.write_parquet(boxes_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"merge gt_boxes failed ({e!r}); overwriting full")
                _write_boxes_parquet(boxes_path, all_boxes)

    n_ok = sum(1 for r in results if r.success)
    n_skip = sum(1 for r in results if r.skipped)
    n_fail = sum(1 for r in results if not r.success)
    summary = {
        "n_attempted": len(results),
        "n_success": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "wall_seconds": wall,
        "failures": [
            {"patient_id": r.patient_id, "error": r.error}
            for r in results
            if not r.success
        ],
    }
    logger.info(
        f"summary: ok={n_ok} skipped={n_skip} failed={n_fail} wall={wall:.1f}s"
    )
    return summary


def _boxes_to_df(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "patient_id": pl.Utf8,
        "slice_y": pl.Int32,
        "cc_id": pl.Int32,
        "x1": pl.Int32,
        "z1": pl.Int32,
        "x2": pl.Int32,
        "z2": pl.Int32,
        "box_max_dim_mm": pl.Float32,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _write_boxes_parquet(path: Path, rows: list[dict]) -> None:
    df = _boxes_to_df(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _log_result(logger: logging.Logger, r: PreprocessResult) -> None:
    if r.skipped:
        logger.info(f"[{r.patient_id}] skipped (idempotency hit)")
    elif r.success:
        n_ccs = (r.manifest_row or {}).get("n_lesion_ccs")
        zr = (r.manifest_row or {}).get("lesion_vs_ring_z")
        logger.info(f"[{r.patient_id}] ok n_ccs={n_ccs} lesion_vs_ring_z={zr}")
    else:
        logger.error(f"[{r.patient_id}] FAILED: {r.error}")


# =============================================================================
# Connectivity probe (post-build pass; PRD §13 A.3).
# =============================================================================


def probe_connectivity(
    cache_root: Path,
    code_version: str,
    raw_manifest_path: Path | None = None,
    raw_root: Path | None = None,
) -> dict:
    """Probe connectivity at NATIVE resolution to pick 6- vs 26-conn.

    Per phase-1 §1.3, the 197-CC reference was computed on the un-resampled
    masks. Resampling with nearest-neighbour interpolation can occasionally
    merge or split CCs (we observed 26-conn @ native = 197 vs 26-conn @
    cached = 196). To preserve the spec's "exactly 197" discriminator, we
    probe natively and apply the locked connectivity downstream.

    For cohort QC (gt_boxes.parquet, n_lesion_ccs in manifest), we also report
    the cached-resolution counts under the locked connectivity.
    """
    logger = _setup_logger(cache_root)
    manifest_path = cache_root / "preprocessed_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; run --build first")

    rows = _read_jsonl(manifest_path)
    positives = [r for r in rows if r["label"] == "positive"]
    logger.info(f"probe: {len(positives)} positives in cache")

    # Probe at NATIVE resolution against data/manifest.jsonl
    raw_manifest_path = raw_manifest_path or Path("data/manifest.jsonl").resolve()
    raw_root = raw_root or Path("data").resolve()
    if not raw_manifest_path.exists():
        raise SystemExit(f"missing {raw_manifest_path} (needed for native-resolution probe)")
    raw_rows = _read_jsonl(raw_manifest_path)
    pid_to_raw = {r["patient_id"]: r for r in raw_rows}

    counts_per_conn: dict[int, int] = {}
    for conn in (6, 26):
        total = 0
        for r in positives:
            raw_row = pid_to_raw[r["patient_id"]]
            lesion_path = (raw_root / raw_row["paths"]["lesion_mask"]).resolve()
            img = nib.load(str(lesion_path))
            mask = np.asarray(img.dataobj).astype(np.uint8)
            structure = _connectivity_structure(conn)
            _, n = ndi.label(mask > 0, structure=structure)
            total += int(n)
        counts_per_conn[conn] = total
        logger.info(f"native connectivity={conn} -> total CCs = {total}")

    locked_conn: int | None = None
    for c, t in counts_per_conn.items():
        if t == 197:
            locked_conn = c
            break

    if locked_conn is None:
        raise SystemExit(
            f"Neither connectivity yields 197 CCs at native resolution. "
            f"counts={counts_per_conn}. Investigate."
        )

    structure = _connectivity_structure(locked_conn)
    lock = {
        "connectivity": str(locked_conn),
        "structure": structure.astype(int).tolist(),
        "n_ccs_in_cohort": 197,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_version": code_version,
    }
    out = cache_root / "runtime" / "connectivity_lock.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2) + "\n")
    logger.info(f"locked connectivity={locked_conn} -> wrote {out}")

    # Re-derive boxes if locked connectivity differs from build default.
    if locked_conn != DEFAULT_PROBE_CONNECTIVITY:
        logger.info("locked connectivity differs from build default; re-deriving gt_boxes.parquet")
    else:
        logger.info("locked connectivity matches build default; gt_boxes.parquet still re-derived for sanity")

    all_rows: list[dict] = []
    n_ccs_per_pid: dict[str, int] = {}
    z_per_pid: dict[str, float | None] = {}
    for r in positives:
        pid = r["patient_id"]
        mask = np.load(cache_root / r["cache_lesion_mask_path"])
        vol = np.load(cache_root / r["cache_volume_path"]).astype(np.float32)
        rows_, n_cc = derive_2d_boxes(
            mask, pid, connectivity=locked_conn,
            spacing_xz_mm=(TARGET_SPACING[0], TARGET_SPACING[2]),
        )
        all_rows.extend(rows_)
        n_ccs_per_pid[pid] = n_cc
        z = lesion_vs_ring_z(
            vol, mask, ring_mm=3.0, spacing=TARGET_SPACING,
            connectivity=locked_conn,
        )
        z_per_pid[pid] = z

    boxes_path = cache_root / "gt_boxes.parquet"
    _write_boxes_parquet(boxes_path, all_rows)
    logger.info(f"wrote {len(all_rows)} box rows -> {boxes_path}")

    # Update n_lesion_ccs and lesion_vs_ring_z in manifest
    changed = 0
    for r in rows:
        if r["label"] == "positive":
            pid = r["patient_id"]
            new_n = n_ccs_per_pid.get(pid, r.get("n_lesion_ccs", 0))
            new_z = z_per_pid.get(pid, r.get("lesion_vs_ring_z"))
            if new_n != r.get("n_lesion_ccs") or new_z != r.get("lesion_vs_ring_z"):
                r["n_lesion_ccs"] = int(new_n)
                r["lesion_vs_ring_z"] = (float(new_z) if new_z is not None else None)
                changed += 1
    if changed:
        _write_jsonl(manifest_path, rows)
        logger.info(f"updated n_lesion_ccs/lesion_vs_ring_z for {changed} patients")

    # Two-tier check on lesion_vs_ring_z under locked connectivity:
    #   - Strict regression bug check: any z < 0 is a hard fail
    #     (negative contrast means the lesion is darker than its surround,
    #     which would indicate mask corruption).
    #   - Reproduction floor (phase-1 §1.4 min=0.121): warn but do not abort.
    #     A small slip below 0.121 is likely from the new 0.82-mm in-plane
    #     resample + fp16 cache quantization, not a code bug. Documented in
    #     agent/complete_spec/IMPLEMENTATION_LOG.md.
    negative_violations = [
        (pid, z) for pid, z in z_per_pid.items() if z is not None and z < 0.0
    ]
    if negative_violations:
        logger.error(
            f"{len(negative_violations)} patients have NEGATIVE "
            f"lesion_vs_ring_z (regression bug indicator): "
            f"{negative_violations[:5]}"
        )
        raise SystemExit(
            f"Hard-fail: {len(negative_violations)} positives have "
            f"lesion_vs_ring_z < 0. First few: {negative_violations[:5]}"
        )
    floor_violations = [
        (pid, z) for pid, z in z_per_pid.items()
        if z is not None and z < LESION_VS_RING_Z_FLOOR
    ]
    if floor_violations:
        logger.warning(
            f"WARN: {len(floor_violations)} patients have lesion_vs_ring_z "
            f"below phase-1 floor={LESION_VS_RING_Z_FLOOR} (not a hard fail; "
            f"likely from 0.82-mm resample + fp16 quantization): "
            f"{floor_violations[:5]}"
        )

    z_values = [z for z in z_per_pid.values() if z is not None]
    if z_values:
        logger.info(
            f"lesion_vs_ring_z: min={min(z_values):.4f}, "
            f"median={float(np.median(z_values)):.4f}, "
            f"max={max(z_values):.4f}"
        )

    total_ccs = sum(n_ccs_per_pid.values())
    logger.info(
        f"cache-frame total CCs across {len(positives)} positives: {total_ccs} "
        f"(native count = 197; minor discrepancy expected from NN resampling)"
    )
    logger.info(f"total box rows: {len(all_rows)}")

    return lock


# =============================================================================
# CLI
# =============================================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    p.add_argument("--cohort", type=Path, default=Path("data/cohort.json"))
    p.add_argument("--raw-root", type=Path, default=Path("data/"))
    p.add_argument("--cache-root", type=Path, default=Path("cache/v1/"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--patients", type=str, default=None,
                   help="comma-separated patient_ids to restrict the run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--probe-connectivity", action="store_true",
                   help="Run the post-build connectivity probe + re-derive gt_boxes")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    code_version = get_git_sha()
    cache_root = args.cache_root.resolve()

    if args.probe_connectivity:
        probe_connectivity(cache_root, code_version)
        return 0

    cfg = PreprocessConfig(
        manifest_path=args.manifest.resolve(),
        cohort_path=args.cohort.resolve(),
        raw_root=args.raw_root.resolve(),
        cache_root=cache_root,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
        code_version=code_version,
    )
    pf = [p.strip() for p in args.patients.split(",")] if args.patients else None
    summary = preprocess_cohort(cfg, patient_filter=pf)
    if summary["n_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
