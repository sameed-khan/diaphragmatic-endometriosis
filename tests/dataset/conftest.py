"""Synthetic mini-cache fixture for Component 3 tests.

We don't depend on the real ``cache/v1`` (Phase 1 may not have run yet).
Instead, we build a tiny stand-in with cache shape ``(40, 20, 40)`` and
target shape ``(36, 16, 36)`` so the geometry math is identical (pad-offset
``= ((40-36)//2, (20-16)//2, (40-36)//2) = (2, 2, 2)`` per axis), at small
scale.

Layout written to ``tmp_path``:

    cache/v1/preprocessed_manifest.jsonl
    cache/v1/volumes/<pid>/volume.npy           (40, 20, 40) float16
    cache/v1/volumes/<pid>/lesion_mask.npy      (positives only) uint8
    cache/v1/border_bands/<pid>.npy             (CV cohort only) (M, 3) int16
    cache/v1/gt_boxes.parquet                   schema PRD §5.2.4

A matching mini ``manifest.jsonl`` and ``cohort.json`` are written to
``tmp_path / 'data'`` so the DataModule can resolve folds itself.

The fixture creates 5 patients:

  - p_pos_cv_0, p_pos_cv_1   (positive, CV cohort, fold 0 and 1 respectively)
  - p_neg_cv_0               (negative, CV cohort, fold 0)
  - p_neg_cv_1               (negative, CV cohort, fold 1)
  - p_pos_holdout            (positive, holdout cohort)

Boxes are placed at deterministic locations on chosen slice_ys so tests can
look them up exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest


CACHE_SHAPE = (40, 20, 40)
TARGET_SHAPE = (36, 16, 36)
PAD_OFFSET = (2, 2, 2)


@dataclass
class FixturePaths:
    root: Path
    cache_root: Path
    manifest_path: Path
    cohort_path: Path
    pids: list[str] = field(default_factory=list)
    holdout_pid: str = ""
    positive_pids: list[str] = field(default_factory=list)
    negative_pids: list[str] = field(default_factory=list)
    boxes_by_key: dict[tuple[str, int], list[tuple[int, int, int, int]]] = field(
        default_factory=dict
    )


def _write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _build_volume(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(CACHE_SHAPE, dtype=np.float32).astype(np.float16)


def _zero_mask() -> np.ndarray:
    return np.zeros(CACHE_SHAPE, dtype=np.uint8)


def _border_band(rng: np.random.Generator) -> np.ndarray:
    n = 50
    xs = rng.integers(0, CACHE_SHAPE[0], size=n, dtype=np.int16)
    ys = rng.integers(0, CACHE_SHAPE[1], size=n, dtype=np.int16)
    zs = rng.integers(0, CACHE_SHAPE[2], size=n, dtype=np.int16)
    return np.stack([xs, ys, zs], axis=1).astype(np.int16)


@pytest.fixture
def synth_cache(tmp_path: Path) -> FixturePaths:
    rng = np.random.default_rng(0)

    fp = FixturePaths(
        root=tmp_path,
        cache_root=tmp_path / "cache" / "v1",
        manifest_path=tmp_path / "data" / "manifest.jsonl",
        cohort_path=tmp_path / "data" / "cohort.json",
    )
    fp.cache_root.mkdir(parents=True, exist_ok=True)
    fp.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    patients: list[dict[str, Any]] = [
        {"pid": "p_pos_cv_0", "label": "positive", "cohort": "cross-validation", "fold": 0},
        {"pid": "p_pos_cv_1", "label": "positive", "cohort": "cross-validation", "fold": 1},
        {"pid": "p_neg_cv_0", "label": "negative", "cohort": "cross-validation", "fold": 0},
        {"pid": "p_neg_cv_1", "label": "negative", "cohort": "cross-validation", "fold": 1},
        {"pid": "p_pos_holdout", "label": "positive", "cohort": "holdout", "fold": None},
    ]

    manifest_lines: list[str] = []
    pre_manifest_lines: list[str] = []
    gt_rows: list[dict[str, Any]] = []

    # Boxes placed at known positions so tests can verify the lookup.
    # Format: pid -> list of (slice_y_cached, x1, z1, x2, z2)
    boxes_plan: dict[str, list[tuple[int, int, int, int, int]]] = {
        "p_pos_cv_0": [
            (8, 5, 6, 11, 14),
            (10, 20, 22, 28, 30),
        ],
        "p_pos_cv_1": [
            (12, 3, 4, 7, 10),
        ],
        "p_pos_holdout": [
            (9, 12, 14, 18, 20),
        ],
    }

    for p in patients:
        pid = p["pid"]
        fp.pids.append(pid)
        if p["label"] == "positive":
            fp.positive_pids.append(pid)
        else:
            fp.negative_pids.append(pid)
        if p["cohort"] == "holdout":
            fp.holdout_pid = pid

        # Volume.
        vol = _build_volume(rng)
        vol_rel = f"volumes/{pid}/volume.npy"
        _write_npy(fp.cache_root / vol_rel, vol)

        # Lesion mask (positives only). Fill 1s at the box footprints so the
        # mask is internally consistent with the gt_boxes table.
        lesion_rel: str | None = None
        if p["label"] == "positive":
            mask = _zero_mask()
            for sy, x1, z1, x2, z2 in boxes_plan.get(pid, []):
                mask[x1:x2, sy, z1:z2] = 1
                gt_rows.append(
                    {
                        "patient_id": pid,
                        "slice_y": int(sy),
                        "cc_id": 1,
                        "x1": int(x1),
                        "z1": int(z1),
                        "x2": int(x2),
                        "z2": int(z2),
                        "box_max_dim_mm": float(max(x2 - x1, z2 - z1)),
                    }
                )
                fp.boxes_by_key.setdefault((pid, int(sy)), []).append(
                    (int(x1), int(z1), int(x2), int(z2))
                )
            lesion_rel = f"volumes/{pid}/lesion_mask.npy"
            _write_npy(fp.cache_root / lesion_rel, mask)

        # Border band (CV only).
        band_rel: str | None = None
        if p["cohort"] == "cross-validation":
            band = _border_band(rng)
            band_rel = f"border_bands/{pid}.npy"
            _write_npy(fp.cache_root / band_rel, band)

        # Manifest row (data/manifest.jsonl). Minimal but sufficient.
        manifest_row = {
            "patient_id": pid,
            "cohort": p["cohort"],
            "label": p["label"],
            "fold": p["fold"],
            "soft_negative": False,
            "paths": {"raw": f"raw/{pid}.nii.gz"},
            "scanner": {"model": "SIGNA Artist", "variant": "A"},
        }
        manifest_lines.append(json.dumps(manifest_row))

        # Preprocessed manifest row.
        pre_row = {
            "patient_id": pid,
            "cohort": p["cohort"],
            "label": p["label"],
            "fold": p["fold"],
            "scanner_model": "SIGNA Artist",
            "variant": "A",
            "cache_volume_path": vol_rel,
            "cache_lesion_mask_path": lesion_rel,
            "cache_border_band_path": band_rel,
            "roi_bbox_post_resample": {
                "x0": 0, "x1": CACHE_SHAPE[0],
                "y0": 0, "y1": CACHE_SHAPE[1],
                "z0": 0, "z1": CACHE_SHAPE[2],
            },
            "pad_offset": {"x": PAD_OFFSET[0], "y": PAD_OFFSET[1], "z": PAD_OFFSET[2]},
            "n_lesion_ccs": len(boxes_plan.get(pid, [])),
            "roi_norm": {"p1": 0.0, "p99": 1.0, "mean": 0.0, "std": 1.0},
            "lesion_vs_ring_z": 0.5 if p["label"] == "positive" else None,
            "raw_sha256": f"raw_{pid}",
            "code_version": "test",
        }
        pre_manifest_lines.append(json.dumps(pre_row))

    # Write data/manifest.jsonl
    fp.manifest_path.write_text("\n".join(manifest_lines) + "\n")

    # cohort.json (only the bits the DataModule reads — currently nothing,
    # but write a valid skeleton so future readers don't break).
    cohort = {
        "version": "1.0",
        "n_patients_total": len(patients),
        "splits": {
            "cross-validation": [p["pid"] for p in patients if p["cohort"] == "cross-validation"],
            "holdout": [p["pid"] for p in patients if p["cohort"] == "holdout"],
        },
    }
    fp.cohort_path.write_text(json.dumps(cohort))

    # preprocessed_manifest.jsonl
    (fp.cache_root / "preprocessed_manifest.jsonl").write_text(
        "\n".join(pre_manifest_lines) + "\n"
    )

    # gt_boxes.parquet
    if gt_rows:
        gt_df = pl.DataFrame(
            gt_rows,
            schema={
                "patient_id": pl.Utf8,
                "slice_y": pl.Int32,
                "cc_id": pl.Int32,
                "x1": pl.Int32,
                "z1": pl.Int32,
                "x2": pl.Int32,
                "z2": pl.Int32,
                "box_max_dim_mm": pl.Float32,
            },
        )
    else:
        gt_df = pl.DataFrame(
            schema={
                "patient_id": pl.Utf8,
                "slice_y": pl.Int32,
                "cc_id": pl.Int32,
                "x1": pl.Int32,
                "z1": pl.Int32,
                "x2": pl.Int32,
                "z2": pl.Int32,
                "box_max_dim_mm": pl.Float32,
            }
        )
    gt_df.write_parquet(fp.cache_root / "gt_boxes.parquet")

    return fp


# ----------------------------------------------------------------------
# In-memory cache helpers — most Dataset tests don't need a DataModule.
# ----------------------------------------------------------------------


def load_in_memory_cache(fp: FixturePaths) -> dict[str, dict[str, Any]]:
    """Materialize ``fp`` into the dict shape the Dataset expects."""
    pre_rows = [
        json.loads(line)
        for line in (fp.cache_root / "preprocessed_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest_rows = [
        json.loads(line)
        for line in fp.manifest_path.read_text().splitlines()
        if line.strip()
    ]
    manifest_lookup = {r["patient_id"]: r for r in manifest_rows}

    cache: dict[str, dict[str, Any]] = {}
    for r in pre_rows:
        pid = r["patient_id"]
        cache[pid] = {
            "volume": np.load(fp.cache_root / r["cache_volume_path"]),
            "lesion_mask": (
                np.load(fp.cache_root / r["cache_lesion_mask_path"])
                if r.get("cache_lesion_mask_path")
                else None
            ),
            "border_band": (
                np.load(fp.cache_root / r["cache_border_band_path"])
                if r.get("cache_border_band_path")
                else None
            ),
            "manifest_row": manifest_lookup[pid],
            "preprocessed_row": r,
        }
    return cache


def build_gt_lookup(fp: FixturePaths) -> dict[tuple[str, int], np.ndarray]:
    df = pl.read_parquet(fp.cache_root / "gt_boxes.parquet")
    out: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for row in df.select(["patient_id", "slice_y", "x1", "z1", "x2", "z2"]).iter_rows():
        pid, sy, x1, z1, x2, z2 = row
        out.setdefault((pid, int(sy)), []).append(
            (float(x1), float(z1), float(x2), float(z2))
        )
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def build_slice_index(
    cache: dict[str, dict[str, Any]],
    pids: list[str],
    gt_lookup: dict[tuple[str, int], np.ndarray],
    slice_y_lo: int,
    slice_y_hi: int,
) -> list[tuple[str, int, bool, str]]:
    out: list[tuple[str, int, bool, str]] = []
    for pid in pids:
        if pid not in cache:
            continue
        label = cache[pid]["manifest_row"]["label"]
        is_pos_vol = label == "positive"
        for sy in range(slice_y_lo, slice_y_hi):
            has_box = (pid, sy) in gt_lookup
            is_pos_slice = bool(is_pos_vol and has_box)
            kind = (
                "pos_slice"
                if is_pos_slice
                else ("neg_slice_pos_vol" if is_pos_vol else "neg_slice_neg_vol")
            )
            out.append((pid, sy, is_pos_slice, kind))
    return out
