"""Microbenchmark each augmentation stage on a single cached volume.

Times paste, geometric (affine + elastic), intensity, box re-derivation, and
5-channel extraction *separately*, plus the full pipeline cost. Reports
mean / p50 / p90 / p99 in ms over N iterations.

Usage:
  uv run python scripts/profile_aug_stages.py --iters 30 \
    [--pid amber_bear_quartz] [--cache-root cache/v1/]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from endo.augmentation.boxes import derive_boxes_from_mask, read_connectivity
from endo.augmentation.geometric import (
    apply_affine_lockstep,
    apply_elastic_lockstep,
    random_affine_2d,
    random_elastic_2d,
)
from endo.augmentation.intensity import intensity_aug
from endo.augmentation.paste import multi_paste_volume
from endo.config.augmentation import (
    AugmentationConfig,
    GeometricConfig,
    IntensityConfig,
    PasteConfig,
)
from endo.lesion_bank import load_bank


def _summary(times_s: list[float]) -> dict:
    a = np.asarray(times_s, dtype=np.float64) * 1000.0  # ms
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "n": int(a.size),
    }


def _time_calls(fn: Callable[[], None], iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="cache/v1/")
    ap.add_argument("--manifest", default="cache/v1/preprocessed_manifest.jsonl")
    ap.add_argument("--pid", default=None,
                    help="patient id; if omitted, picks first positive with mask + band")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-shape", default="384,160,384")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    target_shape = tuple(int(s) for s in args.target_shape.split(","))

    pre_rows = []
    for line in Path(args.manifest).read_text().splitlines():
        line = line.strip()
        if line:
            pre_rows.append(json.loads(line))

    pos_rows = [
        r for r in pre_rows
        if r.get("cache_lesion_mask_path") and r.get("cache_border_band_path")
    ]
    if args.pid is not None:
        row = next(r for r in pre_rows if r["patient_id"] == args.pid)
    else:
        row = pos_rows[0]
    print(f"[load] using pid={row['patient_id']!r}")

    volume_full = np.load(cache_root / row["cache_volume_path"]).astype(np.float32)
    lesion_full = np.load(cache_root / row["cache_lesion_mask_path"]).astype(np.uint8)
    band = np.load(cache_root / row["cache_border_band_path"]).astype(np.int32)

    cx, cy, cz = volume_full.shape
    tx, ty, tz = target_shape
    px, py, pz = (cx - tx) // 2, (cy - ty) // 2, (cz - tz) // 2
    volume = np.ascontiguousarray(volume_full[px:px+tx, py:py+ty, pz:pz+tz])
    lesion_mask = np.ascontiguousarray(lesion_full[px:px+tx, py:py+ty, pz:pz+tz])
    band_local = band.copy()
    band_local[:, 0] -= px
    band_local[:, 1] -= py
    band_local[:, 2] -= pz
    in_range = (
        (band_local[:, 0] >= 0) & (band_local[:, 0] < tx)
        & (band_local[:, 1] >= 0) & (band_local[:, 1] < ty)
        & (band_local[:, 2] >= 0) & (band_local[:, 2] < tz)
    )
    band_local = band_local[in_range].astype(np.int16)

    print(f"[load] volume={volume.shape} fp32 = {volume.nbytes/1e6:.1f} MB, "
          f"lesion_voxels={int(lesion_mask.sum())}, band_voxels={band_local.shape[0]}")

    bank = list(load_bank(cache_root / "lesion_banks" / "current.pkl"))
    print(f"[load] bank size = {len(bank)} donors")

    cfg = AugmentationConfig(
        paste=PasteConfig(p_any_paste=1.0, n_paste_sigma=1.0, n_paste_max=3),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    )

    rng = np.random.default_rng(args.seed)
    connectivity = read_connectivity(cache_root / "runtime" / "connectivity_lock.json")

    # ---- 1) paste (3 attempts forced)
    def fn_paste() -> None:
        v = volume.copy()
        m = lesion_mask.copy()
        multi_paste_volume(
            v, m, band_local.astype(np.int32), bank, cfg.paste, np.random.default_rng(rng.integers(0, 2**31)),
            frame_shape=tuple(int(s) for s in v.shape),
        )

    # ---- 2) geometric affine only
    def fn_affine() -> None:
        M = random_affine_2d(
            np.random.default_rng(rng.integers(0, 2**31)),
            max_rot_deg=cfg.geometric.rotation_deg,
            scale_min=cfg.geometric.scale_min,
            scale_max=cfg.geometric.scale_max,
            max_translate_px_x=cfg.geometric.translation_frac * volume.shape[0],
            max_translate_px_z=cfg.geometric.translation_frac * volume.shape[2],
        )
        apply_affine_lockstep(volume, lesion_mask, M)

    # ---- 3) geometric elastic only
    def fn_elastic() -> None:
        field = random_elastic_2d(
            np.random.default_rng(rng.integers(0, 2**31)),
            alpha=1.0,
            sigma=cfg.geometric.elastic_sigma,
            shape_xz=(volume.shape[0], volume.shape[2]),
            n_control_points=cfg.geometric.elastic_control_points,
        )
        apply_elastic_lockstep(volume, lesion_mask, field)

    # ---- 4) intensity
    def fn_intensity() -> None:
        intensity_aug(volume.copy(), cfg.intensity, np.random.default_rng(rng.integers(0, 2**31)))

    # ---- 5) box re-derivation (one center slice, X-Z plane)
    slice_y = volume.shape[1] // 2
    def fn_boxes() -> None:
        derive_boxes_from_mask(lesion_mask[:, slice_y, :], connectivity=connectivity, min_dim=2)

    # ---- 6) 5-channel extraction
    def fn_extract() -> None:
        triplet = volume[:, slice_y - 2 : slice_y + 3, :]
        out = np.ascontiguousarray(np.transpose(triplet, (1, 2, 0)).astype(np.float32, copy=False))

    # ---- 7) full sample-build cost (mimic pipeline minus paste-mutating issues)
    def fn_full_no_paste() -> None:
        v = volume.copy()
        m = lesion_mask.copy()
        M = random_affine_2d(
            np.random.default_rng(rng.integers(0, 2**31)),
            max_rot_deg=cfg.geometric.rotation_deg,
            scale_min=cfg.geometric.scale_min,
            scale_max=cfg.geometric.scale_max,
            max_translate_px_x=cfg.geometric.translation_frac * v.shape[0],
            max_translate_px_z=cfg.geometric.translation_frac * v.shape[2],
        )
        v, m = apply_affine_lockstep(v, m, M)
        field = random_elastic_2d(
            np.random.default_rng(rng.integers(0, 2**31)),
            alpha=1.0,
            sigma=cfg.geometric.elastic_sigma,
            shape_xz=(v.shape[0], v.shape[2]),
            n_control_points=cfg.geometric.elastic_control_points,
        )
        v, m = apply_elastic_lockstep(v, m, field)
        v = intensity_aug(v, cfg.intensity, np.random.default_rng(rng.integers(0, 2**31)))
        derive_boxes_from_mask(m[:, slice_y, :], connectivity=connectivity, min_dim=2)
        triplet = v[:, slice_y - 2 : slice_y + 3, :]
        _ = np.ascontiguousarray(np.transpose(triplet, (1, 2, 0)).astype(np.float32, copy=False))

    stages = [
        ("paste(3)", fn_paste),
        ("affine_lockstep", fn_affine),
        ("elastic_lockstep", fn_elastic),
        ("intensity", fn_intensity),
        ("box_rederive", fn_boxes),
        ("extract_5ch", fn_extract),
        ("full_no_paste", fn_full_no_paste),
    ]

    report = {}
    for name, fn in stages:
        ts = _time_calls(fn, args.iters, args.warmup)
        s = _summary(ts)
        report[name] = s
        print(f"  {name:<20} mean={s['mean_ms']:8.2f} ms  p50={s['p50_ms']:8.2f}  "
              f"p90={s['p90_ms']:8.2f}  p99={s['p99_ms']:8.2f}")

    print(json.dumps({"pid": row["patient_id"], "iters": args.iters, "stages": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
