"""Microbenchmark the full ``TrainAugmentation.__call__`` pipeline on a real
cached sample. This is the end-to-end aug cost the dataloader actually pays
per sample — useful to compare pre/post Y-slab narrowing.

Usage:
  uv run python scripts/profile_train_aug_call.py --iters 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from endo.augmentation.transform import TrainAugmentation
from endo.config.augmentation import (
    AugmentationConfig,
    GeometricConfig,
    IntensityConfig,
    PasteConfig,
)
from endo.data.samples import Sample


def _summary(times_s: list[float]) -> dict:
    a = np.asarray(times_s, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "n": int(a.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="cache/v1/")
    ap.add_argument("--manifest", default="cache/v1/preprocessed_manifest.jsonl")
    ap.add_argument("--pid", default=None)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
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

    pos_rows = [r for r in pre_rows if r.get("cache_lesion_mask_path") and r.get("cache_border_band_path")]
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
    px, py_, pz = (cx - tx) // 2, (cy - ty) // 2, (cz - tz) // 2
    volume = np.ascontiguousarray(volume_full[px:px+tx, py_:py_+ty, pz:pz+tz])
    lesion_mask = np.ascontiguousarray(lesion_full[px:px+tx, py_:py_+ty, pz:pz+tz])
    band_local = band.copy()
    band_local[:, 0] -= px
    band_local[:, 1] -= py_
    band_local[:, 2] -= pz
    in_range = (
        (band_local[:, 0] >= 0) & (band_local[:, 0] < tx)
        & (band_local[:, 1] >= 0) & (band_local[:, 1] < ty)
        & (band_local[:, 2] >= 0) & (band_local[:, 2] < tz)
    )
    band_local = band_local[in_range].astype(np.int16)
    print(f"[load] volume={volume.shape} fp32 = {volume.nbytes/1e6:.1f} MB, "
          f"lesion_voxels={int(lesion_mask.sum())}, band_voxels={band_local.shape[0]}")

    cfg = AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_sigma=1.0, n_paste_max=3),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    )
    aug = TrainAugmentation(cfg=cfg, cache_root=cache_root, rng_seed=args.seed)
    print(f"[load] bank size = {len(aug.bank)} donors")

    fy = volume.shape[1]
    slice_y = fy // 2

    def fn() -> None:
        s = Sample(
            volume_5ch=np.zeros((5, tz, tx), dtype=np.float32),
            lesion_mask_center=np.zeros((tz, tx), dtype=np.uint8),
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            patient_id=row["patient_id"],
            slice_y=int(slice_y),
            is_positive_volume=True,
            is_positive_slice=True,
            pad_offset=(0, 0, 0),
            volume_full_cropped=volume.copy(),
            lesion_mask_full_cropped=lesion_mask.copy(),
            border_band_coords=band_local,
        )
        aug(s)

    for _ in range(args.warmup):
        fn()
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    s = _summary(times)
    print(f"  TrainAugmentation.__call__   mean={s['mean_ms']:8.2f} ms  "
          f"p50={s['p50_ms']:8.2f}  p90={s['p90_ms']:8.2f}  p99={s['p99_ms']:8.2f}")
    print(json.dumps({"iters": args.iters, "stage": "TrainAugmentation.__call__",
                      "summary_ms": s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
