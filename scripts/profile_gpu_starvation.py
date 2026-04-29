"""Profile GPU starvation in the training stack.

Usage (examples):
  uv run python scripts/profile_gpu_starvation.py \
    --experiment experiments/smoke.py --fold 0 --mode dataloader

  uv run python scripts/profile_gpu_starvation.py \
    --experiment experiments/smoke.py --fold 0 --mode training --steps 200

Notes:
  - Reports CPU batch time, H2D transfer time, GPU step time, and GPU util/mem.
  - Uses NVML (nvidia-ml-py) for per-sample GPU utilization.
  - Does not modify model weights; it runs a short forward/backward loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# Ensure repo root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from endo.config import load_experiment
from endo.data.datamodule import LesionDataModule
from endo.lightning_module import LesionDetectorLM

try:
    import pynvml
except Exception:  # noqa: BLE001
    pynvml = None

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None


def _init_nvml(device_index: int) -> tuple[object, str] | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(h)
        return h, name.decode("utf-8") if isinstance(name, bytes) else str(name)
    except Exception:
        return None


def _gpu_sample(handle: object) -> dict[str, float]:
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": float(util.gpu),
            "mem_util": float(util.memory),
            "mem_used_mb": float(mem.used) / (1024.0 * 1024.0),
        }
    except Exception:
        return {"gpu_util": float("nan"), "mem_util": float("nan"), "mem_used_mb": float("nan")}


def _summary(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if np.isfinite(v)]
    if not clean:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _cpu_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _iter_batches(loader: Iterable, num_batches: int) -> Iterable:
    it = iter(loader)
    for _ in range(num_batches):
        yield next(it)


def run_dataloader_profile(
    experiment_path: Path,
    fold: int,
    device_index: int,
    num_batches: int,
    pin_memory: bool,
    workers: int | None,
    disable_augmentation: bool,
    output_json: Path | None,
):
    experiment = load_experiment(experiment_path)
    if disable_augmentation:
        experiment.augmentation = None
    manifest_override = os.environ.get("PROFILE_MANIFEST")
    if manifest_override:
        dm = LesionDataModule.from_experiment(
            experiment, fold=fold, manifest_path=Path(manifest_override)
        )
    else:
        dm = LesionDataModule.from_experiment(experiment, fold=fold)
    if workers is not None:
        dm.num_workers = int(workers)
    dm.pin_memory = bool(pin_memory)
    dm.setup()

    loader = dm.train_dataloader()
    handle = _init_nvml(device_index)

    cpu_batch_times: list[float] = []
    gpu_util_samples: list[float] = []
    mem_used_samples: list[float] = []

    t_last = time.perf_counter()
    for i, batch in enumerate(_iter_batches(loader, num_batches)):
        t_now = time.perf_counter()
        cpu_batch_times.append(t_now - t_last)
        t_last = t_now
        if handle is not None:
            sample = _gpu_sample(handle[0])
            gpu_util_samples.append(sample["gpu_util"])
            mem_used_samples.append(sample["mem_used_mb"])

    report = {
        "mode": "dataloader",
        "fold": int(fold),
        "num_batches": int(num_batches),
        "pin_memory": bool(pin_memory),
        "num_workers": int(dm.num_workers),
        "augmentation": "disabled" if disable_augmentation else "enabled",
        "cpu_rss_mb": _cpu_rss_mb(),
        "cpu_batch_time_s": _summary(cpu_batch_times),
        "gpu_util": _summary(gpu_util_samples),
        "gpu_mem_used_mb": _summary(mem_used_samples),
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_training_profile(
    experiment_path: Path,
    fold: int,
    device_index: int,
    steps: int,
    warmup: int,
    pin_memory: bool,
    workers: int | None,
    disable_augmentation: bool,
    enable_ema: bool,
    output_json: Path | None,
):
    experiment = load_experiment(experiment_path)
    if disable_augmentation:
        experiment.augmentation = None
    manifest_override = os.environ.get("PROFILE_MANIFEST")
    if manifest_override:
        dm = LesionDataModule.from_experiment(
            experiment, fold=fold, manifest_path=Path(manifest_override)
        )
    else:
        dm = LesionDataModule.from_experiment(experiment, fold=fold)
    if workers is not None:
        dm.num_workers = int(workers)
    dm.pin_memory = bool(pin_memory)
    dm.setup()

    loader = dm.train_dataloader()
    device = torch.device("cuda", device_index) if torch.cuda.is_available() else torch.device("cpu")

    lm = LesionDetectorLM(experiment)
    if enable_ema:
        try:
            from endo.sampler.score_ema import ScoreEMATracker

            lm.score_ema_tracker = ScoreEMATracker(decay=float(experiment.sampler.score_ema_decay))
        except Exception:
            pass
    lm.to(device)
    lm.train()
    optim = torch.optim.AdamW(lm.parameters(), lr=1e-4)

    handle = _init_nvml(device_index)

    cpu_batch_times: list[float] = []
    h2d_times: list[float] = []
    gpu_step_times: list[float] = []
    step_times: list[float] = []
    gpu_util_samples: list[float] = []
    mem_used_samples: list[float] = []

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    it = iter(loader)
    for step in range(steps + warmup):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        # H2D transfer
        _sync()
        t2 = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        _sync()
        t3 = time.perf_counter()

        # GPU step
        _sync()
        t4 = time.perf_counter()
        loss = lm.training_step(batch, batch_idx=step)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        _sync()
        t5 = time.perf_counter()

        if step >= warmup:
            cpu_batch_times.append(t1 - t0)
            h2d_times.append(t3 - t2)
            gpu_step_times.append(t5 - t4)
            step_times.append(t5 - t0)
            if handle is not None:
                sample = _gpu_sample(handle[0])
                gpu_util_samples.append(sample["gpu_util"])
                mem_used_samples.append(sample["mem_used_mb"])

    report = {
        "mode": "training",
        "fold": int(fold),
        "steps": int(steps),
        "warmup": int(warmup),
        "pin_memory": bool(pin_memory),
        "num_workers": int(dm.num_workers),
        "augmentation": "disabled" if disable_augmentation else "enabled",
        "score_ema": bool(enable_ema),
        "cpu_rss_mb": _cpu_rss_mb(),
        "cpu_batch_time_s": _summary(cpu_batch_times),
        "h2d_time_s": _summary(h2d_times),
        "gpu_step_time_s": _summary(gpu_step_times),
        "step_time_s": _summary(step_times),
        "gpu_util": _summary(gpu_util_samples),
        "gpu_mem_used_mb": _summary(mem_used_samples),
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("dataloader", "training"), default="training")
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--no-augment", action="store_true", help="disable TrainAugmentation")
    parser.add_argument("--enable-ema", action="store_true", help="enable ScoreEMATracker updates")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()
    output_json = Path(args.output_json) if args.output_json else None

    if args.mode == "dataloader":
        run_dataloader_profile(
            experiment_path=Path(args.experiment),
            fold=int(args.fold),
            device_index=int(args.device),
            num_batches=int(args.num_batches),
            pin_memory=bool(args.pin_memory),
            workers=args.workers,
            disable_augmentation=bool(args.no_augment),
            output_json=output_json,
        )
        return 0

    run_training_profile(
        experiment_path=Path(args.experiment),
        fold=int(args.fold),
        device_index=int(args.device),
        steps=int(args.steps),
        warmup=int(args.warmup),
        pin_memory=bool(args.pin_memory),
        workers=args.workers,
        disable_augmentation=bool(args.no_augment),
        enable_ema=bool(args.enable_ema),
        output_json=output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
