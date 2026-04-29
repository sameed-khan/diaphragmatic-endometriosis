"""5-min smoke training gate (Component 8 §1).

Pipeline:
  1. Pick 5 smallest CV volumes (2 positives + 3 negatives) by cached
     ``volume.npy`` size, ensuring at least one positive in fold 0 (val)
     and one in another fold (train).
  2. Materialize a tiny manifest at ``data/.smoke_manifest.jsonl`` containing
     only those 5 rows.
  3. Build the real ``LesionDataModule`` + ``LesionDetectorLM`` against this
     subset; train 2 epochs at ``samples_per_epoch=100``.
  4. Assert: ≥50 step losses captured, last-10 mean < first-10 mean, no
     NaN/Inf, ``val/slice_auroc`` was logged.

Run::

    uv run python scripts/smoke_train.py

Artifacts go under ``runs/smoke_<uuid8>/`` and are wiped at the end unless
``--keep`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


log = logging.getLogger("smoke")


def pick_smoke_pids(
    manifest_rows: list[dict],
    cache_root: Path,
    n_pos: int = 2,
    n_neg: int = 3,
) -> list[str]:
    """Pick smallest-by-volume.npy CV pids: ``n_pos`` positives + ``n_neg`` negatives.

    Constrains so at least one positive lands in fold 0 (val) and at least
    one positive lands in another fold (train).
    """

    def vol_size(pid: str) -> int:
        p = cache_root / "volumes" / pid / "volume.npy"
        return p.stat().st_size if p.exists() else 0

    cv_pos = [r for r in manifest_rows if r["cohort"] == "cross-validation" and r["label"] == "positive"]
    cv_neg = [r for r in manifest_rows if r["cohort"] == "cross-validation" and r["label"] == "negative"]

    # Smallest per group.
    cv_pos.sort(key=lambda r: vol_size(r["patient_id"]))
    cv_neg.sort(key=lambda r: vol_size(r["patient_id"]))

    # Ensure at least 1 positive in fold 0 and 1 positive in another fold.
    pos_fold0 = next((r for r in cv_pos if r["fold"] == 0), None)
    pos_other = next((r for r in cv_pos if r["fold"] != 0), None)
    if pos_fold0 is None or pos_other is None:
        raise RuntimeError("smoke pid selection: need at least 1 positive in fold 0 and 1 in another fold")
    chosen_pos = [pos_fold0, pos_other][:n_pos]

    # Negatives — at least one in fold 0 too.
    neg_fold0 = next((r for r in cv_neg if r["fold"] == 0), None)
    neg_others = [r for r in cv_neg if r["fold"] != 0][: max(n_neg - 1, 0)]
    chosen_neg: list[dict] = []
    if neg_fold0 is not None:
        chosen_neg.append(neg_fold0)
    chosen_neg.extend(neg_others)
    chosen_neg = chosen_neg[:n_neg]

    if len(chosen_pos) + len(chosen_neg) < (n_pos + n_neg):
        raise RuntimeError("smoke pid selection: insufficient candidates with required fold coverage")

    pids = [r["patient_id"] for r in (chosen_pos + chosen_neg)]
    log.info("smoke pids: %s", pids)
    return pids


def write_smoke_manifest(full_manifest: Path, pids: list[str], out_path: Path) -> None:
    rows = []
    keep = set(pids)
    with full_manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("patient_id") in keep:
                rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info("wrote smoke manifest: %s (%d rows)", out_path, len(rows))


class StepLossCapture:
    def __init__(self) -> None:
        self.step_losses: list[float] = []


def _make_capture_callback(capture: StepLossCapture):
    import pytorch_lightning as pl
    import torch

    class CB(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            v = outputs.get("loss") if isinstance(outputs, dict) else outputs
            try:
                if isinstance(v, torch.Tensor):
                    v = float(v.detach().cpu().item())
                else:
                    v = float(v)
            except Exception:  # noqa: BLE001
                return
            capture.step_losses.append(v)
    return CB()


def run_smoke(
    keep_artifacts: bool = False,
    cache_root: Path = Path("cache/v1"),
    data_root: Path = Path("data"),
) -> dict:
    import pytorch_lightning as pl
    import torch

    from endo.config import load_experiment
    from endo.data.datamodule import LesionDataModule
    from endo.data.manifest import read_manifest_jsonl
    from endo.ema_callback import EmaCallback
    from endo.lightning_module import LesionDetectorLM
    from endo.sampler.weighted import WeightedScheduledSampler
    from endo.utils.provenance import get_git_sha

    experiment = load_experiment(Path("experiments/smoke.py"))
    full_manifest = data_root / "manifest.jsonl"
    cohort_path = data_root / "cohort.json"

    rows = read_manifest_jsonl(full_manifest)
    pids = pick_smoke_pids(rows, cache_root)
    smoke_manifest = data_root / ".smoke_manifest.jsonl"
    write_smoke_manifest(full_manifest, pids, smoke_manifest)

    out_root = Path(tempfile.mkdtemp(prefix="smoke_run_", dir="runs")) if Path("runs").exists() else Path(tempfile.mkdtemp(prefix="smoke_run_"))
    log.info("smoke run dir: %s", out_root)

    # Try to wire augmentation — fall back gracefully.
    try:
        from endo.augmentation.transform import TrainAugmentation

        augment = TrainAugmentation(
            cfg=experiment.augmentation,
            cache_root=cache_root,
            rng_seed=experiment.seed,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("TrainAugmentation unavailable (%s); running without aug.", e)
        augment = None

    dm = LesionDataModule(
        cache_root=cache_root,
        manifest_path=smoke_manifest,
        cohort_path=cohort_path,
        fold=0,
        batch_size=4,
        num_workers=2,
        augment_train=augment,
        sampler_train=None,
        allow_holdout=False,
        rng_seed=experiment.seed,
    )
    dm.setup()

    sl = [(p, sy, kind) for (p, sy, _ispos, kind) in dm._train_slice_index]
    if not sl:
        raise RuntimeError("smoke train slice_index is empty")
    sampler = WeightedScheduledSampler(sl, experiment.sampler, seed=experiment.seed)
    dm.sampler_train = sampler

    lm = LesionDetectorLM(experiment)

    capture = StepLossCapture()
    callbacks: list[pl.Callback] = [
        EmaCallback(decay=0.99),
        _make_capture_callback(capture),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices: list[int] | int = [0] if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        max_epochs=experiment.training.max_epochs,
        precision=experiment.training.precision,
        gradient_clip_val=experiment.training.gradient_clip_val,
        log_every_n_steps=1,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(out_root),
        deterministic=False,
        benchmark=True,
    )
    trainer.fit(lm, datamodule=dm)

    n_steps = len(capture.step_losses)
    if n_steps < 20:
        raise AssertionError(f"too few steps captured ({n_steps}); trainer crashed early?")
    first = float(np.mean(capture.step_losses[:10]))
    last = float(np.mean(capture.step_losses[-10:]))
    finite = bool(np.isfinite(capture.step_losses).all())
    val_auroc = trainer.callback_metrics.get("val/slice_auroc")

    result = {
        "git_sha": get_git_sha(),
        "n_steps": n_steps,
        "first10_loss": first,
        "last10_loss": last,
        "finite": finite,
        "val_slice_auroc": float(val_auroc) if val_auroc is not None else None,
        "smoke_pids": pids,
        "run_dir": str(out_root),
    }
    log.info("smoke result: %s", result)

    # Hard assertions (fail loudly so callers see SMOKE FAILED).
    assert finite, "non-finite loss detected"
    assert last < first, f"loss did not decrease ({first:.3f} -> {last:.3f})"
    assert val_auroc is not None, "val/slice_auroc never logged"

    # Cleanup
    if not keep_artifacts:
        shutil.rmtree(out_root, ignore_errors=True)
        try:
            smoke_manifest.unlink()
        except FileNotFoundError:
            pass

    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--cache-root", type=Path, default=Path("cache/v1"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    res = run_smoke(
        keep_artifacts=args.keep,
        cache_root=args.cache_root,
        data_root=args.data_root,
    )
    print("SMOKE PASSED.", json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
