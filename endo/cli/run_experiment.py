"""``run_experiment`` — single CLI for the diaphragmatic-endometriosis project.

Subcommands (PRD §4):

  - ``train``           — train the detector for one or more folds
  - ``train_gru``       — Stage-2 GRU rescorer (per fold, two stages)
  - ``eval``            — CV evaluation across all 5 folds
  - ``predict_holdout`` — ad-hoc inference on the 122 holdout patients
  - ``viz``             — per-slice TP/FP/FN visualization
  - ``smoke``           — 5-min integration smoke gate

Conventions:
  * one experiment file per run, located at ``experiments/<name>.py``
  * artifacts under ``runs/<name>_<uuid8>/``
  * fold-as-run; multi-fold execution is sequential by default
  * WandB OFF unless ``--wandb`` (PRD A.9)
  * holdout is allowed only inside the ``predict_holdout`` subcommand
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from endo.config import ExperimentConfig, load_experiment
from endo.utils.provenance import (
    initial_provenance,
    load_provenance,
    save_provenance,
    update_fold_status,
)


log = logging.getLogger("endo.cli")


# =============================================================================
# Helpers
# =============================================================================


def _setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _parse_folds(arg_fold: int | None, arg_folds: str | None) -> list[int]:
    if arg_fold is not None and arg_folds is not None:
        raise SystemExit("--fold and --folds are mutually exclusive")
    if arg_fold is not None:
        if arg_fold not in range(5):
            raise SystemExit(f"--fold must be in [0..4], got {arg_fold}")
        return [int(arg_fold)]
    if arg_folds is None:
        return [0]
    if arg_folds.strip().lower() == "all":
        return [0, 1, 2, 3, 4]
    out = []
    for part in arg_folds.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v not in range(5):
            raise SystemExit(f"--folds entry {v} not in [0..4]")
        out.append(v)
    if not out:
        raise SystemExit("--folds parsed to empty list")
    return sorted(set(out))


def _bootstrap_run_dir(
    experiment_path: Path,
    experiment: ExperimentConfig,
    force_resync: bool = False,
) -> Path:
    """Create or reuse ``runs/<exp>_<uuid8>/``; enforce drift guard."""
    run_dir = experiment.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = run_dir / "experiment.yaml"
    py_copy_path = run_dir / "experiment.py"
    prov_path = run_dir / "provenance.json"

    if yaml_path.exists() and not force_resync:
        prior = ExperimentConfig.from_yaml(yaml_path)
        diffs = experiment.diff(prior)
        if diffs:
            log.error(
                "Experiment drift detected vs %s.\n  %s",
                yaml_path,
                "\n  ".join(diffs[:20]),
            )
            raise SystemExit(
                "Experiment file differs from materialized experiment.yaml. "
                "Use --force-resync if this is intentional."
            )
    else:
        experiment.to_yaml(yaml_path)
        shutil.copy2(experiment_path, py_copy_path)
        save_provenance(prov_path, initial_provenance())

    if not prov_path.exists():
        save_provenance(prov_path, initial_provenance())

    return run_dir


# =============================================================================
# train
# =============================================================================


def _build_datamodule_for_train(
    experiment: ExperimentConfig,
    fold: int,
):
    """Construct a configured ``LesionDataModule`` for training of one fold."""
    # Lazy imports — keep CLI startup snappy and tolerant of missing optional
    # components.
    from endo.data.datamodule import LesionDataModule

    paths = experiment.paths
    train_cfg = experiment.training

    # Try to construct training augmentation. If Component 4 isn't ready or any
    # required artifact (lesion bank) is missing, we fall back to
    # ``augment_train=None`` (no online augmentation).
    augment_train = _try_build_train_augmentation(experiment)

    dm = LesionDataModule(
        cache_root=paths.cache_root,
        manifest_path=paths.data_root / "manifest.jsonl",
        cohort_path=paths.data_root / "cohort.json",
        fold=int(fold),
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        slice_window=train_cfg.slice_window,
        target_input_shape=train_cfg.target_input_shape,
        augment_train=augment_train,
        sampler_train=None,  # filled in after setup() once slice_index is built
        allow_holdout=False,
        rng_seed=experiment.seed,
    )
    return dm


def _try_build_train_augmentation(experiment: ExperimentConfig):
    try:
        from endo.augmentation.transform import TrainAugmentation
    except Exception as e:  # noqa: BLE001
        log.warning(
            "endo.augmentation.transform.TrainAugmentation not importable yet "
            "(%s). Training will run WITHOUT online augmentation.",
            e,
        )
        return None
    try:
        return TrainAugmentation(
            cfg=experiment.augmentation,
            cache_root=experiment.paths.cache_root,
            rng_seed=experiment.seed,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "TrainAugmentation construction failed (%s). Disabling augmentation "
            "for this run.",
            e,
        )
        return None


def _build_sampler(dm, experiment: ExperimentConfig, fold: int):
    """Construct the WeightedScheduledSampler from the dm's slice_index."""
    from endo.sampler.weighted import WeightedScheduledSampler

    # The datamodule emits 4-tuples; the sampler expects 3-tuples (pid, sy, kind).
    sl = [(p, sy, kind) for (p, sy, _ispos, kind) in dm._train_slice_index]
    sampler = WeightedScheduledSampler(
        slice_index=sl,
        cfg=experiment.sampler,
        seed=experiment.seed + 1000 * int(fold),
    )
    return sampler


def _train_one_fold(
    experiment: ExperimentConfig,
    fold: int,
    run_dir: Path,
    use_wandb: bool,
    resume: bool,
) -> dict[str, Any]:
    """Train one fold. Returns a small status dict."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from endo.ema_callback import EmaCallback
    from endo.lightning_module import LesionDetectorLM
    from endo.sampler.periodic_eval import PeriodicDeepEvalCallback

    fold_dir = run_dir / f"fold{fold}"
    ckpt_dir = fold_dir / "ckpts"
    runtime_dir = fold_dir / "runtime"
    deep_eval_dir = runtime_dir / "deep_eval"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    deep_eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. DataModule + sampler.
    dm = _build_datamodule_for_train(experiment, fold)
    dm.setup()
    sampler = _build_sampler(dm, experiment, fold)
    dm.sampler_train = sampler

    # 2. Model.
    lm = LesionDetectorLM(experiment)

    # 3. Wire score-EMA tracker into the LightningModule (Component 5 §5).
    try:
        from endo.sampler.score_ema import ScoreEMATracker

        lm.score_ema_tracker = ScoreEMATracker(
            decay=float(experiment.sampler.score_ema_decay)
        )
    except Exception as e:  # noqa: BLE001
        log.warning("ScoreEMATracker not available (%s) — HNM disabled.", e)

    # 4. Callbacks.
    callbacks: list[pl.Callback] = []
    ema_cb = EmaCallback(decay=experiment.training.ema_decay)
    callbacks.append(ema_cb)
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val/slice_auroc",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    )
    # LearningRateMonitor requires a logger — only add when one is wired up.
    if use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # PeriodicDeepEvalCallback wires the hard-pool + deep-eval cache.
    try:
        train_neg_pids = [
            pid for pid in dm._train_pids
            if (dm._cache.get(pid, {}).get("manifest_row", {}).get("label") == "negative")
        ]
        val_pids = list(dm._val_pids)
        val_volume_labels = {
            pid: int(dm._cache[pid]["manifest_row"].get("label") == "positive")
            for pid in val_pids if pid in dm._cache
        }
        callbacks.append(
            PeriodicDeepEvalCallback(
                sampler_cfg=experiment.sampler,
                run_dir=fold_dir,
                train_neg_pids=train_neg_pids,
                val_pids=val_pids,
                ema_callback=ema_cb,
                val_volume_labels=val_volume_labels,
            )
        )
    except Exception as e:  # noqa: BLE001
        log.warning("PeriodicDeepEvalCallback wiring failed (%s).", e)

    # 5. Logger (default OFF; opt-in WandB).
    logger: Any = False
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                project="diaphragmatic-endometriosis",
                group=f"{experiment.name}_{experiment.short_uuid}",
                name=f"fold{fold}",
                tags=list({**experiment.tags, "fold": str(fold)}.values()),
                save_dir=str(fold_dir),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("WandB logger requested but failed to init: %s", e)
            logger = False

    # 6. Trainer.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices: list[int] | int = [0] if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        max_epochs=experiment.training.max_epochs,
        precision=experiment.training.precision,
        gradient_clip_val=experiment.training.gradient_clip_val,
        log_every_n_steps=experiment.training.log_every_n_steps,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=str(fold_dir),
        deterministic=False,
        benchmark=True,
    )

    # 7. Fit.
    update_fold_status(run_dir / "provenance.json", fold, "running")
    started = time.time()
    ckpt_path = ckpt_dir / "last.ckpt"
    resume_from = str(ckpt_path) if (resume and ckpt_path.exists()) else None

    try:
        trainer.fit(lm, datamodule=dm, ckpt_path=resume_from)
    except Exception:
        update_fold_status(run_dir / "provenance.json", fold, "failed")
        raise

    elapsed = time.time() - started
    update_fold_status(run_dir / "provenance.json", fold, "complete")

    # 8. Write fold_status.json.
    best_metric = float("nan")
    try:
        m = trainer.callback_metrics.get("val/slice_auroc")
        if m is not None:
            best_metric = float(m.detach().cpu().item()) if hasattr(m, "detach") else float(m)
    except Exception:
        pass

    best_ckpt = ckpt_dir / "best.ckpt"
    fold_status = {
        "fold": int(fold),
        "best_val_slice_auroc": best_metric,
        "best_ckpt": str(best_ckpt) if best_ckpt.exists() else None,
        "wall_clock_seconds": elapsed,
        "wandb_used": bool(use_wandb),
    }
    save_provenance(fold_dir / "fold_status.json", fold_status)
    return fold_status


def cmd_train(args: argparse.Namespace) -> int:
    _setup_logging()
    experiment = load_experiment(args.experiment)
    folds = _parse_folds(args.fold, args.folds)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)

    log.info("run_dir=%s folds=%s", run_dir, folds)
    for f in folds:
        log.info("=== Training fold %d ===", f)
        _train_one_fold(experiment, f, run_dir, args.wandb, args.resume)
    return 0


# =============================================================================
# smoke
# =============================================================================


def cmd_smoke(args: argparse.Namespace) -> int:
    _setup_logging()
    # Defer to scripts/smoke_train.run_smoke which performs the smoke pid
    # subset selection, training, and assertions.
    from scripts.smoke_train import run_smoke

    run_smoke(keep_artifacts=False)
    log.info("SMOKE PASSED.")
    return 0


# =============================================================================
# eval / predict_holdout / train_gru / viz — delegating subcommands
# =============================================================================


def cmd_eval(args: argparse.Namespace) -> int:
    _setup_logging()
    experiment = load_experiment(args.experiment)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    try:
        from endo.eval.run_eval import run_cv_evaluation
    except Exception as e:  # noqa: BLE001
        log.error("eval module not available: %s", e)
        return 1
    eval_dir = run_dir / "eval"
    res = run_cv_evaluation(experiment=experiment, use_gru=args.use_gru, eval_dir=eval_dir)
    log.info("CV evaluation done: %s", res)
    return 0


def cmd_predict_holdout(args: argparse.Namespace) -> int:
    _setup_logging()
    experiment = load_experiment(args.experiment)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    try:
        from endo.eval.run_eval import run_holdout_inference
    except Exception as e:  # noqa: BLE001
        log.error("eval module not available: %s", e)
        return 1
    if args.ckpts.strip().lower() == "all":
        ckpts: list[int] | str = [0, 1, 2, 3, 4]
    else:
        ckpts = [int(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    out = run_holdout_inference(
        experiment=experiment,
        ckpts=ckpts,
        use_gru=args.use_gru,
    )
    log.info("holdout invocation dir: %s", out)
    return 0


def cmd_train_gru(args: argparse.Namespace) -> int:
    _setup_logging()
    experiment = load_experiment(args.experiment)
    _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    folds = _parse_folds(args.fold, args.folds)
    stage = args.stage
    try:
        from endo.gru.feature_cache import extract_features_for_fold
        from endo.gru.train import train_gru_for_fold
    except Exception as e:  # noqa: BLE001
        log.error("gru module not available: %s", e)
        return 1
    for f in folds:
        if stage in ("feature_cache", "all"):
            log.info("[fold %d] extracting backbone features", f)
            extract_features_for_fold(experiment, f)
        if stage in ("train", "all"):
            log.info("[fold %d] training GRU", f)
            train_gru_for_fold(experiment, f)
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    _setup_logging()
    experiment = load_experiment(args.experiment)
    _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    folds = _parse_folds(args.fold, args.folds)
    try:
        from endo.viz.run_viz import visualize_predictions_for_fold
    except Exception as e:  # noqa: BLE001
        log.error("viz module not available: %s", e)
        return 1
    for f in folds:
        log.info("[fold %d] rendering visualizations", f)
        visualize_predictions_for_fold(experiment=experiment, fold=f)
    return 0


def cmd_qc_paste(args: argparse.Namespace) -> int:
    _setup_logging()
    log.info("qc_paste is a dev workflow; rendering composites is delegated "
             "to scripts/qc_paste_review.py (advisory only).")
    return 0


# =============================================================================
# Argparse wiring
# =============================================================================


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--experiment", type=str, required=True, help="path to experiments/<name>.py")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--folds", type=str, default=None, help='CSV of fold ids or "all"')
    p.add_argument("--force-resync", action="store_true",
                   help="overwrite experiment.yaml (drift override)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_experiment", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    _add_common(p_train)
    p_train.add_argument("--wandb", action="store_true")
    p_train.add_argument("--resume", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--experiment", type=str, default="experiments/smoke.py")
    p_smoke.add_argument("--device", type=int, default=0)
    p_smoke.set_defaults(func=cmd_smoke)

    p_eval = sub.add_parser("eval")
    _add_common(p_eval)
    p_eval.add_argument("--use-gru", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_pred = sub.add_parser("predict_holdout")
    _add_common(p_pred)
    p_pred.add_argument("--ckpts", type=str, default="all",
                        help='CSV of fold indices or "all"')
    p_pred.add_argument("--use-gru", action="store_true")
    p_pred.set_defaults(func=cmd_predict_holdout)

    p_gru = sub.add_parser("train_gru")
    _add_common(p_gru)
    p_gru.add_argument("--stage", choices=("feature_cache", "train", "all"), default="all")
    p_gru.set_defaults(func=cmd_train_gru)

    p_viz = sub.add_parser("viz")
    _add_common(p_viz)
    p_viz.set_defaults(func=cmd_viz)

    p_qc = sub.add_parser("qc_paste")
    p_qc.set_defaults(func=cmd_qc_paste)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
