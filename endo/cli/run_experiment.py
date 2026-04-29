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
  * WandB controlled via ``LoggingConfig.wandb.enabled`` (config) or
    ``--wandb`` / ``--no-wandb`` (CLI override). See top-level CLAUDE.md
    for the contract.
  * holdout is allowed only inside the ``predict_holdout`` subcommand
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from endo.config import ExperimentConfig, load_experiment
from endo.config.logging import LoggingConfig
from endo.utils.logging_setup import setup_logging
from endo.utils.provenance import (
    initial_provenance,
    save_provenance,
    update_fold_status,
)
from endo.utils.wandb_init import (
    build_wandb_logger,
    build_wandb_run,
    finish_run,
    is_wandb_enabled,
    log_summary,
    upload_artifact,
)


log = logging.getLogger("endo.cli")


# =============================================================================
# Helpers
# =============================================================================


def _apply_cli_logging_overrides(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    """Apply ``--wandb / --no-wandb / --wandb-mode / -v`` overrides in place."""
    cfg: LoggingConfig = experiment.logging
    wandb_arg = getattr(args, "wandb", None)
    if wandb_arg is True:
        cfg.wandb.enabled = True
    elif wandb_arg is False:
        cfg.wandb.enabled = False
    mode = getattr(args, "wandb_mode", None)
    if mode:
        cfg.wandb.mode = mode  # type: ignore[assignment]
    verbose = int(getattr(args, "verbose", 0) or 0)
    if verbose >= 2:
        cfg.file.level_console = "DEBUG"
        cfg.file.level_file = "DEBUG"
    elif verbose == 1:
        cfg.file.level_console = "DEBUG"


def _setup_run_logging(
    experiment: ExperimentConfig, run_dir: Path, fold: int | None = None
) -> None:
    setup_logging(experiment.logging.file, run_dir=run_dir, fold=fold)


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
        # Rewrite the materialized YAML so logging settings track the file
        # (drift-exempt — see ExperimentConfig.diff).
        experiment.to_yaml(yaml_path)
    else:
        experiment.to_yaml(yaml_path)
        shutil.copy2(experiment_path, py_copy_path)
        save_provenance(prov_path, initial_provenance())

    if not prov_path.exists():
        save_provenance(prov_path, initial_provenance())

    return run_dir


def _gpu_summary() -> dict[str, Any]:
    out: dict[str, Any] = {"cuda": False}
    try:
        import torch

        out["cuda"] = bool(torch.cuda.is_available())
        out["torch_version"] = str(torch.__version__)
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            out["device_index"] = int(idx)
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            try:
                props = torch.cuda.get_device_properties(idx)
                out["gpu_total_memory_gib"] = round(props.total_memory / (1024**3), 2)
            except Exception:  # noqa: BLE001
                pass
            try:
                out["cuda_version"] = str(torch.version.cuda)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


def _experiment_config_dict(experiment: ExperimentConfig) -> dict[str, Any]:
    """Serialize the experiment config as a plain dict for W&B config logging."""
    import json as _json

    return _json.loads(experiment.model_dump_json())


# =============================================================================
# train
# =============================================================================


def _build_datamodule_for_train(
    experiment: ExperimentConfig,
    fold: int,
):
    from endo.data.datamodule import LesionDataModule

    paths = experiment.paths
    train_cfg = experiment.training

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
        sampler_train=None,
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
    from endo.sampler.weighted import WeightedScheduledSampler

    sl = [(p, sy, kind) for (p, sy, _ispos, kind) in dm._train_slice_index]
    sampler = WeightedScheduledSampler(
        slice_index=sl,
        cfg=experiment.sampler,
        seed=experiment.seed + 1000 * int(fold),
    )
    return sampler


def _log_dataset_summary(
    wandb_logger,
    experiment: ExperimentConfig,
    dm,
    fold: int,
) -> None:
    """Log a once-per-run dataset / model summary to W&B."""
    if wandb_logger is False or wandb_logger is None:
        return
    try:
        from endo.utils.provenance import get_git_sha

        train_pids = list(getattr(dm, "_train_pids", []))
        val_pids = list(getattr(dm, "_val_pids", []))
        train_index = list(getattr(dm, "_train_slice_index", []))
        val_index = list(getattr(dm, "_val_slice_index", []))

        def _pos_neg_ratio(pids: list[str]) -> float:
            cache = getattr(dm, "_cache", {})
            pos = neg = 0
            for pid in pids:
                row = cache.get(pid, {}).get("manifest_row", {})
                if row.get("label") == "positive":
                    pos += 1
                else:
                    neg += 1
            if neg == 0:
                return float("inf")
            return float(pos) / float(neg)

        gpu = _gpu_summary()
        summary = {
            "n_train_volumes": len(train_pids),
            "n_val_volumes": len(val_pids),
            "n_train_slice_index": len(train_index),
            "n_val_slice_index": len(val_index),
            "pos_neg_volume_ratio_train": _pos_neg_ratio(train_pids),
            "pos_neg_volume_ratio_val": _pos_neg_ratio(val_pids),
            "fold": int(fold),
            "experiment_uuid": experiment.uuid,
            "experiment_short_uuid": experiment.short_uuid,
            "git_sha": get_git_sha(short=True),
            **gpu,
        }
        for k, v in summary.items():
            try:
                wandb_logger.experiment.summary[k] = v
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        log.debug("dataset summary logging skipped (%s)", e)


def _train_one_fold(
    experiment: ExperimentConfig,
    fold: int,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Train one fold. Returns a small status dict."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from endo.ema_callback import EmaCallback
    from endo.lightning_module import LesionDetectorLM
    from endo.sampler.periodic_eval import PeriodicDeepEvalCallback
    from endo.utils.aug_counters import AugStatsCallback
    from endo.utils.step_timer import StepTimerCallback

    fold_dir = run_dir / f"fold{fold}"
    ckpt_dir = fold_dir / "ckpts"
    runtime_dir = fold_dir / "runtime"
    deep_eval_dir = runtime_dir / "deep_eval"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    deep_eval_dir.mkdir(parents=True, exist_ok=True)

    # Re-init logging with the per-fold file handler.
    _setup_run_logging(experiment, run_dir, fold=fold)
    log.info("===== fold %d start =====", fold)

    use_wandb = is_wandb_enabled(experiment.logging)
    wandb_logger = (
        build_wandb_logger(experiment, fold=fold, stage="detector", save_dir=fold_dir)
        if use_wandb
        else False
    )
    if wandb_logger is not False:
        try:
            wandb_logger.experiment.config.update(
                _experiment_config_dict(experiment), allow_val_change=True
            )
        except Exception:  # noqa: BLE001
            pass

    # 1. DataModule + sampler.
    dm = _build_datamodule_for_train(experiment, fold)
    dm.setup()
    sampler = _build_sampler(dm, experiment, fold)
    dm.sampler_train = sampler

    # 2. Model.
    lm = LesionDetectorLM(experiment)

    # 3. Wire score-EMA tracker into the LightningModule.
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
    callbacks.append(StepTimerCallback())
    aug_pipeline = getattr(dm, "augment_train", None)
    if aug_pipeline is not None:
        callbacks.append(AugStatsCallback(aug_pipeline))

    if wandb_logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # PeriodicDeepEvalCallback.
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

    # 5. Trainer.
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
        logger=wandb_logger if wandb_logger is not False else False,
        enable_checkpointing=True,
        default_root_dir=str(fold_dir),
        deterministic=False,
        benchmark=True,
    )

    _log_dataset_summary(wandb_logger, experiment, dm, fold)

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

    # 9. Post-training viz (always when wandb is enabled, otherwise best-effort).
    _maybe_post_training_viz(experiment, fold, run_dir, wandb_logger)

    # 10. Upload best.ckpt + experiment artifacts.
    if wandb_logger is not False:
        try:
            run = wandb_logger.experiment
            upload_artifact(
                run,
                name=f"config-{experiment.short_uuid}",
                artifact_type="config",
                paths=[run_dir / "experiment.yaml", run_dir / "experiment.py"],
                aliases=["current"],
            )
            upload_artifact(
                run,
                name=f"provenance-{experiment.short_uuid}",
                artifact_type="provenance",
                paths=[run_dir / "provenance.json"],
            )
            if (
                experiment.logging.wandb.upload_checkpoints
                and best_ckpt.exists()
            ):
                upload_artifact(
                    run,
                    name=f"detector-fold{fold}-{experiment.short_uuid}",
                    artifact_type="model",
                    paths=[best_ckpt],
                    aliases=["best"],
                )
        except Exception as e:  # noqa: BLE001
            log.warning("artifact upload at fit-end failed (%s)", e)
        finally:
            try:
                wandb_logger.experiment.finish()
            except Exception:  # noqa: BLE001
                pass

    log.info(
        "fold %d finished in %.1fs best_val_slice_auroc=%s",
        fold,
        elapsed,
        f"{best_metric:.4f}" if best_metric == best_metric else "nan",
    )
    return fold_status


def _maybe_post_training_viz(
    experiment: ExperimentConfig,
    fold: int,
    run_dir: Path,
    wandb_logger,
) -> None:
    """Render `<fold_dir>/viz/epoch_post-train/*.png` and upload sampled slice."""
    if wandb_logger is False or wandb_logger is None:
        return
    try:
        from endo.viz.run_viz import sample_tp_fp_fn, visualize_predictions_for_fold

        fold_dir = run_dir / f"fold{fold}"
        viz_dir = fold_dir / "viz" / "epoch_post-train"
        log.info("rendering post-training viz under %s", viz_dir)
        try:
            visualize_predictions_for_fold(
                experiment=experiment,
                fold=fold,
                output_dir=viz_dir,
            )
        except FileNotFoundError as e:
            log.warning("post-train viz skipped (no checkpoint): %s", e)
            return
        if not experiment.logging.wandb.upload_viz_artifacts:
            return
        viz_cfg = experiment.logging.viz
        seed = int(experiment.seed) + 1000 * int(fold)
        sample_dir = sample_tp_fp_fn(
            viz_dir,
            n_tp=viz_cfg.sample_tp_per_fold,
            n_fp=viz_cfg.sample_fp_per_fold,
            n_fn=viz_cfg.sample_fn_per_fold,
            seed=seed,
        )
        upload_artifact(
            wandb_logger.experiment,
            name=f"viz-fold{fold}-{experiment.short_uuid}",
            artifact_type="viz",
            paths=[sample_dir],
            aliases=["latest"],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("post-training viz failed (%s)", e)


def cmd_train(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    _apply_cli_logging_overrides(experiment, args)
    folds = _parse_folds(args.fold, args.folds)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)

    _setup_run_logging(experiment, run_dir, fold=None)
    log.info("===== run start =====")
    log.info(
        "experiment=%s uuid=%s short=%s",
        experiment.name,
        experiment.uuid,
        experiment.short_uuid,
    )
    log.info("run_dir=%s folds=%s", run_dir, folds)
    gpu = _gpu_summary()
    log.info(
        "cuda=%s device=%s name=%s total=%s GiB",
        gpu.get("cuda"),
        gpu.get("device_index"),
        gpu.get("gpu_name"),
        gpu.get("gpu_total_memory_gib"),
    )
    cfg = experiment.logging.wandb
    log.info(
        "wandb=%s project=%s mode=%s",
        "ON" if is_wandb_enabled(experiment.logging) else "OFF",
        cfg.project,
        cfg.mode,
    )

    for f in folds:
        log.info("----- fold %d -----", f)
        _train_one_fold(experiment, f, run_dir, args.resume)
    return 0


# =============================================================================
# smoke
# =============================================================================


def cmd_smoke(args: argparse.Namespace) -> int:
    setup_logging(LoggingConfig().file)
    from scripts.smoke_train import run_smoke

    run_smoke(keep_artifacts=False)
    log.info("SMOKE PASSED.")
    return 0


# =============================================================================
# eval / predict_holdout / train_gru / viz — delegating subcommands
# =============================================================================


def cmd_eval(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    _apply_cli_logging_overrides(experiment, args)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    _setup_run_logging(experiment, run_dir, fold=None)

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = build_wandb_run(
        experiment, fold=None, stage="eval", save_dir=eval_dir
    )
    try:
        try:
            from endo.eval.run_eval import run_cv_evaluation
        except Exception as e:  # noqa: BLE001
            log.error("eval module not available: %s", e)
            return 1

        res = run_cv_evaluation(experiment=experiment, use_gru=args.use_gru, eval_dir=eval_dir)
        log.info("CV evaluation done: %s", res)

        if wandb_run is not None:
            log_summary(
                wandb_run,
                {
                    "run_id": res.get("run_id"),
                    "ensemble_threshold_large": res.get("ensemble_threshold", {}).get("large"),
                    "ensemble_threshold_small": res.get("ensemble_threshold", {}).get("small"),
                    "n_rows": res.get("n_rows"),
                    "use_gru": bool(args.use_gru),
                },
            )
            if experiment.logging.wandb.upload_eval_reports:
                upload_paths = []
                for fname in (
                    "eval_report.csv",
                    "eval_thresholds.json",
                ):
                    p = eval_dir / fname
                    if p.exists():
                        upload_paths.append(p)
                # also include any per-call jsonl from this run
                if res.get("calls_path"):
                    p = Path(res["calls_path"])
                    if p.exists():
                        upload_paths.append(p)
                upload_artifact(
                    wandb_run,
                    name=f"eval-report-{experiment.short_uuid}",
                    artifact_type="eval-report",
                    paths=upload_paths,
                    aliases=["latest"],
                )
    finally:
        finish_run(wandb_run)
    return 0


def cmd_predict_holdout(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    _apply_cli_logging_overrides(experiment, args)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    _setup_run_logging(experiment, run_dir, fold=None)

    if args.ckpts.strip().lower() == "all":
        ckpts: list[int] | str = [0, 1, 2, 3, 4]
    else:
        ckpts = [int(p.strip()) for p in args.ckpts.split(",") if p.strip()]

    holdout_dir = run_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = build_wandb_run(
        experiment, fold=None, stage="holdout", save_dir=holdout_dir
    )
    try:
        try:
            from endo.eval.run_eval import run_holdout_inference
        except Exception as e:  # noqa: BLE001
            log.error("eval module not available: %s", e)
            return 1
        out = run_holdout_inference(
            experiment=experiment,
            ckpts=ckpts,
            use_gru=args.use_gru,
        )
        log.info("holdout invocation dir: %s", out)
        if wandb_run is not None:
            try:
                _log_holdout_summary(wandb_run, out, experiment)
            except Exception as e:  # noqa: BLE001
                log.warning("holdout W&B summary failed (%s)", e)
            if experiment.logging.wandb.upload_eval_reports:
                upload_paths = []
                for fname in (
                    "eval_report.csv",
                    "invocation.json",
                ):
                    p = out / fname
                    if p.exists():
                        upload_paths.append(p)
                # any per_call jsonl
                for p in out.glob("per_call_*.jsonl"):
                    upload_paths.append(p)
                if upload_paths:
                    upload_artifact(
                        wandb_run,
                        name=f"holdout-report-{experiment.short_uuid}",
                        artifact_type="holdout-report",
                        paths=upload_paths,
                        aliases=["latest"],
                    )
    finally:
        finish_run(wandb_run)
    return 0


def _log_holdout_summary(wandb_run, holdout_dir: Path, experiment: ExperimentConfig) -> None:
    """Attach holdout score histogram + summary table to the W&B run."""
    invocation_path = holdout_dir / "invocation.json"
    summary: dict[str, Any] = {}
    if invocation_path.exists():
        try:
            import json as _json

            summary.update(_json.loads(invocation_path.read_text()))
        except Exception:  # noqa: BLE001
            pass
    log_summary(wandb_run, summary)
    # Histogram of per-patient scores (read from eval_report.csv if present).
    csv_path = holdout_dir / "eval_report.csv"
    if not csv_path.exists():
        return
    try:
        import csv as _csv

        scores: list[float] = []
        with csv_path.open("r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    scores.append(float(row.get("value", "nan")))
                except Exception:  # noqa: BLE001
                    continue
        if scores:
            try:
                import wandb  # type: ignore[import]

                wandb_run.log(
                    {"holdout/eval_value_hist": wandb.Histogram(scores)}, commit=True
                )
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def cmd_train_gru(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    _apply_cli_logging_overrides(experiment, args)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    _setup_run_logging(experiment, run_dir, fold=None)
    folds = _parse_folds(args.fold, args.folds)
    stage = args.stage
    try:
        from endo.gru.feature_cache import extract_features_for_fold
        from endo.gru.train import train_gru_for_fold
    except Exception as e:  # noqa: BLE001
        log.error("gru module not available: %s", e)
        return 1
    for f in folds:
        gru_dir = run_dir / f"fold{f}" / "gru"
        gru_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = build_wandb_run(
            experiment, fold=f, stage="gru", save_dir=gru_dir
        )
        try:
            if stage in ("feature_cache", "all"):
                log.info("[fold %d] extracting backbone features", f)
                extract_features_for_fold(experiment, f)
            if stage in ("train", "all"):
                log.info("[fold %d] training GRU", f)
                ckpt_path = train_gru_for_fold(experiment, f)
                if wandb_run is not None and Path(ckpt_path).exists():
                    log_summary(
                        wandb_run,
                        {"gru_ckpt": str(ckpt_path), "fold": int(f)},
                    )
                    if experiment.logging.wandb.upload_checkpoints:
                        upload_artifact(
                            wandb_run,
                            name=f"gru-fold{f}-{experiment.short_uuid}",
                            artifact_type="model",
                            paths=[Path(ckpt_path)],
                            aliases=["best"],
                        )
        finally:
            finish_run(wandb_run)
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    _apply_cli_logging_overrides(experiment, args)
    run_dir = _bootstrap_run_dir(Path(args.experiment), experiment, args.force_resync)
    _setup_run_logging(experiment, run_dir, fold=None)
    folds = _parse_folds(args.fold, args.folds)
    try:
        from endo.viz.run_viz import sample_tp_fp_fn, visualize_predictions_for_fold
    except Exception as e:  # noqa: BLE001
        log.error("viz module not available: %s", e)
        return 1
    for f in folds:
        log.info("[fold %d] rendering visualizations", f)
        viz_dir = run_dir / f"fold{f}" / "viz" / "epoch_post-train"
        out = visualize_predictions_for_fold(
            experiment=experiment, fold=f, output_dir=viz_dir
        )
        wandb_run = build_wandb_run(
            experiment, fold=f, stage="viz", save_dir=run_dir / f"fold{f}"
        )
        try:
            if wandb_run is not None and experiment.logging.wandb.upload_viz_artifacts:
                viz_cfg = experiment.logging.viz
                seed = int(experiment.seed) + 1000 * int(f)
                sample_dir = sample_tp_fp_fn(
                    out,
                    n_tp=viz_cfg.sample_tp_per_fold,
                    n_fp=viz_cfg.sample_fp_per_fold,
                    n_fn=viz_cfg.sample_fn_per_fold,
                    seed=seed,
                )
                upload_artifact(
                    wandb_run,
                    name=f"viz-fold{f}-{experiment.short_uuid}",
                    artifact_type="viz",
                    paths=[sample_dir],
                    aliases=["latest"],
                )
        finally:
            finish_run(wandb_run)
    return 0


def cmd_qc_paste(args: argparse.Namespace) -> int:
    setup_logging(LoggingConfig().file)
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
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--wandb", dest="wandb", action="store_true", default=None,
                     help="force-enable W&B logging (overrides experiment config)")
    grp.add_argument("--no-wandb", dest="wandb", action="store_false", default=None,
                     help="force-disable W&B logging")
    p.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default=None,
        help="override LoggingConfig.wandb.mode",
    )
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v=DEBUG console, -vv=DEBUG console+file")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_experiment", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    _add_common(p_train)
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
