"""W&B logger construction + helper utilities.

Single source of truth for *how* a wandb run is named, grouped, and tagged
for any of the pipeline stages (``detector``, ``gru``, ``eval``, ``holdout``,
``viz``). The CLI calls :func:`build_wandb_logger` to obtain a Lightning
``WandbLogger`` (or ``False`` when wandb is disabled) and uses
:func:`build_wandb_run` for non-Lightning stages (eval / holdout / viz).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from endo.config import ExperimentConfig
from endo.config.logging import LoggingConfig, WandbConfig

log = logging.getLogger(__name__)


_STAGE_DEFAULT_NAME = {
    "detector": "fold{fold}",
    "gru": "fold{fold}-gru",
    "eval": "cv_summary",
    "holdout": "holdout",
    "viz": "viz",
}


def _load_dotenv_once() -> None:
    """Load ``.env`` from the current working directory, exactly once.

    Idempotent — safe to call from every CLI subcommand.
    """
    if os.environ.get("_ENDO_DOTENV_LOADED") == "1":
        return
    try:
        from dotenv import load_dotenv

        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception as e:  # noqa: BLE001
        log.debug("dotenv not loaded (%s)", e)
    os.environ["_ENDO_DOTENV_LOADED"] = "1"


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _experiment_name(experiment: ExperimentConfig, wandb_cfg: WandbConfig) -> str:
    return wandb_cfg.experiment_name or experiment.name


def resolve_group(experiment: ExperimentConfig, logging_cfg: LoggingConfig) -> str:
    wandb_cfg = logging_cfg.wandb
    if wandb_cfg.group:
        return wandb_cfg.group
    exp_name = _experiment_name(experiment, wandb_cfg)
    return f"{exp_name}_{experiment.short_uuid}"


def resolve_run_name(
    experiment: ExperimentConfig,
    logging_cfg: LoggingConfig,
    *,
    stage: str,
    fold: int | None,
) -> str:
    wandb_cfg = logging_cfg.wandb
    exp_name = _experiment_name(experiment, wandb_cfg)
    if wandb_cfg.run_name is not None:
        # The plan defines the holdout sub-run as ``{run_name}-holdout`` when
        # the user has explicitly set ``run_name``. Detector / gru / eval /
        # viz simply use ``run_name`` verbatim.
        if stage == "holdout":
            return f"{wandb_cfg.run_name}-holdout"
        return str(wandb_cfg.run_name)
    template = _STAGE_DEFAULT_NAME.get(stage, stage)
    if "{fold}" in template:
        if fold is None:
            template = template.replace("{fold}", "x")
        else:
            template = template.format(fold=int(fold))
    return f"{exp_name}/{template}"


def resolve_tags(
    experiment: ExperimentConfig,
    logging_cfg: LoggingConfig,
    *,
    stage: str,
    fold: int | None,
) -> list[str]:
    wandb_cfg = logging_cfg.wandb
    tag_set: list[str] = []
    # experiment.tags is a dict[str, str] — values are the public tags.
    for v in experiment.tags.values():
        if v not in tag_set:
            tag_set.append(str(v))
    tag_set.append(f"stage={stage}")
    if fold is not None:
        tag_set.append(f"fold={int(fold)}")
    for t in wandb_cfg.tags:
        if t not in tag_set:
            tag_set.append(str(t))
    return tag_set


def is_wandb_enabled(logging_cfg: LoggingConfig) -> bool:
    cfg = logging_cfg.wandb
    return bool(cfg.enabled) and cfg.mode != "disabled"


def build_wandb_logger(
    experiment: ExperimentConfig,
    *,
    fold: int | None,
    stage: str,
    save_dir: Path,
):
    """Return a Lightning ``WandbLogger`` configured for this stage.

    Returns ``False`` when wandb is disabled (matches Lightning's "no logger"
    sentinel so the CLI can pass the result straight into ``pl.Trainer``).
    """
    logging_cfg: LoggingConfig = experiment.logging
    if not is_wandb_enabled(logging_cfg):
        return False
    _load_dotenv_once()
    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception as e:  # noqa: BLE001
        log.warning("WandB requested but pytorch_lightning.loggers not importable: %s", e)
        return False

    wandb_cfg = logging_cfg.wandb
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        return WandbLogger(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=resolve_group(experiment, logging_cfg),
            name=resolve_run_name(experiment, logging_cfg, stage=stage, fold=fold),
            tags=resolve_tags(experiment, logging_cfg, stage=stage, fold=fold),
            save_dir=str(save_dir),
            mode=wandb_cfg.mode,
            log_model=False,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("WandbLogger init failed (%s); continuing without W&B.", e)
        return False


def build_wandb_run(
    experiment: ExperimentConfig,
    *,
    fold: int | None,
    stage: str,
    save_dir: Path,
):
    """Build a raw ``wandb.run`` for non-Lightning stages.

    Returns the active ``wandb.Run`` object, or ``None`` when wandb is
    disabled or ``wandb`` is not importable.
    """
    logging_cfg: LoggingConfig = experiment.logging
    if not is_wandb_enabled(logging_cfg):
        return None
    _load_dotenv_once()
    try:
        import wandb  # type: ignore[import]
    except Exception as e:  # noqa: BLE001
        log.warning("wandb not importable (%s); continuing without W&B.", e)
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    wandb_cfg = logging_cfg.wandb
    try:
        run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=resolve_group(experiment, logging_cfg),
            name=resolve_run_name(experiment, logging_cfg, stage=stage, fold=fold),
            tags=resolve_tags(experiment, logging_cfg, stage=stage, fold=fold),
            dir=str(save_dir),
            mode=wandb_cfg.mode,
            reinit=True,
            settings=wandb.Settings(
                _disable_stats=not wandb_cfg.log_system_metrics,
            ) if wandb_cfg.log_system_metrics is False else None,
        )
        return run
    except Exception as e:  # noqa: BLE001
        log.warning("wandb.init failed (%s); continuing without W&B.", e)
        return None


def upload_artifact(
    run: Any,
    *,
    name: str,
    artifact_type: str,
    paths: Iterable[Path],
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Upload ``paths`` (files or directories) as a single named artifact."""
    if run is None:
        return False
    try:
        import wandb  # type: ignore[import]
    except Exception:
        return False
    try:
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        for p in paths:
            p = Path(p)
            if not p.exists():
                log.warning("artifact path missing: %s", p)
                continue
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
        run.log_artifact(artifact, aliases=aliases or [])
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("artifact upload failed (%s)", e)
        return False


def log_summary(run: Any, payload: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        for k, v in payload.items():
            run.summary[k] = v
    except Exception as e:  # noqa: BLE001
        log.warning("summary update failed (%s)", e)


def finish_run(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # noqa: BLE001
        pass
