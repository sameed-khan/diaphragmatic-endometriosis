"""Structured logging setup — console + per-fold rotating file handler.

Used by ``endo.cli.run_experiment`` to replace the legacy ``_setup_logging``
helper. One call per ``(run_dir, fold)`` configures the root logger with:

  * a stdout ``StreamHandler`` at ``cfg.level_console``
  * a ``RotatingFileHandler`` at ``cfg.level_file`` writing to either
    ``<run_dir>/run.log`` (top-level) or ``<run_dir>/fold{f}/run.log``.

The file handler captures a structured ``%(asctime)s [%(levelname)s] %(name)s: %(message)s``
record per line, which keeps the file ``grep``-friendly. Tqdm bars stay on
stdout — they NEVER go through the structured handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from endo.config.logging import FileLoggingConfig

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Sentinel attribute used to mark handlers we own so repeated calls don't
# stack up duplicate handlers.
_OWNED_ATTR = "_endo_logging_owned"


def _resolve_level(name: str) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)


def _drop_owned_handlers(root: logging.Logger) -> None:
    for h in list(root.handlers):
        if getattr(h, _OWNED_ATTR, False):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass


def setup_logging(
    cfg: FileLoggingConfig,
    run_dir: Path | None = None,
    fold: int | None = None,
) -> Path | None:
    """Configure the root logger and return the path to the file log (if any).

    Args:
        cfg: file/console log levels + rotation policy.
        run_dir: top-level run directory. If ``None``, only stdout logging
            is configured (file handler is skipped).
        fold: optional fold index. When set, the file log is written to
            ``<run_dir>/fold{fold}/run.log``; otherwise to
            ``<run_dir>/run.log``.

    Returns the resolved file-log path or ``None`` when no file handler was
    installed.
    """
    root = logging.getLogger()
    # Allow a more permissive root level than handler thresholds so DEBUG
    # records can still be filtered by handler.
    handler_levels = (_resolve_level(cfg.level_console), _resolve_level(cfg.level_file))
    root.setLevel(min(handler_levels))

    _drop_owned_handlers(root)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(_resolve_level(cfg.level_console))
    sh.setFormatter(fmt)
    setattr(sh, _OWNED_ATTR, True)
    root.addHandler(sh)

    file_path: Path | None = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        if fold is not None:
            file_dir = run_dir / f"fold{int(fold)}"
        else:
            file_dir = run_dir
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / "run.log"
        fh = logging.handlers.RotatingFileHandler(
            str(file_path),
            maxBytes=int(cfg.rotate_max_bytes),
            backupCount=int(cfg.rotate_backups),
            encoding="utf-8",
        )
        fh.setLevel(_resolve_level(cfg.level_file))
        fh.setFormatter(fmt)
        setattr(fh, _OWNED_ATTR, True)
        root.addHandler(fh)

    return file_path
