"""Tests for endo.utils.logging_setup."""

from __future__ import annotations

import logging
from pathlib import Path

from endo.config import FileLoggingConfig
from endo.utils.logging_setup import setup_logging


def _drop_owned(root: logging.Logger) -> None:
    for h in list(root.handlers):
        if getattr(h, "_endo_logging_owned", False):
            root.removeHandler(h)


def test_setup_logging_writes_per_fold_file(tmp_path: Path) -> None:
    cfg = FileLoggingConfig(level_console="INFO", level_file="DEBUG")
    out = setup_logging(cfg, run_dir=tmp_path, fold=0)
    try:
        assert out is not None and out == tmp_path / "fold0" / "run.log"
        log = logging.getLogger("endo.test")
        log.info("hello")
        log.debug("debugfine")
        # File handler should pick up DEBUG; console only INFO+.
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        text = out.read_text()
        assert "hello" in text
        assert "debugfine" in text
    finally:
        _drop_owned(logging.getLogger())


def test_setup_logging_no_run_dir(tmp_path: Path) -> None:
    cfg = FileLoggingConfig(level_console="INFO", level_file="INFO")
    out = setup_logging(cfg, run_dir=None, fold=None)
    try:
        assert out is None
        # Just ensure stdout handler exists and accepts a record.
        logging.getLogger("endo.test").info("x")
    finally:
        _drop_owned(logging.getLogger())


def test_setup_logging_idempotent(tmp_path: Path) -> None:
    cfg = FileLoggingConfig()
    setup_logging(cfg, run_dir=tmp_path, fold=0)
    setup_logging(cfg, run_dir=tmp_path, fold=0)
    owned = [
        h for h in logging.getLogger().handlers
        if getattr(h, "_endo_logging_owned", False)
    ]
    try:
        # Exactly one stream + one file handler should remain.
        assert len(owned) == 2
    finally:
        _drop_owned(logging.getLogger())
