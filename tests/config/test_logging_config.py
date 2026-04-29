"""Tests for LoggingConfig + drift-exempt behavior."""

from __future__ import annotations

import yaml

from endo.config import (
    ExperimentConfig,
    FileLoggingConfig,
    LoggingConfig,
    WandbConfig,
)


def _base_experiment(**overrides) -> ExperimentConfig:
    kw = dict(uuid="b3a7f1e9-4c8a-4d2b-9f1c-0e6a8b9c1d2e", name="test")
    kw.update(overrides)
    return ExperimentConfig(**kw)


def test_logging_config_defaults_match_plan():
    cfg = LoggingConfig()
    assert cfg.wandb.enabled is False
    assert cfg.wandb.project == "diaphragmatic-endometriosis"
    assert cfg.wandb.mode == "online"
    assert cfg.viz.log_during_training is False
    assert cfg.viz.sample_tp_per_fold == 20


def test_logging_subtree_drift_exempt():
    a = _base_experiment(
        logging=LoggingConfig(wandb=WandbConfig(enabled=False, mode="online"))
    )
    b = _base_experiment(
        logging=LoggingConfig(wandb=WandbConfig(enabled=True, mode="offline"))
    )
    assert a.diff(b) == []
    # Sanity: actual config drift is still detected.
    c = _base_experiment(name="other")
    assert any("name" in d for d in a.diff(c))


def test_logging_yaml_round_trip(tmp_path):
    exp = _base_experiment(
        logging=LoggingConfig(
            file=FileLoggingConfig(level_console="DEBUG", level_file="INFO"),
            wandb=WandbConfig(enabled=True, run_name="custom"),
        )
    )
    yaml_path = tmp_path / "experiment.yaml"
    exp.to_yaml(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())
    assert "logging" in raw
    reloaded = ExperimentConfig.from_yaml(yaml_path)
    assert reloaded.logging.wandb.enabled is True
    assert reloaded.logging.wandb.run_name == "custom"
    assert reloaded.logging.file.level_console == "DEBUG"
