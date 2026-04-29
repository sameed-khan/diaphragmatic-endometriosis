"""Tests for endo.utils.wandb_init.

Cover the pure-Python helpers (group/run-name/tag composition + enable check).
The actual W&B SDK is not exercised here.
"""

from __future__ import annotations

from endo.config import (
    ExperimentConfig,
    LoggingConfig,
    WandbConfig,
)
from endo.utils.wandb_init import (
    is_wandb_enabled,
    resolve_group,
    resolve_run_name,
    resolve_tags,
)


def _experiment(**logging_kw) -> ExperimentConfig:
    return ExperimentConfig(
        uuid="b3a7f1e9-4c8a-4d2b-9f1c-0e6a8b9c1d2e",
        name="myexp",
        tags={"phase": "1", "head": "rtmdet"},
        logging=LoggingConfig(**logging_kw),
    )


def test_is_wandb_enabled_respects_mode():
    exp = _experiment()
    assert is_wandb_enabled(exp.logging) is False

    exp = _experiment(wandb=WandbConfig(enabled=True, mode="online"))
    assert is_wandb_enabled(exp.logging) is True

    exp = _experiment(wandb=WandbConfig(enabled=True, mode="disabled"))
    assert is_wandb_enabled(exp.logging) is False


def test_group_default_includes_short_uuid():
    exp = _experiment()
    g = resolve_group(exp, exp.logging)
    assert g.startswith("myexp_")
    # short_uuid is 8 hex chars.
    assert len(g.split("_", 1)[1]) == 8


def test_run_name_default_per_stage():
    exp = _experiment()
    assert resolve_run_name(exp, exp.logging, stage="detector", fold=0) == "myexp/fold0"
    assert resolve_run_name(exp, exp.logging, stage="eval", fold=None) == "myexp/cv_summary"
    assert resolve_run_name(exp, exp.logging, stage="holdout", fold=None) == "myexp/holdout"
    assert resolve_run_name(exp, exp.logging, stage="gru", fold=2) == "myexp/fold2-gru"


def test_run_name_override_with_holdout_suffix():
    exp = _experiment(
        wandb=WandbConfig(enabled=True, experiment_name="e2e", run_name="run1"),
    )
    assert resolve_run_name(exp, exp.logging, stage="detector", fold=0) == "run1"
    assert resolve_run_name(exp, exp.logging, stage="holdout", fold=None) == "run1-holdout"


def test_tags_include_stage_fold_and_experiment_tags():
    exp = _experiment(wandb=WandbConfig(tags=["extra"]))
    tags = resolve_tags(exp, exp.logging, stage="detector", fold=3)
    assert "stage=detector" in tags
    assert "fold=3" in tags
    assert "extra" in tags
    # experiment tags (values from the dict) get included.
    assert "1" in tags or "rtmdet" in tags
