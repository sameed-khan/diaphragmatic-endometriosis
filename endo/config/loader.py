"""Dynamic loader for experiment .py files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .experiment import ExperimentConfig


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Dynamically import an experiment .py file and return its ExperimentConfig.

    Convention: the file must define a module-level ``experiment: ExperimentConfig``.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    spec = importlib.util.spec_from_file_location("_experiment_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_experiment_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "experiment"):
        raise AttributeError(f"{path}: must define `experiment: ExperimentConfig`")
    obj = module.experiment
    if not isinstance(obj, ExperimentConfig):
        raise TypeError(
            f"{path}: `experiment` must be ExperimentConfig, got {type(obj).__name__}"
        )
    return obj
