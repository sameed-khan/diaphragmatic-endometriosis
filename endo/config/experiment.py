"""Top-level ExperimentConfig (composition of sub-configs)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from .augmentation import AugmentationConfig
from .eval import EvalConfig
from .gru import GRUConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .paths import PathsConfig
from .sampler import SamplerConfig
from .training import TrainingConfig

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class ExperimentConfig(BaseModel):
    """Top-level experiment declaration. One per ``experiments/<name>.py``."""

    uuid: str
    name: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)

    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    gru: GRUConfig = Field(default_factory=GRUConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    seed: int = 42

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _check_uuid_format(self) -> Self:
        if not _UUID4_RE.match(self.uuid):
            raise ValueError(f"uuid must be a uuid4 string, got: {self.uuid!r}")
        return self

    # ─── Identity helpers ──────────────────────────────────────────────

    @property
    def short_uuid(self) -> str:
        return self.uuid.replace("-", "")[:8]

    @property
    def run_dir_name(self) -> str:
        return f"{self.name}_{self.short_uuid}"

    def run_dir(self) -> Path:
        return self.paths.runs_root / self.run_dir_name

    # ─── Serialization ─────────────────────────────────────────────────

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self.model_dump_json())  # ensures Paths→str etc.
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with Path(path).open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def diff(self, other: "ExperimentConfig") -> list[str]:
        """Return a list of dotted paths where ``self`` and ``other`` differ.

        Used by ``run_experiment.py`` to detect drift between the live
        experiment file and the materialized ``runs/<exp>/experiment.yaml``.

        The ``logging.*`` subtree is excluded from drift comparison — toggling
        wandb mode, log levels, or upload gates between resumes does NOT
        trip the drift guard.
        """
        a = json.loads(self.model_dump_json())
        b = json.loads(other.model_dump_json())
        # Drift-exempt: logging configuration may be tweaked between resumes.
        a.pop("logging", None)
        b.pop("logging", None)
        diffs: list[str] = []
        _walk(a, b, prefix="", out=diffs)
        return diffs


def _walk(a, b, prefix: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a:
                out.append(f"{p}: <missing in self> != {b[k]!r}")
            elif k not in b:
                out.append(f"{p}: {a[k]!r} != <missing in other>")
            else:
                _walk(a[k], b[k], p, out)
    else:
        if a != b:
            out.append(f"{prefix}: {a!r} != {b!r}")
