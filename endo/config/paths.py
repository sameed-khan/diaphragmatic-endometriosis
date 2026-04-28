"""Path configuration for an experiment."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_root: Path = Field(default=Path("data/"))
    cache_root: Path = Field(default=Path("cache/v1/"))
    runs_root: Path = Field(default=Path("runs/"))

    lesion_bank: Path | None = None

    model_config = {"arbitrary_types_allowed": True}
