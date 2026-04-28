"""Generic IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any, *, atomic: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2, default=_json_default, sort_keys=True)
        tmp.replace(path)
    else:
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=_json_default, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def _json_default(obj):  # noqa: ANN001
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
