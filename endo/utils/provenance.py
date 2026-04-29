"""Run-provenance helpers (git sha, hostname, fold-status JSON).

Per PRD §5.3.1: every ``runs/<exp>_<uuid8>/`` carries an ``experiment.yaml``
materialized from the ``ExperimentConfig``, an ``experiment.py`` byte-copy of
the source file, and a ``provenance.json`` with run metadata.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


FOLD_STATES = ("pending", "running", "complete", "failed")


def get_git_sha(short: bool = False) -> str:
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short=8")
        cmd.append("HEAD")
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def initial_provenance() -> dict[str, Any]:
    return {
        "git_sha": get_git_sha(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "started_at": now_iso(),
        "fold_status": {str(f): "pending" for f in range(5)},
    }


def load_provenance(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_provenance(path: Path, data: dict[str, Any]) -> None:
    """Atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def update_fold_status(provenance_path: Path, fold: int, state: str) -> None:
    """Atomically set ``fold_status[fold] = state`` in provenance.json."""
    if state not in FOLD_STATES:
        raise ValueError(f"invalid fold state {state!r}; expected one of {FOLD_STATES}")
    data = load_provenance(provenance_path)
    data.setdefault("fold_status", {})[str(int(fold))] = state
    if state in ("complete", "failed"):
        data.setdefault("fold_finished_at", {})[str(int(fold))] = now_iso()
    save_provenance(provenance_path, data)
