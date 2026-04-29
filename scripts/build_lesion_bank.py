"""Build the global, read-only Lesion Bank.

Implements Component 2 of the diaphragmatic-endometriosis pipeline.

Reads ``cache/v1/preprocessed_manifest.jsonl`` (filtered to CV positives),
extracts one :class:`LesionBankEntry` per CC at the connectivity locked by
``cache/v1/runtime/connectivity_lock.json``, and serializes the result to
``cache/v1/lesion_banks/lesion_bank_<git_sha8>.pkl`` along with a provenance
JSON and a ``current.pkl`` symlink.

CLI:
    uv run python scripts/build_lesion_bank.py \
        --cache-root cache/v1/ \
        --workers 8 \
        [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

# Make ``endo`` importable when invoked as ``python scripts/build_lesion_bank.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from endo.lesion_bank import (  # noqa: E402
    LesionBankEntry,
    extract_entries_for_donor,
    save_bank,
)

logger = logging.getLogger("build_lesion_bank")


DEFAULT_CONNECTIVITY = 26


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    """Return the full git SHA at HEAD; fall back to the env if git is missing."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("GIT_SHA", "unknown_" + "0" * 32)


def _read_connectivity_lock(cache_root: Path) -> int:
    """Return the locked connectivity (6 or 26).

    If the lock file is missing we WARN and fall back to 26, per the
    Component 2 build contract.
    """
    lock_path = cache_root / "runtime" / "connectivity_lock.json"
    if not lock_path.exists():
        logger.warning(
            "connectivity_lock.json not found at %s — defaulting to %d-connectivity.",
            lock_path,
            DEFAULT_CONNECTIVITY,
        )
        return DEFAULT_CONNECTIVITY
    with lock_path.open() as f:
        data = json.load(f)
    raw = data.get("connectivity")
    conn = int(raw) if raw is not None else DEFAULT_CONNECTIVITY
    if conn not in (6, 26):
        raise ValueError(f"connectivity_lock.json has invalid connectivity={raw!r}")
    return conn


def _read_donor_manifest(cache_root: Path) -> list[str]:
    """Return the sorted list of donor patient_ids (CV positives only)."""
    manifest_path = cache_root / "preprocessed_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"preprocessed_manifest.jsonl not found at {manifest_path}"
        )
    df = pl.read_ndjson(manifest_path)
    cv_pos = df.filter(
        (pl.col("cohort") == "cross-validation") & (pl.col("label") == "positive")
    )
    pids = sorted(cv_pos.get_column("patient_id").to_list())
    return pids


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _worker_extract(
    patient_id: str, *, cache_root: str, connectivity: int
) -> list[LesionBankEntry]:
    return extract_entries_for_donor(
        patient_id, Path(cache_root), connectivity=connectivity
    )


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_lesion_bank(
    cache_root: Path,
    *,
    workers: int = 8,
    force: bool = False,
    repo_root: Path | None = None,
) -> Path:
    """Top-level build. Returns the path to the (current) bank pkl."""
    cache_root = Path(cache_root).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]

    git_sha = _git_sha(repo_root)
    git_sha8 = git_sha[:8]
    out_dir = cache_root / "lesion_banks"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / f"lesion_bank_{git_sha8}.pkl"
    current_link = out_dir / "current.pkl"
    provenance_path = out_dir / "bank_provenance.json"

    # Idempotency: a bank file matching the current SHA already exists.
    if bank_path.exists() and not force:
        logger.info(
            "Lesion bank for git_sha8=%s already exists at %s; skipping (use --force to rebuild).",
            git_sha8,
            bank_path,
        )
        return bank_path

    connectivity = _read_connectivity_lock(cache_root)
    donor_ids = _read_donor_manifest(cache_root)
    if not donor_ids:
        raise RuntimeError("No CV-positive donors found in preprocessed_manifest.jsonl")

    logger.info(
        "Building lesion bank: %d donor patients, connectivity=%d, workers=%d",
        len(donor_ids),
        connectivity,
        workers,
    )

    started = dt.datetime.now(dt.timezone.utc)
    started_mono = dt.datetime.now()

    worker = partial(
        _worker_extract,
        cache_root=str(cache_root),
        connectivity=connectivity,
    )

    if workers <= 1:
        per_donor = [worker(pid) for pid in donor_ids]
    else:
        with mp.Pool(processes=workers) as pool:
            per_donor = pool.map(worker, donor_ids)

    entries: list[LesionBankEntry] = []
    for pid, per_pid in zip(donor_ids, per_donor):
        if not per_pid:
            logger.error("Donor %s produced 0 CCs — investigate cache integrity.", pid)
        entries.extend(per_pid)
        logger.info("donor=%s n_cc=%d", pid, len(per_pid))

    save_bank(entries, bank_path)
    bank_sha256 = _file_sha256(bank_path)

    # Atomically (re)point current.pkl → bank_path.
    tmp_link = current_link.with_suffix(".pkl.tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(bank_path.name)
    os.replace(tmp_link, current_link)

    elapsed = (dt.datetime.now() - started_mono).total_seconds()

    provenance = {
        "build_timestamp": started.isoformat().replace("+00:00", "Z"),
        "built_at": started.isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "git_sha8": git_sha8,
        "code_version": git_sha,
        "bank_filename": bank_path.name,
        "cache_version": "v1",
        "connectivity": connectivity,
        "n_donors": len(donor_ids),
        "n_donor_patients": len(donor_ids),
        "n_entries": len(entries),
        "n_ccs": len(entries),
        "cohort_filter": "cohort=cross-validation AND label=positive",
        "donor_patient_ids": donor_ids,
        "bank_sha256": bank_sha256,
        "build_seconds": round(elapsed, 3),
    }

    tmp_prov = provenance_path.with_suffix(".json.tmp")
    with tmp_prov.open("w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    os.replace(tmp_prov, provenance_path)

    logger.info(
        "Wrote bank: n_entries=%d donors=%d connectivity=%d sha256=%s elapsed=%.2fs",
        len(entries),
        len(donor_ids),
        connectivity,
        bank_sha256[:12],
        elapsed,
    )
    return bank_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the global lesion bank.")
    p.add_argument("--cache-root", type=Path, default=Path("cache/v1/"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    build_lesion_bank(
        cache_root=args.cache_root,
        workers=args.workers,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
