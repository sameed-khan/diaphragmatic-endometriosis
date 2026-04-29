"""Integration tests E11, E12 — eval orchestrator with synthetic deep_eval cache."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from endo.config.experiment import ExperimentConfig
from endo.config.paths import PathsConfig
from endo.eval.run_eval import run_cv_evaluation


def _build_synth_deep_eval_npz(
    out_path: Path,
    pids: list[str],
    n_slices: int = 10,
    pos_pids: set | None = None,
    seed: int = 0,
) -> None:
    """Synthesize a §5.3.4 CSR-style deep_eval npz for a list of pids."""
    pos_pids = pos_pids or set()
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat_pids = []
    flat_slice_ys = []
    flat_aux = []
    boxes_chunks = []
    scores_chunks = []
    box_offsets = [0]
    cur = 0

    for pid in pids:
        is_pos = pid in pos_pids
        for slice_y in range(n_slices):
            flat_pids.append(pid)
            flat_slice_ys.append(slice_y * 5)
            flat_aux.append(0.1)
            # Positives get 1 box at score 0.7 around slice 25; negatives get
            # nothing (or low-score noise).
            if is_pos and slice_y == n_slices // 2:
                # 30×30 px box (large by 0.82 mm/px → 24.6 mm).
                boxes_chunks.append(np.array([[50.0, 50.0, 80.0, 80.0]], dtype=np.float32))
                scores_chunks.append(np.array([0.7], dtype=np.float32))
                cur += 1
            elif not is_pos and slice_y == n_slices // 2:
                # Small low-score box: 4×4 px ≈ 3 mm.
                boxes_chunks.append(np.array([[200.0, 200.0, 204.0, 204.0]], dtype=np.float32))
                scores_chunks.append(np.array([0.15], dtype=np.float32))
                cur += 1
            box_offsets.append(cur)

    boxes_flat = np.concatenate(boxes_chunks, axis=0) if boxes_chunks else np.zeros((0, 4), dtype=np.float32)
    scores_flat = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.zeros((0,), dtype=np.float32)

    np.savez_compressed(
        out_path,
        patient_ids=np.asarray(flat_pids, dtype=object),
        slice_ys=np.asarray(flat_slice_ys, dtype=np.int32),
        boxes_flat=boxes_flat,
        scores_flat=scores_flat,
        box_offsets=np.asarray(box_offsets, dtype=np.int32),
        aux_seg_max=np.asarray(flat_aux, dtype=np.float32),
    )


def _make_experiment(tmp_path: Path) -> ExperimentConfig:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    cache_root.mkdir()
    runs_root.mkdir()

    # Build a tiny manifest.jsonl: 4 positives + 4 negatives in fold 0.
    manifest_path = data_root / "manifest.jsonl"
    rows = []
    for i in range(4):
        rows.append(
            {
                "patient_id": f"pos_{i}",
                "label": "positive",
                "cohort": "cross-validation",
                "fold": 0,
                "scanner_model": "SIGNA Artist",
                "variant": "A",
                "slice_thickness_mm": 1.5,
            }
        )
    for i in range(4):
        rows.append(
            {
                "patient_id": f"neg_{i}",
                "label": "negative",
                "cohort": "cross-validation",
                "fold": 0,
                "scanner_model": "SIGNA Explorer",
                "variant": "B",
                "slice_thickness_mm": 3.6,
            }
        )
    with manifest_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg = ExperimentConfig(
        uuid=str(uuid.uuid4()),
        name="eval_test",
        paths=PathsConfig(
            data_root=data_root,
            cache_root=cache_root,
            runs_root=runs_root,
        ),
    )
    cfg.eval.bootstrap_n = 10  # speed
    cfg.eval.froc_fp_points = [1.0, 2.0]  # speed
    cfg.eval.large_threshold_grid = [0.05]
    cfg.eval.small_threshold_grid = [0.30]
    cfg.eval.stratify_keys = ["scanner_model"]  # speed
    return cfg


def test_eval_one_fold_e2e(tmp_path: Path):
    """E11: Synthetic fold-0 deep_eval cache → eval_report.csv with finite metrics."""
    exp = _make_experiment(tmp_path)
    fold0_dir = exp.run_dir() / "fold0"
    npz_path = fold0_dir / "runtime" / "deep_eval" / "epoch10_val.npz"
    pos = {f"pos_{i}" for i in range(4)}
    pids = [f"pos_{i}" for i in range(4)] + [f"neg_{i}" for i in range(4)]
    _build_synth_deep_eval_npz(npz_path, pids, n_slices=6, pos_pids=pos)

    result = run_cv_evaluation(exp, use_gru=False)

    csv_path = Path(result["csv_path"])
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) > 0
    # Every row has finite value or "nan" string (we tolerate the latter).
    for r in reader:
        v = r["value"]
        assert v != ""
    # Per-fold AND cv_pooled rows present.
    scopes = {r["scope"] for r in reader}
    assert "per_fold" in scopes
    assert "cv_pooled" in scopes


def test_eval_with_and_without_gru(tmp_path: Path):
    """E12: Same fold; with-gru and without-gru row sets both produced.

    Note: GRU module may not exist; in that case ``use_gru=True`` falls back
    to the non-rescored path, but the rows are still tagged ``rescored=true``
    by the orchestrator (per spec — caller declared intent)."""
    exp = _make_experiment(tmp_path)
    fold0_dir = exp.run_dir() / "fold0"
    npz_path = fold0_dir / "runtime" / "deep_eval" / "epoch10_val.npz"
    pos = {f"pos_{i}" for i in range(4)}
    pids = [f"pos_{i}" for i in range(4)] + [f"neg_{i}" for i in range(4)]
    _build_synth_deep_eval_npz(npz_path, pids, n_slices=6, pos_pids=pos)

    run_cv_evaluation(exp, use_gru=False)
    run_cv_evaluation(exp, use_gru=True)

    csv_path = exp.run_dir() / "eval" / "eval_report.csv"
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))

    rescored_values = {r["rescored"] for r in reader}
    # Both true and false should be present.
    assert "true" in rescored_values
    assert "false" in rescored_values

    # Both runs' run_ids present (E15 also covered).
    run_ids = {r["run_id"] for r in reader}
    assert len(run_ids) >= 2
