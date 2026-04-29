"""Integration tests E11, E12 — eval orchestrator with fresh-inference path
(audit 2026-04-29). The inference helper is monkeypatched so these tests
don't require ckpts or a GPU; the focus is on the metric/threshold/JSONL
plumbing.
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from endo.config.experiment import ExperimentConfig
from endo.config.paths import PathsConfig
from endo.eval import run_eval as run_eval_mod
from endo.eval.run_eval import run_cv_evaluation
from endo.inference_pass import SliceScore


# Keep the eval canvas tiny in tests: scipy CC and picai_eval CC scale with the
# number of voxels and dominate runtime otherwise.
_TEST_VOLUME_SHAPE = (40, 96, 96)  # (Y, Z, X)


class _FakeDM:
    """Minimal DataModule stand-in for ``_gt_masks_from_dm``."""

    def __init__(self, val_pids: list[str], pos_pids: set[str]):
        self._val_pids = list(val_pids)
        # (X, Y, Z) shape mirrors LesionDataModule contract; downsized for speed.
        self.target_input_shape = (96, 40, 96)
        self.cache_shape = (104, 48, 104)
        self._cache: dict[str, dict] = {}
        cx, cy, cz = self.cache_shape
        for pid in val_pids:
            if pid in pos_pids:
                m = np.zeros((cx, cy, cz), dtype=np.uint8)
                cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2
                m[cx2 - 3 : cx2 + 3, cy2 - 2 : cy2 + 2, cz2 - 3 : cz2 + 3] = 1
                self._cache[pid] = {"lesion_mask": m}
            else:
                self._cache[pid] = {"lesion_mask": None}


def _synth_slice_scores_for_pids(
    pids: list[str], pos_pids: set[str], n_slices: int = 4
) -> dict[str, list[SliceScore]]:
    out: dict[str, list[SliceScore]] = {}
    for pid in pids:
        is_pos = pid in pos_pids
        slices: list[SliceScore] = []
        for i in range(n_slices):
            slice_y = 4 + i * 4  # within (Y=40) canvas
            if is_pos and i == n_slices // 2:
                # ~16 px box → ~13 mm at 0.82 mm/px (above split=10mm).
                boxes = np.array([[40.0, 40.0, 56.0, 56.0]], dtype=np.float32)
                scores = np.array([0.7], dtype=np.float32)
                aux = 0.5
            elif not is_pos and i == n_slices // 2:
                boxes = np.array([[80.0, 80.0, 84.0, 84.0]], dtype=np.float32)
                scores = np.array([0.15], dtype=np.float32)
                aux = 0.1
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                aux = 0.05
            slices.append(
                SliceScore(
                    patient_id=pid,
                    slice_y=int(slice_y),
                    boxes=boxes,
                    scores=scores,
                    aux_seg_max=float(aux),
                )
            )
        out[pid] = slices
    return out


def _make_experiment(tmp_path: Path, n_folds_with_data: int = 5) -> ExperimentConfig:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    cache_root.mkdir()
    runs_root.mkdir()

    manifest_path = data_root / "manifest.jsonl"
    rows: list[dict] = []
    for fold in range(5):
        # 2 pos + 2 neg per fold (kept small so picai_eval CC stays cheap).
        for i in range(2):
            rows.append(
                {
                    "patient_id": f"f{fold}_pos_{i}",
                    "label": "positive",
                    "cohort": "cross-validation",
                    "fold": fold,
                    "scanner_model": "SIGNA Artist",
                    "variant": "A",
                    "slice_thickness_mm": 1.5,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "patient_id": f"f{fold}_neg_{i}",
                    "label": "negative",
                    "cohort": "cross-validation",
                    "fold": fold,
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
    cfg.eval.bootstrap_n = 5
    cfg.eval.froc_fp_points = [1.0, 2.0]
    cfg.eval.large_threshold_grid = [0.05]
    cfg.eval.small_threshold_grid = [0.30]
    cfg.eval.stratify_keys = ["scanner_model"]

    # Drop a fake best.ckpt for each fold we want to "succeed."
    run_dir = cfg.run_dir()
    for fold in range(n_folds_with_data):
        ckpt_dir = run_dir / f"fold{fold}" / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "best.ckpt").write_text("fake")
    return cfg


def _patch_fresh_inference(monkeypatch, exp: ExperimentConfig, pos_pids_by_fold):
    """Replace ``_run_fresh_inference_for_fold`` with a synthetic generator."""

    def _fake(experiment, fold, ckpt_choice, cfg):
        pids = sorted(
            [f"f{fold}_pos_{i}" for i in range(2)] + [f"f{fold}_neg_{i}" for i in range(2)]
        )
        pos_pids = pos_pids_by_fold.get(fold, set())
        slice_scores = _synth_slice_scores_for_pids(pids, pos_pids)
        dm = _FakeDM(pids, pos_pids)
        return slice_scores, pids, dm

    monkeypatch.setattr(run_eval_mod, "_run_fresh_inference_for_fold", _fake)
    # Shrink the eval canvas — picai_eval CC scales with voxel count and
    # dominates runtime if we keep the production (160, 384, 384) shape.
    monkeypatch.setattr(run_eval_mod, "_DEFAULT_VOLUME_SHAPE", _TEST_VOLUME_SHAPE)


def test_eval_one_fold_e2e(tmp_path: Path, monkeypatch):
    """E11: Fresh inference path produces a non-empty eval_report.csv with
    finite metrics across all 5 folds."""
    exp = _make_experiment(tmp_path)
    pos = {fold: {f"f{fold}_pos_{i}" for i in range(2)} for fold in range(5)}
    _patch_fresh_inference(monkeypatch, exp, pos)

    result = run_cv_evaluation(exp, use_gru=False)

    csv_path = Path(result["csv_path"])
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) > 0
    scopes = {r["scope"] for r in reader}
    assert "per_fold" in scopes
    assert "cv_pooled" in scopes

    thr = json.loads(Path(result["thresholds_path"]).read_text())
    assert thr["tuning_policy"] == "cross_fold_leave_one_out"
    assert set(thr["per_fold_thresholds"].keys()) == {"0", "1", "2", "3", "4"}

    # Per-call JSONL exists and is non-empty.
    calls_path = Path(result["calls_path"])
    assert calls_path.exists()
    lines = [json.loads(l) for l in calls_path.read_text().splitlines() if l.strip()]
    assert len(lines) > 0
    types = {l["call_type"] for l in lines}
    assert types <= {"tp", "fp", "fn"}


def test_eval_with_and_without_gru(tmp_path: Path, monkeypatch):
    """E12: Both rescored=true and rescored=false rowsets present."""
    exp = _make_experiment(tmp_path)
    pos = {fold: {f"f{fold}_pos_{i}" for i in range(2)} for fold in range(5)}
    _patch_fresh_inference(monkeypatch, exp, pos)

    run_cv_evaluation(exp, use_gru=False)
    run_cv_evaluation(exp, use_gru=True)

    csv_path = exp.run_dir() / "eval" / "eval_report.csv"
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    rescored_values = {r["rescored"] for r in reader}
    assert "true" in rescored_values
    assert "false" in rescored_values

    run_ids = {r["run_id"] for r in reader}
    assert len(run_ids) >= 2
