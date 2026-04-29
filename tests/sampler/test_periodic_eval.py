"""Unit tests for PeriodicDeepEvalCallback (S12, S13, S14)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest
import torch

from endo.config.sampler import SamplerConfig
from endo.sampler.periodic_eval import PeriodicDeepEvalCallback
from endo.sampler.weighted import WeightedScheduledSampler


# ─── synthetic stand-ins for model + datamodule ──────────────────────


@dataclass
class _FakeBatch:
    volume_5ch: torch.Tensor
    patient_ids: list[str]
    slice_ys: torch.Tensor


class _FakeDetector(torch.nn.Module):
    """Returns deterministic per-slice scores keyed by patient_id."""

    image_size = (8, 8)

    def __init__(self) -> None:
        super().__init__()
        # Need at least one parameter so .parameters() returns a device.
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    @property
    def head(self):
        # inference_pass calls detector.head.predict — alias to self.
        return self

    def forward(self, vol: torch.Tensor):
        b = vol.shape[0]
        # Two FPN levels, one class.
        cls = [
            torch.zeros(b, 1, 4, 4),
            torch.zeros(b, 1, 2, 2),
        ]
        bbox = [
            torch.zeros(b, 4, 4, 4),
            torch.zeros(b, 4, 2, 2),
        ]
        aux = torch.zeros(b, 1, 8, 8)
        return cls, bbox, aux

    def predict(self, cls_scores, bbox_preds, image_size, **kw):
        b = cls_scores[0].shape[0]
        # Synthetic boxes: 1 box per image with score scaled by index.
        out = []
        for i in range(b):
            score = float(0.1 + 0.05 * (i % 5))
            out.append(
                {
                    "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                    "scores": torch.tensor([score]),
                    "labels": torch.tensor([0], dtype=torch.long),
                }
            )
        return out


class _FakePLModule(torch.nn.Module):
    """Just enough of pl.LightningModule's surface for the callback."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeDetector()
        self._logged: dict[str, float] = {}

    def log_dict(self, d: dict, **_kw) -> None:
        self._logged.update({k: float(v) for k, v in d.items()})


class _FakeDataModule:
    def __init__(self, slice_grid: list[tuple[str, int]]) -> None:
        # slice_grid: ordered list of (pid, slice_y) used for inference.
        self._slice_grid = slice_grid

    def inference_dataloader(self, patient_ids: list[str]):
        wanted = set(patient_ids)
        items = [(pid, sy) for (pid, sy) in self._slice_grid if pid in wanted]

        def gen() -> Iterator[_FakeBatch]:
            bs = 4
            for k in range(0, len(items), bs):
                chunk = items[k : k + bs]
                pids = [pid for pid, _ in chunk]
                sys_ = torch.tensor([sy for _, sy in chunk], dtype=torch.int64)
                vol = torch.zeros(len(chunk), 5, 8, 8, dtype=torch.float32)
                yield _FakeBatch(volume_5ch=vol, patient_ids=pids, slice_ys=sys_)

        return gen()


def _make_trainer(
    *,
    epoch: int,
    sampler: WeightedScheduledSampler | None,
    datamodule: _FakeDataModule,
) -> SimpleNamespace:
    dl = SimpleNamespace(sampler=sampler, dataset=SimpleNamespace(slice_index=sampler._slice_index)) if sampler is not None else None
    return SimpleNamespace(
        current_epoch=epoch,
        train_dataloader=dl,
        datamodule=datamodule,
    )


# ─── tests ───────────────────────────────────────────────────────────


def _build_slice_grid() -> tuple[list[tuple[str, int, str]], list[tuple[str, int]], list[str], list[str]]:
    """5 train negatives x 6 slices each + 3 val pids x 4 slices each."""
    slice_index: list[tuple[str, int, str]] = []
    train_neg_pids = [f"tn{i:02d}" for i in range(5)]
    val_pids = [f"v{i:02d}" for i in range(3)]
    # Add at least one positive so the sampler has a non-empty pos pool.
    slice_index.append(("p_pos", 0, "pos_slice"))
    sy = 0
    for pid in train_neg_pids:
        for s in range(6):
            slice_index.append((pid, sy, "neg_slice_neg_vol"))
            sy += 1
    # Inference grid (order matters for the dataloader).
    grid: list[tuple[str, int]] = [
        (pid, s) for pid in train_neg_pids for s in range(6)
    ] + [(pid, s) for pid in val_pids for s in range(4)]
    return slice_index, grid, train_neg_pids, val_pids


def test_S12_callback_skips_pre_start_epoch(tmp_path: Path) -> None:
    """S12: at epoch 5 with start_epoch 10, callback is a no-op."""
    cfg = SamplerConfig(deep_eval_start_epoch=10, deep_eval_refresh_every_epochs=10)
    slice_index, grid, train_neg_pids, val_pids = _build_slice_grid()
    sampler = WeightedScheduledSampler(slice_index, SamplerConfig(samples_per_epoch=10), seed=0)
    cb = PeriodicDeepEvalCallback(
        sampler_cfg=cfg,
        run_dir=tmp_path,
        train_neg_pids=train_neg_pids,
        val_pids=val_pids,
    )
    pl_module = _FakePLModule()
    dm = _FakeDataModule(grid)
    trainer = _make_trainer(epoch=5, sampler=sampler, datamodule=dm)

    cb.on_validation_epoch_end(trainer, pl_module)
    assert not (tmp_path / "runtime" / "hard_negatives.json").exists()
    assert not (tmp_path / "runtime" / "deep_eval").exists() or not any(
        (tmp_path / "runtime" / "deep_eval").iterdir()
    )


def test_S13_callback_writes_hard_negatives_json(tmp_path: Path) -> None:
    """S13: at start epoch the JSON is written with the right schema."""
    cfg = SamplerConfig(
        deep_eval_start_epoch=10,
        deep_eval_refresh_every_epochs=10,
        hard_pool_top_k=8,
        samples_per_epoch=10,
    )
    slice_index, grid, train_neg_pids, val_pids = _build_slice_grid()
    sampler = WeightedScheduledSampler(slice_index, cfg, seed=0)
    cb = PeriodicDeepEvalCallback(
        sampler_cfg=cfg,
        run_dir=tmp_path,
        train_neg_pids=train_neg_pids,
        val_pids=val_pids,
        score_threshold=0.0,
    )
    pl_module = _FakePLModule()
    dm = _FakeDataModule(grid)
    trainer = _make_trainer(epoch=10, sampler=sampler, datamodule=dm)

    cb.on_validation_epoch_end(trainer, pl_module)

    p = tmp_path / "runtime" / "hard_negatives.json"
    assert p.exists(), "hard_negatives.json should be written at start epoch"
    payload = json.loads(p.read_text())
    assert set(payload.keys()) == {
        "epoch_written",
        "model_checkpoint_epoch",
        "slice_indices",
        "n_slices",
        "score_threshold",
    }
    assert payload["epoch_written"] == 10
    assert payload["model_checkpoint_epoch"] == 10
    assert payload["n_slices"] == len(payload["slice_indices"])
    assert payload["n_slices"] <= cfg.hard_pool_top_k
    assert all(isinstance(i, int) for i in payload["slice_indices"])
    # Sampler should now have a non-empty hard pool reflecting the file.
    assert sampler.hard_pool_size == payload["n_slices"]


def test_S14_callback_writes_deep_eval_npz(tmp_path: Path) -> None:
    """S14: at start epoch the npz is written with the §5.3.4 arrays."""
    cfg = SamplerConfig(
        deep_eval_start_epoch=10,
        deep_eval_refresh_every_epochs=10,
        samples_per_epoch=10,
    )
    slice_index, grid, train_neg_pids, val_pids = _build_slice_grid()
    sampler = WeightedScheduledSampler(slice_index, cfg, seed=0)
    cb = PeriodicDeepEvalCallback(
        sampler_cfg=cfg,
        run_dir=tmp_path,
        train_neg_pids=train_neg_pids,
        val_pids=val_pids,
    )
    pl_module = _FakePLModule()
    dm = _FakeDataModule(grid)
    trainer = _make_trainer(epoch=10, sampler=sampler, datamodule=dm)

    cb.on_validation_epoch_end(trainer, pl_module)

    npz_path = tmp_path / "runtime" / "deep_eval" / "epoch10_val.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as z:
        assert set(z.files) == {
            "patient_ids",
            "slice_ys",
            "boxes_flat",
            "scores_flat",
            "box_offsets",
            "aux_seg_max",
        }
        n_slices = z["slice_ys"].shape[0]
        # 3 val patients × 4 slices = 12.
        assert n_slices == 12
        assert z["box_offsets"].shape == (n_slices + 1,)
        assert z["aux_seg_max"].shape == (n_slices,)
        assert z["boxes_flat"].shape[1] == 4
        # Offsets monotonically nondecreasing, ending at boxes_flat length.
        offs = z["box_offsets"]
        assert int(offs[0]) == 0
        assert int(offs[-1]) == int(z["boxes_flat"].shape[0])
        assert (np.diff(offs) >= 0).all()


def test_callback_refresh_cadence(tmp_path: Path) -> None:
    """Sanity: fires at epoch 10, 20, 30 but not 11/15/25."""
    cfg = SamplerConfig(deep_eval_start_epoch=10, deep_eval_refresh_every_epochs=10)
    slice_index, grid, train_neg_pids, val_pids = _build_slice_grid()
    cb = PeriodicDeepEvalCallback(
        sampler_cfg=cfg,
        run_dir=tmp_path,
        train_neg_pids=train_neg_pids,
        val_pids=val_pids,
    )
    assert cb._should_run(9) is False
    assert cb._should_run(10) is True
    assert cb._should_run(11) is False
    assert cb._should_run(15) is False
    assert cb._should_run(20) is True
    assert cb._should_run(25) is False
