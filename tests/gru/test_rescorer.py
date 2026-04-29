"""Unit tests for the GRU rescorer (G1, G3, G4, G6, G7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from endo.config.gru import GRUConfig
from endo.gru.rescorer import (
    GRURescorer,
    rescore_detector_outputs,
    volume_score,
)


# ----------------------------------------------------------------------
# G1 — forward shape.
# ----------------------------------------------------------------------


def test_gru_forward_shape() -> None:
    cfg = GRUConfig(input_dim=768, hidden_dim=64, num_layers=1, bidirectional=True)
    model = GRURescorer(cfg).eval()
    feats = torch.randn(4, 50, 768)
    mask = torch.ones(4, 50, dtype=torch.bool)
    with torch.no_grad():
        out = model(feats, mask)
    assert out.shape == (4, 50)
    assert torch.all((out >= 0) & (out <= 1))


# ----------------------------------------------------------------------
# G3 — volume_score: max + topk match manual computation.
# ----------------------------------------------------------------------


def test_volume_score_max_and_topk() -> None:
    # B=2, T=6 — known values, all valid.
    probs = torch.tensor(
        [
            [0.1, 0.4, 0.9, 0.2, 0.7, 0.3],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    mask = torch.ones_like(probs, dtype=torch.bool)

    vol_max = volume_score(probs, mask, agg="max")
    assert torch.allclose(vol_max, torch.tensor([0.9, 0.2]))

    # top-3 mean.
    vol_topk = volume_score(probs, mask, agg="topk", k=3)
    expected = torch.tensor([(0.9 + 0.7 + 0.4) / 3.0, 0.2])
    assert torch.allclose(vol_topk, expected, atol=1e-6)


# ----------------------------------------------------------------------
# G4 — volume_score respects mask (padding doesn't influence reduction).
# ----------------------------------------------------------------------


def test_volume_score_respects_mask() -> None:
    # Row 0 valid for first 3 entries; high padded values are ignored.
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.99, 0.99, 0.99]])
    mask = torch.tensor([[True, True, True, False, False, False]])

    vol_max = volume_score(probs, mask, agg="max")
    assert torch.allclose(vol_max, torch.tensor([0.3]))

    vol_topk = volume_score(probs, mask, agg="topk", k=3)
    assert torch.allclose(vol_topk, torch.tensor([(0.1 + 0.2 + 0.3) / 3.0]), atol=1e-6)

    # Top-k larger than valid count: only the valid entries contribute.
    vol_topk_5 = volume_score(probs, mask, agg="topk", k=5)
    assert torch.allclose(vol_topk_5, torch.tensor([(0.1 + 0.2 + 0.3) / 3.0]), atol=1e-6)


# ----------------------------------------------------------------------
# G6 — rescore_detector_outputs multiplies by per-slice probability.
# ----------------------------------------------------------------------


def _write_synthetic_npz(path: Path, n_slices: int, slice_ys: list[int]) -> None:
    feats = np.zeros((n_slices, 768), dtype=np.float16)
    np.savez_compressed(
        path,
        feats=feats,
        slice_ys=np.asarray(slice_ys, dtype=np.int32),
        patient_label=np.int8(0),
    )


def _save_constant_p_ckpt(path: Path, p: float, cfg: GRUConfig) -> None:
    """Build a GRURescorer whose per-slice output is approximately p everywhere.

    We zero the GRU and rely on the head bias = logit(p). With dropout disabled
    at eval, all timesteps map to sigmoid(bias) = p (exactly, in fp32).
    """
    model = GRURescorer(cfg)
    with torch.no_grad():
        for n, q in model.gru.named_parameters():
            q.zero_()
        model.head.weight.zero_()
        # logit(p) sets sigmoid(bias) = p
        logit_p = float(np.log(p / (1.0 - p)))
        model.head.bias.fill_(logit_p)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.model_dump(),
            "epoch": 0,
            "val_auroc": 0.5,
        },
        str(path),
    )


def test_rescore_multiplies_scores(tmp_path: Path) -> None:
    cfg = GRUConfig(input_dim=768, hidden_dim=8, num_layers=1, bidirectional=True)
    ckpt = tmp_path / "gru.pt"
    _save_constant_p_ckpt(ckpt, p=0.5, cfg=cfg)

    npz = tmp_path / "patient.npz"
    _write_synthetic_npz(npz, n_slices=4, slice_ys=[10, 11, 12, 13])

    boxes_per_slice = {
        10: {"boxes": np.array([[1.0, 2.0, 3.0, 4.0]]), "scores": np.array([0.8, 0.6])},
        12: {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "scores": np.array([0.4])},
    }
    rescored = rescore_detector_outputs(ckpt, npz, boxes_per_slice)

    assert set(rescored.keys()) == {10, 12}
    np.testing.assert_allclose(rescored[10]["scores"], np.array([0.4, 0.3]), atol=1e-5)
    np.testing.assert_allclose(rescored[12]["scores"], np.array([0.2]), atol=1e-5)
    # Boxes pass through.
    np.testing.assert_array_equal(rescored[10]["boxes"], boxes_per_slice[10]["boxes"])


# ----------------------------------------------------------------------
# G7 — slice missing from feature cache → score unchanged.
# ----------------------------------------------------------------------


def test_rescore_handles_missing_slice(tmp_path: Path) -> None:
    cfg = GRUConfig(input_dim=768, hidden_dim=8, num_layers=1, bidirectional=True)
    ckpt = tmp_path / "gru.pt"
    _save_constant_p_ckpt(ckpt, p=0.5, cfg=cfg)

    npz = tmp_path / "patient.npz"
    _write_synthetic_npz(npz, n_slices=2, slice_ys=[10, 11])

    boxes_per_slice = {
        10: {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "scores": np.array([0.8])},
        99: {"boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "scores": np.array([0.7])},
    }
    rescored = rescore_detector_outputs(ckpt, npz, boxes_per_slice)
    np.testing.assert_allclose(rescored[10]["scores"], np.array([0.4]), atol=1e-5)
    np.testing.assert_allclose(rescored[99]["scores"], np.array([0.7]), atol=1e-5)
