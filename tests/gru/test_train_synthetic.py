"""G.INT.2 — synthetic correlated dataset → val AUROC > 0.7 in 5 epochs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from endo.config.gru import GRUConfig
from endo.gru.rescorer import GRURescorer, volume_score
from endo.gru.train import GRUFeatureDataset, gru_collate


def _make_synthetic_patient(
    out_path: Path,
    label: int,
    n_slices: int,
    rng: np.random.Generator,
    signal_strength: float = 4.0,
) -> None:
    """Per-patient features carry the volume label in a few privileged
    dimensions. Positive volumes have a strong positive shift on a few
    "lesion" slices; negatives are pure noise.
    """
    feats = rng.standard_normal((n_slices, 768)).astype(np.float32) * 0.3

    if label == 1:
        n_lesion = int(rng.integers(2, 6))
        lesion_idx = rng.choice(n_slices, size=n_lesion, replace=False)
        # First 8 channels carry a strong signal on lesion slices.
        feats[lesion_idx, :8] += signal_strength
    np.savez_compressed(
        out_path,
        feats=feats.astype(np.float16),
        slice_ys=np.arange(n_slices, dtype=np.int32),
        patient_label=np.int8(label),
    )


def test_train_gru_synthetic_correlation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # 30 train + 10 val patients per class. Variable sequence length.
    def populate(out_dir: Path, n_per_class: int, prefix: str) -> list[Path]:
        paths: list[Path] = []
        for i in range(n_per_class):
            for label in (0, 1):
                T = int(rng.integers(20, 60))
                p = out_dir / f"{prefix}_{label}_{i}.npz"
                _make_synthetic_patient(p, label, T, rng)
                paths.append(p)
        return paths

    train_paths = populate(train_dir, n_per_class=30, prefix="t")
    val_paths = populate(val_dir, n_per_class=10, prefix="v")

    cfg = GRUConfig(
        input_dim=768,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True,
        dropout_input=0.0,
        epochs=5,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=8,
        top_k=3,
        aux_loss_weight=0.1,
    )

    torch.manual_seed(0)
    train_ds = GRUFeatureDataset(train_paths)
    val_ds = GRUFeatureDataset(val_paths)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=gru_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=gru_collate
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRURescorer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    eps = 1e-6
    for _epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            feats = batch["feats"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            probs = model(feats, mask)
            vol_max = volume_score(probs, mask, agg="max").clamp(eps, 1 - eps)
            vol_topk = volume_score(probs, mask, agg="topk", k=cfg.top_k).clamp(
                eps, 1 - eps
            )
            loss = torch.nn.functional.binary_cross_entropy(
                vol_max, labels
            ) + cfg.aux_loss_weight * torch.nn.functional.binary_cross_entropy(
                vol_topk, labels
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluate val AUROC.
    model.eval()
    scores: list[float] = []
    gts: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["feats"].to(device)
            mask = batch["mask"].to(device)
            probs = model(feats, mask)
            vol = volume_score(probs, mask, agg="max")
            scores.extend(vol.detach().cpu().numpy().tolist())
            gts.extend(int(x) for x in batch["labels"].numpy().tolist())

    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(gts, scores))
    assert auroc > 0.7, f"val AUROC {auroc:.3f} below 0.7 threshold"
