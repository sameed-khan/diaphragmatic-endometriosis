"""GRU training loop (Component 6.5 §7).

Loads per-patient ``feature_cache/<pid>.npz`` files for the fold's
training set (the *other 4 folds*' patients) and validation set, trains
the GRU with BCE on a max-aggregated volume score, and writes
``runs/<exp>/fold{f}/gru/ckpt.pt`` plus ``gru_provenance.json``.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from endo.config import ExperimentConfig
from endo.config.gru import GRUConfig
from endo.data.manifest import manifest_by_pid, read_manifest_jsonl
from endo.gru.rescorer import GRURescorer, volume_score, write_gru_provenance

log = logging.getLogger(__name__)


def _default_output_dir(experiment: ExperimentConfig, fold: int) -> Path:
    return experiment.run_dir() / f"fold{fold}" / "gru"


def _git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha or None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Dataset / collate.
# ----------------------------------------------------------------------


class GRUFeatureDataset(Dataset):
    """Loads a list of per-patient .npz feature caches into RAM."""

    def __init__(self, npz_paths: list[Path]) -> None:
        if not npz_paths:
            raise ValueError("GRUFeatureDataset received empty npz_paths")
        self.entries: list[dict[str, Any]] = []
        for p in npz_paths:
            data = np.load(str(p))
            feats = np.asarray(data["feats"], dtype=np.float32)
            label = int(np.asarray(data["patient_label"]).item())
            self.entries.append(
                {
                    "patient_id": p.stem,
                    "feats": feats,
                    "label": label,
                }
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        e = self.entries[idx]
        return {
            "patient_id": e["patient_id"],
            "feats": torch.from_numpy(e["feats"]).float(),
            "label": torch.tensor(float(e["label"]), dtype=torch.float32),
        }


def gru_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sequences and produce a (B, T) bool mask."""
    feats = [b["feats"] for b in batch]
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    if lengths.max().item() == 0:
        raise ValueError("All sequences in batch have length 0")
    max_T = int(lengths.max().item())
    B = len(batch)
    D = feats[0].shape[1]
    padded = torch.zeros((B, max_T, D), dtype=torch.float32)
    mask = torch.zeros((B, max_T), dtype=torch.bool)
    for i, f in enumerate(feats):
        T = f.shape[0]
        padded[i, :T] = f
        mask[i, :T] = True
    labels = torch.stack([b["label"] for b in batch])
    return {
        "feats": padded,
        "mask": mask,
        "labels": labels,
        "lengths": lengths,
        "patient_ids": [b["patient_id"] for b in batch],
    }


# ----------------------------------------------------------------------
# Patient-set discovery (cross-fold features).
# ----------------------------------------------------------------------


def _discover_feature_cache_paths(
    experiment: ExperimentConfig,
    val_fold: int,
) -> tuple[list[Path], list[Path]]:
    """Return (train_npz_paths, val_npz_paths).

    Train = patients with ``cohort=='cross-validation'`` and ``fold != val_fold``;
    val = same cohort, ``fold == val_fold``.
    The ``feature_cache/<pid>.npz`` is sourced from each patient's *own*
    fold directory (i.e. the fold for which they were the validation set).
    """
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    rows = read_manifest_jsonl(manifest_path)
    lookup = manifest_by_pid(rows)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for pid, row in lookup.items():
        if row.get("cohort") != "cross-validation":
            continue
        f = int(row.get("fold", -1))
        npz = experiment.run_dir() / f"fold{f}" / "gru" / "feature_cache" / f"{pid}.npz"
        if f == val_fold:
            val_paths.append(npz)
        else:
            train_paths.append(npz)
    train_paths.sort()
    val_paths.sort()
    return train_paths, val_paths


# ----------------------------------------------------------------------
# Training step.
# ----------------------------------------------------------------------


def _bce_volume_loss(
    per_slice_probs: Tensor,
    mask: Tensor,
    labels: Tensor,
    cfg: GRUConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Volume BCE on max-aggregation + auxiliary top-k mean BCE."""
    eps = 1e-6
    vol_max = volume_score(per_slice_probs, mask, agg="max")
    vol_topk = volume_score(per_slice_probs, mask, agg="topk", k=cfg.top_k)

    vol_max_c = vol_max.clamp(eps, 1.0 - eps)
    vol_topk_c = vol_topk.clamp(eps, 1.0 - eps)

    loss_max = F.binary_cross_entropy(vol_max_c, labels)
    loss_topk = F.binary_cross_entropy(vol_topk_c, labels)
    total = loss_max + cfg.aux_loss_weight * loss_topk
    return total, {
        "loss_max": float(loss_max.detach().item()),
        "loss_topk": float(loss_topk.detach().item()),
        "loss_total": float(total.detach().item()),
    }


@torch.no_grad()
def _evaluate(
    model: GRURescorer,
    loader: DataLoader,
    device: str,
) -> tuple[float, list[float], list[int]]:
    model.eval()
    scores: list[float] = []
    gts: list[int] = []
    for batch in loader:
        feats = batch["feats"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        probs = model(feats, mask)
        vol = volume_score(probs, mask, agg="max")
        scores.extend(vol.detach().cpu().numpy().tolist())
        gts.extend([int(x) for x in labels.detach().cpu().numpy().tolist()])
    if len(set(gts)) < 2:
        return 0.0, scores, gts
    return float(roc_auc_score(gts, scores)), scores, gts


# ----------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------


def train_gru_for_fold(
    experiment: ExperimentConfig,
    fold: int,
    output_dir: Path | None = None,
    device: str = "cuda",
    train_npz_paths: list[Path] | None = None,
    val_npz_paths: list[Path] | None = None,
    seed: int = 42,
) -> Path:
    """Train the GRU rescorer for one fold; write ``ckpt.pt`` + provenance.

    Returns the path to the written ckpt.
    """
    out_dir = Path(output_dir) if output_dir is not None else _default_output_dir(experiment, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    provenance_path = out_dir / "gru_provenance.json"

    if train_npz_paths is None or val_npz_paths is None:
        discovered_train, discovered_val = _discover_feature_cache_paths(experiment, fold)
        train_npz_paths = train_npz_paths or discovered_train
        val_npz_paths = val_npz_paths or discovered_val

    missing = [p for p in train_npz_paths + val_npz_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} feature_cache files (first 3): {missing[:3]}"
        )

    cfg = experiment.gru
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = GRUFeatureDataset(train_npz_paths)
    val_ds = GRUFeatureDataset(val_npz_paths)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=gru_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=gru_collate,
        num_workers=0,
    )

    model = GRURescorer(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auroc = -1.0
    best_epoch = -1
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            feats = batch["feats"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            probs = model(feats, mask)
            loss, _ = _bce_volume_loss(probs, mask, labels, cfg)
            optim.zero_grad()
            loss.backward()
            optim.step()

        val_auroc, _, _ = _evaluate(model, val_loader, device)
        log.info("[fold %d] epoch %d val_auroc=%.4f", fold, epoch, val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        k: v for k, v in cfg.model_dump().items()
                    },
                    "epoch": epoch,
                    "val_auroc": float(val_auroc),
                },
                str(ckpt_path),
            )

    write_gru_provenance(
        provenance_path,
        config=cfg,
        git_sha=_git_sha(),
        val_auroc=float(best_auroc),
        epoch=int(best_epoch),
        extra={
            "fold": fold,
            "n_train_patients": len(train_ds),
            "n_val_patients": len(val_ds),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    return ckpt_path
