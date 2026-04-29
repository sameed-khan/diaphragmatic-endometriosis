"""GRU rescorer model + rescoring helper (Component 6.5 §6, §8).

The GRU consumes a sequence of GAP-pooled stage-3 backbone features per
slice and emits a per-slice probability that the slice contains a lesion.
At eval time the per-slice probability multiplies each detector box's score
on that slice (PRD §6.10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from endo.config.gru import GRUConfig


class GRURescorer(nn.Module):
    """BiGRU over per-slice features → per-slice presence probability.

    forward(feats, mask) -> (B, T) probabilities in [0, 1].
    The mask is *not* used inside forward (PyTorch's GRU has no native
    masking and zero-padded timesteps don't break a BiGRU's outputs at
    valid positions because we run the full sequence). Callers must
    re-apply ``mask`` before any reduction (max / top-k / mean).
    """

    def __init__(self, cfg: GRUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dropout = nn.Dropout(cfg.dropout_input)
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        gru_out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.head = nn.Linear(gru_out_dim, 1)

    def forward(self, feats: Tensor, mask: Tensor | None = None) -> Tensor:
        """feats: (B, T, input_dim) float; mask: (B, T) bool. Returns (B, T) probs."""
        x = self.input_dropout(feats)
        h, _ = self.gru(x)  # (B, T, gru_out_dim)
        logits = self.head(h).squeeze(-1)  # (B, T)
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask.to(probs.dtype)
        return probs

    def logits(self, feats: Tensor) -> Tensor:
        """Return raw logits (B, T) without mask multiplication."""
        x = self.input_dropout(feats)
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def volume_score(
    per_slice_probs: Tensor,
    mask: Tensor,
    agg: str = "topk",
    k: int = 3,
) -> Tensor:
    """Aggregate per-slice probabilities into a (B,) volume score.

    Parameters
    ----------
    per_slice_probs : (B, T) probabilities in [0, 1].
    mask : (B, T) bool. ``True`` for valid timesteps; padded positions ``False``.
    agg : ``"max"`` or ``"topk"``.
    k : top-k count (used when ``agg == "topk"``).

    The ``mask`` is enforced by replacing padded positions with ``-inf`` before
    reduction so they cannot influence max/top-k.
    """
    if per_slice_probs.dim() != 2 or mask.dim() != 2:
        raise ValueError(
            f"Expected (B, T) probs and mask, got {per_slice_probs.shape=}, {mask.shape=}"
        )
    if per_slice_probs.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch probs={per_slice_probs.shape} mask={mask.shape}"
        )

    neg_inf = torch.finfo(per_slice_probs.dtype).min
    masked = per_slice_probs.masked_fill(~mask, neg_inf)

    if agg == "max":
        return masked.max(dim=1).values

    if agg == "topk":
        T = per_slice_probs.shape[1]
        # Per-row valid count → effective k.
        valid_counts = mask.sum(dim=1).clamp(min=1)
        k_clamped = int(min(k, T))
        topk_vals, _ = masked.topk(k_clamped, dim=1)
        # For rows where valid_count < k, only the first ``valid_count`` entries
        # in topk_vals are real; the rest are -inf and must be excluded.
        idx = torch.arange(k_clamped, device=topk_vals.device)
        valid_in_topk = idx[None, :] < valid_counts[:, None]  # (B, k_clamped)
        topk_safe = topk_vals.masked_fill(~valid_in_topk, 0.0)
        denom = valid_in_topk.sum(dim=1).clamp(min=1).to(topk_vals.dtype)
        return topk_safe.sum(dim=1) / denom

    raise ValueError(f"Unknown agg {agg!r}; expected 'max' or 'topk'")


# ----------------------------------------------------------------------
# Rescoring helper for Component 7.
# ----------------------------------------------------------------------


def _build_rescorer_from_ckpt(gru_ckpt_path: str | Path) -> GRURescorer:
    ckpt = torch.load(str(gru_ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict, GRUConfig):
        cfg = cfg_dict
    else:
        # Filter to known fields so older / extended dumps still load.
        valid = set(GRUConfig.model_fields.keys())
        cfg = GRUConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = GRURescorer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def rescore_detector_outputs(
    gru_ckpt_path: str | Path,
    feature_cache_path: str | Path,
    detector_boxes_per_slice: Mapping[int, dict],
) -> dict[int, dict]:
    """Multiply each detector box's score by the GRU's per-slice probability.

    Parameters
    ----------
    gru_ckpt_path : path to ``runs/<exp>/fold{f}/gru/ckpt.pt``.
    feature_cache_path : path to a single patient's ``feature_cache/<pid>.npz``.
    detector_boxes_per_slice : ``{slice_y: {"boxes": ..., "scores": ndarray}}``.

    Returns
    -------
    Same shape as input. ``boxes`` are passed through; ``scores`` are
    multiplied by ``p_t`` for that slice. If a slice is missing from the
    feature cache, its scores are left unchanged (PRD §6.10 fallback).
    """
    model = _build_rescorer_from_ckpt(gru_ckpt_path)

    cache = np.load(str(feature_cache_path))
    feats = torch.from_numpy(np.asarray(cache["feats"], dtype=np.float32)).unsqueeze(0)
    slice_ys = np.asarray(cache["slice_ys"], dtype=np.int64).tolist()

    mask = torch.ones((1, feats.shape[1]), dtype=torch.bool)
    with torch.no_grad():
        probs = model(feats, mask).squeeze(0).cpu().numpy()  # (N,)
    p_by_slice = {int(sy): float(p) for sy, p in zip(slice_ys, probs)}

    rescored: dict[int, dict] = {}
    for sy, item in detector_boxes_per_slice.items():
        sy_int = int(sy)
        scores = np.asarray(item["scores"])
        if sy_int in p_by_slice:
            new_scores = scores * p_by_slice[sy_int]
        else:
            new_scores = scores.copy()
        rescored[sy_int] = {"boxes": item["boxes"], "scores": new_scores}
    return rescored


def rescore_slice_scores(
    slice_scores: Mapping[str, list],
    *,
    ckpt_path: str | Path,
    feature_dir: str | Path,
) -> dict[str, list]:
    """Apply GRU rescoring to a ``{pid: list[SliceScore]}`` mapping.

    Used by Component 7 ``run_eval`` when ``--use-gru`` is set. Loads the
    fold's GRU ckpt once, then per-patient looks up the feature cache npz
    and multiplies each slice's box scores by the GRU probability for that
    slice. Missing patients / slices are passed through unchanged.
    """
    from endo.inference_pass import SliceScore

    model = _build_rescorer_from_ckpt(ckpt_path)
    feature_dir = Path(feature_dir)

    out: dict[str, list[SliceScore]] = {}
    for pid, slices in slice_scores.items():
        npz = feature_dir / f"{pid}.npz"
        if not npz.exists():
            out[pid] = list(slices)
            continue
        cache = np.load(str(npz))
        feats = torch.from_numpy(np.asarray(cache["feats"], dtype=np.float32)).unsqueeze(0)
        slice_ys = np.asarray(cache["slice_ys"], dtype=np.int64).tolist()
        mask = torch.ones((1, feats.shape[1]), dtype=torch.bool)
        with torch.no_grad():
            probs = model(feats, mask).squeeze(0).cpu().numpy()
        p_by_slice = {int(sy): float(p) for sy, p in zip(slice_ys, probs)}

        new_slices: list[SliceScore] = []
        for s in slices:
            p = p_by_slice.get(int(s.slice_y))
            if p is None:
                new_slices.append(s)
            else:
                new_slices.append(
                    SliceScore(
                        patient_id=s.patient_id,
                        slice_y=s.slice_y,
                        boxes=s.boxes,
                        scores=s.scores * p,
                        aux_seg_max=float(s.aux_seg_max * p),
                    )
                )
        out[pid] = new_slices
    return out


# ----------------------------------------------------------------------
# Provenance helper.
# ----------------------------------------------------------------------


def write_gru_provenance(
    output_path: Path,
    config: GRUConfig,
    git_sha: str | None,
    val_auroc: float,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "config": json.loads(config.model_dump_json()),
        "git_sha": git_sha,
        "val_auroc": float(val_auroc),
        "epoch": int(epoch),
    }
    if extra:
        payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
