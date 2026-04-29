"""GRU feature cache (Component 6.5 §5).

For each fold-f validation patient, runs the frozen Stage-1 backbone on
every valid slice's 5-channel input, GAP-pools the last backbone stage
feature, and writes ``runs/<exp>/fold{f}/gru/feature_cache/<pid>.npz``.

Schema per patient (PRD §5.3.5):
  feats:         (N_valid_slices, 768) float16
  slice_ys:      (N_valid_slices,) int32
  patient_label: () int8 (0 negative, 1 positive)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from endo.config import ExperimentConfig
from endo.data.datamodule import LesionDataModule
from endo.data.manifest import manifest_by_pid, read_manifest_jsonl
from endo.lightning_module import LesionDetectorLM

log = logging.getLogger(__name__)


def _default_output_dir(experiment: ExperimentConfig, fold: int) -> Path:
    return experiment.run_dir() / f"fold{fold}" / "gru" / "feature_cache"


def _default_ckpt_path(experiment: ExperimentConfig, fold: int) -> Path:
    return experiment.run_dir() / f"fold{fold}" / "ckpts" / "best.ckpt"


def _expected_pids(
    experiment: ExperimentConfig, fold: int
) -> tuple[list[str], dict[str, dict]]:
    """Return (validation_pids_for_fold, manifest_lookup)."""
    manifest_path = experiment.paths.data_root / "manifest.jsonl"
    rows = read_manifest_jsonl(manifest_path)
    lookup = manifest_by_pid(rows)
    val_pids = sorted(
        pid
        for pid, row in lookup.items()
        if row.get("cohort") == "cross-validation" and int(row.get("fold", -1)) == fold
    )
    return val_pids, lookup


def _build_datamodule(experiment: ExperimentConfig, fold: int) -> LesionDataModule:
    cache_root = experiment.paths.cache_root
    data_root = experiment.paths.data_root
    dm = LesionDataModule(
        cache_root=cache_root,
        manifest_path=data_root / "manifest.jsonl",
        cohort_path=data_root / "cohort.json",
        fold=fold,
        batch_size=8,
        num_workers=0,
        allow_holdout=False,
    )
    dm.setup()
    return dm


def _load_detector_with_ema(ckpt_path: Path, device: str, experiment=None) -> LesionDetectorLM:
    """Load the LightningModule and overlay EMA weights when present.

    ``LesionDetectorLM.__init__`` takes a positional ``exp_cfg``, so
    ``Lightning.load_from_checkpoint`` can't auto-instantiate it. We
    construct the module with ``experiment`` (or the saved hparams as
    fallback), then load the state dict manually.
    """
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if experiment is None:
        from endo.config import ExperimentConfig

        hp = raw.get("hyper_parameters", {}) or {}
        exp_payload = hp.get("experiment")
        if exp_payload is None:
            raise ValueError(
                "Experiment config not provided and not found in checkpoint hyper_parameters"
            )
        experiment = ExperimentConfig.model_validate(exp_payload)

    lm = LesionDetectorLM(experiment)
    lm.load_state_dict(raw["state_dict"], strict=False)

    ema_sd = raw.get("ema_state_dict")
    if ema_sd is not None:
        try:
            lm.model.load_state_dict(ema_sd, strict=True)
            log.info("Loaded EMA weights from %s", ckpt_path)
        except Exception as e:  # pragma: no cover — fallback path
            log.warning("EMA weights present but could not be loaded (%s); using live weights", e)
    else:
        log.warning("No ema_state_dict in %s; using live weights", ckpt_path)
    lm.eval()
    lm.to(device)
    return lm


@torch.no_grad()
def _extract_for_pid(
    pid: str,
    dm: LesionDataModule,
    lm: LesionDetectorLM,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats[N,768] fp16, slice_ys[N] int32)."""
    loader = dm.inference_dataloader([pid])
    feats_chunks: list[np.ndarray] = []
    slice_ys_all: list[int] = []
    for batch in loader:
        x = batch.volume_5ch.to(device)
        # Backbone returns a list of stage feature maps for ConvNeXt-tiny
        # (features_only). Last stage is index -1 with 768 channels at stride 32.
        feats_pyramid = lm.model.backbone(x)
        last = feats_pyramid[-1]  # (B, 768, h, w)
        gap = F.adaptive_avg_pool2d(last, 1).flatten(1)  # (B, 768)
        feats_chunks.append(gap.detach().cpu().numpy().astype(np.float16))
        slice_ys_all.extend(int(s) for s in batch.slice_ys.detach().cpu().tolist())
    if not feats_chunks:
        return np.zeros((0, 768), dtype=np.float16), np.zeros((0,), dtype=np.int32)
    feats = np.concatenate(feats_chunks, axis=0)
    slice_ys = np.asarray(slice_ys_all, dtype=np.int32)
    return feats, slice_ys


def extract_features_for_pids(
    experiment: ExperimentConfig,
    fold: int,
    *,
    pids: Iterable[str],
    output_dir: Path,
    ckpt_path: Path,
    device: str = "cuda",
) -> Path:
    """Build a feature cache directory at ``output_dir`` for ``pids`` using
    the detector ``ckpt_path``.

    Used by the post-training eval to rebuild the GRU feature cache from a
    chosen ``best``/``last`` checkpoint instead of re-using stale per-fold
    caches that were built from training-time monitoring weights.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt}")

    lm = _load_detector_with_ema(ckpt, device, experiment=experiment)
    dm = _build_datamodule(experiment, fold)

    _, lookup = _expected_pids(experiment, fold)
    target_pids = sorted(set(pids))
    for pid in target_pids:
        out_path = out_dir / f"{pid}.npz"
        feats, slice_ys = _extract_for_pid(pid, dm, lm, device)
        label = lookup.get(pid, {}).get("label", "negative")
        patient_label = np.int8(1 if label == "positive" else 0)
        np.savez_compressed(
            out_path,
            feats=feats,
            slice_ys=slice_ys,
            patient_label=patient_label,
        )
        log.info(
            "Wrote %s (%d slices, label=%d)",
            out_path,
            feats.shape[0],
            int(patient_label),
        )
    return out_dir


def extract_features_for_fold(
    experiment: ExperimentConfig,
    fold: int,
    output_dir: Path | None = None,
    device: str = "cuda",
    force: bool = False,
    pids: Iterable[str] | None = None,
    ckpt_path: Path | None = None,
) -> Path:
    """Build feature cache npz files for the fold's validation patients.

    Parameters
    ----------
    experiment : ExperimentConfig describing the run.
    fold : fold index (0..4).
    output_dir : default ``runs/<exp>/fold{f}/gru/feature_cache/``.
    device : torch device.
    force : if False, skip when all expected ``<pid>.npz`` already exist.
    pids : restrict to a specific pid set (else: all fold-{f} val pids).
    ckpt_path : override checkpoint location (default: the fold's best.ckpt).

    Returns the output directory.
    """
    out_dir = Path(output_dir) if output_dir is not None else _default_output_dir(
        experiment, fold
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    val_pids, lookup = _expected_pids(experiment, fold)
    target_pids = sorted(set(pids)) if pids is not None else val_pids

    expected_files = [out_dir / f"{pid}.npz" for pid in target_pids]
    if not force and target_pids and all(p.exists() for p in expected_files):
        log.info(
            "Feature cache already populated at %s (%d files); skip (use force=True to rebuild)",
            out_dir,
            len(expected_files),
        )
        return out_dir

    ckpt = Path(ckpt_path) if ckpt_path is not None else _default_ckpt_path(experiment, fold)
    if not ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt}")

    lm = _load_detector_with_ema(ckpt, device, experiment=experiment)
    dm = _build_datamodule(experiment, fold)

    for pid in target_pids:
        out_path = out_dir / f"{pid}.npz"
        if not force and out_path.exists():
            log.info("Skip existing %s", out_path)
            continue
        feats, slice_ys = _extract_for_pid(pid, dm, lm, device)
        label = lookup.get(pid, {}).get("label", "negative")
        patient_label = np.int8(1 if label == "positive" else 0)
        np.savez_compressed(
            out_path,
            feats=feats,
            slice_ys=slice_ys,
            patient_label=patient_label,
        )
        log.info("Wrote %s (%d slices, label=%d)", out_path, feats.shape[0], int(patient_label))

    return out_dir
