"""``LesionDataModule`` — Lightning DataModule wrapping ``LesionDataset``.

Loads the entire preprocessed cache eagerly into RAM in ``setup()``, builds
the train/val ``slice_index``, and exposes ``train_dataloader`` /
``val_dataloader`` / ``inference_dataloader``.

Holdout protection (PRD §6.6, spec §11) is enforced two ways:

  1. ``setup()`` refuses to load any holdout patient unless ``allow_holdout``.
  2. ``inference_dataloader(patient_ids)`` re-checks against the cohort's
     known holdout pids and raises if any leak in.

The cache is a sibling artifact built by Component 1 (``cache/v1/...``); see
PRD §5.2 for the on-disk schema.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning as pl_lightning
from torch.utils.data import DataLoader, Sampler

from endo.data.collate import collate_fn
from endo.data.dataset import LesionDataset
from endo.data.manifest import (
    fold_split,
    manifest_by_pid,
    read_manifest_jsonl,
)
from endo.data.samples import Sample


class HoldoutAccessError(RuntimeError):
    """Raised when a holdout patient enters a code path with ``allow_holdout=False``."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LesionDataModule(pl_lightning.LightningDataModule):
    def __init__(
        self,
        cache_root: Path,
        manifest_path: Path,
        cohort_path: Path,
        fold: int,
        batch_size: int = 8,
        num_workers: int = 8,
        slice_window: int = 5,
        target_input_shape: tuple[int, int, int] = (384, 160, 384),
        cache_shape: tuple[int, int, int] = (408, 174, 408),
        augment_train: Callable[[Sample], Sample] | None = None,
        sampler_train: Sampler[int] | None = None,
        allow_holdout: bool = False,
        rng_seed: int = 42,
        persistent_workers: bool | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root)
        self.manifest_path = Path(manifest_path)
        self.cohort_path = Path(cohort_path)
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.slice_window = slice_window
        self.target_input_shape = tuple(target_input_shape)
        self.cache_shape = tuple(cache_shape)
        self.augment_train = augment_train
        self.sampler_train = sampler_train
        self.allow_holdout = allow_holdout
        self.rng_seed = rng_seed
        self.pin_memory = pin_memory
        # ``persistent_workers`` requires num_workers > 0. Default to True iff so.
        self.persistent_workers = (
            persistent_workers if persistent_workers is not None else (num_workers > 0)
        )

        # populated by setup()
        self._cache: dict[str, dict[str, Any]] = {}
        self._train_pids: list[str] = []
        self._val_pids: list[str] = []
        self._holdout_pids: set[str] = set()
        self._all_known_pids: set[str] = set()
        self._gt_lookup: dict[tuple[str, int], np.ndarray] = {}
        self._train_slice_index: list[tuple[str, int, bool, str]] = []
        self._val_slice_index: list[tuple[str, int, bool, str]] = []
        self._train_dataset: LesionDataset | None = None
        self._val_dataset: LesionDataset | None = None
        self._is_setup = False

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        # 1. Read manifest + cohort, derive (train, val, holdout) pids.
        manifest_rows = read_manifest_jsonl(self.manifest_path)
        manifest_lookup = manifest_by_pid(manifest_rows)
        train_pids, val_pids, holdout_pids = fold_split(manifest_rows, self.fold)
        self._train_pids = list(train_pids)
        self._val_pids = list(val_pids)
        self._holdout_pids = set(holdout_pids)
        self._all_known_pids = set(manifest_lookup.keys())

        # 2. Read preprocessed manifest (per-patient cache rows).
        pre_path = self.cache_root / "preprocessed_manifest.jsonl"
        pre_rows = _read_jsonl(pre_path)
        pre_lookup: dict[str, dict[str, Any]] = {r["patient_id"]: r for r in pre_rows}

        # 3. Decide which patients to load.
        load_pids: list[str] = list(self._train_pids) + list(self._val_pids)
        if self.allow_holdout:
            load_pids = list(load_pids) + list(self._holdout_pids)
        else:
            # Holdout guard: refuse if any holdout pid was somehow requested.
            overlap = self._holdout_pids.intersection(load_pids)
            if overlap:
                raise HoldoutAccessError(
                    f"Refusing to load holdout patients {sorted(overlap)} "
                    f"with allow_holdout=False."
                )

        # 4. Eager-load every needed patient.
        self._cache = {}
        for pid in load_pids:
            if pid not in pre_lookup:
                raise FileNotFoundError(
                    f"patient_id {pid!r} missing from {pre_path} (preprocessed cache)."
                )
            row = pre_lookup[pid]
            volume = np.load(self.cache_root / row["cache_volume_path"])  # fp16
            lesion_mask: np.ndarray | None
            if row.get("cache_lesion_mask_path"):
                lesion_mask = np.load(self.cache_root / row["cache_lesion_mask_path"])
            else:
                lesion_mask = None
            border_band: np.ndarray | None
            if row.get("cache_border_band_path"):
                border_band = np.load(self.cache_root / row["cache_border_band_path"])
            else:
                border_band = None
            self._cache[pid] = {
                "volume": volume,
                "lesion_mask": lesion_mask,
                "border_band": border_band,
                "manifest_row": manifest_lookup[pid],
                "preprocessed_row": row,
            }

        # 5. Read gt_boxes.parquet and build the (pid, slice_y) -> boxes lookup.
        gt_path = self.cache_root / "gt_boxes.parquet"
        if gt_path.exists():
            gt_df = pl.read_parquet(gt_path)
            self._gt_lookup = self._build_gt_lookup(gt_df)
        else:
            self._gt_lookup = {}

        # 6. Build slice_index for train and val.
        tx, ty, tz = self.target_input_shape
        cx, cy, cz = self.cache_shape
        py = (cy - ty) // 2  # cache pad-offset on y
        half = self.slice_window // 2
        # Iterate cached slice indices that, when crop is centered (jy=0), lie
        # inside the valid window. This is the simplest deterministic
        # parameterization; jitter at sampling time may push the *target* slice
        # by ±jy_max, but the dataset checks bounds at __getitem__ and any
        # invalid combination would raise (the indexer guarantees it can't).
        slice_y_lo = py + half
        slice_y_hi = py + ty - half  # exclusive

        self._train_slice_index = self._build_slice_index(
            self._train_pids, slice_y_lo, slice_y_hi
        )
        self._val_slice_index = self._build_slice_index(
            self._val_pids, slice_y_lo, slice_y_hi
        )

        # 7. Construct datasets.
        self._train_dataset = LesionDataset(
            patient_ids=self._train_pids,
            cache=self._cache,
            gt_boxes_by_pid_slice=self._gt_lookup,
            slice_index=self._train_slice_index,
            target_input_shape=self.target_input_shape,
            slice_window=self.slice_window,
            augment=self.augment_train,
            rng_seed=self.rng_seed,
            cache_shape=self.cache_shape,
        )
        self._val_dataset = LesionDataset(
            patient_ids=self._val_pids,
            cache=self._cache,
            gt_boxes_by_pid_slice=self._gt_lookup,
            slice_index=self._val_slice_index,
            target_input_shape=self.target_input_shape,
            slice_window=self.slice_window,
            augment=None,
            rng_seed=self.rng_seed,
            cache_shape=self.cache_shape,
        )
        self._is_setup = True

    # ------------------------------------------------------------------
    # Public dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "call setup() first"
        kwargs: dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        if self.sampler_train is not None:
            kwargs["sampler"] = self.sampler_train
        else:
            kwargs["shuffle"] = True
        return DataLoader(self._train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "call setup() first"
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def inference_dataloader(self, patient_ids: list[str]) -> DataLoader:
        """Build a sequential dataloader over the requested patients' slices.

        Holdout guard re-fires here. ``patient_ids`` are checked against the
        cohort's holdout set and against ``self.allow_holdout`` before any
        cache access. Order is ``(patient_id ASC, slice_y ASC)``.
        """
        if not self._is_setup:
            self.setup()

        if not self.allow_holdout:
            overlap = self._holdout_pids.intersection(patient_ids)
            if overlap:
                raise HoldoutAccessError(
                    f"Refusing to load holdout patients {sorted(overlap)} "
                    f"with allow_holdout=False."
                )

        # Make sure each pid is loaded into cache.
        missing = [p for p in patient_ids if p not in self._cache]
        if missing:
            raise KeyError(
                f"inference_dataloader requested pids that were not loaded "
                f"in setup(): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        tx, ty, tz = self.target_input_shape
        cx, cy, cz = self.cache_shape
        py = (cy - ty) // 2
        half = self.slice_window // 2
        slice_y_lo = py + half
        slice_y_hi = py + ty - half

        slice_index = self._build_slice_index(
            sorted(set(patient_ids)), slice_y_lo, slice_y_hi
        )

        ds = LesionDataset(
            patient_ids=sorted(set(patient_ids)),
            cache=self._cache,
            gt_boxes_by_pid_slice=self._gt_lookup,
            slice_index=slice_index,
            target_input_shape=self.target_input_shape,
            slice_window=self.slice_window,
            augment=None,
            rng_seed=self.rng_seed,
            cache_shape=self.cache_shape,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_gt_lookup(
        self, gt_df: pl.DataFrame
    ) -> dict[tuple[str, int], np.ndarray]:
        out: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
        # Required schema columns (per PRD §5.2.4): patient_id, slice_y, x1, z1, x2, z2.
        for row in gt_df.select(["patient_id", "slice_y", "x1", "z1", "x2", "z2"]).iter_rows():
            pid, sy, x1, z1, x2, z2 = row
            key = (pid, int(sy))
            out.setdefault(key, []).append((float(x1), float(z1), float(x2), float(z2)))
        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}

    def _build_slice_index(
        self,
        pids: list[str],
        slice_y_lo: int,
        slice_y_hi: int,
    ) -> list[tuple[str, int, bool, str]]:
        out: list[tuple[str, int, bool, str]] = []
        for pid in pids:
            entry = self._cache.get(pid)
            if entry is None:
                # Holdout pid not loaded — skip (only reachable from setup, which
                # is guarded).
                continue
            label = entry["manifest_row"].get("label", "negative")
            is_positive_volume = label == "positive"
            for sy in range(slice_y_lo, slice_y_hi):
                has_box = (pid, sy) in self._gt_lookup
                is_positive_slice = bool(is_positive_volume and has_box)
                if is_positive_slice:
                    kind = "pos_slice"
                elif is_positive_volume:
                    kind = "neg_slice_pos_vol"
                else:
                    kind = "neg_slice_neg_vol"
                out.append((pid, sy, is_positive_slice, kind))
        return out
