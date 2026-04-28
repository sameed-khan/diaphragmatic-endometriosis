"""Weighted, epoch-aware slice sampler with hard-negative substitution.

Implements PRD §5.1 / Component 5 §4. Class mix decays linearly over
``decay_epochs`` from ``(pos_frac_start, *, *)`` to ``(pos_frac_end, *, *)``;
at each step a uniform draw routes the index to one of three pools:

  - ``pos_idx``  — positive-slice dataset indices.
  - ``nipv_idx`` — negative slices inside positive volumes.
  - ``ninv_idx`` — negative slices inside negative volumes.

Negative-in-negative-volume draws are the source of hard-negative
substitution: when ``epoch >= hard_pool_start_epoch`` and a non-empty hard
pool has been set via :meth:`set_hard_pool`, a fraction
``hard_pool_substitution_rate`` of those draws are replaced with a uniform
random draw from the hard pool.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from endo.config.sampler import SamplerConfig
from endo.utils.seeding import derive_seed

log = logging.getLogger(__name__)


_KIND_POS = "pos_slice"
_KIND_NIPV = "neg_slice_pos_vol"
_KIND_NINV = "neg_slice_neg_vol"
_VALID_KINDS = {_KIND_POS, _KIND_NIPV, _KIND_NINV}


class WeightedScheduledSampler(Sampler[int]):
    """Yields dataset indices for one epoch with class-mix decay + HNM.

    ``slice_index`` is ``[(patient_id, slice_y, kind), ...]`` parallel to the
    Dataset's slice list — the sampler's emitted integers index directly into
    that list.
    """

    def __init__(
        self,
        slice_index: Sequence[tuple[str, int, str]],
        cfg: SamplerConfig,
        seed: int = 42,
    ) -> None:
        self._slice_index = list(slice_index)
        self.cfg = cfg
        self.seed = int(seed)

        # Partition once at construction.
        pos_idx: list[int] = []
        nipv_idx: list[int] = []
        ninv_idx: list[int] = []
        for i, (_, _, kind) in enumerate(self._slice_index):
            if kind == _KIND_POS:
                pos_idx.append(i)
            elif kind == _KIND_NIPV:
                nipv_idx.append(i)
            elif kind == _KIND_NINV:
                ninv_idx.append(i)
            else:
                raise ValueError(
                    f"Invalid slice kind {kind!r} at index {i}; "
                    f"expected one of {sorted(_VALID_KINDS)}"
                )
        self._pos_idx = np.asarray(pos_idx, dtype=np.int64)
        self._nipv_idx = np.asarray(nipv_idx, dtype=np.int64)
        self._ninv_idx = np.asarray(ninv_idx, dtype=np.int64)

        if self._pos_idx.size == 0:
            log.warning(
                "WeightedScheduledSampler: pos_idx is empty; "
                "falling back to all-negative sampling.",
            )
        if self._nipv_idx.size == 0 and self._ninv_idx.size == 0:
            raise ValueError(
                "WeightedScheduledSampler: both negative pools are empty.",
            )

        self.epoch: int = 0
        self._hard_pool: np.ndarray = np.empty((0,), dtype=np.int64)

    # ─── epoch / hard-pool plumbing ───────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_hard_pool(self, indices: Sequence[int]) -> None:
        """Replace the hard pool with ``indices`` (dataset-level)."""
        arr = np.asarray(list(indices), dtype=np.int64).ravel()
        if arr.size > 0:
            n = len(self._slice_index)
            if (arr < 0).any() or (arr >= n).any():
                raise ValueError(
                    f"hard pool contains out-of-range indices for dataset of size {n}",
                )
        self._hard_pool = arr

    @property
    def hard_pool_size(self) -> int:
        return int(self._hard_pool.size)

    # ─── schedule ────────────────────────────────────────────────────

    def current_p_pos(self) -> float:
        cfg = self.cfg
        if cfg.decay_epochs <= 0:
            t = 1.0
        else:
            t = min(self.epoch / float(cfg.decay_epochs), 1.0)
        p = cfg.pos_frac_start + t * (cfg.pos_frac_end - cfg.pos_frac_start)
        # Clamp to schedule envelope (handles either direction of pos_frac_*).
        lo, hi = sorted((cfg.pos_frac_start, cfg.pos_frac_end))
        return float(min(max(p, lo), hi))

    # ─── core protocol ───────────────────────────────────────────────

    def __len__(self) -> int:
        if self.cfg.epoch_mode == "full_pass":
            return len(self._slice_index)
        return int(self.cfg.samples_per_epoch)

    def __iter__(self) -> Iterator[int]:
        cfg = self.cfg
        rng = np.random.default_rng(derive_seed(self.seed, self.epoch))

        n_total = len(self)
        p_pos = self.current_p_pos() if self._pos_idx.size > 0 else 0.0
        # If positive pool is empty, redistribute its budget to negatives.
        nipv_share = cfg.neg_in_pos_vol_share
        # Edge case: degenerate negative pools.
        nipv_empty = self._nipv_idx.size == 0
        ninv_empty = self._ninv_idx.size == 0

        use_hard_pool = (
            self.epoch >= cfg.hard_pool_start_epoch and self._hard_pool.size > 0
        )
        sub_rate = float(cfg.hard_pool_substitution_rate)

        # Pre-roll uniform randoms in chunks for speed; still single-process so
        # this is the parent process's RNG state.
        rolls = rng.random(n_total)
        rolls2 = rng.random(n_total) if use_hard_pool else None

        for i in range(n_total):
            r = rolls[i]
            if r < p_pos and self._pos_idx.size > 0:
                yield int(self._pos_idx[rng.integers(self._pos_idx.size)])
                continue

            # Negative budget. Split into NIPV vs NINV within remaining mass.
            # Avoid a divide-by-zero when p_pos == 1.
            denom = max(1.0 - p_pos, 1e-12)
            r_neg = (r - p_pos) / denom  # in [0, 1) for the negative branch
            pick_nipv = r_neg < nipv_share

            if pick_nipv and not nipv_empty:
                yield int(self._nipv_idx[rng.integers(self._nipv_idx.size)])
                continue
            if not pick_nipv and not ninv_empty:
                # Maybe substitute from hard pool.
                if use_hard_pool and rolls2[i] < sub_rate:
                    yield int(self._hard_pool[rng.integers(self._hard_pool.size)])
                else:
                    yield int(self._ninv_idx[rng.integers(self._ninv_idx.size)])
                continue
            # Fallback: requested pool empty → use the other negative pool, then hard pool.
            if not nipv_empty:
                yield int(self._nipv_idx[rng.integers(self._nipv_idx.size)])
            elif not ninv_empty:
                yield int(self._ninv_idx[rng.integers(self._ninv_idx.size)])
            elif use_hard_pool:
                yield int(self._hard_pool[rng.integers(self._hard_pool.size)])
            else:
                # Final fallback: positives (only reachable if everything else empty).
                yield int(self._pos_idx[rng.integers(self._pos_idx.size)])
