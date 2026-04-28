"""Per-(patient_id, slice_y) exponential moving average of training-step
max-detector-score, restricted to negative slices.

PRD invariant I.8.3: only negative slices are tracked. Positive slices are a
no-op so the hard-negative-mining pool is exclusively negatives.
"""

from __future__ import annotations

from typing import Tuple


SliceKey = Tuple[str, int]


class ScoreEMATracker:
    """Cheap, in-memory EMA store, keyed by (patient_id, slice_y).

    The tracker is updated by the LightningModule's ``training_step`` once per
    sample with the batch's max detector score for that slice.

    Memory: ~75K negative slices * ~50 bytes/entry = ~4 MB. Trivial.
    """

    def __init__(self, decay: float = 0.9) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay!r}")
        self.decay = float(decay)
        self._ema: dict[SliceKey, float] = {}

    # ─── core API ─────────────────────────────────────────────────────

    def update(
        self,
        key: SliceKey,
        score: float,
        *,
        is_positive_slice: bool,
    ) -> None:
        """Update EMA for ``key`` with ``score``.

        No-op when ``is_positive_slice`` is True (PRD I.8.3). The first update
        for a key seeds ``ema = score``; subsequent updates apply the standard
        EMA recurrence ``ema_new = decay * ema_old + (1 - decay) * score``.
        """
        if is_positive_slice:
            return
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = float(score)
        else:
            self._ema[key] = self.decay * prev + (1.0 - self.decay) * float(score)

    def top_k(self, k: int = 1000) -> list[SliceKey]:
        """Return the ``k`` keys with highest EMA score (descending)."""
        if k <= 0 or not self._ema:
            return []
        items = sorted(self._ema.items(), key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in items[:k]]

    # ─── dunder / persistence ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ema)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - convenience
        return key in self._ema

    def get(self, key: SliceKey) -> float | None:
        return self._ema.get(key)

    def state_dict(self) -> dict:
        # Stash as a list of [pid, sy, ema] triples so this round-trips through
        # JSON / torch checkpoint dicts cleanly.
        return {
            "decay": self.decay,
            "entries": [
                [pid, int(sy), float(v)] for (pid, sy), v in self._ema.items()
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd.get("decay", self.decay))
        entries = sd.get("entries", [])
        self._ema = {(str(pid), int(sy)): float(v) for pid, sy, v in entries}
