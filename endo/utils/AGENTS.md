# `endo/utils/` — generic helpers (seeding, IO, provenance)

No domain logic. Anything reusable across components that doesn't fit a specific subpackage.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `seeding.py` | `derive_seed(*ints) -> int` — deterministic 64-bit seed derivation via blake2b. Used by `WeightedScheduledSampler` (per-epoch seed) and `TrainAugmentation` (per-sample seed). |
| `io.py` | Atomic JSON write helper. |
| `provenance.py` | `get_git_sha`, `now_iso`, `initial_provenance()` (returns `{git_sha, hostname, platform, python_version, python_executable, started_at, fold_status: {0..4: "pending"}}`), `load_provenance` / `save_provenance` (atomic), `update_fold_status(path, fold, state)` where `state ∈ {"pending", "running", "complete", "failed"}`. |

## Contracts

- **`update_fold_status` is the only writer** of `runs/<exp>/provenance.json[fold_status]`. Concurrent writes (e.g. multi-fold parallelism) are not safe — Phase 4 runs folds sequentially per default.
- **`derive_seed` is order-sensitive**: `derive_seed(42, "pid_x", 7)` is NOT equal to `derive_seed(42, 7, "pid_x")`. Don't reorder args in callers.

## Invariants

- I.8.7 (atomic fold-status updates) is enforced here via temp-file + `os.replace`.

## Don't

- Don't put torch / numpy domain-specific helpers here. If it's tied to `Sample` or `Batch`, it belongs in `endo/data/`. If it's tied to a Lightning callback, it belongs alongside the callback.
