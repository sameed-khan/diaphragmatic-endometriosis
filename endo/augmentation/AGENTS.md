# `endo/augmentation/` — online training augmentation

Implements Component 4 (`agent/complete_spec/04_augmentation.md`) plus PRD I.8.8 (the `(B, 5, Z=384, X=384)` shape contract). All scipy.ndimage; no MONAI (A.7).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `TrainAugmentation` and the per-stage callables. |
| `paste.py` | `sample_n_pastes(rng, p_any_paste, n_paste_sigma, n_paste_max)` (clipped half-Gaussian; `P(n=0) = 1 - p_any_paste`). `select_paste_site(border_band_coords, occupancy_mask, donor_extent, rng)` returns a center voxel from the right-hemidiaphragm 2 mm shell that fits the donor + avoids existing lesions (50 attempts, else `None`). `apply_paste(volume, lesion_mask, donor_entry, site, paste_local_intensity_stats)` rescales donor intensity to match the local cohort std and soft-blends through the donor's 1 mm shell. `multi_paste_volume(...)` orchestrates `n` pastes with collision avoidance. |
| `geometric.py` | `random_affine_2d` + `apply_affine_lockstep` (in-plane only — X/Z, NOT Y). `random_elastic_2d(rng, alpha, sigma, shape)` produces a single 2D `(2, Z, X)` displacement field; `apply_elastic_lockstep` reuses it for every Y slice (T1.13 Y-coherent invariant). `geometric_aug` composes affine + elastic. |
| `intensity.py` | `random_brightness_contrast`, sign-preserving `random_gamma`, `random_gaussian_noise`, and the `intensity_aug` composer. Volume only — masks pass through. |
| `boxes.py` | `read_connectivity(cache_root)` — reads `cache/v1/runtime/connectivity_lock.json`, defaults to 26 with a WARN per A.3. `derive_boxes_from_mask` (per-slice 2D), `derive_all_boxes` (3D label → per-Y bboxes), `clamp_box_to_frame` (drops 1-voxel sub-pixel CCs). |
| `transform.py` | `TrainAugmentation` callable — the production composition. Pipeline: paste → geometric → intensity → re-derive boxes (center slice only) → 5-channel `(C=5, Z=384, X=384)` extraction with the spec §9 transpose. Per-call seeded by `sha256(rng_seed, patient_id, slice_y)`. Lazy-builds `cache/v1/runtime/cohort_local_std.json` on first construction (uses 3×3×1 box-stddev on CV-negative border-band voxels). |

## Contracts

- **Input/output**: `__call__(sample: Sample) -> Sample`. Mutates the sample in place semantically: writes back `volume_5ch`, `lesion_mask_center`, `boxes`, `labels`. Sets `volume_full_cropped`, `lesion_mask_full_cropped`, `border_band_coords` to `None` after consumption (they're only forwarded to the augmentation pipeline at training time).
- **Coordinate frame** (PRD I.8.8): boxes are `(x1, z1, x2, z2)` where `x ∈ [0, 384)` is the model's W axis and `z ∈ [0, 384)` is the H axis. The 5-channel extraction at `slice_y` is `tensor[c, z_pixel, x_pixel] = volume_full_cropped[x_pixel, slice_y - 2 + c, z_pixel]`. Channel 2 = center slice.
- **Connectivity contract (A.3)**: box re-derivation MUST use the connectivity locked in `cache/v1/runtime/connectivity_lock.json`. If absent, default 26 with a warning — but in production preprocessing always writes the lock file.
- **Bank contract**: Loads `current.pkl` (or the path in `experiment.paths.lesion_bank` when set). If the bank is missing at construction time, paste is silently disabled (warn + skip) so smoke and pre-bank CLI runs proceed.
- **Determinism**: `__call__` is deterministic given `(rng_seed, patient_id, slice_y)`. Don't introduce hidden global RNG state.

## Invariants checked by tests

T1.1-T1.7 (paste counts, sites, no overlap, intensity match, soft-blend continuity), T1.11-T1.13 (geometric lockstep, in-plane-only, Y-coherent elastic), T1.16-T1.19 (box re-derivation matches mask, sub-pixel CC drop, 5-ch shape, channel-2 alignment), T2.1/T2.4/T2.5 (paste centroid near border-band, right-side only, in-volume bounds).

## Don't

- Don't move geometric augmentations across the Y axis (breaks T1.12).
- Don't replace the cached `cohort_local_std.json` with a per-batch recompute — the constant is part of the cache contract (PRD §5.2.6).
- Don't alter the 5-channel shape contract — `LesionDetector` and the FPN are wired to `(B, 5, 384, 384)`.
