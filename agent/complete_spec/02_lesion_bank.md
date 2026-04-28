# Component 2 — Lesion Bank Builder

**Status:** Spec locked, ready for implementation.
**Owner script:** `scripts/build_lesion_bank.py`
**Date:** 2026-04-27
**Companion:** Implements §6.3.1 of `agent/training_pipeline_decisions_phase1.md`. Consumed by Component 4 (Augmentation) at training time.

---

## 1. Purpose

Build a **single global, read-only lesion bank** that contains the donor-side payload for the lesion copy-paste augmentation (§6.3 of the decision doc). The bank is constructed once after Component 1 completes, holds the lesion morphology + intensity statistics for every connected component (CC) across all CV positives, and is loaded into RAM by every dataloader worker at training time.

The bank contains **only the donor side** of the copy-paste operation. Target-side work (paste-site selection, target-local stats, soft-blend compositing) lives in Component 4.

---

## 2. Scope

**In scope:**

- Read cached `volume.npy` and `lesion_mask.npy` for every patient with `cohort == cross-validation` AND `label == positive` from `preprocessed_manifest.csv`. **Holdout positives never enter the bank.**
- Run `scipy.ndimage.label` (3D, 6-connectivity) on each lesion mask to extract CCs.
- For each CC, build a tight-bounding-box payload with mask, intensities, 1 mm outer shell, centroid, intensity stats, and physical extent.
- Serialize all CC entries to a single pickle file with a code-version-tagged filename.
- Emit a provenance JSON tracking inputs, counts, and a content hash.
- Be idempotent and parallel.

**Out of scope:**

- Per-fold filtering. The bank is global; per-fold training reads the same file.
- Target-side compositing, paste-site selection, intensity rescaling at paste time. → Component 4.
- Holdout positives. → Never included; protected by `cohort != 'holdout'` filter at load time.

### Design note — deliberate validation leak

A patient in fold `f`'s validation set (e.g., `cohort=cross-validation`, `fold=f`, `label=positive`) has their CCs in the global bank. During fold `f`'s training, those CCs can be pasted into *other* patients' volumes as synthetic positives. The patient's own *image* is never seen by the model during training — only their lesion shape and intensity profile. This is an accepted leak in exchange for maximum donor diversity (~155 CCs vs ~124 per-fold). It does NOT compromise the holdout boundary — the 22 holdout positives are excluded from the bank by the cohort filter.

---

## 3. Inputs

| Input | Path | Used for |
|---|---|---|
| Preprocessed cache (volumes + lesion masks) | `cache/v1/volumes/<patient_id>/...` | CC extraction + intensity sampling |
| Preprocessed manifest | `cache/v1/preprocessed_manifest.csv` | Patient list with cohort, label, fold |
| Code version | `cache/v1/code_version.txt` | Cache-key participation |

---

## 4. Outputs (downstream contract)

```
cache/v1/lesion_banks/
├── lesion_bank_<git_sha8>.pkl           # The bank itself
└── bank_provenance.json                 # Build metadata
```

`<git_sha8>` is the first 8 chars of `git rev-parse HEAD` at build time. A new code revision produces a new pkl filename — stale banks cannot be silently loaded.

### 4.1 Bank file schema

The pkl file contains a single Python list `entries: list[LesionBankEntry]`:

```python
@dataclass(frozen=True)
class LesionBankEntry:
    donor_patient_id: str                  # FK to preprocessed_manifest.csv
    donor_cc_id: int                       # 1..n_cc within donor (matches gt_boxes.parquet cc_id)
    tight_mask: np.ndarray                 # uint8  (Δx, Δy, Δz) — CC foreground (1) elsewhere (0)
    tight_intensities: np.ndarray          # float32 (Δx, Δy, Δz) — post-z-score values inside CC; 0 outside
    tight_shell_mask: np.ndarray           # uint8  (Δx, Δy, Δz) — 1 mm anisotropic outer dilation, exclusive of CC
    centroid_offset_in_tight: tuple[int, int, int]  # CC centroid expressed in tight-bbox local coords
    z_extent_voxels: int                   # span along axis 1 (slice axis); 1..maximum from §1.3
    intensity_mean: float                  # mean of post-z-score values inside CC
    intensity_std: float                   # std of post-z-score values inside CC
    physical_extent_mm: tuple[float, float, float]  # (Δx*0.82, Δy*1.5, Δz*0.82); QC field
```

All array dtypes are fixed; downstream code can rely on them.

### 4.2 `bank_provenance.json` schema

```json
{
  "build_timestamp": "2026-04-27T12:34:56Z",
  "git_sha": "<full SHA>",
  "git_sha8": "<8-char>",
  "bank_filename": "lesion_bank_<git_sha8>.pkl",
  "cache_version": "v1",
  "preprocessed_code_version": "<from cache/v1/code_version.txt>",
  "n_donor_patients": 86,
  "n_ccs": 157,
  "cohort_filter": "cohort=cross-validation AND label=positive",
  "donor_patient_ids": ["...", "..."],
  "bank_sha256": "<sha256 of the .pkl file>",
  "build_seconds": 12.4
}
```

---

## 5. Pipeline

```
build_lesion_bank(cfg) →
    1. Load preprocessed_manifest.csv, filter to cohort='cross-validation' AND label='positive'
    2. Verify 86 patients selected; assert holdout positives excluded
    3. Idempotency check: compute cache_key, if matches existing bank_provenance.json record → exit
    4. multiprocessing.Pool(workers): for each donor patient, build per-patient CC list
    5. Concatenate all CC lists → entries (preserves donor diversity)
    6. Pickle entries to cache/v1/lesion_banks/lesion_bank_<git_sha8>.pkl
    7. Compute bank_sha256, write bank_provenance.json
```

### 5.1 Per-patient CC extraction

```python
def extract_ccs_for_donor(patient_id: str, cache_root: Path) -> list[LesionBankEntry]:
    volume = np.load(cache_root / "volumes" / patient_id / "volume.npy", mmap_mode="r")
    lesion_mask = np.load(cache_root / "volumes" / patient_id / "lesion_mask.npy", mmap_mode="r")

    cc_labels, n_cc = scipy.ndimage.label(lesion_mask, structure=np.ones((3, 3, 3)))  # 26-connectivity
    # NOTE: §1.3 reports 197 CCs across 108 patients; use whichever connectivity reproduces that count.
    # Default to 26-connectivity unless EDA reproduction reveals 6-connectivity was used.

    entries = []
    for cc_id in range(1, n_cc + 1):
        bbox = scipy.ndimage.find_objects(cc_labels == cc_id)[0]  # tight bbox
        cc_mask_full = (cc_labels == cc_id).astype(np.uint8)
        tight_mask = cc_mask_full[bbox]
        tight_intensities = volume[bbox].astype(np.float32) * tight_mask  # zero outside CC

        # 1 mm outer shell, anisotropic; exclusive of the CC itself
        # Compute on a slightly padded tight bbox so the shell isn't truncated at edges
        pad = (2, 1, 2)  # ~1.6 mm × 1.5 mm × 1.6 mm of pad — covers a 1 mm dilation
        padded_mask = np.pad(tight_mask, [(p, p) for p in pad])
        dist_outside = scipy.ndimage.distance_transform_edt(
            ~padded_mask.astype(bool),
            sampling=(0.82, 1.5, 0.82),
        )
        shell_padded = ((dist_outside > 0) & (dist_outside <= 1.0)).astype(np.uint8)
        # Crop shell back to the same shape as tight_mask
        shell_tight = shell_padded[pad[0]:-pad[0], pad[1]:-pad[1], pad[2]:-pad[2]]

        # CC voxels for stats
        cc_vals = volume[bbox][tight_mask.astype(bool)].astype(np.float32)
        intensity_mean = float(cc_vals.mean())
        intensity_std = float(cc_vals.std())

        # Centroid in tight-bbox local coords
        coords = np.argwhere(tight_mask.astype(bool))
        centroid = tuple(int(round(c)) for c in coords.mean(axis=0))

        z_extent_voxels = int(tight_mask.any(axis=(0, 2)).sum())
        physical_extent_mm = (
            tight_mask.shape[0] * 0.82,
            tight_mask.shape[1] * 1.5,
            tight_mask.shape[2] * 0.82,
        )

        entries.append(LesionBankEntry(
            donor_patient_id=patient_id,
            donor_cc_id=cc_id,
            tight_mask=tight_mask,
            tight_intensities=tight_intensities,
            tight_shell_mask=shell_tight,
            centroid_offset_in_tight=centroid,
            z_extent_voxels=z_extent_voxels,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            physical_extent_mm=physical_extent_mm,
        ))

    return entries
```

### 5.2 CC-connectivity reproduction note

§1.3 reports exactly 197 CCs across 108 positives. The bank build must reproduce this count over all 108 positives (CV + holdout). Run a one-time check during bank build:

- Run `scipy.ndimage.label` with both 6- and 26-connectivity over all 108 positives.
- Whichever produces 197 (matching §1.3) is the connectivity to use.
- Hardcode the choice; assert at build time.

---

## 6. CLI & invocation

```bash
uv run python scripts/build_lesion_bank.py \
    --cache-root /scratch/pioneer/users/sak185/diaphragmatic-endometriosis/cache/v1 \
    --workers 8 \
    [--force]            # ignore idempotency cache
    [--dry-run]          # print plan, write nothing
```

No fold flag; the bank is global. No spacing/shape flags; those are pinned by the cache version.

---

## 7. Idempotency

Cache key:

```python
cache_key = sha256(
    sorted(donor_patient_id) +
    sorted(volume.npy sha256 for each donor) +
    code_version
).hexdigest()
```

If `bank_provenance.json` records a row with this `cache_key` AND the `bank_filename` exists, skip. Else build.

`--force` overrides.

---

## 8. Parallelism

`multiprocessing.Pool(processes=8)` over donor patients. Each worker reads its donor's `volume.npy` and `lesion_mask.npy` via mmap, returns a list of `LesionBankEntry`. Parent concatenates and pickles. Memory budget per worker is bounded by one donor volume at a time (~120 MB at fp16 for the 408×174×408 cache shape) — trivial.

---

## 9. Test plan

Tests live in `tests/lesion_bank/`. Run via `uv run pytest tests/lesion_bank/`.

### 9.1 Unit tests (synthetic)

| Test | Setup | Assertion |
|---|---|---|
| `test_extract_single_cc_shape` | Synthetic volume + lesion_mask = single 4×3×4 block at known coords | Returns 1 entry; `tight_mask.shape == (4, 3, 4)`; `physical_extent_mm == (3.28, 4.5, 3.28)` |
| `test_extract_disjoint_ccs` | Two non-touching blocks of shapes (3,2,3) and (2,2,2) | Returns 2 entries with correct shapes |
| `test_centroid_offset_in_tight` | Block 4×3×4, fully filled | `centroid_offset_in_tight == (1 or 2, 1, 1 or 2)` (allow integer rounding) |
| `test_intensity_stats_correctness` | Volume = constant 1.5 inside CC, 0 outside | `intensity_mean == 1.5`, `intensity_std == 0.0` |
| `test_shell_excludes_cc` | Single voxel CC at center of tight bbox | `tight_shell_mask & tight_mask` is all-zero |
| `test_shell_thickness_anisotropic` | Single voxel CC | Shell thickness in voxels matches anisotropic 1 mm: ~1 vox in X/Z (0.82 mm), ~0 vox in Y (1.5 mm > 1 mm) — verify shell exists in X/Z but is thinner in Y |
| `test_intensities_outside_cc_zero` | Filled CC + non-zero intensities elsewhere | `tight_intensities` is non-zero only where `tight_mask == 1` |
| `test_z_extent_correct` | CC spanning slices y=2..5 (4 slices) | `z_extent_voxels == 4` |
| `test_idempotency_skip` | Build once, then again with same code/data | Second call exits without re-pickling |

### 9.2 Integration test (real cache, 1 fold-0 training donor)

| Test | Assertion |
|---|---|
| `test_real_one_donor_extracts` | Build bank from a single positive patient; `len(entries) == n_lesion_ccs from preprocessed_manifest.csv for that patient` |
| `test_real_intensity_stats_finite` | All entries have finite, non-NaN `intensity_mean` and `intensity_std > 0` |
| `test_real_tight_mask_nonempty` | `tight_mask.sum() > 0` for all entries |

### 9.3 Cohort-level test (full bank build)

Acceptance gate before moving to Component 3:

1. **Donor count:** exactly 86 (matches CV positives in §1.1).
2. **CC count:** total entries in `[140, 180]` (point estimate ~157 = 86/108 × 197; allow ±15% for connectivity-rule sensitivity).
3. **Connectivity reproducibility check** (§5.2) executed and connectivity hardcoded.
4. **No empty entries:** every `tight_mask.sum() > 0`, `intensity_std > 0`.
5. **Holdout protection:** `donor_patient_ids` set has zero intersection with `manifest[cohort=='holdout'].patient_id`.
6. **Provenance complete:** `bank_provenance.json` valid, `bank_sha256` present.
7. **Idempotency:** re-run with no changes is a no-op.

---

## 10. Logging

Per-donor: patient_id, n_cc extracted, per-CC physical extents.
Cohort summary: total CCs, distribution of CC physical extents (quantiles), build wall-clock, bank file size, `bank_sha256`.

---

## 11. Failure modes

| Failure | Detection | Action |
|---|---|---|
| 0 entries for a donor with `n_lesion_ccs > 0` in manifest | per-donor count mismatch | Hard-fail; investigate cache integrity |
| Holdout patient appears in entries | §9.3 #5 | Hard-fail; investigate cohort filter |
| `intensity_std == 0` for a non-degenerate CC | §9.3 #4 | Hard-fail; investigate cached volume |
| `n_cc == 0` for a positive patient | per-donor zero | Hard-fail; investigate lesion mask |
| Connectivity mismatch with §1.3 (197) | §5.2 | Investigate source of EDA count; resolve before locking connectivity |

---

## 12. Estimated wall-clock

- Per-donor: ~50 ms (mmap-read + 1–3 CC label + per-CC mask + shell + stats).
- 86 donors with 8 workers: ~1 s wall-clock.
- Test suite: < 30 s total.

---

## 13. Acceptance checklist (Component 2 done)

- [ ] `scripts/build_lesion_bank.py` exists with the CLI in §6.
- [ ] All §9.1 unit tests pass.
- [ ] §9.2 integration test passes on a real donor.
- [ ] Full bank build satisfies all §9.3 gates.
- [ ] `LesionBankEntry` dataclass importable from `src/lesion_bank.py` for downstream use.
- [ ] Re-running is a no-op (§7 idempotency).

When this checklist is green, Component 3 (Dataset + DataModule) can begin.
