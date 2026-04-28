# Component 4 — Augmentation Stack

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/augmentation.py`
**Date:** 2026-04-27
**Companion:** Implements §6 (all subsections) of `agent/training_pipeline_decisions_phase1.md`. Plugged into Component 3 (Dataset) via the `augment` callable argument.

---

## 1. Purpose

Implement the on-the-fly training augmentation stack: lesion copy-paste (target side), geometric jitter (in-plane affine + elastic), intensity perturbation, and the post-aug pipeline that re-derives boxes and extracts the 5-channel slice triplet. Disabled at validation and inference.

Augmentation is the **single highest-EV component** for closing the gap to RSNA targets given only 86 CV training positives. Get this right.

---

## 2. Scope

**In scope:**

- Lesion paste (target side): paste-site selection from `border_band`, target-local stats, soft-blend compositing, online lesion-mask update.
- Multi-paste schedule: `Bernoulli(0.5)` outer × `HalfGaussian` inner clipped to `[1, 7]` with mode at 1. Non-overlapping site constraint.
- Geometric aug: in-plane (XZ) rotation ±10°, scale 0.9–1.1, translation ±5%, light elastic (σ=2, ~8 control points). Lockstep across `(volume, lesion_mask)`. Coherent across Y slices.
- Intensity aug: γ ∈ [0.8, 1.2], multiplicative bias ∈ [0.9, 1.1], gaussian noise σ=0.01. Volume only.
- Box re-derivation from final augmented `lesion_mask` via `scipy.ndimage.label`.
- 5-channel slice extraction at the sampled center index.
- 4-tier test gate: unit + automated metric tests + agentic visual review + human review.

**Out of scope:**

- Hard-negative mining (Component 5).
- Sampler scheduling (Component 5).
- Validation/inference path (no augmentation; Component 3 short-circuits).

---

## 3. Public API

```python
@dataclass
class PasteConfig:
    p_any_paste: float = 0.5
    n_paste_sigma: float = 1.0           # half-gaussian σ; mode at 1
    n_paste_max: int = 7
    site_local_std_threshold: float = 2.0   # reject sites with local std > 2× cohort-median local std
    cohort_median_local_std: float = ...    # populated at construction from a one-time cache scan
    overlap_buffer_voxels: int = 0          # 0 = strict non-overlap; >0 = enforce gap

@dataclass
class GeometricConfig:
    rotation_deg: float = 10.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    translation_frac: float = 0.05
    elastic_sigma: float = 2.0
    elastic_control_points: int = 8

@dataclass
class IntensityConfig:
    gamma_min: float = 0.8
    gamma_max: float = 1.2
    bias_min: float = 0.9
    bias_max: float = 1.1
    noise_sigma: float = 0.01

class TrainAugmentation:
    """Composable augmentation callable for training.
       Construct once per DataModule; loaded into all dataloader workers via fork."""
    def __init__(
        self,
        lesion_bank: list[LesionBankEntry],
        paste_cfg: PasteConfig,
        geom_cfg: GeometricConfig,
        intensity_cfg: IntensityConfig,
        slice_window: int = 5,
        rng_seed: int | None = None,   # None → derived from torch.utils.data worker seed
    ): ...

    def __call__(self, sample: Sample) -> Sample:
        """Mutates and returns sample. Order: paste → geometric → intensity → re-derive boxes → extract 5ch.
           Sample input: full-cropped (384, 160, 384) volume + lesion_mask.
           Sample output: volume_5ch (5, 384, 384), boxes (N, 4), labels (N,)."""
        ...
```

`TrainAugmentation` is constructed once in `LesionDataModule.setup()` and passed to the train Dataset. Each forked worker sees the same lesion bank via copy-on-write.

---

## 4. Augmentation order (recap from Component 3 §5)

```
input Sample (post Component 3 sub-crop + border-band translation):
    volume_full_cropped: (384, 160, 384) float32 [upcast from fp16 at this point]
    lesion_mask_full_cropped: (384, 160, 384) uint8
    border_band_coords: (M, 3) int16 in cropped coords
    slice_y_target: int       # the sampled center index k (in cropped frame)

Step 3: LesionPaste
        Modifies volume + lesion_mask in place. Adds 0..7 synthetic lesions.

Step 4: GeometricAug
        Applies in-plane affine + elastic to (volume, lesion_mask) lockstep.

Step 5: IntensityAug
        Applies γ, bias, noise to volume only.

Step 6: Re-derive boxes
        scipy.ndimage.label on lesion_mask; extract 2D bboxes at slice_y_target.

Step 7: Extract 5-channel triplet
        Slice volume[:, k-2:k+3, :] → transpose to (5, X, Z) = (5, 384, 384).

output Sample:
    volume_5ch: (5, 384, 384) float32
    lesion_mask_center: (384, 384) uint8
    boxes: (N, 4) float32 — derived from slice_y_target only
    labels: (N,) int64 — all zeros
```

---

## 5. Lesion paste (Step 3) — algorithm

### 5.1 Sample n_pastes

```python
def sample_n_pastes(rng) -> int:
    if rng.random() >= self.paste_cfg.p_any_paste:
        return 0
    x = abs(rng.normal(0, self.paste_cfg.n_paste_sigma))
    n = int(round(x)) + 1
    return min(n, self.paste_cfg.n_paste_max)
```

Distribution at default σ=1.0: P(n=1)≈0.38, P(n=2)≈0.31, P(n=3)≈0.13, P(n=4)≈0.04, P(n=5+)≈0.014. Combined with `p_any_paste=0.5`, expected pastes per sample ≈ 0.93.

### 5.2 Per-paste algorithm (run n_pastes times)

```
For attempt in range(MAX_ATTEMPTS_PER_PASTE = 20):
  1. Pick voxel (x*, y*, z*) uniformly from sample.border_band_coords.
  2. Compute local 3-mm-shell std at (x*, y*, z*) on volume_full_cropped.
     Reject if local_std > paste_cfg.site_local_std_threshold * cohort_median_local_std.
  3. Pick donor LesionBankEntry uniformly from self.lesion_bank.
  4. Translate donor.tight_mask so its centroid_offset_in_tight maps to (x*, y*, z*).
     Result: paste_mask in volume coords (zeros except where donor CC sits).
  5. If paste_mask intersects (sample.lesion_mask_full_cropped > 0)
        OR paste_mask intersects any previously-placed paste_mask in this sample:
     Continue to next attempt.
  6. Compute target-local intensity stats:
     target_shell = binary_dilation(paste_mask, radius_mm=3) AND NOT paste_mask
     target_local_mean = volume_full_cropped[target_shell].mean()
     target_local_std  = volume_full_cropped[target_shell].std()
  7. Rescale donor intensities:
     donor_normed = (donor.tight_intensities - donor.intensity_mean) / donor.intensity_std
     injected = donor_normed * target_local_std + target_local_mean   # shape: tight bbox of donor
  8. Composite (overwrite ONLY lesion voxels):
     volume_full_cropped[paste_mask_full_indices] = injected[donor_local_indices]
  9. Soft-blend at the lesion's outer 1-mm shell:
     translated_shell = translate(donor.tight_shell_mask, target_voxel)
     For voxel v in translated_shell:
        d_outside_mm = distance from v to nearest paste_mask voxel (in mm)
        α(v) = max(0, 1 - d_outside_mm)   # linear ramp 1→0 across the 1 mm shell
        volume_full_cropped[v] = α(v) * injected_at_v + (1 - α(v)) * volume_full_cropped[v]
 10. Update sample.lesion_mask_full_cropped:
     sample.lesion_mask_full_cropped |= paste_mask
 11. Record paste_mask in placed_pastes (for non-overlap check on subsequent attempts).
 12. Break out of attempt loop. SUCCESS.

If MAX_ATTEMPTS_PER_PASTE exhausted with no successful placement: skip this paste, move to next.
```

`MAX_ATTEMPTS_PER_PASTE = 20` is conservative; in practice the right_band has tens of thousands of valid voxels per volume, so success on first try is almost guaranteed.

### 5.3 Out-of-bounds handling

If a translated `paste_mask` extends past the `(384, 160, 384)` frame (donor centroid too close to the edge), it gets clipped. Reject any paste with > 25% clipped voxels (donor CC is mostly outside the volume — bad).

### 5.4 `cohort_median_local_std` (one-time computation)

At first construction of `TrainAugmentation`, run a one-time scan over the cache: for each volume, sample 100 random voxels in the volume's `border_band`; compute the 3-mm-shell std at each; record cohort-wide median. Cache the result to `cache/v1/runtime/cohort_local_std.json` so subsequent constructions skip the scan. If cache exists, load it.

---

## 6. Geometric aug (Step 4) — algorithm

### 6.1 Affine (rotation + scale + translation)

Applied in-plane (XZ), uniformly across all 174 Y slices.

- Rotation θ ∈ Uniform(-10°, +10°) around the Y axis.
- Scale s ∈ Uniform(0.9, 1.1), isotropic in XZ.
- Translation t_x ∈ Uniform(-19.2, +19.2) voxels (5% of 384), t_z same.

Implementation: build a 3×3 affine matrix in (X, Z) coords, then apply via `scipy.ndimage.affine_transform` to each Y slice (or via a single 4×4 affine on the 3D volume with identity in Y — same result, library-dependent).

- `volume`: bilinear interpolation (`order=1`).
- `lesion_mask`: nearest neighbor (`order=0`).

### 6.2 Elastic deformation

In-plane elastic field, coherent across Y slices.

```python
def elastic_field(shape_xz, sigma, n_control_points, rng):
    dx = rng.normal(0, sigma, size=(n_control_points, n_control_points))
    dz = rng.normal(0, sigma, size=(n_control_points, n_control_points))
    # Upsample to full XZ shape via bicubic
    dx_full = ndimage.zoom(dx, [shape_xz[0]/n, shape_xz[1]/n], order=3)
    dz_full = ndimage.zoom(dz, [shape_xz[0]/n, shape_xz[1]/n], order=3)
    return dx_full, dz_full   # shape (X, Z) each
```

Apply same `(dx_full, dz_full)` field to every Y slice via `scipy.ndimage.map_coordinates`. Volume: linear; lesion_mask: nearest.

### 6.3 Combined apply

For implementation efficiency, compose the affine and elastic into a single coordinate map and apply in one `map_coordinates` call per (volume, mask).

### 6.4 Library choice

Spec requires: in-plane only, lockstep across (volume, lesion_mask), coherent across Y. Acceptable implementations:

- Hand-rolled scipy (above).
- MONAI `RandAffine` with `rotate_range=(0, π/18, 0)` + `Rand2DElastic` applied per-Y-slice (must use the same seed across slices for coherence). Awkward; not recommended.
- Kornia (GPU-native, but we're operating pre-collate in numpy). Move to GPU augmentation later if CPU bottleneck.

Default: hand-rolled scipy. Keeps Component 4 self-contained.

---

## 7. Intensity aug (Step 5)

Applied to `volume_full_cropped` only. After geometric aug.

```python
volume = volume * mult_bias              # mult_bias ~ U(0.9, 1.1)
volume = sign(volume) * |volume|^gamma   # gamma ~ U(0.8, 1.2)
volume = volume + rng.normal(0, 0.01, volume.shape)
```

The γ correction is applied with sign preservation since z-scored values include negatives.

---

## 8. Box re-derivation (Step 6)

After all geometric/intensity aug:

```python
labels_3d, n_cc = scipy.ndimage.label(
    sample.lesion_mask_full_cropped,
    structure=np.ones((3, 3, 3))   # use the same connectivity as Component 2
)

boxes = []
for cc_id in range(1, n_cc + 1):
    if not (labels_3d[:, slice_y_target, :] == cc_id).any():
        continue   # CC doesn't intersect target slice
    xs, zs = np.where(labels_3d[:, slice_y_target, :] == cc_id)
    boxes.append([xs.min(), zs.min(), xs.max() + 1, zs.max() + 1])

boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
labels = np.zeros(len(boxes), dtype=np.int64)
```

Connectivity must match Component 2's locked choice (`np.ones((3,3,3))` = 26-connectivity, default).

**Skip boxes with `max_dim < 2 voxels`** — likely artefacts of warped-mask sub-pixel residue. Track skipped count for QC; warn if > 5% of CCs are skipped on average (suggests over-aggressive aug).

---

## 9. 5-channel slice extraction (Step 7)

```python
k = sample.slice_y_target
triplet = sample.volume_full_cropped[:, k-2:k+3, :]   # shape (X, 5, Z)
volume_5ch = triplet.transpose(1, 2, 0).astype(np.float32)   # (5, Z, X) = (5, 384, 384)
lesion_mask_center = sample.lesion_mask_full_cropped[:, k, :].T.astype(np.uint8)  # (Z, X)
```

**Tensor convention at the model boundary:** output is `(5, Z, X)` so PyTorch H/W maps as `H=Z` (axis 2 in cache, anatomical I-S, vertical in coronal view), `W=X` (axis 0 in cache, anatomical R-L, horizontal). This makes our cached box format `(x1, z1, x2, z2)` map directly to PyTorch's `(W_min, H_min, W_max, H_max)` with no permutation needed at the detector boundary. Bonus: a "horizontal flip" in image space then equals R-L flip — exactly the flip we forbid (per §1.3 right-side prior). The 5-channel axis is in position 0 of the per-sample tensor (position 1 after batching).

`lesion_mask_center` is also transposed to `(Z, X)` for consistency.

Box re-derivation in §8 produces boxes already in `(x1, z1, x2, z2)` cache convention — no edit needed there because the model interprets them as `(W, H, W, H)` per the alignment above.

---

## 10. RNG strategy

- Each `TrainAugmentation.__call__` uses a per-call `np.random.Generator` derived from a base seed XOR'd with the worker_id and the sample idx. Ensures per-sample reproducibility AND distinct streams across workers AND distinct streams across epochs (PyTorch DataLoader's `worker_init_fn` increments the seed each epoch).
- Seed source: `torch.utils.data.get_worker_info().seed` + `idx`.

---

## 11. Test plan — 4 tiers

### Tier 1 — automated unit tests (`tests/augmentation/`)

All synthetic, no GPU, < 30 s total.

| # | Test | Assertion |
|---|---|---|
| T1.1 | `test_sample_n_pastes_distribution` | 100K samples; P(0) ≈ 0.5; conditional on >0, mode is 1; max is ≤ 7 |
| T1.2 | `test_sample_n_pastes_seeded_reproducible` | Two RNGs with same seed yield identical sequences |
| T1.3 | `test_paste_site_inside_border_band` | 100 paste attempts; every successful site is in `border_band_coords` |
| T1.4 | `test_paste_no_overlap_with_existing` | Pre-place a lesion; assert no paste lands on it |
| T1.5 | `test_paste_no_overlap_between_pastes` | n_pastes=5; assert no two paste_masks intersect |
| T1.6 | `test_paste_intensity_match_local_stats` | Paste a uniform-intensity donor; assert pasted region's mean ≈ target_local_mean ± 0.1 |
| T1.7 | `test_paste_soft_blend_continuity` | Paste; sample voxels at 0.5 mm outside paste boundary; assert no value change > 1.5σ |
| T1.8 | `test_paste_mask_updated` | Paste; assert `lesion_mask` includes paste_mask voxels |
| T1.9 | `test_paste_clipped_oob_rejected` | Donor centroid placed 1 voxel from edge; assert >25% clip → paste rejected |
| T1.10 | `test_paste_zero_pastes_no_op` | `n_pastes=0`; volume + mask unchanged |
| T1.11 | `test_geometric_lockstep` | Apply identity-near affine; assert `lesion_mask` voxels still align with `volume` foreground |
| T1.12 | `test_geometric_in_plane_only` | Apply non-trivial rotation; assert no voxel moved across Y axis |
| T1.13 | `test_geometric_y_coherent` | Apply elastic; assert displacement field at slice y=10 == field at slice y=100 |
| T1.14 | `test_intensity_only_volume` | Apply intensity aug; assert `lesion_mask` unchanged |
| T1.15 | `test_intensity_gamma_sign_preserved` | Volume with negative values + γ=0.8; assert no NaN, sign preserved |
| T1.16 | `test_box_rederivation_matches_mask` | Synthetic post-aug mask with known CCs at known positions; derived boxes match |
| T1.17 | `test_box_skip_subpixel_artifacts` | Mask with 1-voxel-wide CC; assert dropped from box list with warning logged |
| T1.18 | `test_5ch_slice_extraction_shape` | Sample at k=80; assert `volume_5ch.shape == (5, 384, 384)` |
| T1.19 | `test_5ch_center_channel_alignment` | `volume_5ch[2]` equals `volume_full_cropped[:, k, :]` |
| T1.20 | `test_full_pipeline_smoke` | Run `TrainAugmentation(...)` on a synthetic sample; output shapes and dtypes correct |

### Tier 2 — automated metric tests on real composites (`tests/augmentation/test_real_composites.py`)

Loads real cache + real lesion bank. Generates 100 paste composites with fixed seed.

| # | Test | Assertion |
|---|---|---|
| T2.1 | `test_paste_centroid_near_liver_border` | ≥ 95% of paste centroids within 3 mm (in physical space) of a true `border_band` voxel |
| T2.2 | `test_paste_intensity_distribution_realistic` | Histogram of post-paste lesion-region intensities vs histogram of native-lesion intensities; KS test p > 0.05 |
| T2.3 | `test_paste_no_step_at_boundary` | At paste boundary, mean |intensity_jump| over a 1-voxel shell < 1.5 × within-paste σ |
| T2.4 | `test_paste_right_side_only` | All paste centroids have x > liver_centroid_x for each volume |
| T2.5 | `test_no_paste_outside_volume_bounds` | All updated `lesion_mask` voxels in `[0, 384) × [0, 160) × [0, 384)` |

### Tier 3 — agentic visual review gate (`scripts/qc_paste_agentic_review.py`)

Run-once before training begins. Generates 30 paste composites + invokes Claude API as the visual-judgment subagent.

```
1. Sample 30 (target_volume, n_pastes, paste_seeds) tuples — stratified across:
     - target volume scanner (Artist vs Explorer)
     - target volume variant (A vs B)
     - n_pastes ∈ {1, 2, 3, 5}
     - target volume label (negative vs positive)
2. For each: produce the post-paste volume + lesion_mask.
3. Render a triptych PNG per composite:
     - Center slice y=k_paste with paste overlay (red contour) on original volume (grayscale)
     - The 5-channel context (5 strips: k-2..k+2)
     - A coronal MIP showing all paste sites in 3D context
4. For each PNG, send to Claude API (Sonnet 4.6 or higher, vision-capable) with the rubric below.
5. Parse JSON response per composite. Aggregate.
6. Write QC manifest cache/v1/runtime/qc_paste_review.json with per-composite scores + flagged issues.
7. Render single HTML page cache/v1/runtime/qc_paste_review.html with all 30 composites + scores.
8. Acceptance gate: ≥ 80% of composites must score "PASS" (overall_score ≥ 3 AND no critical_issue flag).
```

Subagent rubric (sent as prompt with each PNG):

```
You are a radiologist's assistant reviewing synthetic diaphragmatic-endometriosis lesion paste augmentations for an MRI dataset. Score this composite on a 1–5 scale across four axes:

1. Anatomical plausibility (is the paste location consistent with diaphragmatic-endometriosis anatomy — right hemidiaphragm, near liver dome / border)?
2. Intensity realism (does the paste blend naturally — no obvious step at boundary, no implausible brightness)?
3. Through-plane coherence (do the 5 context slices show a believable 3D lesion shape, not a single flat insertion)?
4. Edge artifacts (any visible boundary halos, copy-paste seams, geometry breaks)?

Return STRICT JSON:
{
  "anatomical_plausibility": 1-5,
  "intensity_realism": 1-5,
  "through_plane_coherence": 1-5,
  "edge_artifacts": 1-5,
  "overall_score": 1-5,
  "critical_issue": true/false,
  "notes": "<one sentence>"
}

A "critical_issue" flag means: the composite would mislead a real radiologist, OR the paste lands somewhere clearly non-diaphragmatic (e.g., deep liver parenchyma, lung air).
```

Implementation notes:

- API key via `ANTHROPIC_API_KEY` env var (required).
- Cost budget: ~$0.30 total (30 calls × $0.01).
- Network requirement: HPC node must have outbound HTTPS to api.anthropic.com.
- If network unavailable: degrade to **Tier 3 stub**: render the 30 composites + write a placeholder JSON marking all "REQUIRES_HUMAN_REVIEW", and surface this prominently. Tier 4 still runs; treat Tier 3 as advisory.

### Tier 4 — human review gate

After Tier 3 completes, you (the human) open `cache/v1/runtime/qc_paste_review.html` and:

- Visually inspect all 30 composites alongside the agentic scores.
- Sign off by writing a row to `cache/v1/runtime/qc_human_signoff.json`:
  ```json
  {
    "reviewer": "Sameed Khan",
    "signoff_timestamp": "...",
    "signoff_status": "APPROVED" | "BLOCKED",
    "review_notes": "..."
  }
  ```
- If `BLOCKED`: file specific issues; engineering revisits paste algorithm before training begins.

The training entrypoint (`train.py`) checks for `qc_human_signoff.json` with `signoff_status == APPROVED` AND a freshness check (signoff timestamp newer than `qc_paste_review.json`). If absent or stale, training refuses to start with a clear message.

---

## 12. Acceptance gate (all 4 tiers)

Before Component 5 begins:

1. All Tier 1 unit tests pass.
2. All Tier 2 metric tests pass on real cache.
3. Tier 3 agentic review runs to completion; ≥ 80% composites scored PASS.
4. Tier 4 human signoff present and APPROVED.
5. `TrainAugmentation` instantiable from `src/augmentation.py` and integrates with `LesionDataModule` via the `augment_train` argument.

---

## 13. Logging

Per-batch (debug-level): n_pastes per sample, paste site coords, target-local stats, retries.
Per-epoch (info-level): mean n_pastes, paste-success rate, mean retries, mean post-aug box count.

---

## 14. Failure modes

| Failure | Detection | Action |
|---|---|---|
| Paste retry exhaustion (MAX_ATTEMPTS) common | per-epoch retry rate > 10% | Investigate border_band size or local-std threshold |
| Box re-derivation drops > 5% of CCs on average | Tier 1 / per-epoch metric | Geometric aug too aggressive; reduce ranges |
| Tier 3 API call fails | exception in qc script | Degrade to stub mode, surface for human review |
| Human signoff missing at training start | `train.py` precheck | Refuse to start; print path to QC HTML |
| Soft-blend creates NaNs in volume | per-batch sentinel | Hard-fail; investigate distance transform anisotropy |

---

## 15. Wall-clock budget

- Per-sample augmentation: < 50 ms target on CPU. Profile after first integration.
- Tier 1 tests: < 30 s.
- Tier 2 tests: < 2 min (loads real cache; runs 100 composites).
- Tier 3 review: < 10 min wall-clock (30 composite renders + 30 API calls).
- Cohort `cohort_median_local_std` computation (one-time at construction): < 60 s; cached afterward.

---

## 16. Acceptance checklist (Component 4 done)

- [ ] `src/augmentation.py` exists with the API in §3.
- [ ] All Tier 1 unit tests pass.
- [ ] All Tier 2 metric tests pass on real cache.
- [ ] Tier 3 agentic review runs end-to-end with ≥ 80% PASS.
- [ ] Tier 4 human signoff workflow tested (block + approve paths).
- [ ] `train.py` refuses to start without valid signoff (precheck verified).
- [ ] DataModule + augmentation integration test passes (one batch with non-zero pastes, valid output shapes).

When this checklist is green, Component 5 (Sampler + Hard-Negative Mining) can begin.
