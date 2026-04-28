# Component 7 — Post-Training Evaluation

**Status:** Spec locked, ready for implementation.
**Owner files:** `src/eval/wbf.py`, `src/eval/froc.py`, `src/eval/metrics.py`, `src/eval/threshold_search.py`, `eval.py`, `eval_holdout.py`
**Date:** 2026-04-27
**Companion:** Implements §8, §9 of `agent/training_pipeline_decisions_phase1.md`. Consumes Component 5's deep-eval cache, Component 6's checkpoints, and Component 6.5's GRU rescorers. Final stage before abstract draft.

---

## 1. Purpose

Compute the volume-level metrics that go into the RSNA abstract: volume AUROC, FROC@{0.125, 0.25, 0.5, 1, 2, 4, 8} FP/vol, AP@IoU=0.3, with patient-level bootstrap 95% CIs and scanner/variant/slice-thickness stratified breakdowns. Two entrypoints:

- **`eval.py`** — pooled 5-fold CV evaluation (per fold + cohort-pooled).
- **`eval_holdout.py`** — one-shot 5-model ensemble inference on the 122-patient holdout. Touched **exactly once**.

---

## 2. Scope

**In scope:**

- 3D WBF aggregation of per-slice boxes (`weighted_boxes_fusion_3d` from `ensemble-boxes`).
- FROC via `picai_eval` with bootstrap CIs.
- Volume AUROC + AP@IoU=0.3 + bootstrap CIs.
- Stratified breakdowns: scanner, variant, slice-thickness bin.
- Per-fold and CV-pooled WBF score thresholds via grid search.
- Ensemble inference on holdout: each fold's detector + (optionally) that fold's GRU; aggregated via WBF across all 5 models' boxes.
- CSV-only output (`eval_report.csv`); presentation layer is explicitly out of scope.

**Out of scope:**

- Holdout boundary enforcement at any level above the DataModule guard (Component 3 §11).
- Markdown / LaTeX / figure generation — deferred.
- Re-running training. Component 7 only consumes already-trained artifacts.

---

## 3. Inputs

| Input | Path | Used for |
|---|---|---|
| Detector checkpoints | `runs/baseline_fold{0..4}/ckpts/best.ckpt` | Inference on val (CV) and on holdout (ensemble) |
| GRU checkpoints | `cache/v1/gru_ckpts/fold{0..4}.pt` | Optional rescoring (--use-gru) |
| Deep-eval cache | `cache/v1/runtime/deep_eval/epoch{N}_val.npz` (per fold, latest is best) | Pre-computed per-slice val scores; avoids re-inference for `eval.py` |
| GT boxes | `cache/v1/gt_boxes.parquet` | FROC hit criterion |
| Lesion masks | `cache/v1/volumes/<pid>/lesion_mask.npy` | FROC alternative hit criterion (centroid-in-mask) |
| Manifest | `cache/v1/preprocessed_manifest.csv` | Cohort, fold, scanner, variant, slice-thickness for stratification |
| GRU feature caches | `cache/v1/gru_features/fold{0..4}/` | Used at rescoring time |
| Holdout volumes | `cache/v1/volumes/<pid>/` (cohort=holdout) | Holdout ensemble inference |

---

## 4. Outputs

### 4.1 `eval_report.csv` (single global file written by both entrypoints)

One row per `(metric, scope, fold, stratum, rescored)`:

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | E.g., `cv_2026_04_28_a1b2c3d` (timestamp + git sha) |
| `entrypoint` | enum | `cv` \| `holdout` |
| `metric` | enum | `volume_auroc` \| `sens_at_2fp` \| `cpm` \| `ap_iou_30` \| `sens_at_<X>fp` (for X in {0.125, 0.25, 0.5, 1, 4, 8}) |
| `scope` | enum | `per_fold` \| `cv_pooled` \| `holdout` |
| `fold` | int \| null | 0–4 for `per_fold`; null otherwise |
| `stratum_kind` | enum \| null | `scanner` \| `variant` \| `slice_thickness_bin` \| null (overall) |
| `stratum_value` | str \| null | E.g., `SIGNA Artist`, `A`, `<=2mm` |
| `rescored` | bool | Whether GRU rescoring was applied |
| `value` | float | Point estimate |
| `ci_lower_95` | float | Bootstrap 95% CI lower |
| `ci_upper_95` | float | Bootstrap 95% CI upper |
| `n_patients` | int | Number of patients contributing |
| `n_lesions` | int | Number of GT lesions in scope |
| `code_version` | str | Git SHA at eval time |

Append-only. Each `eval.py` / `eval_holdout.py` run adds rows under a fresh `run_id`. Old runs preserved.

### 4.2 `eval_thresholds.json` (per-run sidecar)

```json
{
  "run_id": "cv_2026_04_28_a1b2c3d",
  "per_fold_thresholds": {"0": {"large": 0.05, "small": 0.30}, "1": {...}, ...},
  "ensemble_threshold": {"large": 0.04, "small": 0.28}
}
```

Used by `eval_holdout.py` to apply the CV-pooled ensemble threshold.

---

## 5. Pipeline

### 5.1 `eval.py` — CV evaluation

```
For each fold f in 0..4:
  1. Load deep_eval cache for fold f (most recent epoch).
     If --use-gru: rescore each slice's boxes via fold-f GRU.
  2. Per-fold WBF threshold grid search on (fold f) val set:
       Sweep large_threshold ∈ {0.01, 0.03, 0.05, 0.10}
       Sweep small_threshold ∈ {0.10, 0.20, 0.30, 0.40, 0.50}
       Score: maximize sens@2FP/vol on the val set.
       Store best (large, small) thresholds.
  3. Apply per-fold threshold + WBF to fold-f val set; produce per-volume box list + scores.
  4. Compute per-fold metrics (volume AUROC, FROC, AP, stratified):
       - picai_eval for FROC + AUROC with patient-level bootstrap (N=fold_val_size, 1000 resamples).
       - sklearn for AP@IoU=0.3.
  5. Append rows to eval_report.csv with scope=per_fold, fold=f.

Pool across folds:
  6. Concatenate all 5 folds' (volume_score, gt_label, lesion_list) tuples.
  7. CV-pooled WBF threshold grid search on the concatenated val set.
  8. Apply CV-pooled threshold to all 5 folds' boxes; recompute pooled metrics.
  9. Append rows with scope=cv_pooled, fold=null.
 10. Per stratum (scanner, variant, slice_thickness_bin):
       - Filter pooled volumes to that stratum
       - Recompute volume AUROC, sens@2FP, AP — append rows with stratum_kind/value.
 11. If --use-gru: repeat steps 1–10 with rescored=true, rows appended separately.
 12. Write eval_thresholds.json.
```

### 5.2 `eval_holdout.py` — one-shot ensemble

```
Precheck: refuse to start unless eval_report.csv has cv_pooled rows AND --i-mean-it flag set.

1. Set DataModule allow_holdout=True (ONLY here).
2. Load all 5 detector checkpoints (best.ckpt) + EMA weights into 5 LesionDetectorLM instances.
3. If --use-gru: load all 5 GRU checkpoints.
4. For each holdout patient (122 total):
     a. Build inference dataloader for this single patient.
     b. For each of the 5 models:
          - Run inference on every valid slice → list of (slice_y, boxes, scores) per model.
          - If --use-gru: extract backbone features per slice → rescore via that model's fold's GRU.
     c. Concatenate boxes across all 5 models (with model_id tag for diagnostics).
     d. Apply ensemble_threshold from eval_thresholds.json.
     e. Run 3D WBF over the concatenated 5-model boxes.
     f. Compute volume_score = max(post_WBF_confidences) (or top-k mean).
5. Compute holdout metrics (volume AUROC, FROC, AP, stratified) with bootstrap CIs.
6. Append rows to eval_report.csv with scope=holdout, fold=null, run_id matching the precheck.
```

The `--i-mean-it` flag is a deliberate gate against accidental holdout touches. The script also writes `cache/v1/runtime/holdout_touched_<run_id>.json` recording the run for audit.

---

## 6. Library choices and primitives

### 6.1 3D WBF (`src/eval/wbf.py`)

```python
from ensemble_boxes import weighted_boxes_fusion_3d

def wbf_aggregate_volume(
    boxes_per_slice: dict[int, dict],   # {slice_y: {boxes, scores}}
    n_slices_total: int,
    iou_thr: float = 0.3,
    skip_box_thr: float = 0.01,
    weights: list[float] | None = None,   # for ensemble: [1, 1, 1, 1, 1]
    large_threshold: float = 0.05,
    small_threshold: float = 0.30,
    box_size_threshold_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (boxes_3d, scores) post-WBF.
       boxes_3d in (x1, y1, z1, x2, y2, z2) with y normalized to [0,1] over n_slices_total."""

    # Build per-source list:
    #   For ensemble: 5 lists (one per model), each containing all that model's boxes across slices.
    #   For single-model: one list.
    # Each box (x1, z1, x2, z2) on slice_y becomes (x1/W, slice_y/n_slices, z1/H, x2/W, (slice_y+1)/n_slices, z2/H).
    # Then call weighted_boxes_fusion_3d.
    # Apply box-size-dependent threshold:
    #   - large boxes: scores ≥ large_threshold pass
    #   - small boxes: scores ≥ small_threshold pass
    # Filter and return.
    ...
```

### 6.2 FROC + AUROC (`src/eval/froc.py`)

```python
from picai_eval import evaluate

def compute_volume_metrics(
    volume_predictions: list[dict],   # {patient_id, volume_score, boxes_post_wbf, scores_post_wbf}
    gt_lesions: list[dict],           # {patient_id, lesion_mask | lesion_boxes}
    fp_per_vol_points: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Returns:
       {
         'volume_auroc': {'value', 'ci_lower', 'ci_upper'},
         'sens_at_2fp': {...},
         'cpm': {...},
         'froc_curve_fp': [...], 'froc_curve_sens': [...],
         'sens_at_each_fp': {0.125: {...}, 0.25: {...}, ...},
       }
       """
    ...
```

`picai_eval.evaluate` handles patient-level bootstrap natively (N resamples = `bootstrap_n`, with-replacement at the patient level).

**Hit criterion:** centroid-in-mask. A predicted box is a true positive if its center voxel falls inside any GT lesion mask voxel for that patient.

### 6.3 AP@IoU=0.3 (`src/eval/metrics.py`)

```python
def ap_at_iou_30(predictions, gts, iou_thr: float = 0.3) -> dict:
    """Per-volume mAP at IoU=0.3 (lower than COCO's 0.5 because lesions are tiny).
       Patient-level bootstrap 95% CI."""
    # Use sklearn.metrics.average_precision_score on the concatenated detection list,
    # filtered by IoU>=iou_thr matching to GT.
    ...
```

### 6.4 Threshold grid search (`src/eval/threshold_search.py`)

```python
def grid_search_wbf_thresholds(
    val_predictions_per_volume: list[dict],
    val_gt_per_volume: list[dict],
    large_threshold_grid: list[float] = [0.01, 0.03, 0.05, 0.10],
    small_threshold_grid: list[float] = [0.10, 0.20, 0.30, 0.40, 0.50],
    target_metric: str = "sens_at_2fp",
) -> dict:
    """Returns the (large_threshold, small_threshold) pair that maximizes target_metric.
       Naive cartesian-product grid search (5×5=25 combos × ~2s each = ~50 s)."""
    ...
```

---

## 7. Stratified breakdowns

For each `(metric, stratum_kind, stratum_value)`:

1. Filter the cv_pooled prediction set to volumes matching `stratum_value`.
2. Recompute the metric on that subset.
3. Bootstrap CI: resample only within the stratum.

Strata enumerated:

- **`scanner`**: `SIGNA Artist`, `SIGNA Explorer`
- **`variant`**: `A`, `B`
- **`slice_thickness_bin`**: `<=2mm` (Variant A native ~1.5 mm reconstruction), `>2mm` (Variant B native 3.6 mm)

Per §8.3 of decision doc, these are the breakdowns radiology reviewers will request.

---

## 8. CLI

```bash
# CV evaluation, with and without GRU rescoring
uv run python eval.py \
    --runs-dir runs/ \
    --cache-root /scratch/.../cache/v1 \
    --use-gru \
    --output-csv eval_report.csv

# Holdout one-shot (single touch)
uv run python eval_holdout.py \
    --runs-dir runs/ \
    --cache-root /scratch/.../cache/v1 \
    --use-gru \
    --output-csv eval_report.csv \
    --thresholds eval_thresholds.json \
    --i-mean-it
```

---

## 9. Test plan

Tests in `tests/eval/`. Run via `uv run pytest tests/eval/`.

### 9.1 Unit tests (synthetic)

| # | Test | Assertion |
|---|---|---|
| E1 | `test_wbf_aggregates_overlapping` | 3 overlapping boxes → 1 fused box; score is weighted mean |
| E2 | `test_wbf_keeps_disjoint` | 2 non-overlapping boxes → 2 boxes returned |
| E3 | `test_wbf_box_size_threshold` | Large box at score 0.06 (above large_thr=0.05) and small box at score 0.06 (below small_thr=0.30) → only large box returned |
| E4 | `test_wbf_3d_z_normalization` | Box on slice 50 of 100 → z=0.5 ± 0.005 in normalized coords |
| E5 | `test_compute_volume_metrics_smoke` | 10 vols (5 pos, 5 neg) with synthetic predictions → returns dict with all expected keys; no NaN |
| E6 | `test_bootstrap_ci_widens_with_fewer_patients` | Compute CI on 50 vs 200 patients (synth); 50-patient CI is wider |
| E7 | `test_ap_iou_30_perfect_predictions` | Predictions match GT exactly → AP = 1.0 |
| E8 | `test_ap_iou_30_no_predictions` | Empty predictions → AP = 0.0 |
| E9 | `test_threshold_grid_search_finds_optimum` | Synthetic dataset where best threshold is known → grid search returns it |
| E10 | `test_stratified_breakdown_filters` | Construct mock pooled set with 60% Artist, 40% Explorer; stratified Artist breakdown uses only Artist patients |

### 9.2 Integration tests

| # | Test | Assertion |
|---|---|---|
| E11 | `test_eval_one_fold_e2e` | Run `eval.py --folds 0` on real fold-0 deep-eval cache; produces eval_report.csv with correct schema; metrics finite |
| E12 | `test_eval_with_and_without_gru` | Same fold; with-gru and without-gru rows both appear; row counts equal |
| E13 | `test_eval_holdout_refuses_without_flag` | Run `eval_holdout.py` without `--i-mean-it`; raises with clear message |
| E14 | `test_eval_holdout_refuses_without_cv_rows` | Run `eval_holdout.py` against empty eval_report.csv; raises (must run CV eval first) |
| E15 | `test_eval_csv_append_only` | Run `eval.py` twice; second run preserves first run's rows; new rows have different run_id |

### 9.3 Acceptance gate

Before declaring Stage 1 complete:

1. All §9.1 unit tests pass.
2. All §9.2 integration tests pass.
3. `eval.py` runs end-to-end on all 5 folds; produces eval_report.csv with cv_pooled rows.
4. `eval.py --use-gru` produces an additional `rescored=true` row set.
5. CV-pooled volume AUROC ≥ 0.80 (per §3 doc target — soft acceptance, not hard).
6. CV-pooled sens@2FP ≥ 0.70 (per §3 doc target — soft).
7. Stratified breakdowns produced for all 3 stratification axes.
8. Bootstrap CIs sensible (not zero-width, not nonsensical).

If §9.3 #5 or #6 fails: investigate per the §13 risk register of the decision doc; do not falsify metrics. Holdout still gets its one shot.

---

## 10. Logging

`eval.log`:
- Per-fold inference time (if re-inferring; else "loaded from cache")
- Per-fold WBF threshold (chosen pair + grid search trace)
- Per-fold metrics with CIs
- CV-pooled metrics with CIs
- Stratified breakdowns

`holdout_touched_<run_id>.json`:
- Timestamp, run_id, eval_report.csv row IDs added
- The 5 detector ckpt SHAs used in the ensemble
- The 5 GRU ckpt SHAs (if --use-gru)

---

## 11. Failure modes

| Failure | Detection | Action |
|---|---|---|
| Deep-eval cache missing for a fold | precheck | Re-run inference using fold's best.ckpt — slower but works. Log to eval.log. |
| picai_eval version drift | import-time | Pin version in pyproject.toml to known-good (>=2.1) |
| Holdout touched twice without manual override | `holdout_touched_*.json` exists | Refuse second run unless `--re-touch-holdout-i-am-sure` flag set (debug only) |
| WBF returns 0 boxes for all volumes | sanity check at end of WBF call | Hard-fail; threshold too high or score distribution issue |
| Volume AUROC = 0.5 ± 0.05 (random) | acceptance gate | Soft-warn; investigate detector training |

---

## 12. Wall-clock

- `eval.py` (CV, all 5 folds, both with and without GRU): ~10 min total. Most time in 1000-resample bootstrap × 7 metric points × 5 folds × 2 modes ≈ 70K metric computations. picai_eval is fast.
- `eval_holdout.py`: ~15 min (5-model ensemble inference on 122 holdout volumes + WBF + bootstrap).

---

## 13. Acceptance checklist (Component 7 done)

- [ ] All `src/eval/*.py` modules + `eval.py` + `eval_holdout.py` exist with the APIs in §6.
- [ ] All §9.1 unit tests pass.
- [ ] All §9.2 integration tests pass.
- [ ] `eval.py` produces `eval_report.csv` with cv_pooled + per_fold + stratified rows.
- [ ] `eval.py --use-gru` adds rescored rows.
- [ ] `eval_holdout.py` refuses without `--i-mean-it` flag (verified).
- [ ] `eval_holdout.py` refuses without prior cv_pooled rows (verified).
- [ ] `holdout_touched_<run_id>.json` written on first holdout run.
- [ ] `eval_thresholds.json` written with per-fold + ensemble thresholds.

When this checklist is green, Component 8 (smoke test + viz) can begin (or it can be implemented in parallel since it only depends on Components 3 + 6).
