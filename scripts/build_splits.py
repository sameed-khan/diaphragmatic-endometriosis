"""Stage 3a: build frozen holdout + 5-fold CV splits for the cohort.

Pool design (post-exclusions):
  - 108 true positives (have an existing radiologist mask file)
  - 4,890 true negatives (canonical WATER series present)
  -    57 soft negatives (workplan reclassified mask-less ex-positives)
  -    29 FAT-only patients (no WATER canonical) — assigned phase2_unsupervised
        directly; not splits-eligible.

Stratification keys:
  - Positives: manufacturer_model_name only (thickness bin collapsed because
    Explorer-thick has only 6 patients and Artist-thick has 0).
  - Negatives: manufacturer_model_name x slice_thickness_bin (<=4mm vs >4mm)
    on the canonical sequence. Best-effort proportional rounding (largest
    remainder); tiny strata may end up with 0 in some folds.

Splits are deterministic given seed=42. Re-running produces identical output.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


SEED = 42
N_FOLDS = 5
HOLDOUT_POS = 22
HOLDOUT_NEG = 100
PHASE1_NEG_TOTAL = 500
CV_POS = 86
CV_NEG = PHASE1_NEG_TOTAL - HOLDOUT_NEG  # 400
THICKNESS_THRESHOLD_MM = 4.0


def proportional_alloc(stratum_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Allocate `target` items across strata proportional to sizes.
    Largest-remainder rounding so allocations sum to exactly `target`,
    capped per-stratum at the stratum's size."""
    total = sum(stratum_sizes.values())
    if total == 0 or target == 0:
        return {k: 0 for k in stratum_sizes}
    raw = {k: target * v / total for k, v in stratum_sizes.items()}
    floors = {k: min(int(v), stratum_sizes[k]) for k, v in raw.items()}
    remaining = target - sum(floors.values())
    fracs = sorted(((raw[k] - floors[k], k) for k in raw), key=lambda x: (-x[0], x[1]))
    i = 0
    while remaining > 0 and i < len(fracs) * 4:
        _, k = fracs[i % len(fracs)]
        if floors[k] < stratum_sizes[k]:
            floors[k] += 1
            remaining -= 1
        i += 1
    return floors


def stratified_split(
    pid_by_stratum: dict[str, list[str]],
    holdout: int,
    cv: int,
    n_folds: int,
    rng: np.random.Generator,
    leftover_label: str,
) -> dict[str, str]:
    """Per-stratum: shuffle, slice off `holdout` then `cv`, distribute the cv
    slice round-robin across n_folds. The remainder gets `leftover_label`."""
    sizes = {k: len(v) for k, v in pid_by_stratum.items()}
    holdout_alloc = proportional_alloc(sizes, holdout)
    remaining_sizes = {k: sizes[k] - holdout_alloc[k] for k in sizes}
    cv_alloc = proportional_alloc(remaining_sizes, cv)

    out: dict[str, str] = {}
    # Process strata in deterministic order for reproducibility
    for k in sorted(pid_by_stratum.keys()):
        pids = sorted(pid_by_stratum[k])
        idx = np.arange(len(pids))
        rng.shuffle(idx)
        shuffled = [pids[i] for i in idx]
        h, c = holdout_alloc[k], cv_alloc[k]
        for p in shuffled[:h]:
            out[p] = "holdout"
        # Round-robin CV slice into folds. Within a stratum, cycle 0..n_folds-1
        # so per-fold counts within-stratum differ by at most 1.
        for i, p in enumerate(shuffled[h:h + c]):
            out[p] = f"fold{i % n_folds}"
        for p in shuffled[h + c:]:
            out[p] = leftover_label
    return out


def thickness_bin(t: float | None) -> str:
    if t is None:
        return "unknown"
    return "thin" if t <= THICKNESS_THRESHOLD_MM else "thick"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--pre-scan-index", type=Path, required=True)
    ap.add_argument("--alignment-audit", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--holdout-pos", type=int, default=HOLDOUT_POS)
    ap.add_argument("--holdout-neg", type=int, default=HOLDOUT_NEG)
    ap.add_argument("--phase1-neg", type=int, default=PHASE1_NEG_TOTAL)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    cv_pos = 108 - args.holdout_pos  # all positives go in CV+holdout
    cv_neg = args.phase1_neg - args.holdout_neg

    workplan = pl.read_csv(args.workplan, infer_schema_length=10000)
    prescan = pl.read_csv(args.pre_scan_index, infer_schema_length=10000)

    # Per-patient canonical-series stratification metadata.
    canon = (workplan.filter(pl.col("role") == "canonical")
             .select(["cohort", "patient_id", "source_series_path",
                      "soft_negative"]))
    canon = canon.join(
        prescan.select(["series_path", "manufacturer_model_name",
                        "slice_thickness_mm"]),
        left_on="source_series_path", right_on="series_path", how="left",
    ).with_columns(
        pl.col("slice_thickness_mm").map_elements(
            thickness_bin, return_dtype=pl.String).alias("thickness_bin"),
    )

    # Patients with WATER canonical: split-eligible. FAT-only patients (no
    # canonical row) get phase2_unsupervised directly.
    canon_pids = set(canon["patient_id"].to_list())
    all_pids = set(workplan["patient_id"].to_list())
    no_canonical_pids = sorted(all_pids - canon_pids)

    # Positives pool (true positives, 108)
    pos_rows = canon.filter(pl.col("cohort") == "pos")
    pos_by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in pos_rows.iter_rows(named=True):
        # Collapsed thickness bin: stratify positives by model only.
        key = f"model={r['manufacturer_model_name']}"
        pos_by_stratum[key].append(r["patient_id"])

    # Negatives pool (true neg + soft neg, 4,890+57 = 4,947)
    neg_rows = canon.filter(pl.col("cohort") == "neg")
    neg_by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in neg_rows.iter_rows(named=True):
        key = (f"model={r['manufacturer_model_name']}"
               f"|thick={r['thickness_bin']}")
        neg_by_stratum[key].append(r["patient_id"])

    rng_pos = np.random.default_rng(args.seed)
    rng_neg = np.random.default_rng(args.seed + 1)  # independent draw stream

    pos_assign = stratified_split(
        pos_by_stratum, args.holdout_pos, cv_pos, args.n_folds, rng_pos,
        leftover_label="phase2_unsupervised",  # shouldn't trigger for pos (108 = 22+86)
    )
    neg_assign = stratified_split(
        neg_by_stratum, args.holdout_neg, cv_neg, args.n_folds, rng_neg,
        leftover_label="phase2_unsupervised",
    )

    # Combine
    assignments: dict[str, str] = {}
    assignments.update(pos_assign)
    assignments.update(neg_assign)
    for pid in no_canonical_pids:
        assignments[pid] = "phase2_unsupervised"

    # Soft-negative pid set (for the manifest annotation)
    soft_neg_pids = sorted(
        canon.filter(pl.col("soft_negative"))["patient_id"].unique().to_list()
    )

    # === Build summary ===
    # cohort_for_summary: pos | neg_true | neg_soft | neg_no_canonical
    def cohort_label(pid: str, cohort: str, soft: bool) -> str:
        if pid in canon_pids:
            if cohort == "pos":
                return "pos"
            return "neg_soft" if soft else "neg_true"
        return "neg_no_canonical"

    # Build per-pid metadata used by summary
    pid_meta: dict[str, dict] = {}
    for r in canon.iter_rows(named=True):
        pid_meta[r["patient_id"]] = {
            "cohort": r["cohort"],
            "soft_negative": bool(r["soft_negative"]),
            "manufacturer_model_name": r["manufacturer_model_name"] or "",
            "thickness_bin": r["thickness_bin"],
        }
    for pid in no_canonical_pids:
        pid_meta[pid] = {"cohort": "neg", "soft_negative": False,
                         "manufacturer_model_name": "",
                         "thickness_bin": "no_canonical"}

    summary_rows = []
    counts: dict[tuple, int] = defaultdict(int)
    for pid, label in assignments.items():
        m = pid_meta[pid]
        ck = cohort_label(pid, m["cohort"], m["soft_negative"])
        key = (ck, m["manufacturer_model_name"], m["thickness_bin"], label)
        counts[key] += 1
    for (ck, model, tb, label), n in sorted(counts.items()):
        summary_rows.append({
            "cohort_class": ck,
            "manufacturer_model_name": model,
            "thickness_bin": tb,
            "split": label,
            "n_patients": n,
        })
    summary_df = pl.DataFrame(summary_rows)

    # === Subset patient lists for SLURM ===
    phase1_labels = {"holdout"} | {f"fold{i}" for i in range(args.n_folds)}
    phase1_pids = sorted(p for p, lbl in assignments.items() if lbl in phase1_labels)
    phase2_pids = sorted(p for p, lbl in assignments.items() if lbl == "phase2_unsupervised")

    # === Write outputs ===
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits_doc = {
        "seed": args.seed,
        "n_folds": args.n_folds,
        "stratification_keys": {
            "positives": ["manufacturer_model_name"],
            "negatives": ["manufacturer_model_name", "slice_thickness_bin"],
        },
        "thickness_bin_rule": f"<={THICKNESS_THRESHOLD_MM}mm vs >{THICKNESS_THRESHOLD_MM}mm on canonical sequence",
        "thickness_bin_collapsed_for_positives": True,
        "phase1_targets": {
            "holdout_pos": args.holdout_pos,
            "holdout_neg": args.holdout_neg,
            "cv_pos": cv_pos,
            "cv_neg": cv_neg,
            "phase1_total": args.holdout_pos + cv_pos + args.phase1_neg,
        },
        "pool_sizes": {
            "true_positives": int(pos_rows.height),
            "true_negatives": int(neg_rows.filter(~pl.col("soft_negative")).height),
            "soft_negatives": int(neg_rows.filter(pl.col("soft_negative")).height),
            "fat_only_no_canonical": len(no_canonical_pids),
        },
        "generated_at": date.today().isoformat(),
        "soft_negative_pids": soft_neg_pids,
        "assignments": dict(sorted(assignments.items())),
        "summary": summary_rows,
    }
    (args.output_dir / "splits.json").write_text(json.dumps(splits_doc, indent=2))

    summary_df.write_csv(args.output_dir / "splits_summary.csv")
    (args.output_dir / "subset_phase1.txt").write_text(
        "\n".join(phase1_pids) + "\n")
    (args.output_dir / "subset_phase2.txt").write_text(
        "\n".join(phase2_pids) + "\n")

    print(f"Wrote splits.json ({len(assignments)} patient assignments)")
    print(f"Wrote splits_summary.csv ({summary_df.height} rows)")
    print(f"Wrote subset_phase1.txt ({len(phase1_pids)} patients)")
    print(f"Wrote subset_phase2.txt ({len(phase2_pids)} patients)")
    print()
    print("Per-split totals:")
    by_split = defaultdict(int)
    for label in assignments.values():
        by_split[label] += 1
    for label in ["holdout"] + [f"fold{i}" for i in range(args.n_folds)] + ["phase2_unsupervised"]:
        print(f"  {label:>22}: {by_split[label]}")


if __name__ == "__main__":
    main()
