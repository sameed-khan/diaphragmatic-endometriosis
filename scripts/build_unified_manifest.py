"""Phase-0 data unification.

Collapses the historical three-file convention
    data/manifest.csv         (5060 rows × 42 cols, mostly bloat)
    data/sidecars.jsonl       (608 rows, BIDS sidecars)
    data/splits.json          (5084-entry assignments + globals)
    data/patient_id_mapping.csv (5089 rows, ANON ↔ mnemonic)

into the unified, mnemonic-keyed format
    data/manifest.jsonl       (608 rows, one per phase-1 patient)
    data/cohort.json          (global splits/strat metadata)
    data/_archive/anon_id_mapping.csv  (full forensic ANON map)

and moves the originals to data/_legacy/.

Field-naming changes (deliberate; see PRD §5.1):
    legacy            unified
    bucket            cohort        ("cross-validation" | "holdout")
    cohort            label         ("positive" | "negative")
    split=foldN       fold=N        (int 0..4 | null for holdout)

Run:
    uv run python scripts/build_unified_manifest.py --dry-run
    uv run python scripts/build_unified_manifest.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

LEGACY_FILES = [
    "manifest.csv",
    "sidecars.jsonl",
    "splits.json",
    "patient_id_mapping.csv",
]

OUT_MANIFEST = DATA / "manifest.jsonl"
OUT_COHORT = DATA / "cohort.json"
OUT_ARCHIVE = DATA / "_archive" / "anon_id_mapping.csv"
OUT_LEGACY = DATA / "_legacy"


# ─── Helpers ───────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _classify_variant(series_description: str) -> str:
    """A: 1.5mm reconstruction, B: 3.6mm. Per phase-1 §1.2."""
    if not series_description:
        return "unknown"
    s = series_description.upper()
    # Variant B = "WATER: COR DIAFRAGMA T1 LAVA …" on Explorer
    if "DIAFRAGMA" in s and "LAVA" in s:
        return "B"
    # Variant A = "WATER: COR LAVA DIAF." (and FLEX NAV variants)
    if "LAVA" in s:
        return "A"
    return "unknown"


def _safe_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_shape(s) -> list[int] | None:
    """manifest.csv `shape` is x-separated, e.g. '512x110x512'."""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    sep = "x" if "x" in s else ","
    try:
        return [int(x.strip(" ()[]")) for x in s.split(sep) if x.strip(" ()[]")]
    except Exception:
        return None


def _split_to_fold(split: str) -> tuple[str, int | None]:
    """Map legacy `split` (fold0..4 / holdout) to (cohort, fold)."""
    if split == "holdout":
        return "holdout", None
    if split.startswith("fold"):
        return "cross-validation", int(split[4:])
    raise ValueError(f"unexpected split value for phase-1 row: {split!r}")


# ─── Main pipeline ─────────────────────────────────────────────────────────

def build(dry_run: bool) -> int:
    # 1. Read sources
    if not (DATA / "manifest.csv").exists():
        print(f"ERROR: data/manifest.csv missing — already migrated?", file=sys.stderr)
        return 1

    manifest = pl.read_csv(DATA / "manifest.csv")
    splits = json.loads((DATA / "splits.json").read_text())
    pid_map = pl.read_csv(DATA / "patient_id_mapping.csv")

    sidecars: dict[str, dict] = {}
    with (DATA / "sidecars.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            sidecars[rec["mnemonic_id"]] = rec

    # 2. Filter manifest to phase-1 (the cohort we actually use)
    phase1 = manifest.filter(pl.col("phase") == "phase1")
    n = phase1.shape[0]
    if n != 608:
        print(f"ERROR: expected 608 phase-1 rows, got {n}", file=sys.stderr)
        return 1

    # Cross-check: every phase-1 patient is transferred
    not_transferred = phase1.filter(~pl.col("transferred_to_home"))
    if not_transferred.shape[0] != 0:
        print(
            f"ERROR: {not_transferred.shape[0]} phase-1 rows have transferred_to_home=False",
            file=sys.stderr,
        )
        return 1

    print(f"[1/6] loaded {n} phase-1 rows")

    # 3. Cross-walk: every phase-1 patient must appear in sidecars + splits.assignments
    rows: list[dict] = []
    fold_counter: dict[str, int] = {}
    label_counter: dict[str, int] = {}

    for row in phase1.iter_rows(named=True):
        mnemonic = row["mnemonic_id"]
        anon = row["anon_id"]

        # Sidecar lookup
        if mnemonic not in sidecars:
            print(f"ERROR: no sidecar for {mnemonic}", file=sys.stderr)
            return 1
        sc = sidecars[mnemonic]
        if sc["anon_id"] != anon:
            print(
                f"ERROR: anon_id mismatch for {mnemonic}: manifest={anon} sidecar={sc['anon_id']}",
                file=sys.stderr,
            )
            return 1

        # Splits lookup (keyed by ANON)
        if anon not in splits["assignments"]:
            print(f"ERROR: no splits.assignments entry for ANON {anon} ({mnemonic})", file=sys.stderr)
            return 1
        split_value = splits["assignments"][anon]

        cohort, fold = _split_to_fold(split_value)
        # Cross-check vs manifest
        manifest_split = row["split"]
        if manifest_split != split_value:
            print(
                f"ERROR: split mismatch for {mnemonic}: manifest={manifest_split} splits.json={split_value}",
                file=sys.stderr,
            )
            return 1
        manifest_bucket = row["bucket"]
        if manifest_bucket != cohort:
            print(
                f"ERROR: bucket vs derived cohort mismatch for {mnemonic}: "
                f"bucket={manifest_bucket} derived={cohort}",
                file=sys.stderr,
            )
            return 1

        # Label: legacy `cohort` column = our new `label`
        label = row["cohort"]
        if label not in {"positive", "negative"}:
            print(f"ERROR: unexpected label for {mnemonic}: {label!r}", file=sys.stderr)
            return 1

        # Geometry
        bids = sc.get("sidecar", {})
        shape = _parse_shape(row["shape"])
        zoom_y_mm = _safe_float(bids.get("SpacingBetweenSlices"))  # NIfTI zoom_y per phase-1 §1.2
        # Important: the *NIfTI* zoom_y is what matters for physical math.
        # We do NOT trust manifest.slice_thickness_mm (DICOM tag) per phase-1 §1.2.
        # SpacingBetweenSlices in BIDS is the closest proxy from the sidecar; in practice
        # for these volumes the canonical NIfTI zoom_y will be re-read by the preprocessor.
        # We still record this here as a hint.

        pixel_x = _safe_float(row["pixel_spacing_x_mm"])
        pixel_z = _safe_float(row["pixel_spacing_y_mm"])  # axis-2 spacing

        # Variant
        series_desc = row.get("series_description") or bids.get("SeriesDescription", "")
        variant = _classify_variant(series_desc)

        # Liver ROI bbox
        roi_bbox = {
            "x0": _safe_int(row["roi_bbox_x0"]),
            "x1": _safe_int(row["roi_bbox_x1"]),
            "y0": _safe_int(row["roi_bbox_y0"]),
            "y1": _safe_int(row["roi_bbox_y1"]),
            "z0": _safe_int(row["roi_bbox_z0"]),
            "z1": _safe_int(row["roi_bbox_z1"]),
            "extent_x_mm": _safe_float(row["roi_bbox_extent_x_mm"]),
            "extent_y_mm": _safe_float(row["roi_bbox_extent_y_mm"]),
            "extent_z_mm": _safe_float(row["roi_bbox_extent_z_mm"]),
        }

        # Hot DICOM fields (raised for convenience) + full bids object preserved
        dicom_block = {
            "echo_time_s": _safe_float(bids.get("EchoTime")),
            "repetition_time_s": _safe_float(bids.get("RepetitionTime")),
            "flip_angle": _safe_float(bids.get("FlipAngle")),
            "scanning_sequence": bids.get("ScanningSequence"),
            "image_type": bids.get("ImageType"),
            "bids": bids,
        }

        unified = {
            "patient_id": mnemonic,
            "cohort": cohort,
            "label": label,
            "fold": fold,
            "soft_negative": bool(row["soft_negative"]),
            "paths": {
                "raw": row["raw_path"],
                "lesion_mask": row["lesion_mask_path"] if row["lesion_mask_path"] else None,
                "liver_mask": row["liver_mask_path"] if row["liver_mask_path"] else None,
                "liver_roi": row["liver_roi_path"] if row["liver_roi_path"] else None,
            },
            "hashes": {
                "raw_sha256": row["sha256_raw"],
                "liver_mask_sha256": row["liver_mask_sha256"]
                if row["liver_mask_sha256"]
                else None,
            },
            "geometry": {
                # shape: [x, y_slices, z]; y_slices is the through-plane axis (coronal)
                "shape": shape,
                # n_slices = shape[1] (through-plane). manifest.n_slices_actual is unreliable
                # — sometimes records an in-plane dimension. Trust shape[1] when present.
                "n_slices": shape[1] if (shape and len(shape) == 3) else None,
                # In-plane spacings: only a HINT. Preprocessor reads NIfTI zooms directly
                # (per phase-1 doc §1.2: never trust manifest.slice_thickness_mm).
                "pixel_spacing_xz_mm_hint": [pixel_x, pixel_z],
                "slice_spacing_mm_bids_hint": zoom_y_mm,
                "orientation": "RAS",
            },
            "scanner": {
                "manufacturer": bids.get("Manufacturer", "GE"),
                "model": row["scanner_model"],
                "magnetic_field_strength_t": _safe_float(row["magnetic_field_strength"]),
                "variant": variant,
                "series_description": series_desc,
            },
            "liver_roi_bbox": roi_bbox,
            "dicom": dicom_block,
            "provenance": {
                "migration_timestamp": row["migration_timestamp"],
                "anon_id": anon,
                "selected_subvolume": bool(row["selected_subvolume"]),
                "had_multi_canonical": bool(row["had_multi_canonical"]),
                "volume_index": _safe_int(row["volume_index"]),
            },
        }

        # Per-row invariant checks
        if label == "positive" and unified["paths"]["lesion_mask"] is None:
            print(f"ERROR: positive {mnemonic} has no lesion_mask_path", file=sys.stderr)
            return 1
        if label == "negative" and unified["paths"]["lesion_mask"] is not None:
            print(f"ERROR: negative {mnemonic} has lesion_mask_path", file=sys.stderr)
            return 1
        if cohort == "cross-validation" and fold not in {0, 1, 2, 3, 4}:
            print(f"ERROR: CV {mnemonic} has fold={fold!r}", file=sys.stderr)
            return 1
        if cohort == "holdout" and fold is not None:
            print(f"ERROR: holdout {mnemonic} has fold={fold!r}", file=sys.stderr)
            return 1

        rows.append(unified)
        key = f"{cohort}/{label}/fold{fold}" if fold is not None else f"{cohort}/{label}"
        fold_counter[key] = fold_counter.get(key, 0) + 1
        label_counter[label] = label_counter.get(label, 0) + 1

    print(f"[2/6] cross-walked all {len(rows)} rows")
    print("      label totals:", label_counter)
    print("      cohort × label × fold:")
    for k in sorted(fold_counter):
        print(f"        {k:50s} {fold_counter[k]}")

    # 4. Build cohort.json
    summary_keyed = {}
    for fold_idx in range(5):
        n_pos = fold_counter.get(f"cross-validation/positive/fold{fold_idx}", 0)
        n_neg = fold_counter.get(f"cross-validation/negative/fold{fold_idx}", 0)
        summary_keyed[f"fold{fold_idx}"] = {"n": n_pos + n_neg, "pos": n_pos, "neg": n_neg}
    n_h_pos = fold_counter.get("holdout/positive", 0)
    n_h_neg = fold_counter.get("holdout/negative", 0)
    summary_keyed["holdout"] = {"n": n_h_pos + n_h_neg, "pos": n_h_pos, "neg": n_h_neg}

    # Soft negatives → mnemonic
    anon_to_mn = dict(zip(pid_map["anon_id"].to_list(), pid_map["mnemonic_id"].to_list()))
    soft_negative_mnemonics = sorted(
        anon_to_mn[a] for a in splits.get("soft_negative_pids", []) if a in anon_to_mn
    )

    cohort_meta = {
        "version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_version": _git_sha(),
        "n_patients_total": len(rows),
        "splits": {
            "seed": splits["seed"],
            "n_folds": splits["n_folds"],
            "stratification": {
                "positives": splits["stratification_keys"]["positives"],
                "negatives": splits["stratification_keys"]["negatives"],
                "thickness_bin_rule": splits.get("thickness_bin_rule"),
                "thickness_bin_collapsed_for_positives": splits.get(
                    "thickness_bin_collapsed_for_positives"
                ),
            },
            "frozen_at": splits.get("generated_at"),
        },
        "phase1_targets": splits.get("phase1_targets", {}),
        "fold_summary": summary_keyed,
        "n_soft_negatives": len(soft_negative_mnemonics),
        "soft_negative_pids": soft_negative_mnemonics,
        "schema_url": "see agent/complete_spec/00_PRD.md §5.1",
    }
    print(f"[3/6] cohort.json composed")

    # 5. Build archive (full ANON↔mnemonic for forensics)
    archive_rows = pid_map.with_columns(
        pl.col("anon_id").is_in(set(splits["assignments"].keys())).alias("in_assignments")
    ).with_columns(
        pl.col("anon_id")
        .is_in(set(r["provenance"]["anon_id"] for r in rows))
        .alias("used_in_phase1")
    )
    print(f"[4/6] archive table composed: {archive_rows.shape[0]} rows")

    # 6. Write outputs
    if dry_run:
        print(f"\n[DRY RUN] would write:")
        print(f"  {OUT_MANIFEST}  ({len(rows)} rows)")
        print(f"  {OUT_COHORT}")
        print(f"  {OUT_ARCHIVE}  ({archive_rows.shape[0]} rows)")
        print(f"  move legacy → {OUT_LEGACY}/  ({', '.join(LEGACY_FILES)})")
        print(f"\nSample manifest row:")
        print(json.dumps(rows[0], indent=2)[:1500])
        return 0

    # Real write
    OUT_LEGACY.mkdir(exist_ok=True, parents=True)
    OUT_ARCHIVE.parent.mkdir(exist_ok=True, parents=True)

    with OUT_MANIFEST.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[5/6] wrote {OUT_MANIFEST} ({len(rows)} rows)")

    OUT_COHORT.write_text(json.dumps(cohort_meta, indent=2))
    print(f"      wrote {OUT_COHORT}")

    archive_rows.write_csv(OUT_ARCHIVE)
    print(f"      wrote {OUT_ARCHIVE} ({archive_rows.shape[0]} rows)")

    # Compute sha for forensics
    h = hashlib.sha256(OUT_MANIFEST.read_bytes()).hexdigest()[:16]
    print(f"      manifest.jsonl sha256[:16] = {h}")

    # Move legacy
    for fname in LEGACY_FILES:
        src = DATA / fname
        if src.exists():
            dst = OUT_LEGACY / fname
            shutil.move(str(src), str(dst))
            print(f"      moved {src.name} → _legacy/{fname}")

    print(f"[6/6] DONE.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-0 data unification.")
    parser.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    args = parser.parse_args()
    sys.exit(build(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
