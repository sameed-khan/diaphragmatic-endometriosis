"""Stage 2: classify, filter, select canonicals, build workplan.csv.

Operates on the consolidated layout (positive/, negative/) — inter-batch dedup
and pos-overlap filtering happen at consolidation time, not here.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from _common import EXCLUDED_PIDS


def select_canonical(rows, freq):
    """Pick the canonical WATER row for one patient using the documented tiebreak."""
    return sorted(rows, key=lambda r: (
        -freq.get(r["series_description"], 0),
        -int(r["n_dcm_files"] or 0),
        r["series_description"],
    ))[0]


def write_csv_or_empty(rows, path, schema_columns):
    if rows:
        pl.DataFrame(rows).write_csv(path)
    else:
        pl.DataFrame(schema={c: pl.String for c in schema_columns}).write_csv(path)


def common_cols(r, cohort, out_subdir):
    return {
        "cohort": cohort, "patient_id": r["patient_id"],
        "source_series_path": r["series_path"],
        "source_series_description": r["series_description"],
        "image_type_full": r["image_type_full"],
        "image_type_token": r["image_type_token"],
        "n_dcm_files_in_source": r["n_dcm_files"],
        "output_subdir": out_subdir,
        "soft_negative": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-scan-index", type=Path, required=True)
    ap.add_argument("--existing-nifti", type=Path, required=True)
    ap.add_argument("--existing-masks", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    df = pl.read_csv(args.pre_scan_index, infer_schema_length=10000)

    skipped, workplan, alignment = [], [], []

    # Drop hard-excluded patients (per _common.EXCLUDED_PIDS) before anything else.
    excluded_list = list(EXCLUDED_PIDS)
    for r in df.filter(pl.col("patient_id").is_in(excluded_list)).iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "cohort": r["cohort"],
                        "source_path": r["series_path"],
                        "reason": "excluded_no_visible_lesion_on_canonical",
                        "detail": "see scripts/_common.py EXCLUDED_PIDS"})
    df = df.filter(~pl.col("patient_id").is_in(excluded_list))

    # Drop empties (log).
    for r in df.filter(pl.col("read_status") == "empty_folder").iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "cohort": r["cohort"],
                        "source_path": r["series_path"],
                        "reason": "empty_folder", "detail": ""})
    df = df.filter(pl.col("read_status") == "ok")

    # Drop non-Dixon ImageType.
    is_dixon = pl.col("image_type_full").fill_null("").str.contains("DIXON")
    for r in df.filter(~is_dixon).iter_rows(named=True):
        skipped.append({"patient_id": r["patient_id"], "cohort": r["cohort"],
                        "source_path": r["series_path"],
                        "reason": "non_dixon",
                        "detail": r["image_type_full"]})
    df = df.filter(is_dixon)

    # SeriesDescription frequency tally (WATER only) for canonical tiebreak.
    water_all = df.filter(pl.col("image_type_token") == "WATER")
    if water_all.height > 0:
        freq_df = water_all["series_description"].value_counts()
        sd_freq = dict(zip(freq_df["series_description"].to_list(),
                           freq_df["count"].to_list()))
    else:
        sd_freq = {}

    def emit_patient(rows, cohort):
        pid = rows[0]["patient_id"]
        out_subdir = (f"nifti_neg/{pid}" if cohort == "neg"
                      else f"nifti_pos/{pid}")
        token_groups = defaultdict(list)
        for r in rows:
            token_groups[r["image_type_token"]].append(r)
        water_rows = token_groups.get("WATER", [])
        if water_rows:
            canonical = select_canonical(water_rows, sd_freq)
            alts = [r for r in water_rows
                    if r["series_path"] != canonical["series_path"]]
            alts.sort(key=lambda r: r["series_description"])
            workplan.append({**common_cols(canonical, cohort, out_subdir),
                             "role": "canonical",
                             "output_basename": "water_canonical"})
            for i, r in enumerate(alts, start=1):
                workplan.append({**common_cols(r, cohort, out_subdir),
                                 "role": "alt",
                                 "output_basename": f"water_alt_{i:02d}"})
        for i, r in enumerate(sorted(token_groups.get("FAT", []),
                                      key=lambda x: x["series_description"]), start=1):
            workplan.append({**common_cols(r, cohort, out_subdir),
                             "role": "fat",
                             "output_basename": f"fat_{i:02d}"})
        for token, prefix in (("IN_PHASE", "inphase"), ("OUT_PHASE", "outphase")):
            for i, r in enumerate(sorted(token_groups.get(token, []),
                                          key=lambda x: x["series_description"]), start=1):
                workplan.append({**common_cols(r, cohort, out_subdir),
                                 "role": prefix,
                                 "output_basename": f"{prefix}_{i:02d}"})

    for cohort_name in ("pos", "neg"):
        cohort_df = df.filter(pl.col("cohort") == cohort_name)
        by_patient = defaultdict(list)
        for r in cohort_df.iter_rows(named=True):
            by_patient[r["patient_id"]].append(r)
        for pid in sorted(by_patient.keys()):
            emit_patient(by_patient[pid], cohort_name)

    # Mask-source mapping for positives: emit ONE ROW PER EXISTING MASK FILE.
    # We do not try to assign masks to canonical here (the existing nifti's series
    # description in the filename does not always match the canonical pick, and
    # plain-style filenames are ambiguous). Instead, realign_masks.py does the
    # actual shape-match against freshly-converted output volumes at runtime.
    # Skip excluded pids in the alignment audit too.
    pos_pids = {w["patient_id"] for w in workplan if w["cohort"] == "pos"} - EXCLUDED_PIDS
    nifti_files = list(args.existing_nifti.glob("*.nii.gz"))
    nifti_by_pid = defaultdict(list)
    for f in nifti_files:
        m = re.match(r"^(ANON[A-F0-9]+)(?:_(.*))?\.nii\.gz$", f.name)
        if m:
            nifti_by_pid[m.group(1)].append((f, m.group(2)))

    for pid in sorted(pos_pids):
        candidates = nifti_by_pid.get(pid, [])
        if not candidates:
            alignment.append({
                "patient_id": pid,
                "mask_filename": "",
                "mask_source_path": "",
                "filename_suffix": "",
                "reason": "no_existing_nifti_or_mask",
            })
            continue
        for nifti_file, suffix in sorted(candidates, key=lambda x: x[0].name):
            mask_source = args.existing_masks / nifti_file.name
            alignment.append({
                "patient_id": pid,
                "mask_filename": nifti_file.name,
                "mask_source_path": str(mask_source) if mask_source.exists() else "",
                "filename_suffix": suffix or "",
                "reason": "" if mask_source.exists() else "mask_file_not_found",
            })

    # === Reclassify mask-less positives as "soft negatives" ===
    # A positive patient with no existing mask file is not a usable supervised
    # case. Per user direction (2026-04-26), these patients are reclassified
    # into the negative cohort with output_subdir nifti_neg/<pid> and a
    # soft_negative=True flag, so the on-disk data organization reflects that
    # they are not positives. They remain traceable via source_series_path
    # (still under .../input/positive/...) and the soft_negative flag.
    pos_pids_with_mask = {a["patient_id"] for a in alignment
                          if a["mask_source_path"]}
    soft_negative_pids = pos_pids - pos_pids_with_mask
    for w in workplan:
        if w["patient_id"] in soft_negative_pids:
            w["cohort"] = "neg"
            w["output_subdir"] = f"nifti_neg/{w['patient_id']}"
            w["soft_negative"] = True
    # Drop reclassified pids from alignment_audit (they no longer have a mask
    # and they're no longer positives).
    alignment = [a for a in alignment if a["patient_id"] not in soft_negative_pids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_or_empty(workplan, args.output_dir / "workplan.csv",
        ["cohort", "patient_id", "source_series_path",
         "source_series_description", "image_type_full", "image_type_token",
         "n_dcm_files_in_source", "output_subdir", "role", "output_basename",
         "soft_negative"])
    write_csv_or_empty(skipped, args.output_dir / "skipped.csv",
        ["patient_id", "cohort", "source_path", "reason", "detail"])
    write_csv_or_empty(alignment, args.output_dir / "alignment_audit.csv",
        ["patient_id", "mask_filename", "mask_source_path",
         "filename_suffix", "reason"])

    n_neg = sum(1 for w in workplan if w["cohort"] == "neg" and not w["soft_negative"])
    n_pos = sum(1 for w in workplan if w["cohort"] == "pos")
    n_soft = sum(1 for w in workplan if w["soft_negative"])
    n_pos_unique = len({a["patient_id"] for a in alignment})
    n_align_with_mask = sum(1 for a in alignment if a["mask_source_path"])
    print(f"workplan: {len(workplan)} rows "
          f"({n_neg} true neg + {n_soft} soft_neg rows, {n_pos} pos rows)")
    print(f"  soft_neg unique pids: {len(soft_negative_pids)}")
    print(f"skipped:  {len(skipped)} rows")
    print(f"alignment: {len(alignment)} mask candidates across "
          f"{n_pos_unique} positives ({n_align_with_mask} with a mask file)")


if __name__ == "__main__":
    main()
