"""Convert a chunk of patients from workplan.csv. Called from SLURM or pilot."""
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

MANIFEST_PART_FIELDS = [
    "cohort", "patient_id", "source_series_path", "role",
    "output_subdir", "output_basename", "exit_code",
    "produced_files", "stderr_excerpt",
]


def convert_one(row, output_root: Path, work: Path) -> dict:
    src = Path(row["source_series_path"])
    out_dir = output_root / row["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    local_src = work / "in" / row["patient_id"] / src.name
    local_src.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, local_src / f.name)

    local_out = work / "out" / row["output_subdir"]
    local_out.mkdir(parents=True, exist_ok=True)

    cmd = ["dcm2niix", "-z", "y", "-b", "y", "-ba", "n",
           "-f", row["output_basename"], "-o", str(local_out), str(local_src)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    produced = sorted(local_out.glob(f"{row['output_basename']}*.nii.gz"))

    for f in produced:
        shutil.copy2(f, out_dir / f.name)
        json_sib = f.with_suffix("").with_suffix(".json")
        if json_sib.exists():
            shutil.copy2(json_sib, out_dir / json_sib.name)

    shutil.rmtree(local_src, ignore_errors=True)
    shutil.rmtree(local_out, ignore_errors=True)
    return {
        **{k: row[k] for k in ["cohort", "patient_id",
                               "source_series_path", "role", "output_subdir",
                               "output_basename"]},
        "exit_code": proc.returncode,
        "produced_files": ";".join(f.name for f in produced),
        "stderr_excerpt": proc.stderr[-500:] if proc.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--manifest-part", type=Path, required=True,
                    help="Where to write per-task manifest CSV")
    ap.add_argument("--patient-list", type=Path,
                    help="Optional file with one ANONID per line; only convert these")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--task-count", type=int, default=1,
                    help="Total array tasks; this task processes rows where "
                         "(row_idx %% task_count) == (task_id-1)")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="Per-task scratch base; defaults to /tmp")
    args = ap.parse_args()

    df = pl.read_csv(args.workplan, infer_schema_length=10000)
    if args.patient_list and args.patient_list.exists():
        keep = set(args.patient_list.read_text().split())
        df = df.filter(pl.col("patient_id").is_in(list(keep)))
    if args.task_count > 1:
        df = df.with_row_index("_idx").filter(
            (pl.col("_idx") % args.task_count) == (args.task_id - 1)
        ).drop("_idx")

    scratch_base = str(args.scratch_dir) if args.scratch_dir else "/tmp"
    Path(scratch_base).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"dcm2niix_{os.getpid()}_", dir=scratch_base))
    args.manifest_part.parent.mkdir(parents=True, exist_ok=True)

    # Resume: index prior successful rows so a re-run of a timed-out / killed
    # task picks up where it left off. Key on (output_subdir, output_basename)
    # — that pair uniquely identifies the on-disk target. Failures (exit_code != 0)
    # are NOT in the skip set, so they get retried automatically.
    done_keys: set[tuple[str, str]] = set()
    if args.manifest_part.exists() and args.manifest_part.stat().st_size > 0:
        with args.manifest_part.open() as f:
            for row in csv.DictReader(f):
                if row.get("exit_code") == "0" and row.get("produced_files"):
                    done_keys.add((row["output_subdir"], row["output_basename"]))
        write_mode, write_header = "a", False
    else:
        write_mode, write_header = "w", True

    print(f"Task {args.task_id}/{args.task_count}: {df.height} series assigned; "
          f"{len(done_keys)} already done (resume); scratch={work}", flush=True)

    with args.manifest_part.open(write_mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_PART_FIELDS)
        if write_header:
            w.writeheader()
        n_skipped = 0
        n_processed = 0
        for i, row in enumerate(df.iter_rows(named=True), start=1):
            if (row["output_subdir"], row["output_basename"]) in done_keys:
                n_skipped += 1
                continue
            try:
                rec = convert_one(row, args.output_root, work)
            except Exception as e:
                rec = {**{k: row.get(k, "") for k in MANIFEST_PART_FIELDS},
                       "exit_code": -1, "produced_files": "",
                       "stderr_excerpt": repr(e)[:500]}
            w.writerow(rec)
            f.flush()
            n_processed += 1
            if n_processed % 25 == 0:
                print(f"  ...{n_processed} new + {n_skipped} skipped of {df.height}",
                      flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Task {args.task_id}/{args.task_count} done; "
          f"{n_processed} new conversions, {n_skipped} skipped (resume); "
          f"manifest: {args.manifest_part}", flush=True)


if __name__ == "__main__":
    main()
