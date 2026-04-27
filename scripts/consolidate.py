"""Consolidate /home/jjs374/DiaE → /scratch/.../input/{positive,negative,nifti,masks}.

New canonical layout (replaces the prior dicom/ + dicom_neg{1..5}/ multi-batch tree):

    <output_root>/positive/<ANONID>/<series>/<*.dcm>
    <output_root>/negative/<ANONID>/<series>/<*.dcm>
    <output_root>/nifti/<ANONID>[_<series>].nii.gz   (the prior radiologist-conv NIfTIs)
    <output_root>/masks/<ANONID>[_<series>].{nii.gz,csv}  (radiologist masks)

Dedup rules applied at copy time:
  * "dicom/Dicom upload/" — exact duplicate of dicom/<ANONID>; excluded.
  * dicom_neg5/_mapeamento_ids.csv, all .DS_Store — excluded.
  * Cross-batch neg dups: same ANONID in multiple dicom_neg<N> → keep the batch
    with the most series subfolders; alphabetical batch tiebreak.
  * Pos/neg overlap (4 known ANONIDs) → positive wins; not copied to negative/.

Decisions are logged to <output_root>/consolidation.csv (audit trail).
"""
import argparse
import csv
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import POSITIVE_OVERLAP_IDS

NEG_BATCHES = ("dicom_neg1", "dicom_neg2", "dicom_neg3", "dicom_neg4", "dicom_neg5")
POS_BATCH = "dicom"
EXCLUDED_NAMES = {".DS_Store", "_mapeamento_ids.csv", "Dicom upload"}


def is_anon_id(name: str) -> bool:
    return name.startswith("ANON")


def list_patient_dirs(batch_dir: Path):
    """Yield ANON-named patient subdirs of `batch_dir` (no nesting tricks)."""
    if not batch_dir.is_dir():
        return
    for p in sorted(batch_dir.iterdir()):
        if p.is_dir() and is_anon_id(p.name) and p.name not in EXCLUDED_NAMES:
            yield p


def n_series(patient_dir: Path) -> int:
    try:
        return sum(1 for s in patient_dir.iterdir() if s.is_dir())
    except OSError:
        return 0


def copy_patient(args):
    """Copy one patient dir to dest. rsync excludes .DS_Store. Returns dict log row."""
    src, dest, cohort, batch, alts = args
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--exclude=.DS_Store", "--exclude=._*",
           f"{src}/", f"{dest}/"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return {
        "patient_id": src.name, "cohort": cohort, "batch_picked": batch,
        "source_path": str(src), "dest_path": str(dest),
        "alternates_dropped": ";".join(alts),
        "rsync_exit": proc.returncode,
        "elapsed_s": f"{elapsed:.2f}",
        "stderr_excerpt": proc.stderr[-300:] if proc.returncode else "",
    }


def copy_flat(args):
    """rsync a single flat directory (nifti/ or masks/). Returns log dict."""
    label, src, dest = args
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--exclude=.DS_Store", "--exclude=._*",
           f"{src}/", f"{dest}/"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "patient_id": label, "cohort": "flat", "batch_picked": "",
        "source_path": str(src), "dest_path": str(dest),
        "alternates_dropped": "",
        "rsync_exit": proc.returncode,
        "elapsed_s": f"{time.time()-t0:.2f}",
        "stderr_excerpt": proc.stderr[-300:] if proc.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/home/jjs374/DiaE"))
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    out = args.output_root
    log_path = args.log or (out / "consolidation.csv")

    # ---- Plan: positives ----
    pos_dir = args.input_root / POS_BATCH
    pos_patients = list(list_patient_dirs(pos_dir))
    pos_pids = {p.name for p in pos_patients}
    print(f"[pos] {len(pos_patients)} ANON patient dirs in {pos_dir}", flush=True)

    # ---- Plan: negatives (with dedup) ----
    neg_by_pid: dict[str, list[Path]] = {}  # pid -> [dir1, dir2, ...]
    for batch_name in NEG_BATCHES:
        for p in list_patient_dirs(args.input_root / batch_name):
            neg_by_pid.setdefault(p.name, []).append(p)

    # Drop pos overlaps (positive wins)
    overlap_drops = 0
    for pid in list(neg_by_pid.keys()):
        if pid in pos_pids or pid in POSITIVE_OVERLAP_IDS:
            del neg_by_pid[pid]
            overlap_drops += 1

    # Pick canonical per pid: max series; tiebreak alphabetical batch.
    neg_canonical: list[tuple[str, Path, str, list[str]]] = []
    cross_batch_dropped = 0
    for pid, dirs in neg_by_pid.items():
        ranked = sorted(dirs, key=lambda d: (-n_series(d), d.parent.name))
        winner = ranked[0]
        alts = [str(d) for d in ranked[1:]]
        neg_canonical.append((pid, winner, winner.parent.name, alts))
        cross_batch_dropped += len(alts)

    print(f"[neg] {len(neg_by_pid)} unique ANONIDs after pos-overlap drop "
          f"({overlap_drops} dropped); {cross_batch_dropped} alternates dropped to dedup",
          flush=True)

    # ---- Build the work list ----
    jobs = []
    for p in pos_patients:
        dest = out / "positive" / p.name
        jobs.append((p, dest, "positive", POS_BATCH, []))
    for pid, src, batch, alts in neg_canonical:
        dest = out / "negative" / pid
        jobs.append((src, dest, "negative", batch, alts))

    # Flat dirs (nifti, masks) — single-job each, run via copy_flat.
    flat_jobs = [
        ("nifti_dir", args.input_root / "nifti", out / "nifti"),
        ("masks_dir", args.input_root / "masks", out / "masks"),
    ]

    # Resumable: use the prior consolidation.csv as ground truth for what's been
    # confirmed-complete (rows are appended after rsync exits 0). For each patient:
    #   - if patient_id is in the prior CSV with rsync_exit=0, skip (and re-emit
    #     that row into the new CSV)
    #   - if dest dir exists but no CSV row, it was in-flight at kill — wipe it
    #     and re-rsync
    #   - otherwise (no dest, no row), fresh rsync
    prior_done: dict[str, dict] = {}  # patient_id -> prior CSV row
    if log_path.exists():
        with log_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("rsync_exit") == "0" and row.get("cohort") in ("positive", "negative"):
                    prior_done[row["patient_id"]] = row
        print(f"[resume] {len(prior_done)} patients confirmed-complete in prior consolidation.csv",
              flush=True)

    pending_jobs = []
    for src, dest, cohort, batch, alts in jobs:
        pid = src.name
        if pid in prior_done:
            continue  # skip; will copy CSV row through later
        if dest.exists():
            # In-flight at kill; wipe and redo.
            shutil.rmtree(dest, ignore_errors=True)
        pending_jobs.append((src, dest, cohort, batch, alts))

    # Flat dirs: skip if a "_dir" row was already logged.
    pending_flat = [j for j in flat_jobs if j[0] not in prior_done]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "cohort", "batch_picked", "source_path", "dest_path",
              "alternates_dropped", "rsync_exit", "elapsed_s", "stderr_excerpt"]
    print(f"[copy] {len(pending_jobs)}/{len(jobs)} patient dirs pending, "
          f"{len(pending_flat)}/{len(flat_jobs)} flat dirs pending; "
          f"{args.workers} workers", flush=True)
    t0 = time.time()
    # Append mode: prior rows preserved as-is; new rows appended.
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        n_done, n_failed = 0, 0
        with mp.Pool(args.workers) as pool:
            for rec in pool.imap_unordered(copy_patient, pending_jobs, chunksize=4):
                w.writerow(rec)
                f.flush()
                n_done += 1
                if rec["rsync_exit"] != 0:
                    n_failed += 1
                if n_done % 250 == 0:
                    rate = n_done / (time.time() - t0)
                    print(f"  [{n_done}/{len(pending_jobs)}] {rate:.1f} patients/s, "
                          f"{n_failed} failed so far", flush=True)
            for rec in pool.imap_unordered(copy_flat, pending_flat):
                w.writerow(rec)
                f.flush()
                n_done += 1
                if rec["rsync_exit"] != 0:
                    n_failed += 1

    elapsed = time.time() - t0
    print(f"[done] {n_done} jobs in {elapsed/60:.1f} min "
          f"({n_failed} failures); log → {log_path}", flush=True)


if __name__ == "__main__":
    main()
