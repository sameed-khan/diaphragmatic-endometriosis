"""
Generate liver masks for the 608 Phase-1-transferred patients via TotalSegmentator
(task=total_mr, roi_subset=liver, multilabel, full-res normal mode), in parallel.

The script is dry-run by default. Pass --execute to actually invoke TotalSegmentator.

Outputs (under --data-root):
    liver_masks/<bucket>/<cohort>/<mnemonic>_liver_mask.nii.gz   # binary uint8 0/1
    _pipeline/failures.csv      # one row per failed/empty patient
    _pipeline/qc_warnings.csv   # masks with < 1000 voxels (kept, but flagged)
    _pipeline/timing_test.csv   # per-patient wall time, when --limit/--patients/--retry-failed
    _pipeline/timing_full.csv   # per-patient wall time, on a full run
    _pipeline/vram_log_test.csv # only when --vram-monitor is set
    _pipeline/pipeline_run.log  # appended free-text log

After a full run (no --limit/--patients/--retry-failed), the manifest gets three
columns populated for every transferred patient: liver_mask_path, liver_mask_sha256,
liver_voxel_count.

Idempotent: skips patients whose target mask already exists with size > 1 KB,
unless --force is given.

Per agent/totalseg-plan.md §6.1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
from tqdm import tqdm

EXPECTED_TRANSFER_COUNT = 608
DEFAULT_WORKERS = 6
SUBPROCESS_TIMEOUT_S = 600  # 10 min per patient ceiling
TINY_MASK_THRESHOLD = 1000  # voxels — flagged as qc warning, still kept
MIN_NON_TRIVIAL_FILE_BYTES = 1024  # 1 KB skip threshold for idempotent skip
VRAM_PEAK_LIMIT_MB = 41 * 1024  # 41 GB on a 46 GB L40S — the test-phase halt threshold

# Direct path to the TotalSegmentator entrypoint that lives in the uv-installed
# tool venv (torch 2.11.0+cu128, GPU-visible). Do NOT use `uvx TotalSegmentator`
# — that resolves to a separate ephemeral venv with an incompatible torch and
# silently falls back to CPU.
DEFAULT_TOTALSEG_BIN = str(Path.home() / ".local/bin/TotalSegmentator")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root containing manifest.csv, raw/, liver_masks/.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="Number of concurrent TotalSegmentator workers (default: 6).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N patients (sorted by mnemonic_id).")
    ap.add_argument("--patients", type=Path, default=None,
                    help="CSV listing mnemonic_ids to process (one column 'mnemonic_id', "
                         "or one mnemonic per line if no header).")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Process only patients listed in _pipeline/failures.csv.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run patients whose output already exists.")
    ap.add_argument("--vram-monitor", action="store_true",
                    help="Spawn a background nvidia-smi sampler that writes vram_log_test.csv.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually invoke TotalSegmentator. Without this flag, dry-run only.")
    ap.add_argument("--totalseg-bin", type=str, default=DEFAULT_TOTALSEG_BIN,
                    help=f"Path to the TotalSegmentator entrypoint (default: {DEFAULT_TOTALSEG_BIN}).")
    ap.add_argument("--device", type=str, default="gpu",
                    help="TotalSegmentator -d device (gpu|cpu|mps|gpu:0|...). Default: gpu.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def binarize_inplace(nifti_path: Path) -> int:
    """Load NIfTI, cast (data > 0) to uint8, save back, return foreground voxel count."""
    img = nib.load(nifti_path)
    data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.header.set_data_dtype(np.uint8)
    out.header.set_slope_inter(1.0, 0.0)
    nib.save(out, nifti_path)
    return int(data.sum())


def gpu_visible_to_totalseg(totalseg_bin: str) -> bool:
    """Quick subprocess check: does the TotalSegmentator interpreter see CUDA?

    Resolves the python interpreter from the same venv as `totalseg_bin` (i.e.,
    the bin/ directory next to the entrypoint).
    """
    bin_path = Path(totalseg_bin)
    py = bin_path.parent / "python"
    # If --totalseg-bin is a symlink in ~/.local/bin, follow it to the real venv.
    if bin_path.is_symlink():
        py = Path(os.readlink(bin_path)).parent / "python"
        if not py.is_absolute():
            py = bin_path.parent / py
    if not py.exists():
        # Fallback to the canonical install location.
        py = Path.home() / ".local/share/uv/tools/totalsegmentator/bin/python"
    if not py.exists():
        return False
    try:
        proc = subprocess.run(
            [str(py), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout.strip() == "True"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def load_failures(failures_csv: Path) -> set[str]:
    if not failures_csv.exists():
        return set()
    df = pl.read_csv(failures_csv)
    if "mnemonic_id" not in df.columns:
        return set()
    return set(df["mnemonic_id"].to_list())


def load_patients_csv(path: Path) -> list[str]:
    """Accept a single-column file (with or without header) or a multi-column CSV
    that has a 'mnemonic_id' column."""
    text = path.read_text().strip()
    if not text:
        return []
    first_line = text.splitlines()[0]
    if "," in first_line:
        df = pl.read_csv(path)
        if "mnemonic_id" not in df.columns:
            raise SystemExit(f"--patients CSV {path} has no mnemonic_id column")
        return df["mnemonic_id"].to_list()
    # one mnemonic per line, possibly with a header line "mnemonic_id"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].lower() == "mnemonic_id":
        lines = lines[1:]
    return lines


# ---------------------------------------------------------------------------
# Build the work queue
# ---------------------------------------------------------------------------

def build_queue(
    args: argparse.Namespace,
    transferred: pl.DataFrame,
    pipeline_dir: Path,
    liver_masks_root: Path,
) -> list[dict]:
    """Returns a list of dicts: {mnemonic_id, anon_id, bucket, cohort, src, dst}."""
    if args.patients:
        wanted = set(load_patients_csv(args.patients))
        if not wanted:
            raise SystemExit(f"--patients {args.patients} is empty")
        df = transferred.filter(pl.col("mnemonic_id").is_in(list(wanted)))
        missing = wanted - set(df["mnemonic_id"].to_list())
        if missing:
            print(f"WARN: {len(missing)} mnemonics from --patients are not in the manifest "
                  f"(or not transferred); first: {sorted(missing)[0]}")
    elif args.retry_failed:
        failed = load_failures(pipeline_dir / "failures.csv")
        if not failed:
            print("--retry-failed: no failures.csv or no rows in it; nothing to do.")
            return []
        df = transferred.filter(pl.col("mnemonic_id").is_in(list(failed)))
    else:
        df = transferred

    df = df.sort("mnemonic_id")

    queue: list[dict] = []
    for r in df.iter_rows(named=True):
        bucket = r["bucket"]
        cohort = r["cohort"]
        mnem = r["mnemonic_id"]
        src = args.data_root / "raw" / bucket / cohort / f"{mnem}.nii.gz"
        dst = liver_masks_root / bucket / cohort / f"{mnem}_liver_mask.nii.gz"
        queue.append({
            "mnemonic_id": mnem,
            "anon_id": r["anon_id"],
            "bucket": bucket,
            "cohort": cohort,
            "src": src,
            "dst": dst,
        })

    if not args.force:
        queue = [q for q in queue if not (q["dst"].exists() and q["dst"].stat().st_size > MIN_NON_TRIVIAL_FILE_BYTES)]

    if args.limit is not None:
        queue = queue[: args.limit]

    return queue


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(item: dict) -> dict:
    """Worker entry: shells out to TotalSegmentator for one patient, then binarizes."""
    mnem = item["mnemonic_id"]
    src = Path(item["src"])
    dst = Path(item["dst"])
    totalseg_bin = item["totalseg_bin"]
    device = item["device"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        totalseg_bin,
        "-i", str(src),
        "-o", str(dst),
        "-ta", "total_mr",
        "-rs", "liver",
        "--ml",
        "-q",
        "-d", device,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S)
        rc = proc.returncode
        stderr = (proc.stderr or "")[-500:]
    except subprocess.TimeoutExpired as e:
        return {
            "mnemonic_id": mnem, "anon_id": item["anon_id"], "cohort": item["cohort"],
            "exit_code": -1, "wall_seconds": time.time() - t0,
            "voxel_count": 0, "stderr": f"timeout after {SUBPROCESS_TIMEOUT_S}s: {e}",
            "status": "failure", "reason": "timeout",
        }
    wall = time.time() - t0

    if rc != 0:
        return {
            "mnemonic_id": mnem, "anon_id": item["anon_id"], "cohort": item["cohort"],
            "exit_code": rc, "wall_seconds": wall,
            "voxel_count": 0, "stderr": stderr,
            "status": "failure", "reason": f"exit_code_{rc}",
        }

    if not dst.exists():
        return {
            "mnemonic_id": mnem, "anon_id": item["anon_id"], "cohort": item["cohort"],
            "exit_code": 0, "wall_seconds": wall,
            "voxel_count": 0, "stderr": stderr,
            "status": "failure", "reason": "no_output_file",
        }

    voxel_count = binarize_inplace(dst)
    if voxel_count == 0:
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        return {
            "mnemonic_id": mnem, "anon_id": item["anon_id"], "cohort": item["cohort"],
            "exit_code": 0, "wall_seconds": wall,
            "voxel_count": 0, "stderr": stderr,
            "status": "failure", "reason": "empty_mask",
        }

    status = "qc_warning" if voxel_count < TINY_MASK_THRESHOLD else "ok"
    return {
        "mnemonic_id": mnem, "anon_id": item["anon_id"], "cohort": item["cohort"],
        "exit_code": 0, "wall_seconds": wall,
        "voxel_count": voxel_count, "stderr": "",
        "status": status, "reason": "tiny_mask" if status == "qc_warning" else "",
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def append_csv_row(path: Path, row: dict, header: list[str]):
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def remove_failure_rows(failures_csv: Path, mnemonics: set[str]):
    """Re-write failures.csv excluding the given mnemonics (used after retry success)."""
    if not failures_csv.exists() or not mnemonics:
        return
    df = pl.read_csv(failures_csv)
    if "mnemonic_id" not in df.columns:
        return
    df2 = df.filter(~pl.col("mnemonic_id").is_in(list(mnemonics)))
    if df2.height == 0:
        failures_csv.unlink()
    else:
        tmp = failures_csv.with_suffix(".csv.tmp")
        df2.write_csv(tmp)
        tmp.replace(failures_csv)


# ---------------------------------------------------------------------------
# VRAM monitor
# ---------------------------------------------------------------------------

class VramSampler:
    """Thread-based VRAM sampler.

    Uses nvidia-smi --query-compute-apps=used_memory (per-process; this driver/host
    returns 0 for the per-GPU memory.used query, so we sum across compute apps every
    2 seconds). Writes one integer MB per line to out_path.
    """

    def __init__(self, out_path: Path, interval_s: float = 2.0):
        self.out_path = out_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb: int = 0

    def _sample(self) -> int:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0
        total = 0
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                total += int(line)
            except ValueError:
                pass
        return total

    def _loop(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", buffering=1) as f:
            while not self._stop.is_set():
                mb = self._sample()
                if mb > self.peak_mb:
                    self.peak_mb = mb
                f.write(f"{mb}\n")
                self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)


def start_vram_monitor(out_path: Path) -> VramSampler | None:
    if shutil.which("nvidia-smi") is None:
        print("WARN: nvidia-smi not found; --vram-monitor is a no-op.")
        return None
    sampler = VramSampler(out_path)
    sampler.start()
    return sampler


def stop_vram_monitor(sampler: VramSampler | None) -> None:
    if sampler is None:
        return
    sampler.stop()


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------

def update_manifest_after_full_run(data_root: Path, liver_masks_root: Path) -> None:
    """For every transferred patient with a liver_mask on disk, populate three columns:
    liver_mask_path, liver_mask_sha256, liver_voxel_count. Atomic via .tmp + rename."""
    manifest_path = data_root / "manifest.csv"
    m = pl.read_csv(manifest_path, infer_schema_length=10000)

    # Add columns if they don't exist yet (preserve any prior values for non-transferred rows).
    new_cols = {
        "liver_mask_path": pl.Utf8,
        "liver_mask_sha256": pl.Utf8,
        "liver_voxel_count": pl.Int64,
    }
    for col, dtype in new_cols.items():
        if col not in m.columns:
            if dtype == pl.Int64:
                m = m.with_columns(pl.lit(None, dtype=pl.Int64).alias(col))
            else:
                m = m.with_columns(pl.lit("", dtype=pl.Utf8).alias(col))

    paths: list[str] = []
    shas: list[str] = []
    counts: list[int | None] = []

    for r in m.iter_rows(named=True):
        if not r["transferred_to_home"]:
            paths.append(r.get("liver_mask_path") or "")
            shas.append(r.get("liver_mask_sha256") or "")
            counts.append(r.get("liver_voxel_count"))
            continue
        bucket = r["bucket"]
        cohort = r["cohort"]
        mnem = r["mnemonic_id"]
        rel = f"liver_masks/{bucket}/{cohort}/{mnem}_liver_mask.nii.gz"
        full = data_root / rel
        if full.exists() and full.stat().st_size > MIN_NON_TRIVIAL_FILE_BYTES:
            paths.append(rel)
            shas.append(sha256_file(full))
            img = nib.load(full)
            counts.append(int((np.asarray(img.dataobj) > 0).sum()))
        else:
            paths.append("")
            shas.append("")
            counts.append(None)

    m = m.with_columns([
        pl.Series("liver_mask_path", paths, dtype=pl.Utf8),
        pl.Series("liver_mask_sha256", shas, dtype=pl.Utf8),
        pl.Series("liver_voxel_count", counts, dtype=pl.Int64),
    ])
    tmp = manifest_path.with_suffix(".csv.tmp")
    m.write_csv(tmp)
    tmp.replace(manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_root = args.data_root.resolve()
    liver_masks_root = data_root / "liver_masks"
    pipeline_dir = data_root / "_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    is_subset_run = bool(args.limit or args.patients or args.retry_failed)
    timing_path = pipeline_dir / ("timing_test.csv" if is_subset_run else "timing_full.csv")
    failures_path = pipeline_dir / "failures.csv"
    qc_path = pipeline_dir / "qc_warnings.csv"
    log_path = pipeline_dir / "pipeline_run.log"
    vram_path = pipeline_dir / "vram_log_test.csv"

    started_at = datetime.now().isoformat(timespec="seconds")

    if args.execute:
        print(">>> EXECUTE — TotalSegmentator will be invoked. <<<")
    else:
        print(">>> DRY-RUN — no work will be done. Pass --execute to proceed. <<<")

    # ---- Load manifest ----
    manifest_path = data_root / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    m = pl.read_csv(manifest_path, infer_schema_length=10000)
    transferred = m.filter(pl.col("transferred_to_home"))
    if transferred.height != EXPECTED_TRANSFER_COUNT:
        print(f"WARN: transferred_to_home rows = {transferred.height}, "
              f"expected {EXPECTED_TRANSFER_COUNT}")

    queue = build_queue(args, transferred, pipeline_dir, liver_masks_root)

    # Sanity: source files exist (pre-flight)
    missing_src = [q["src"] for q in queue if not q["src"].exists()]
    if missing_src:
        raise SystemExit(f"Pre-flight: {len(missing_src)} source NIfTIs missing; "
                         f"first: {missing_src[0]}")

    # ---- Dry-run summary ----
    n_total = transferred.height
    n_existing = sum(1 for r in transferred.iter_rows(named=True)
                     if (liver_masks_root / r["bucket"] / r["cohort"] / f"{r['mnemonic_id']}_liver_mask.nii.gz").exists())
    print(f"\n=== Plan ===")
    print(f"  Manifest rows (transferred):   {n_total}")
    print(f"  Outputs already present:       {n_existing}")
    print(f"  In work queue (after filters): {len(queue)}")
    if args.limit:
        print(f"  --limit:                       {args.limit}")
    if args.patients:
        print(f"  --patients:                    {args.patients}")
    if args.retry_failed:
        print(f"  --retry-failed:                yes ({len(queue)} from failures.csv)")
    print(f"  --force:                       {args.force}")
    print(f"  --workers:                     {args.workers}")
    print(f"  --device:                      {args.device}")
    print(f"  Timing log:                    {timing_path}")
    if args.vram_monitor:
        print(f"  VRAM log:                      {vram_path}")
    if queue:
        print(f"  Estimated wall (80s/patient):  {len(queue) * 80 / args.workers / 60:.1f} min "
              f"({len(queue) * 80 / args.workers:.0f} s) at {args.workers} workers")
        print(f"  First 3 destinations:")
        for q in queue[:3]:
            print(f"    {q['mnemonic_id']:30s} -> {q['dst'].relative_to(data_root)}")

    if not args.execute:
        print("\nDry-run complete.")
        return 0

    if not queue:
        print("\nNothing to do — queue is empty.")
        return 0

    # ---- Pre-flight: GPU visible to TotalSegmentator ----
    print("\n=== Pre-flight ===")
    if not gpu_visible_to_totalseg(args.totalseg_bin):
        raise SystemExit("Pre-flight: TotalSegmentator's torch does not see CUDA. Aborting.")
    print("  TotalSegmentator GPU: visible")
    print(f"  Pipeline dir:         {pipeline_dir}")

    # ---- Spawn VRAM monitor if requested ----
    vram_sampler = None
    if args.vram_monitor:
        vram_sampler = start_vram_monitor(vram_path)
        if vram_sampler is not None:
            print(f"  VRAM sampler:         interval=2s, log={vram_path}")

    with open(log_path, "a") as logf:
        logf.write(f"\n=== run_totalseg.py @ {started_at} ===\n")
        logf.write(f"queue_size={len(queue)} workers={args.workers} "
                   f"limit={args.limit} patients={args.patients} retry={args.retry_failed}\n")

    # ---- Execute via spawn pool ----
    work_items = [{
        "mnemonic_id": q["mnemonic_id"],
        "anon_id": q["anon_id"],
        "src": str(q["src"]),
        "dst": str(q["dst"]),
        "bucket": q["bucket"],
        "cohort": q["cohort"],
        "totalseg_bin": args.totalseg_bin,
        "device": args.device,
    } for q in queue]

    cohort_lookup = {q["mnemonic_id"]: q["cohort"] for q in queue}
    thickness_lookup = {
        r["mnemonic_id"]: r["slice_thickness_mm"]
        for r in transferred.iter_rows(named=True)
    }

    timing_header = ["mnemonic_id", "cohort", "thickness_mm", "wall_seconds", "voxel_count", "exit_code"]
    failures_header = ["mnemonic_id", "anon_id", "phase", "reason", "stderr_excerpt", "timestamp"]
    qc_header = ["mnemonic_id", "voxel_count", "timestamp"]

    n_ok = n_warn = n_fail = 0
    fixed_mnemonics: set[str] = set()

    ctx = mp.get_context("spawn")
    t_start = time.time()
    try:
        with ctx.Pool(processes=args.workers) as pool:
            iterator = pool.imap_unordered(_run_one, work_items)
            for result in tqdm(iterator, total=len(work_items), unit="pat", desc="TotalSeg"):
                mnem = result["mnemonic_id"]
                ts = datetime.now().isoformat(timespec="seconds")
                cohort = cohort_lookup.get(mnem, "")
                thickness = thickness_lookup.get(mnem, "")
                append_csv_row(timing_path, {
                    "mnemonic_id": mnem,
                    "cohort": cohort,
                    "thickness_mm": thickness,
                    "wall_seconds": f"{result['wall_seconds']:.2f}",
                    "voxel_count": result["voxel_count"],
                    "exit_code": result["exit_code"],
                }, timing_header)

                if result["status"] == "ok":
                    n_ok += 1
                    fixed_mnemonics.add(mnem)
                elif result["status"] == "qc_warning":
                    n_warn += 1
                    fixed_mnemonics.add(mnem)
                    append_csv_row(qc_path, {
                        "mnemonic_id": mnem,
                        "voxel_count": result["voxel_count"],
                        "timestamp": ts,
                    }, qc_header)
                else:  # failure
                    n_fail += 1
                    append_csv_row(failures_path, {
                        "mnemonic_id": mnem,
                        "anon_id": result["anon_id"],
                        "phase": "totalseg",
                        "reason": result["reason"],
                        "stderr_excerpt": result["stderr"].replace("\n", " ").replace("\r", " "),
                        "timestamp": ts,
                    }, failures_header)
    finally:
        if vram_sampler is not None:
            stop_vram_monitor(vram_sampler)

    wall_total = time.time() - t_start

    # ---- If retry-failed succeeded, drop those rows from failures.csv ----
    if args.retry_failed and fixed_mnemonics:
        remove_failure_rows(failures_path, fixed_mnemonics)

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  attempted:   {len(work_items)}")
    print(f"  ok:          {n_ok}")
    print(f"  qc_warnings: {n_warn} (kept, see qc_warnings.csv)")
    print(f"  failures:    {n_fail} (see failures.csv)")
    print(f"  wall total:  {wall_total/60:.1f} min  (mean {wall_total/max(1, len(work_items)):.1f} s/pat)")

    # ---- VRAM peak readout ----
    peak_mb: int | None = None
    if vram_sampler is not None:
        peak_mb = vram_sampler.peak_mb
        print(f"  peak VRAM:   {peak_mb} MB ({peak_mb / 1024:.1f} GB)")

    with open(log_path, "a") as logf:
        logf.write(f"finished @ {datetime.now().isoformat(timespec='seconds')} "
                   f"ok={n_ok} warn={n_warn} fail={n_fail} wall={wall_total:.0f}s "
                   f"peak_vram_mb={peak_mb}\n")

    # ---- Test-phase guidance ----
    if is_subset_run:
        if peak_mb is not None and peak_mb >= VRAM_PEAK_LIMIT_MB:
            print(f"\nTEST PHASE WARNING — peak VRAM {peak_mb / 1024:.1f} GB "
                  f">= {VRAM_PEAK_LIMIT_MB / 1024:.0f} GB threshold. "
                  f"Consider reducing --workers before the full run.")
        elif n_fail == 0:
            print(f"\nTEST PHASE OK — safe to proceed with --workers {args.workers} on the full set.")
        else:
            print(f"\nTEST PHASE: {n_fail} failures observed; investigate before scaling up.")

    # ---- Manifest update on full runs (and retries) only ----
    if not args.limit and not args.patients:
        print("\n=== Updating manifest ===")
        update_manifest_after_full_run(data_root, liver_masks_root)
        print(f"  Updated columns liver_mask_path / liver_mask_sha256 / liver_voxel_count "
              f"in {manifest_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
