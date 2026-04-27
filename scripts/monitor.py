"""Live progress monitor for the SLURM array. Run from the head node."""
import argparse
import glob
import subprocess
import time
from pathlib import Path

import polars as pl


def squeue_state(job_id: str) -> dict:
    """Returns counts of array tasks in each state."""
    out = subprocess.run(
        ["squeue", "-j", job_id, "--noheader", "-r", "-o", "%T"],
        capture_output=True, text=True)
    states = out.stdout.split()
    return {s: states.count(s) for s in set(states)}


def manifest_progress(output_root: Path, manifest_glob: str) -> tuple[int, int]:
    parts = glob.glob(str(output_root / manifest_glob))
    if not parts:
        return 0, 0
    done = sum(sum(1 for _ in open(p)) - 1 for p in parts)
    failed = 0
    for p in parts:
        try:
            df = pl.read_csv(p, infer_schema_length=10000)
            if "exit_code" in df.columns:
                failed += df.filter(pl.col("exit_code") != 0).height
        except Exception:
            pass
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True, help="SLURM job ID (the %A part)")
    ap.add_argument("--workplan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--cohort", choices=["neg", "pos", "all"], default="all")
    ap.add_argument("--patient-list", type=Path,
                    help="Optional file with one patient ID per line; scopes "
                         "the expected-count to these patients only")
    ap.add_argument("--manifest-glob", default="manifest_part_*.csv",
                    help="Glob (under --output-root) for manifest_part files "
                         "to count toward 'done'. Use 'manifest_part_phase2_*.csv' "
                         "to exclude prior phases.")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    plan = pl.read_csv(args.workplan, infer_schema_length=10000)
    if args.cohort != "all":
        plan = plan.filter(pl.col("cohort") == args.cohort)
    if args.patient_list and args.patient_list.exists():
        keep = set(args.patient_list.read_text().split())
        plan = plan.filter(pl.col("patient_id").is_in(list(keep)))
    expected = plan.height

    history = []
    print(f"Monitoring SLURM job {args.job_id}; expected {expected} conversions")
    print(f"{'time':<19} {'done':>13} {'failed':>7} {'rate/min':>9} {'ETA':>10} {'queue states':<40}")
    while True:
        states = squeue_state(args.job_id)
        done, failed = manifest_progress(args.output_root, args.manifest_glob)
        now = time.time()
        history.append((now, done))
        history = [(t, d) for t, d in history if now - t <= 600]
        if len(history) >= 2:
            dt = history[-1][0] - history[0][0]
            dd = history[-1][1] - history[0][1]
            rate = (dd / dt * 60) if dt > 0 else 0
            eta_sec = ((expected - done) / (rate / 60)) if rate > 0 else float("inf")
            eta = (f"{eta_sec/3600:5.1f}h" if eta_sec < float("inf") and eta_sec > 3600
                   else f"{eta_sec/60:6.1f}m" if eta_sec < float("inf")
                   else "    --")
        else:
            rate, eta = 0, "  --"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sstr = " ".join(f"{k}={v}" for k, v in sorted(states.items()))
        print(f"{ts}  {done:>6}/{expected:<6}  {failed:>5}  {rate:>7.1f}  {eta:>9}  {sstr}")
        if not states:
            print("Job no longer in queue. Final progress reported above.")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
