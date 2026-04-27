"""Consolidate per-patient BIDS sidecars into data/sidecars.jsonl.

For every transferred patient (608 in Phase 1), reads raw/<bucket>/<cohort>/<mnemonic>.json
and emits one JSON object per line to data/sidecars.jsonl with the original sidecar
contents wrapped under a `sidecar` key plus provenance fields (mnemonic_id, anon_id,
bucket, split, cohort, raw_path).

After the JSONL is written and verified, deletes the per-patient .json files
(idempotent; the JSONL is the new source of truth) and drops `raw_json_path`
from data/manifest.csv since the column no longer points anywhere.

Dry-run by default. Pass --execute to actually delete files and rewrite the manifest.
"""

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/home/sak185/dia-endo-conversion/data"))
    ap.add_argument("--execute", action="store_true",
                    help="Delete per-patient JSONs and rewrite manifest. "
                         "Without this, only the JSONL is written (safe to re-run).")
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    manifest_path = data_root / "manifest.csv"
    jsonl_path = data_root / "sidecars.jsonl"

    m = pl.read_csv(manifest_path, infer_schema_length=10000)
    transferred = m.filter(pl.col("transferred_to_home")).sort("mnemonic_id")
    print(f"Transferred patients in manifest: {transferred.height}")

    rows_out: list[dict] = []
    sources_to_delete: list[Path] = []
    missing: list[str] = []

    for r in transferred.iter_rows(named=True):
        json_relpath = r["raw_json_path"]
        if not json_relpath:
            missing.append(r["mnemonic_id"])
            continue
        json_path = data_root / json_relpath
        if not json_path.exists():
            missing.append(f"{r['mnemonic_id']} ({json_path})")
            continue
        with open(json_path) as f:
            sidecar = json.load(f)
        rows_out.append({
            "mnemonic_id": r["mnemonic_id"],
            "anon_id": r["anon_id"],
            "bucket": r["bucket"],
            "split": r["split"],
            "cohort": r["cohort"],
            "raw_path": r["raw_path"],
            "sidecar": sidecar,
        })
        sources_to_delete.append(json_path)

    if missing:
        print(f"WARNING: {len(missing)} sidecars missing or unreferenced:")
        for m_id in missing[:5]:
            print(f"  {m_id}")
        if len(missing) > 5:
            print(f"  ... ({len(missing) - 5} more)")

    print(f"Sidecars read: {len(rows_out)}")

    print(f"Writing JSONL to {jsonl_path}")
    with open(jsonl_path, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Verify line count matches
    with open(jsonl_path) as f:
        n_lines = sum(1 for _ in f)
    if n_lines != len(rows_out):
        raise SystemExit(f"JSONL line count {n_lines} != row count {len(rows_out)}")
    print(f"JSONL verified: {n_lines} lines, {jsonl_path.stat().st_size/1024:.1f} KB")

    if not args.execute:
        print("\nDry-run. JSONL written. Per-patient JSONs NOT deleted, manifest NOT rewritten.")
        print("Re-run with --execute to finalize.")
        return

    # Round-trip parse to make sure the JSONL is readable before destruction.
    with open(jsonl_path) as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                assert "sidecar" in obj and "mnemonic_id" in obj
            except Exception as e:
                raise SystemExit(f"JSONL parse failure at line {i}: {e!r}")
    print("JSONL round-trip parse: ok")

    print(f"Deleting {len(sources_to_delete)} per-patient JSONs...")
    for p in sources_to_delete:
        p.unlink()

    # Drop raw_json_path column from manifest.
    if "raw_json_path" in m.columns:
        m_new = m.drop("raw_json_path")
        m_new.write_csv(manifest_path)
        print(f"Dropped raw_json_path column from {manifest_path}")
    else:
        print("raw_json_path column already absent; no manifest change.")

    print("\n=== Done ===")
    print(f"sidecars.jsonl rows:  {n_lines}")
    print(f"JSON files deleted:   {len(sources_to_delete)}")


if __name__ == "__main__":
    main()
