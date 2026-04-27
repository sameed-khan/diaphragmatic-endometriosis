"""
Generate human-readable mnemonic patient IDs.

Maps each ANON* patient ID to a unique 3-word identifier
in the pattern: adjective_animal_noun

Scheme:
- Deterministic via SHA-256 hashing of the ANON ID
- Different 8-char chunks of the hash select indices into each word list
- Collision resolution by incrementing an attempt counter and re-hashing
- Word lists sourced from scripts/wordlists.json (immutable)

Output: a two-column CSV (anon_id, mnemonic_id) sorted by anon_id.
The mapping is treated as immutable once written — the script refuses
to overwrite an existing mapping unless --force is passed, because
regenerating with a different wordlist or hashing scheme would shift
names and break every downstream reference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


def load_wordlists(path: Path) -> tuple[list[str], list[str], list[str]]:
    with open(path) as f:
        wl = json.load(f)

    adjectives = wl["adjectives"]
    animals = wl["animals"]
    nouns = wl["nouns"]

    for name, lst in [("adjectives", adjectives), ("animals", animals), ("nouns", nouns)]:
        dupes = [w for w in lst if lst.count(w) > 1]
        if dupes:
            raise ValueError(f"Duplicate words in {name}: {set(dupes)}")

    return adjectives, animals, nouns


def generate_name(
    patient_id: str,
    adjectives: list[str],
    animals: list[str],
    nouns: list[str],
    attempt: int = 0,
) -> str:
    """Deterministically generate a 3-word name from a patient ID using hashing."""
    seed = f"{patient_id}:{attempt}"
    h = hashlib.sha256(seed.encode()).hexdigest()

    adj_idx = int(h[0:8], 16) % len(adjectives)
    animal_idx = int(h[8:16], 16) % len(animals)
    noun_idx = int(h[16:24], 16) % len(nouns)

    return f"{adjectives[adj_idx]}_{animals[animal_idx]}_{nouns[noun_idx]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest", type=Path, required=True,
        help="Source manifest.csv whose patient_id column is the universe of IDs to name.",
    )
    ap.add_argument(
        "--wordlists", type=Path,
        default=Path(__file__).parent / "wordlists.json",
        help="Path to wordlists.json (default: scripts/wordlists.json next to this file).",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="Where to write patient_id_mapping.csv.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing mapping (DANGEROUS — shifts all names and breaks downstream).",
    )
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"{args.output} already exists. Refusing to overwrite without --force "
            f"(regenerating would shift names and break downstream references)."
        )

    adjectives, animals, nouns = load_wordlists(args.wordlists)
    capacity = len(adjectives) * len(animals) * len(nouns)
    print(f"Wordlist sizes: {len(adjectives)} adj x {len(animals)} animals x {len(nouns)} nouns")
    print(f"Total capacity: {capacity:,} unique combinations")

    df = pl.read_csv(args.manifest, infer_schema_length=10000)
    if "patient_id" not in df.columns:
        raise SystemExit(f"Source manifest {args.manifest} has no `patient_id` column.")
    patient_ids = sorted(df["patient_id"].unique().to_list())
    print(f"Patients to name: {len(patient_ids)}")
    print(f"Utilization: {len(patient_ids) / capacity * 100:.4f}%")

    used_names: set[str] = set()
    mapping = []
    collisions = 0

    for pid in patient_ids:
        attempt = 0
        while True:
            name = generate_name(pid, adjectives, animals, nouns, attempt)
            if name not in used_names:
                used_names.add(name)
                mapping.append({"anon_id": pid, "mnemonic_id": name})
                if attempt > 0:
                    collisions += 1
                break
            attempt += 1
            if attempt > 100:
                raise RuntimeError(f"Cannot generate unique name for {pid}")

    out = pl.DataFrame(mapping).sort("anon_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.output)

    print(f"\nMapping saved to {args.output}")
    print(f"Collisions resolved: {collisions}")
    print(f"\nSample mappings:")
    for row in out.head(10).iter_rows(named=True):
        print(f"  {row['anon_id']}  ->  {row['mnemonic_id']}")

    assert out["mnemonic_id"].n_unique() == len(out), "Duplicate mnemonic IDs!"
    print(f"\nAll {len(out)} names are unique.")


if __name__ == "__main__":
    main()
