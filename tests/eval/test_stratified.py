"""Test E10 — stratified breakdowns filter to the right subset."""

from __future__ import annotations

import numpy as np

from endo.config.eval import EvalConfig
from endo.eval.stratified import stratify_metrics


def test_stratified_breakdown_filters():
    """E10: 60% Artist + 40% Explorer manifest. Artist breakdown should
    include only Artist patients."""
    n_artist_pos = 6
    n_artist_neg = 6
    n_explorer_pos = 4
    n_explorer_neg = 4

    rng = np.random.default_rng(0)
    preds: dict[str, dict] = {}
    labels: dict[str, int] = {}
    manifest_rows: list[dict] = []

    def _make(pid, label, scanner, variant):
        preds[pid] = {
            "fused_boxes": np.zeros((0, 5), dtype=np.float32),
            "fused_scores": np.zeros((0,), dtype=np.float32),
            "score": float(np.clip(rng.normal(0.7 if label else 0.3, 0.1), 0, 1)),
        }
        labels[pid] = label
        manifest_rows.append(
            {"patient_id": pid, "scanner_model": scanner, "variant": variant, "slice_thickness_mm": 1.5 if variant == "A" else 3.6}
        )

    for i in range(n_artist_pos):
        _make(f"a_pos_{i}", 1, "SIGNA Artist", "A")
    for i in range(n_artist_neg):
        _make(f"a_neg_{i}", 0, "SIGNA Artist", "A")
    for i in range(n_explorer_pos):
        _make(f"e_pos_{i}", 1, "SIGNA Explorer", "B")
    for i in range(n_explorer_neg):
        _make(f"e_neg_{i}", 0, "SIGNA Explorer", "B")

    cfg = EvalConfig(bootstrap_n=50, stratify_keys=["scanner_model", "variant", "slice_thickness_bin"])
    results = stratify_metrics(preds, labels, manifest_rows, eval_cfg=cfg)

    # Find the SIGNA Artist scanner stratum.
    artist = [r for r in results if r["stratum_kind"] == "scanner_model" and r["stratum_value"] == "SIGNA Artist"]
    assert len(artist) == 1
    assert artist[0]["n_patients"] == n_artist_pos + n_artist_neg

    explorer = [r for r in results if r["stratum_kind"] == "scanner_model" and r["stratum_value"] == "SIGNA Explorer"]
    assert len(explorer) == 1
    assert explorer[0]["n_patients"] == n_explorer_pos + n_explorer_neg

    # All scanners should sum to total population.
    scanner_total = sum(r["n_patients"] for r in results if r["stratum_kind"] == "scanner_model")
    assert scanner_total == len(preds)

    # Slice-thickness bin stratification is also produced.
    bins = {r["stratum_value"] for r in results if r["stratum_kind"] == "slice_thickness_bin"}
    assert bins == {"<=2mm", ">2mm"}
