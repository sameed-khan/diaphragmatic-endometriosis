# `tests/` — pytest tree mirroring `endo/` + `scripts/`

Run via `uv run pytest tests/ -q`. The full suite passes 114 tests + 1 skipped (the lesion-bank real-cache integration test, gated on cache existence) + smoke integration gated similarly. Production-target file: `pyproject.toml [tool.pytest.ini_options].testpaths = ["tests"]`.

## Layout

| Path | Coverage |
|---|---|
| `__init__.py` | Empty package marker. |
| `augmentation/` | T1.1–T1.7 (paste counts, sites, no overlap, intensity match, soft-blend continuity), T1.11–T1.13 (geometric lockstep, in-plane-only, Y-coherent elastic), T1.16–T1.19 (box re-derivation, sub-pixel CC drop, 5-ch shape contract), T2.1, T2.4, T2.5. `conftest.py` provides synthetic `Sample` / `LesionBankEntry` fixtures. |
| `dataset/` | D1–D13 from PRD §11.3. D11–D13 are the holdout-guard tests. |
| `eval/` | E1, E3 (WBF), E5, E6 (metrics + bootstrap), E9 (threshold search), E10 (stratified), E11 (one-fold E2E), E12 (rescored vs non-rescored row sets), E15 (CSV append-only), plus a synthetic `EvalReportRow` round-trip and a `eval_thresholds.json` writer test. |
| `gru/` | G1, G3, G4, G6, G7 (rescorer mechanics) + G.INT.2 (synthetic correlated dataset → val AUROC > 0.7 in 5 epochs). |
| `lesion_bank/` | L1–L9 (CC extraction, shell construction, intensity stats, idempotency) + a real-cache integration test gated on `pytest.mark.skipif`. |
| `model/` | M1–M15 backbone / FPN / head / loss tests. M8 is a smoke-shape test on the vendored assigner (downgraded from byte-parity because mmdet isn't installable on Py3.12+uv — restore parity if mmdet ever ships). M.INT.* run the head end-to-end on a real fold-0 batch. |
| `preprocessing/` | P1.1–P1.11 unit tests on the helper functions; P1.INT.1–P1.INT.7 integration tests on a 2-volume fixture. |
| `sampler/` | S1–S6 (sampler decay + hard-pool), S8–S10 (score EMA), S12–S14 (callback gating, hard-negatives JSON, deep-eval npz roundtrip), S.INT.* real-cache. The `_FakeDetector` in `test_periodic_eval.py` exposes a `head` property aliasing `self` so the production `inference_pass.detector.head.predict(...)` call resolves. |
| `smoke/` | `test_smoke.py` — `test_pick_smoke_pids_synthetic` (always runs) + `test_smoke_runs_to_completion_real_cache` (skipif cache missing). The real-cache test runs the full smoke training (~9 min) and asserts SM1-SM4. |
| `viz/` | V1, V3, V4, V5 (event tagging) + a render smoke. |

## Conventions

- **One test file per spec section** where possible. Test IDs match the PRD §11 table.
- **Synthetic fixtures** for unit tests; real-cache integration tests are gated with `pytest.mark.skipif(not CACHE_PATH.exists(), reason="...")`.
- **No GPU-required tests in the unit lane** — model tests use `torch.cuda.is_available()` to switch to CPU when needed.
- **Don't access cache files** outside the `*.INT.*` integration tests — keep the unit lane runnable on a fresh checkout.

## Don't

- Don't add tests that import from `scripts/` directly. If you need to test a script, refactor the testable bit into `endo/...` and call it from the script. The smoke test imports `scripts.smoke_train` because that's the one script-shaped seam — keep it the only exception.
- Don't extend `_FakeDetector` to subclass `LesionDetector` — the whole point is a tiny stub that exercises the contract surface (`forward`, `head.predict`, `parameters().device`).
