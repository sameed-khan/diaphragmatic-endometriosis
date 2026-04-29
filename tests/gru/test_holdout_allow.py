"""Verify that the holdout-rescoring path forwards ``allow_holdout=True``
to ``LesionDataModule``.

Regression test for the bug where ``predict_holdout --use-gru`` would fail
with ``Refusing to load holdout patients ... with allow_holdout=False``
because ``endo.gru.feature_cache._build_datamodule`` hardcoded
``allow_holdout=False``.

These tests don't run inference — they monkeypatch ``LesionDataModule``
inside ``endo.gru.feature_cache`` to record the kwarg it was constructed
with, then assert the contract:

    * ``_build_datamodule(...)`` defaults to ``allow_holdout=False``.
    * ``_build_datamodule(..., allow_holdout=True)`` is honored.
    * ``extract_features_for_pids(...)`` defaults to ``allow_holdout=False``.
    * ``extract_features_for_pids(..., allow_holdout=True)`` is honored.
"""

from __future__ import annotations

import uuid as _uuid
from pathlib import Path

import pytest

from endo.config import ExperimentConfig
from endo.gru import feature_cache as fc


class _FakeDM:
    """Captures the kwargs ``LesionDataModule(...)`` was built with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def setup(self) -> None:
        return None


def _make_experiment(tmp_path: Path) -> ExperimentConfig:
    # Minimal, real ExperimentConfig instance — only ``paths.{data_root,cache_root}``
    # and ``run_dir()`` get touched on the path under test.
    exp = ExperimentConfig(
        name="unittest_gru_holdout_allow",
        uuid=str(_uuid.uuid4()),
    )
    exp.paths.data_root = tmp_path / "data"
    exp.paths.cache_root = tmp_path / "cache"
    (tmp_path / "data").mkdir()
    (tmp_path / "cache").mkdir()
    # Touch the manifest/cohort files so any `Path` checks downstream don't blow up
    (exp.paths.data_root / "manifest.jsonl").write_text("")
    (exp.paths.data_root / "cohort.json").write_text("{}")
    return exp


def test_build_datamodule_default_is_false(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "LesionDataModule", _FakeDM)
    _FakeDM.last_kwargs = None
    exp = _make_experiment(tmp_path)
    fc._build_datamodule(exp, fold=0)
    assert _FakeDM.last_kwargs is not None
    assert _FakeDM.last_kwargs["allow_holdout"] is False


def test_build_datamodule_allow_holdout_true_is_forwarded(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "LesionDataModule", _FakeDM)
    _FakeDM.last_kwargs = None
    exp = _make_experiment(tmp_path)
    fc._build_datamodule(exp, fold=0, allow_holdout=True)
    assert _FakeDM.last_kwargs is not None
    assert _FakeDM.last_kwargs["allow_holdout"] is True


def test_extract_features_for_pids_forwards_allow_holdout(tmp_path, monkeypatch):
    """End-to-end signature check: passing allow_holdout=True through the
    public entrypoint must reach LesionDataModule with the same value.
    """
    monkeypatch.setattr(fc, "LesionDataModule", _FakeDM)
    # Bypass the heavy detector-load step — the test only cares about the DM kwargs.
    monkeypatch.setattr(
        fc, "_load_detector_with_ema", lambda ckpt, device, experiment=None: object()
    )
    # Short-circuit the per-pid extraction loop so we never touch a real loader / GPU.
    monkeypatch.setattr(
        fc,
        "_extract_for_pid",
        lambda pid, dm, lm, device: (__import__("numpy").zeros((0, 768), dtype="float16"),
                                     __import__("numpy").zeros((0,), dtype="int32")),
    )

    exp = _make_experiment(tmp_path)
    fake_ckpt = tmp_path / "best.ckpt"
    fake_ckpt.write_bytes(b"")  # existence check only
    out_dir = tmp_path / "out"

    _FakeDM.last_kwargs = None
    fc.extract_features_for_pids(
        exp,
        fold=0,
        pids=["arctic_ferret_grove"],
        output_dir=out_dir,
        ckpt_path=fake_ckpt,
        device="cpu",
        allow_holdout=True,
    )
    assert _FakeDM.last_kwargs is not None
    assert _FakeDM.last_kwargs["allow_holdout"] is True

    # And the default path should still be False (non-holdout call sites).
    _FakeDM.last_kwargs = None
    fc.extract_features_for_pids(
        exp,
        fold=0,
        pids=["some_cv_pid"],
        output_dir=out_dir,
        ckpt_path=fake_ckpt,
        device="cpu",
    )
    assert _FakeDM.last_kwargs is not None
    assert _FakeDM.last_kwargs["allow_holdout"] is False


def test_holdout_rescore_caller_passes_allow_holdout_true(monkeypatch, tmp_path):
    """Belt-and-suspenders: the only legitimate caller (``_try_gru_rescore_holdout``)
    must invoke extract_features_for_pids with allow_holdout=True.
    """
    from endo.eval import run_eval

    captured: dict = {}

    def _fake_extract(experiment, fold, *, pids, output_dir, ckpt_path, device, allow_holdout=False):
        captured["allow_holdout"] = allow_holdout
        captured["pids"] = list(pids)
        return Path(output_dir)

    def _fake_rescore(slice_scores, *, ckpt_path, feature_dir):
        return slice_scores

    # Patch the symbols at their import site inside run_eval (they're imported lazily
    # inside the function body, so we patch the modules they live in).
    monkeypatch.setattr(
        "endo.gru.feature_cache.extract_features_for_pids", _fake_extract
    )
    monkeypatch.setattr(
        "endo.gru.rescorer.rescore_slice_scores", _fake_rescore
    )

    exp = _make_experiment(tmp_path)
    # Create the fake GRU ckpt so the early-exit guard is bypassed.
    gru_ckpt = exp.run_dir() / "fold0" / "gru" / "ckpt.pt"
    gru_ckpt.parent.mkdir(parents=True, exist_ok=True)
    gru_ckpt.write_bytes(b"")

    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()

    out = run_eval._try_gru_rescore_holdout(
        experiment=exp,
        fold=0,
        ckpt_path=tmp_path / "best.ckpt",
        holdout_pids=["arctic_ferret_grove"],
        holdout_dir=holdout_dir,
        slice_scores={},
    )
    assert captured.get("allow_holdout") is True
    assert captured["pids"] == ["arctic_ferret_grove"]
    assert out == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
