# 00_PRD — Diaphragmatic Endometriosis Detector (Production Reference)

**Author:** Planning agent.
**Date:** 2026-04-28.
**Status:** Authoritative cross-component synthesis. Components 01–08 implement specific subsystems; this PRD is the contract that binds them. **Phase 0 has been executed by the planning agent. The implementation agent picks up at Phase 0d (uv sync + MMDet vendoring) and proceeds through Phase 8.**
**Audience:** The single document a brand-new engineering agent reads first. After this document, the agent reads `01_preprocessing.md … 08_smoke_and_viz.md` for component-internal detail.

---

## Table of contents

| § | Section |
|---|---|
| 1 | System overview |
| 2 | Repository organization |
| 3 | Experiment configuration system |
| 4 | CLI surface |
| 5 | Data contracts |
| 6 | Runtime contracts (Python interfaces) |
| 7 | Invariants — post-preprocessing |
| 8 | Invariants — at training time |
| 9 | Invariants — at evaluation time |
| 10 | End-to-end execution sequence (Phase 0–8) |
| 11 | Test invariants table |
| 12 | Resource accounting |
| 13 | Open issues, spec amendments, deviations |
| 14 | Glossary |

---

## 1. System overview

### 1.1 Goal

Train a 2.5D MR detector for diaphragmatic endometriosis lesions on 608 GE 1.5 T 3D Dixon LAVA WATER coronal volumes (108 positives, 500 negatives). Targets, on patient-level 5-fold CV:

- **Volume AUROC ≥ 0.80** (at-least-one lesion in volume).
- **Sensitivity ≥ 0.70 at 2 FP/volume**.

Hardware: single L40S 46 GB. Wall-clock budget: one week. Stage-1 (detector) + Stage-2 (GRU rescorer) ≤ 25 GPU-h.

### 1.2 Top-level architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │ data/  (frozen post-migration; PHASE 0 unified format)   │
                    │   manifest.jsonl  cohort.json  raw/  liver_*/  lesion_*/  │
                    └─────────────────────────┬────────────────────────────────┘
                                              │
                ┌─────────────────────────────┴─────────────────────────────┐
                ▼                                                           │
   ┌────────────────────────┐                                               │
   │ Component 1            │  cache version-keyed, EXPERIMENT-INDEPENDENT  │
   │ scripts/preprocess.py  │──► cache/v1/volumes/<pid>/{volume.npy,        │
   │                        │     lesion_mask.npy}                          │
   │ analyze_inplane_       │    cache/v1/border_bands/<pid>.npy            │
   │ spacing.py (one-time)  │    cache/v1/gt_boxes.parquet                  │
   │                        │    cache/v1/preprocessed_manifest.jsonl       │
   └─────────┬──────────────┘                                               │
             │                                                              │
             ▼                                                              │
   ┌────────────────────────┐                                               │
   │ Component 2            │                                               │
   │ scripts/build_         │──► cache/v1/lesion_banks/                     │
   │ lesion_bank.py         │     {lesion_bank_<sha8>.pkl, current.pkl,     │
   │                        │      bank_provenance.json}                    │
   └────────────────────────┘                                               │
                                                                            │
   ┌──────────────────────────────────────────────────────────────────────┐ │
   │ Component 4 dev workflow (one-off paste QC; not part of API)         │ │
   │ scripts/qc_paste_review.py + Claude Code subagent (Task tool)        │ │
   │   → cache/v1/runtime/qc_paste_review.{json,html} (advisory)          │ │
   └──────────────────────────────────────────────────────────────────────┘ │
                                                                            │
═══════════════════════════════════════════════════════════════════════════ │
                                                                            │
   experiments/<name>.py  ───►  src.cli.run_experiment  ◄────────────────── ┘
                                       │
                ┌──────────────────────┼─────────────────────────┐
                │ for fold in 0..4:    │                         │
                ▼                      ▼                         ▼
   ┌────────────────────────┐ ┌───────────────────────┐ ┌─────────────────────┐
   │ Component 3+4+5+6      │ │ Component 6.5         │ │ Component 7         │
   │ Lightning Trainer      │ │ GRU rescorer          │ │ CV + holdout eval   │
   │  • LesionDataModule    │ │  • feature_cache      │ │  • WBF, FROC, AUROC │
   │  • TrainAugmentation   │ │  • train_gru          │ │  • bootstrap CIs    │
   │  • WeightedScheduled-  │ │                       │ │  • stratified       │
   │    Sampler             │ └───────────┬───────────┘ │  • CSV-only output  │
   │  • LesionDetectorLM    │             │             └──────────┬──────────┘
   │  • PeriodicDeepEval    │             │                        │
   │  • EmaCallback         │             │                        │
   │                        │             │                        │
   │ outputs:               │             │                        │
   │   runs/<exp>/fold{f}/  │             │                        │
   │     ckpts/best.ckpt    │             │                        │
   │     runtime/           │             │                        │
   │       hard_negs.json   │             │                        │
   │       deep_eval/*.npz  │             │                        │
   └─────────┬──────────────┘             │                        │
             │                            │                        │
             └─► best.ckpt ───────────────┘ → ckpt + features ─────┘
                                              ↓
                                              runs/<exp>/eval/
                                                eval_report.csv (cv_pooled)
                                              runs/<exp>/holdout/run_<id>/
                                                eval_report.csv (holdout)

   Component 8: scripts/smoke_train.py + scripts/visualize_predictions.py
   (optional QC at any time; outputs under runs/<exp>/fold{f}/viz/)
```

### 1.3 Component map

| # | Spec file | Owner code (production) | Purpose in one line |
|---|---|---|---|
| 1 | `01_preprocessing.md` | `scripts/preprocess.py`, `scripts/analyze_inplane_spacing.py` | Resample → ROI z-score → crop+pad → cache `.npy` + GT boxes + border bands. |
| 2 | `02_lesion_bank.md` | `scripts/build_lesion_bank.py`, `src/lesion_bank.py` | Single global donor bank for paste augmentation. |
| 3 | `03_dataset_datamodule.md` | `src/data/dataset.py`, `src/data/datamodule.py` | RAM-resident slice-level Dataset + Lightning DataModule + holdout guard. |
| 4 | `04_augmentation.md` | `src/augmentation/transform.py`, `src/augmentation/{paste,geometric,intensity,boxes}.py` | Online lesion paste + geometric + intensity + box re-derivation + 5-channel slice extraction. |
| 5 | `05_sampler_hnm.md` | `src/sampler/{weighted,score_ema,periodic_eval}.py`, `src/inference_pass.py` | Weighted/scheduled sampling + per-batch loss-EMA + every-10-epoch deep-eval refresh. |
| 6 | `06_model_training.md` | `src/model/*.py`, `src/lightning_module.py`, `src/ema_callback.py` | Backbone + FPN + RTMDet head + aux seg head + Lightning module. |
| 6.5 | `06_5_gru_rescorer.md` | `src/gru/{feature_cache,rescorer,train}.py` | Stage-2 BiGRU on frozen-detector backbone features. |
| 7 | `07_post_training_eval.md` | `src/eval/*.py` | CV + holdout volume metrics (FROC, AUROC, AP, bootstrap CIs, stratified). |
| 8 | `08_smoke_and_viz.md` | `scripts/smoke_train.py`, `scripts/visualize_predictions.py` | 5-min integration smoke + per-slice TP/FP/FN visualization. |

### 1.4 What this PRD adds on top of the 8 specs

- **Unified data layer** — replaces `manifest.csv` + `sidecars.jsonl` + `splits.json` with a single mnemonic-keyed `data/manifest.jsonl` + `data/cohort.json`. (Phase 0a, executed.)
- **Experiment configuration system** — Pydantic-based, `.py`-file experiments, fold-as-run, no Hydra, no CLI overrides. Modeled on rsi.
- **Run-output tree** — `runs/<exp>/fold{f}/` ownership of all model-dependent artifacts. Several specs originally placed these under `cache/v1/`; the PRD relocates them (see §13 spec amendments).
- **Cross-component contracts** — explicit data-on-disk and Python-interface contracts that the 8 specs reference but never centralize.
- **Invariants** — what must be true after preprocessing, during training, and during evaluation.
- **Phase plan** — the implementation agent's autonomous execution sequence Phase 0d → 8.
- **Spec amendments** — every place the PRD overrides one of the 8 specs (CC connectivity, QC signoff, runtime artifact paths, anthropic dep, MONAI vs scipy, etc.).

---

## 2. Repository organization

### 2.1 End-state folder layout

```
diaphragmatic-endometriosis/
├── CLAUDE.md                            # operational notes (uv, polars, quotagrp)
├── README.md
├── LICENSE
├── pyproject.toml                       # Python 3.12, pinned ML deps (Phase 0b ✅)
├── uv.lock
├── .python-version → 3.12               # Phase 0b ✅
├── .gitignore                           # cache/, runs/, outputs/, wandb/, .env (Phase 0b ✅)
├── .env                                 # WANDB_API_KEY (gitignored)
├── Justfile
│
├── data/                                # AUTHORITATIVE input contract
│   ├── manifest.jsonl                   # 608 rows, mnemonic-keyed (Phase 0a ✅)
│   ├── cohort.json                      # global splits/strat metadata (Phase 0a ✅)
│   ├── _archive/anon_id_mapping.csv     # full ANON↔mnemonic, forensic-only (Phase 0a ✅)
│   ├── _legacy/                         # original {manifest.csv, sidecars.jsonl,
│   │                                    #           splits.json, patient_id_mapping.csv}
│   ├── raw/{cross-validation,holdout}/{positive,negative}/<pid>.nii.gz
│   ├── lesion_masks/, liver_masks/, liver_rois/
│   ├── _pipeline/                       # legacy pipeline artifacts (gitignored)
│   ├── CLAUDE.md, README.md
│
├── eda/                                 # frozen post-migration; reference only
│   └── ...
│
├── agent/                               # planning artifacts
│   ├── training_pipeline_decisions_phase1.md   # AUTHORITATIVE source-of-truth
│   ├── complete_spec/
│   │   ├── 00_PRD.md                    # ← THIS DOCUMENT
│   │   ├── 01_preprocessing.md … 08_smoke_and_viz.md
│   │   └── HANDOFF.md
│   └── eda_synthesis.md, research_*.md
│
├── scripts/                             # cache-construction + dev workflows
│   ├── analyze_inplane_spacing.py       # one-time → constant in preprocess.py
│   ├── preprocess.py                    # Component 1 entrypoint
│   ├── build_lesion_bank.py             # Component 2 entrypoint
│   ├── qc_paste_review.py               # Component 4 dev workflow (PNG render only;
│   │                                    #   review is via Claude Code subagent)
│   ├── smoke_train.py                   # Component 8 smoke gate
│   ├── visualize_predictions.py         # Component 8 viz tool
│   ├── build_unified_manifest.py        # Phase 0a one-shot migration ✅
│   └── ... (existing: build_splits.py, migrate_*.py, run_totalseg.py, etc.)
│
├── src/                                 # importable as package `endo`
│   ├── __init__.py
│   ├── config/                          # Pydantic-based experiment configs
│   │   ├── experiment.py                # ExperimentConfig + sub-configs
│   │   ├── model.py                     # ModelConfig
│   │   ├── training.py                  # TrainingConfig
│   │   ├── sampler.py                   # SamplerConfig
│   │   ├── augmentation.py              # PasteConfig, GeometricConfig, IntensityConfig
│   │   ├── gru.py                       # GRUConfig, GRUTrainConfig
│   │   ├── eval.py                      # EvalConfig
│   │   ├── paths.py                     # PathsConfig (cache_root, runs_root)
│   │   └── loader.py                    # load_experiment(path)
│   ├── data/
│   │   ├── manifest.py                  # read_manifest_jsonl, read_cohort_json
│   │   ├── dataset.py                   # LesionDataset
│   │   ├── datamodule.py                # LesionDataModule
│   │   ├── samples.py                   # Sample, Batch dataclasses
│   │   └── collate.py                   # custom collate_fn
│   ├── augmentation/
│   │   ├── transform.py                 # TrainAugmentation
│   │   ├── paste.py
│   │   ├── geometric.py
│   │   ├── intensity.py
│   │   └── boxes.py
│   ├── lesion_bank.py                   # LesionBankEntry + load/save
│   ├── sampler/
│   │   ├── weighted.py                  # WeightedScheduledSampler
│   │   ├── score_ema.py                 # ScoreEMATracker
│   │   └── periodic_eval.py             # PeriodicDeepEvalCallback
│   ├── model/
│   │   ├── detector.py                  # LesionDetector
│   │   ├── fpn.py                       # 4-level FPN
│   │   ├── rtmdet_head.py               # VENDORED from mmdet (Phase 0d)
│   │   ├── assigner.py                  # VENDORED DynamicSoftLabelAssigner (Phase 0d)
│   │   ├── aux_seg_head.py
│   │   └── losses.py                    # compute_total_loss, dice_bce
│   ├── lightning_module.py              # LesionDetectorLM
│   ├── ema_callback.py                  # EmaCallback (timm ModelEmaV3)
│   ├── inference_pass.py                # SHARED inference primitive
│   ├── gru/
│   │   ├── feature_cache.py
│   │   ├── rescorer.py
│   │   └── train.py
│   ├── eval/
│   │   ├── wbf.py
│   │   ├── froc.py
│   │   ├── metrics.py
│   │   ├── threshold_search.py
│   │   └── stratified.py
│   ├── viz/
│   │   ├── tagging.py
│   │   └── render.py
│   ├── cli/
│   │   ├── run_experiment.py            # main CLI entrypoint
│   │   └── precheck.py
│   └── utils/
│       ├── seeding.py
│       ├── io.py
│       └── provenance.py
│
├── experiments/                         # ONE FILE PER EXPERIMENT
│   ├── baseline_rtmdet_p2.py            # Week-1 production baseline
│   ├── ablation_no_paste.py             # paste=0 ablation (phase-1 §12 candidate)
│   └── smoke.py                         # tiny config used by smoke script
│
├── tests/                               # mirrors src/ + scripts/
│   ├── preprocessing/  ├── lesion_bank/  ├── dataset/
│   ├── augmentation/   ├── sampler/      ├── model/
│   ├── gru/            ├── eval/         ├── viz/
│   └── smoke/
│
├── cache/                               # gitignored; cache-version-keyed
│   └── v1/
│       ├── code_version.txt
│       ├── preprocessed_manifest.jsonl
│       ├── gt_boxes.parquet
│       ├── volumes/<pid>/{volume.npy, lesion_mask.npy}
│       ├── border_bands/<pid>.npy
│       ├── lesion_banks/{lesion_bank_<sha8>.pkl, current.pkl→…, bank_provenance.json}
│       ├── runtime/
│       │   ├── cohort_local_std.json
│       │   ├── qc_paste_review.{json,html}
│       │   └── connectivity_lock.json    # see §7 invariant I.7
│       └── preprocessing.log
│
├── runs/                                # gitignored; experiment-keyed
│   └── <exp_name>_<uuid8>/
│       ├── experiment.yaml              # frozen materialized ExperimentConfig
│       ├── experiment.py                # COPY of source experiments/<name>.py
│       ├── provenance.json              # git sha, host, started_at, finished_at
│       ├── fold0/
│       │   ├── ckpts/{best.ckpt, last.ckpt}
│       │   ├── runtime/
│       │   │   ├── hard_negatives.json
│       │   │   └── deep_eval/epoch{n}_val.npz
│       │   ├── gru/{feature_cache/<pid>.npz, ckpt.pt}
│       │   └── viz/{*.png, manifest.csv}
│       ├── fold1/, fold2/, fold3/, fold4/
│       ├── eval/{eval_report.csv, eval_thresholds.json, eval.log}
│       └── holdout/run_<timestamp_uuid8>/
│           ├── eval_report.csv
│           └── invocation.json
│
├── outputs/                             # gitignored; ad-hoc tooling outputs
└── logs/ → /scratch/.../logs            # symlink for SLURM
```

### 2.2 What is gitignored

`cache/`, `runs/`, `outputs/`, `wandb/`, `.venv/`, `.env`, `.claude/`, `data/raw/`, `data/lesion_masks/`, `data/liver_masks/`, `data/liver_rois/`, `data/_pipeline/`, `data/_legacy/`, `data/_archive/`. The committed `data/` payload after Phase 0 is just `manifest.jsonl`, `cohort.json`, `CLAUDE.md`, `README.md`.

### 2.3 What is authoritative vs derived

- **Authoritative input:** `data/manifest.jsonl`, `data/cohort.json`, `data/raw/`, `data/lesion_masks/`, `data/liver_masks/`, `data/liver_rois/`. Frozen post-migration.
- **Authoritative spec:** `agent/training_pipeline_decisions_phase1.md` for locked decisions; `agent/complete_spec/00_PRD.md` (this doc) + `01..08.md` for implementation contracts.
- **Derived (regenerable):** everything in `cache/`, `runs/`, `outputs/`. The cache rebuilds via `preprocess.py` + `build_lesion_bank.py`; runs rebuild by re-training.

---

## 3. Experiment configuration system

### 3.1 Philosophy

Modeled on the rsi pattern (`packages/needle/experiments/needle/*.py`):

1. **One file per experiment.** `experiments/<name>.py` declares one `experiment: ExperimentConfig` object. To run a sweep, copy-paste the file.
2. **Pydantic, not dataclass.** Validation, YAML round-trip serialization, free schema documentation via type hints.
3. **No CLI overrides.** No `--learning-rate 5e-4`. If you want a different LR, copy the file. CLI flags are limited to `--fold`, `--device`, etc. — orchestration knobs, not config knobs.
4. **Immutable after first run.** Once `runs/<exp>/<exp>.yaml` is written, re-invoking with edited Python errors unless `--force-resync`. Prevents silent config drift across folds.
5. **Composition, not inheritance.** `ExperimentConfig` composes sub-configs (`ModelConfig`, `TrainingConfig`, `SamplerConfig`, `AugmentationConfig`, `GRUConfig`, `EvalConfig`, `PathsConfig`). No `BaseExperimentConfig → SubclassExperimentConfig`.

### 3.2 Dataclass tree

```python
# src/config/experiment.py
from pydantic import BaseModel, Field
from .model import ModelConfig
from .training import TrainingConfig
from .sampler import SamplerConfig
from .augmentation import AugmentationConfig, PasteConfig, GeometricConfig, IntensityConfig
from .gru import GRUConfig
from .eval import EvalConfig
from .paths import PathsConfig

class ExperimentConfig(BaseModel):
    """Top-level experiment declaration. One per experiments/<name>.py."""

    # ─── Identity ──────────────────────────────────────────────────
    uuid: str          # uuid4 string, pinned by hand at file creation time
    name: str          # short slug, e.g. "baseline-rtmdet-p2"
    description: str   # markdown, free-form
    tags: dict[str, str] = Field(default_factory=dict)

    # ─── Component configs (composition) ───────────────────────────
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    sampler: SamplerConfig
    augmentation: AugmentationConfig
    gru: GRUConfig
    eval: EvalConfig

    # ─── Reproducibility ───────────────────────────────────────────
    seed: int = 42

    # ─── Serialization ─────────────────────────────────────────────
    def to_yaml(self, path: Path) -> None: ...
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig": ...

    @model_validator(mode="after")
    def _check_uuid_format(self) -> Self: ...
    @model_validator(mode="after")
    def _check_paths_exist(self) -> Self: ...   # cache_root etc.
```

Each sub-config file mirrors the inline dataclass declarations from the 8 component specs (e.g., `PasteConfig` from Component 4 §3, `SamplerConfig` from Component 5 §4). The implementation agent ports these from the specs verbatim.

### 3.3 Experiment file template

```python
# experiments/baseline_rtmdet_p2.py
"""Week-1 production baseline.

ConvNeXt-tiny + custom 4-level FPN with P2 + vendored RTMDet head + aux seg head.
Lesion copy-paste augmentation (p=0.5, multi-paste). Stage-1 detector + Stage-2 GRU.
Target: volume AUROC ≥ 0.80, sens@2FP ≥ 0.70 on patient-level 5-fold CV.
"""

from pathlib import Path
from endo.config import (
    ExperimentConfig, ModelConfig, TrainingConfig, SamplerConfig,
    AugmentationConfig, PasteConfig, GeometricConfig, IntensityConfig,
    GRUConfig, EvalConfig, PathsConfig,
)

experiment = ExperimentConfig(
    uuid="b3a7f1e9-4c8a-4d2b-9f1c-0e6a8b9c1d2e",   # uuid4(), pinned by hand
    name="baseline-rtmdet-p2",
    description=(
        "## Week-1 Production Baseline\n"
        "RTMDet-S head + ConvNeXt-tiny backbone + 4-level FPN with P2 + aux seg head.\n"
        "Lesion copy-paste augmentation (p=0.5). 5-fold CV. GRU rescorer.\n\n"
        "Targets: volume AUROC ≥ 0.80, sens@2FP ≥ 0.70.\n"
        "See agent/training_pipeline_decisions_phase1.md.\n"
    ),
    tags={"phase": "1", "head": "rtmdet", "backbone": "convnext_tiny", "p2": "true"},

    paths=PathsConfig(
        data_root=Path("data/"),
        cache_root=Path("cache/v1/"),
        runs_root=Path("runs/"),
    ),
    model=ModelConfig(
        backbone_name="convnext_tiny.fb_in22k",
        in_channels=5,
        fpn_channels=256,
        fpn_strides=(4, 8, 16, 32),
        head_n_classes=1,
        head_stacked_convs=2,
        aux_seg_channels=64,
    ),
    training=TrainingConfig(
        max_epochs=60,
        batch_size=8,
        base_lr=2e-4,
        min_lr=1e-6,
        weight_decay=0.05,
        warmup_epochs=1,
        aux_seg_weight=0.3,
        ema_decay=0.999,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    ),
    sampler=SamplerConfig(
        epoch_mode="fixed_count",
        samples_per_epoch=6000,
        pos_frac_start=0.50, pos_frac_end=0.25, decay_epochs=30,
        neg_in_pos_vol_share=0.50,
        hard_pool_substitution_rate=0.30,
        hard_pool_start_epoch=5,
        deep_eval_refresh_every_epochs=10,
        deep_eval_start_epoch=10,
    ),
    augmentation=AugmentationConfig(
        paste=PasteConfig(p_any_paste=0.5, n_paste_sigma=1.0, n_paste_max=7),
        geometric=GeometricConfig(),
        intensity=IntensityConfig(),
    ),
    gru=GRUConfig(input_dim=768, hidden_dim=128, bidirectional=True, dropout_input=0.3,
                  epochs=20, lr=1e-3, weight_decay=0.01),
    eval=EvalConfig(
        use_gru=True,
        bootstrap_n=1000, bootstrap_seed=42,
        large_threshold_grid=[0.01, 0.03, 0.05, 0.10],
        small_threshold_grid=[0.10, 0.20, 0.30, 0.40, 0.50],
    ),
    seed=42,
)
```

### 3.4 `load_experiment` loader

```python
# src/config/loader.py
import importlib.util, sys
from pathlib import Path
from .experiment import ExperimentConfig

def load_experiment(path: str | Path) -> ExperimentConfig:
    """Dynamically import an experiment .py file and return its ExperimentConfig.

    Convention: the file must define a module-level `experiment: ExperimentConfig`.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")
    spec = importlib.util.spec_from_file_location("_experiment_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_experiment_module"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "experiment"):
        raise AttributeError(f"{path} must define `experiment: ExperimentConfig`")
    if not isinstance(module.experiment, ExperimentConfig):
        raise TypeError(f"{path}: `experiment` must be an ExperimentConfig instance")
    return module.experiment
```

### 3.5 Experiment / fold / run semantics

| Concept | Definition | Lives at |
|---|---|---|
| **Experiment** | A specific configuration declared in `experiments/<name>.py`. Identified by `(name, uuid)`. | One `.py` file. |
| **Run** | A single training pass = one experiment × one fold. Identified by `(name, uuid, fold)`. | One `runs/<name>_<uuid8>/fold{f}/` directory. |
| **Fold** | The validation partition index. Fold N = patients in `manifest.fold == N` are the val set; the OTHER 4 folds form the training set. | Directory naming + DataLoader filter. |
| **Holdout invocation** | An ad-hoc inference run on the 122 holdout patients using one or more checkpoints. Not a "run" in the same sense — produces no checkpoints. | `runs/<name>_<uuid8>/holdout/run_<timestamp_uuid8>/`. |

Per-fold patient assignment (locked in `data/cohort.json` and `manifest.jsonl`):

| `--fold` | Train (CV) | Val (CV) | Holdout (loaded only by `predict_holdout`) |
|---|---|---|---|
| 0 | folds {1,2,3,4} = 386 | fold 0 = 100 | not loaded |
| 1 | folds {0,2,3,4} = 387 | fold 1 = 99 | not loaded |
| 2 | folds {0,1,3,4} = 390 | fold 2 = 96 | not loaded |
| 3 | folds {0,1,2,4} = 390 | fold 3 = 96 | not loaded |
| 4 | folds {0,1,2,3} = 391 | fold 4 = 95 | not loaded |

`runs/<exp>/fold0/` holds artifacts from the run **where fold 0 was the validation set**.

### 3.6 WandB integration

- **Off by default.** No logging until the run-time flag is passed. This protects against polluting the dashboard with smoke tests and aborted runs. Do not turn on unless user explicitly specifies to.
- Opt-in: `--wandb` flag on `run_experiment.py train` (or `WANDB_MODE=online` env). Until you've landed one successful 10-epoch run, leave WandB off.
- WandB layout when on:
  - `project = "diaphragmatic-endometriosis"`
  - `group   = f"{experiment.name}_{experiment.uuid[:8]}"`
  - `name    = f"fold{fold}"`
  - `tags    = experiment.tags ∪ {"fold": str(fold)}`
  - `config  = experiment.model_dump(mode="json")`
- Smoke and viz scripts NEVER log to WandB by themselves.
- The viz script's W&B integration (Component 8 §2.7) only activates if `WANDB_RUN_ID` is set by an outer caller; it does not start a run.

### 3.7 Output organization per experiment

```
runs/<name>_<uuid8>/
├── experiment.yaml          # canonical materialized config (single source of truth)
├── experiment.py            # source file, copied at first invocation
├── provenance.json          # {git_sha, hostname, python_version, started_at, fold_status}
│
├── fold{0..4}/              # fold-as-run; each fold is independent
│   ├── ckpts/{best.ckpt, last.ckpt}
│   ├── runtime/
│   │   ├── hard_negatives.json
│   │   └── deep_eval/epoch{10,20,30,40,50,60}_val.npz
│   ├── gru/
│   │   ├── feature_cache/<pid>.npz
│   │   ├── ckpt.pt
│   │   └── gru_provenance.json
│   ├── viz/
│   │   ├── *.png
│   │   └── manifest.csv
│   ├── train.log
│   └── fold_status.json     # {started_at, finished_at, best_val_auroc, ckpt_path}
│
├── eval/                    # CV evaluation aggregates across all 5 folds
│   ├── eval_report.csv
│   ├── eval_thresholds.json
│   └── eval.log
│
└── holdout/                 # ad-hoc holdout inferences; one subdir per invocation
    └── run_<timestamp>_<uuid8>/
        ├── invocation.json  # {ckpts_used, gru_used, fold_subset, started_at, …}
        ├── eval_report.csv  # rows with scope=holdout
        └── eval.log
```

---

## 4. CLI surface

### 4.1 Subcommands of `run_experiment.py`

A single script — `src/cli/run_experiment.py`, invoked via `uv run python -m endo.cli.run_experiment`. Subcommands:

| Subcommand | Purpose |
|---|---|
| `train` | Train the detector for one or more folds. |
| `train_gru` | Train the GRU rescorer per fold (after detector training is done). |
| `eval` | Run CV evaluation on the experiment's 5 folds. |
| `predict_holdout` | Run inference on holdout patients (single ckpt or ensemble). |
| `viz` | Run the per-slice prediction visualization for a fold. |
| `smoke` | Run the 5-min smoke training gate. |
| `qc_paste` | Render the 30 paste-composite PNGs for human/agent review (dev workflow). |

### 4.2 Flags

```
Common to most subcommands:
  --experiment PATH       # path to experiments/<name>.py
  --device N              # CUDA device index (single GPU pinning)
  --fold {0..4}           # single fold to run
  --folds CSV             # multiple folds, e.g. "0,1,2,3,4" or "all"
  --devices CSV           # one device per fold (parallel multi-fold), e.g. "0,1,2,3,4"
  --force-resync          # overwrite runs/<exp>/experiment.yaml on edited file (use sparingly)

train-only:
  --wandb                 # opt in to WandB logging (default OFF)
  --resume                # resume from runs/<exp>/fold{f}/ckpts/last.ckpt

predict_holdout-only:
  --ckpts CSV             # comma-sep list of fold indices to load, or "all"
  --use-gru               # apply GRU rescoring (requires gru ckpts present)
```

### 4.3 Multi-GPU pattern

Single-process, single-fold, single-GPU — the architecture's parallelism axis is **folds**, not data parallelism within a fold. Multi-fold parallelism is implemented by spawning multiple processes:

```bash
# Single GPU: all 5 folds sequential
uv run python -m endo.cli.run_experiment train --experiment experiments/baseline.py --folds all --device 0

# Multi-GPU on a future cluster: fan-out via --devices
uv run python -m endo.cli.run_experiment train --experiment experiments/baseline.py --folds all --devices 0,1,2,3,4
# Internally: multiprocessing.spawn(5 processes), each with CUDA_VISIBLE_DEVICES=N

# Manual cross-shell parallelism (works even without --devices):
uv run python -m endo.cli.run_experiment train --experiment experiments/baseline.py --fold 0 --device 0 &
uv run python -m endo.cli.run_experiment train --experiment experiments/baseline.py --fold 1 --device 1 &
```

No DDP, no SLURM strategy, no FSDP. The model fits on one GPU; data parallelism within a fold is unnecessary at our scale.

---

## 5. Data contracts

This section enumerates every artifact that crosses a component boundary. The contract is: **producer writes the file according to this schema; consumers may rely on the schema being honored exactly.**

### 5.1 INPUT contracts (data/, frozen post-Phase-0)

#### 5.1.1 `data/manifest.jsonl`

- **Producer:** `scripts/build_unified_manifest.py` (Phase 0a, executed).
- **Consumers:** `scripts/preprocess.py`, `src/data/manifest.py`, every downstream component.
- **Layout:** one JSON object per line. UTF-8. 608 lines total.
- **Key:** `patient_id` (mnemonic, primary key everywhere downstream).
- **Schema (locked):**

```jsonc
{
  "patient_id": str,                       // mnemonic, unique across manifest
  "cohort": "cross-validation" | "holdout",
  "label": "positive" | "negative",
  "fold": 0|1|2|3|4 | null,                // null iff cohort=="holdout"
  "soft_negative": bool,                   // 57 patients reclassified positive→negative
  "paths": {                               // all relative to data/
    "raw":         "raw/.../<pid>.nii.gz",
    "lesion_mask": "lesion_masks/.../<pid>_lesion_mask.nii.gz" | null,  // null iff label=="negative"
    "liver_mask":  "liver_masks/.../<pid>_liver_mask.nii.gz",
    "liver_roi":   "liver_rois/.../<pid>_liver_roi.nii.gz"
  },
  "hashes": {
    "raw_sha256":         str,             // hex sha256 of raw .nii.gz (idempotency key)
    "liver_mask_sha256":  str | null
  },
  "geometry": {
    "shape":          [int, int, int]|null,   // (X, Y_slices, Z) in NATIVE pre-resample voxel coords
    "n_slices":       int|null,                // == shape[1]; through-plane axis
    "pixel_spacing_xz_mm_hint": [float|null, float|null],  // HINT only; preprocessor reads NIfTI
    "slice_spacing_mm_bids_hint": float|null,  // HINT only; preprocessor uses NIfTI zoom_y
    "orientation":    "RAS"
  },
  "scanner": {
    "manufacturer":   "GE",
    "model":          "SIGNA Artist" | "SIGNA Explorer",
    "magnetic_field_strength_t": 1.5,
    "variant":        "A" | "B" | "unknown",   // A=1.5mm reconstruction, B=3.6mm
    "series_description": str
  },
  "liver_roi_bbox": {
    "x0": int, "x1": int, "y0": int, "y1": int, "z0": int, "z1": int,
    "extent_x_mm": float, "extent_y_mm": float, "extent_z_mm": float
  },
  "dicom": {
    "echo_time_s":      float|null,
    "repetition_time_s":float|null,
    "flip_angle":       float|null,
    "scanning_sequence": str|null,
    "image_type":       [str, ...]|null,
    "bids":             { /* full BIDS sidecar object verbatim */ }
  },
  "provenance": {
    "migration_timestamp": str (ISO-8601),
    "anon_id":             str,                // for forensic traceability ONLY
    "selected_subvolume":  bool,
    "had_multi_canonical": bool,
    "volume_index":        int|null
  }
}
```

- **Invariants** (`scripts/build_unified_manifest.py` enforces all of these on write):

  | I.1.1 | exactly 608 lines |
  | I.1.2 | every `patient_id` is unique |
  | I.1.3 | every `(cohort=="holdout") ⇔ (fold is null)` |
  | I.1.4 | every `(label=="positive") ⇔ (paths.lesion_mask is not null)` |
  | I.1.5 | fold counts sum to 100, 99, 96, 96, 95 (matches `cohort.json.fold_summary`) |
  | I.1.6 | label distribution: 108 positive, 500 negative |
  | I.1.7 | scanner.model ∈ {SIGNA Artist (369), SIGNA Explorer (239)} |
  | I.1.8 | scanner.variant ∈ {A (495), B (113), unknown (0)} |
  | I.1.9 | for every row, all paths.* point to existing files under `data/` |
  | I.1.10 | every `provenance.anon_id` round-trips against `_archive/anon_id_mapping.csv` |

#### 5.1.2 `data/cohort.json`

- **Producer:** `scripts/build_unified_manifest.py`.
- **Consumers:** `scripts/build_splits.py` (if ever re-run), eval stratification, `provenance.json` materialization.
- **Schema:**

```jsonc
{
  "version": "1.0",
  "generated_at": ISO-8601,
  "code_version": str,                    // git sha
  "n_patients_total": 608,
  "splits": {
    "seed": 42,
    "n_folds": 5,
    "stratification": {
      "positives": ["manufacturer_model_name"],
      "negatives": ["manufacturer_model_name", "slice_thickness_bin"],
      "thickness_bin_rule": "<=4.0mm vs >4.0mm on canonical sequence",
      "thickness_bin_collapsed_for_positives": true
    },
    "frozen_at": ISO-8601                 // when build_splits.py originally ran
  },
  "phase1_targets": {"cv_pos":86,"cv_neg":400,"holdout_pos":22,"holdout_neg":100},
  "fold_summary": {
    "fold0": {"n":100,"pos":18,"neg":82},
    ...
    "holdout": {"n":122,"pos":22,"neg":100}
  },
  "n_soft_negatives": 57,
  "soft_negative_pids": [str, ...]        // mnemonic ids
}
```

#### 5.1.3 `data/_archive/anon_id_mapping.csv`

- **Producer:** `scripts/build_unified_manifest.py`.
- **Consumers:** **none in the training stack.** Forensic-only — used to trace a mnemonic back to the original DICOM directory.
- **Schema:** `(anon_id: str, mnemonic_id: str, in_assignments: bool, used_in_phase1: bool)`. 5,089 rows.

#### 5.1.4 Raw NIfTI / mask files under `data/raw/`, `data/lesion_masks/`, `data/liver_masks/`, `data/liver_rois/`

- **Producer:** historical migration (`scripts/migrate_local_copy_to_data.py`, `scripts/run_totalseg.py`, etc.). Frozen.
- **Consumer:** `scripts/preprocess.py` only.
- **Constraints:** all 608 are RAS, shape `(512, N, 512)` with axis-1 the through-plane axis, GE 1.5T. Liver ROI is 20-mm dilation of the TotalSeg liver mask.

### 5.2 CACHE contracts (cache/v1/, EXPERIMENT-INDEPENDENT)

The cache is keyed on `(preprocessing code SHA, target_spacing, target_shape, raw_sha256)`. Multiple experiments share a single cache.

#### 5.2.1 `cache/v1/preprocessed_manifest.jsonl`

- **Producer:** `scripts/preprocess.py` (Component 1).
- **Consumers:** `src/data/datamodule.py`, `scripts/build_lesion_bank.py`, `src/eval/*`.
- **Layout:** JSONL, one row per processed patient. **Note:** Component 1 spec §4.2 originally specified CSV; PRD amends to JSONL for convention consistency (see §13 amendment A.1).
- **Schema:** all fields from Component 1 §4.2 plus:

```jsonc
{
  "patient_id": str,                       // FK to data/manifest.jsonl
  "cohort":     "cross-validation"|"holdout",
  "label":      "positive"|"negative",
  "fold":       int|null,
  "scanner_model": "SIGNA Artist"|"SIGNA Explorer",
  "variant":    "A"|"B"|"unknown",
  "cache_volume_path":      str,           // relative to cache/v1/
  "cache_lesion_mask_path": str|null,
  "cache_border_band_path": str|null,
  "roi_bbox_post_resample": {"x0":int,"x1":int,"y0":int,"y1":int,"z0":int,"z1":int},
  "pad_offset": {"x":int,"y":int,"z":int},
  "n_lesion_ccs": int,                     // 0 for negatives
  "roi_norm": {"p1":float,"p99":float,"mean":float,"std":float},
  "lesion_vs_ring_z": float|null,          // null for negatives
  "raw_sha256": str,
  "code_version": str
}
```

- **Invariants:**

  | I.2.1 | exactly 608 rows |
  | I.2.2 | every `patient_id` joins back to `data/manifest.jsonl` |
  | I.2.3 | every `cache_volume_path` points to a `(408, 174, 408)` float16 array |
  | I.2.4 | every `cache_lesion_mask_path` (positives) points to a `(408, 174, 408)` uint8 array in {0, 1} |
  | I.2.5 | every CV `cache_border_band_path` exists; every holdout `cache_border_band_path` is null |
  | I.2.6 | sum of `n_lesion_ccs` = 197 (matches phase-1 §1.3 exactly) |
  | I.2.7 | for every positive: `lesion_vs_ring_z >= 0.121` (regression check vs phase-1 §1.4 min) |
  | I.2.8 | all `code_version` equal; all `raw_sha256` distinct |

#### 5.2.2 `cache/v1/volumes/<patient_id>/{volume.npy, lesion_mask.npy}`

- **Producer:** `scripts/preprocess.py`.
- **Consumers:** `src/data/datamodule.py`, `scripts/build_lesion_bank.py`, `src/inference_pass.py`, `src/gru/feature_cache.py`.
- **`volume.npy`:** shape `(408, 174, 408)`, dtype `float16`, ROI z-scored. Center-padded with 0 (= cohort mean post-z-score). Axes `(X, Y_slices, Z)`.
- **`lesion_mask.npy`:** positives only. Shape `(408, 174, 408)`, dtype `uint8` in `{0, 1}`. Same coordinate frame.
- **Liver mask is NOT in the cache** — it is consumed only inside Component 1 to derive `border_band` and discarded.

#### 5.2.3 `cache/v1/border_bands/<patient_id>.npy`

- **Producer:** `scripts/preprocess.py`.
- **Consumers:** `src/data/datamodule.py` → `src/augmentation/transform.py`.
- **Layout:** shape `(M, 3)`, dtype `int16`, columns `(x, y, z)` voxel coords in the cached `(408, 174, 408)` frame. Right-hemidiaphragm 2-mm shell only.
- **Coverage:** present for all CV cohort patients (positives + negatives, 486 files). **NOT** present for the 122 holdout patients (paste augmentation never targets holdout).

#### 5.2.4 `cache/v1/gt_boxes.parquet`

- **Producer:** `scripts/preprocess.py`.
- **Consumers:** `src/data/dataset.py`, `src/eval/wbf.py`, `src/eval/froc.py`.
- **Schema:** PRD locks the full schema (Component 1 §4.1 listed only a subset):

  | Column | Type | Notes |
  |---|---|---|
  | `patient_id` | string | FK to manifest |
  | `slice_y` | int32 | center-slice index in cropped+padded `(408, 174, 408)` frame |
  | `cc_id` | int32 | 1..n_cc within the patient (matches CC ordering from `scipy.ndimage.label`) |
  | `x1`, `z1`, `x2`, `z2` | int32 | half-open box coords; `x in [0, 408)`, `z in [0, 408)` |
  | `box_max_dim_mm` | float32 | `max((x2-x1)*0.82, (z2-z1)*0.82)` |

- **Invariants:**

  | I.3.1 | total CC count over distinct `(patient_id, cc_id)` pairs = 197 |
  | I.3.2 | total row count ∈ `[1300, 1450]` (matches phase-1 §1.3 ≈ 1,365) |
  | I.3.3 | every `(x1, z1, x2, z2)` satisfies `0 <= x1 < x2 <= 408` AND `0 <= z1 < z2 <= 408` |
  | I.3.4 | every `slice_y ∈ [0, 174)` |
  | I.3.5 | every `patient_id` exists in `preprocessed_manifest.jsonl` with `label == "positive"` |

#### 5.2.5 `cache/v1/lesion_banks/`

- **Producer:** `scripts/build_lesion_bank.py` (Component 2).
- **Consumers:** `src/augmentation/transform.py` via `src/lesion_bank.py`.
- **Files:**
  - `lesion_bank_<git_sha8>.pkl` — pickled `list[LesionBankEntry]`. Schema in §6.4.
  - `current.pkl` — symlink to the most-recent SHA-keyed pkl. **DataModules and TrainAugmentation load `current.pkl` exclusively** unless an experiment explicitly overrides via `paths.lesion_bank`.
  - `bank_provenance.json` — build metadata, see Component 2 §4.2.
- **Invariants:**

  | I.4.1 | `bank_provenance.json` lists exactly 86 donor patients (matches phase-1 §1.1 CV positives) |
  | I.4.2 | `donor_patient_ids ∩ holdout_patient_ids = ∅` (cohort filter enforced) |
  | I.4.3 | total CC count ∈ `[140, 180]` (point estimate ~157, ±15% for connectivity sensitivity) |
  | I.4.4 | `bank_provenance.json` connectivity field matches `cache/v1/runtime/connectivity_lock.json.connectivity` |

#### 5.2.6 `cache/v1/runtime/cohort_local_std.json`

- **Producer:** `src/augmentation/transform.py` (lazy; first time `TrainAugmentation` is constructed against this cache).
- **Consumer:** `src/augmentation/transform.py` (paste-site rejection threshold, Component 4 §5.4).
- **Schema:** `{"cohort_median_local_std": float, "n_volumes_sampled": int, "samples_per_volume": int, "computed_at": ISO-8601, "code_version": str}`.

#### 5.2.7 `cache/v1/runtime/qc_paste_review.{json,html}`

- **Producer:** `scripts/qc_paste_review.py` (dev workflow; PNG render + Claude Code subagent review via Task tool).
- **Consumer:** human reviewer (advisory only — **NO code path gates on these files**, see §13 amendment A.2).
- Tier-3/Tier-4 QC are dev workflow artifacts. They are not part of the production API.

#### 5.2.8 `cache/v1/runtime/connectivity_lock.json` (NEW per PRD)

- **Producer:** `scripts/preprocess.py`'s connectivity probe (one-time at first cache build, see §13 amendment A.3).
- **Consumers:** `scripts/build_lesion_bank.py`, `src/augmentation/boxes.py` (for online box re-derivation).
- **Schema:** `{"connectivity": "6"|"26", "structure": [[ ... ]], "n_ccs_in_cohort": int, "computed_at": ISO-8601, "code_version": str}`.
- **Invariant I.5.1:** `n_ccs_in_cohort == 197` (matches phase-1 §1.3 — this is the exact discriminator that picks 6- vs 26-connectivity).

### 5.3 RUN contracts (runs/<exp>/, EXPERIMENT-DEPENDENT)

#### 5.3.1 `runs/<exp>/{experiment.yaml, experiment.py, provenance.json}`

- **Producer:** `src/cli/run_experiment.py` first-invocation bootstrap.
- **Consumers:** all subsequent fold runs of the same experiment; `eval`, `predict_holdout`, `viz`.
- **`experiment.yaml`:** materialized `ExperimentConfig.to_yaml()`. Single source of truth for "what config did this run use."
- **`experiment.py`:** byte-for-byte copy of `experiments/<name>.py` at first invocation.
- **`provenance.json`:** `{git_sha, hostname, python_version, python_executable, started_at, fold_status: {0..4: "pending"|"running"|"complete"|"failed"}}`. Updated atomically per fold.

**Drift detection:** subsequent invocations reload `experiments/<name>.py` and compare against `experiment.yaml`. Any field difference → error unless `--force-resync`. This prevents the situation where you edit a config halfway through a 5-fold sweep and the folds train on different configs.

#### 5.3.2 `runs/<exp>/fold{f}/ckpts/`

- **Producer:** `pl.callbacks.ModelCheckpoint` inside `train_one_fold`.
- **Consumers:** `predict_holdout`, `gru/feature_cache`, `viz`, `eval`.
- **Files:**
  - `best.ckpt` — the checkpoint with the highest `val/slice_auroc` seen during training. Persists EMA shadow alongside live weights via `EmaCallback.on_save_checkpoint`.
  - `last.ckpt` — the most recent checkpoint (for resume).
- **Standard Lightning checkpoint format:** state_dict, optimizer_state, lr_scheduler_state, hyper_parameters, plus `ema_state_dict`.

#### 5.3.3 `runs/<exp>/fold{f}/runtime/hard_negatives.json`

- **Producer:** `PeriodicDeepEvalCallback` (Component 5 §6) — refreshes every 10 epochs starting at epoch 10.
- **Consumer:** `WeightedScheduledSampler` (Component 5 §4) at epoch boundaries.
- **Schema:** `{"epoch_written": int, "model_checkpoint_epoch": int, "slice_indices": [int, ...], "n_slices": int, "score_threshold": float}`.
- **Replacement protocol:** atomic — write to `.tmp`, then `os.replace`. Sampler reads at `__iter__` time; if the file is missing or corrupt, sampler treats hard pool as empty (logs warning, continues).
- **Path correction vs Component 5 spec:** original spec placed this under `cache/v1/runtime/`. PRD relocates to `runs/<exp>/fold{f}/runtime/` because it is model-dependent (see §13 amendment A.4).

#### 5.3.4 `runs/<exp>/fold{f}/runtime/deep_eval/epoch{n}_val.npz`

- **Producer:** `PeriodicDeepEvalCallback`.
- **Consumer:** `src/eval/froc.py`, `src/eval/wbf.py` (loaded by `eval` subcommand to avoid re-inference).
- **Schema:** Component 5 §8.2 — compressed `np.savez_compressed` with arrays `patient_ids` (str), `slice_ys` (int32), `boxes_flat` (float32 (M,4)), `scores_flat` (float32 (M,)), `box_offsets` (int32 CSR), `aux_seg_max` (float32 per slice).
- **Path correction:** also moved out of `cache/v1/` (§13 amendment A.4).

#### 5.3.5 `runs/<exp>/fold{f}/gru/feature_cache/<pid>.npz`

- **Producer:** `src/gru/feature_cache.py` (Component 6.5).
- **Consumer:** `src/gru/train.py`, `src/gru/rescorer.py` (at eval time).
- **Schema:** `feats: (N_valid_slices, 768) float16` (GAP-pooled stage-3 backbone features), `slice_ys: (N_valid_slices,) int32`, `patient_label: () int8`.
- **Path correction:** Component 6.5 spec placed under `cache/v1/gru_features/fold{f}/`. PRD relocates (§13 amendment A.4).

#### 5.3.6 `runs/<exp>/fold{f}/gru/ckpt.pt`

- **Producer:** `src/gru/train.py`.
- **Consumer:** `src/gru/rescorer.py` at `eval --use-gru` and `predict_holdout --use-gru`.
- **Schema:** `{state_dict, config (GRUConfig dump), epoch, val_auroc}`.
- **Path correction:** Component 6.5 placed at `cache/v1/gru_ckpts/fold{f}.pt`. PRD relocates (§13 amendment A.4).

#### 5.3.7 `runs/<exp>/fold{f}/viz/`

- **Producer:** `scripts/visualize_predictions.py` (Component 8).
- **Consumer:** human inspection.
- **Files:** `{positive,negative}_<pid>_{tp,fp,fn}_slice<y>.png` + `manifest.csv`.

#### 5.3.8 `runs/<exp>/eval/`

- **Producer:** `eval` subcommand (Component 7).
- **Consumer:** human inspection; `predict_holdout` reads `eval_thresholds.json` to apply CV-pooled threshold.
- **`eval_report.csv`:** schema in Component 7 §4.1. Append-only; multiple eval runs add new rows under fresh `run_id`s.
- **`eval_thresholds.json`:** schema in Component 7 §4.2.

#### 5.3.9 `runs/<exp>/holdout/run_<timestamp>_<uuid8>/`

- **Producer:** `predict_holdout` subcommand (Component 7).
- **Consumer:** human inspection.
- **Files:** `eval_report.csv` (rows scoped `holdout`), `invocation.json`, `eval.log`.
- **Holdout discipline:** the only enforcement is the DataModule guard (`allow_holdout=False` by default; `predict_holdout` is the sole caller setting it `True`). Per Q1.4 in planning: NO global lockfile, NO `--i-mean-it`. Each invocation is a fresh subdir; re-running just adds another. The "touch holdout once" rule is enforced by user discipline (§13 amendment A.5).

---

## 6. Runtime contracts (Python interfaces)

These are the cross-component Python APIs. Implementation agent honors these signatures exactly.

### 6.1 `ExperimentConfig` and `load_experiment`

See §3.2–3.4. The experiment file convention is the single contract between `experiments/` and the `run_experiment.py` CLI.

### 6.2 `Sample` dataclass (Component 3 §4.1)

```python
@dataclass
class Sample:
    volume_5ch: np.ndarray         # (5, 384, 384) float32
    lesion_mask_center: np.ndarray # (384, 384) uint8
    boxes: np.ndarray              # (N, 4) float32, (x1, z1, x2, z2)
    labels: np.ndarray             # (N,) int64, all 0
    patient_id: str
    slice_y: int                   # in cropped+padded (384, 160, 384) frame
    is_positive_volume: bool
    is_positive_slice: bool
    pad_offset: tuple[int, int, int]
    # Forwarded to augmentation only (None at val/inference):
    volume_full_cropped: np.ndarray | None       # (384, 160, 384) float32
    lesion_mask_full_cropped: np.ndarray | None  # (384, 160, 384) uint8
    border_band_coords: np.ndarray | None        # (M, 3) int16 in cropped frame
```

### 6.3 `Batch` dataclass (Component 3 §4.2)

```python
@dataclass
class Batch:
    volume_5ch: torch.Tensor        # (B, 5, 384, 384) float32
    lesion_mask_center: torch.Tensor  # (B, 384, 384) uint8
    boxes:  list[torch.Tensor]      # length B; per-image (N_i, 4)
    labels: list[torch.Tensor]      # length B; per-image (N_i,)
    patient_ids: list[str]
    slice_ys: torch.Tensor          # (B,) int64
    is_positive_volume: torch.Tensor  # (B,) bool
    is_positive_slice:  torch.Tensor  # (B,) bool
```

The custom collate_fn in `src/data/collate.py` produces this. RTMDet head accepts `list[Tensor]` for boxes/labels (variable N per image).

### 6.4 `LesionBankEntry` dataclass (Component 2 §4.1)

```python
@dataclass(frozen=True)
class LesionBankEntry:
    donor_patient_id: str
    donor_cc_id: int
    tight_mask:       np.ndarray   # (Δx, Δy, Δz) uint8
    tight_intensities: np.ndarray  # (Δx, Δy, Δz) float32, post-z-score
    tight_shell_mask: np.ndarray   # (Δx, Δy, Δz) uint8, 1mm anisotropic outer dilation
    centroid_offset_in_tight: tuple[int, int, int]
    z_extent_voxels:  int
    intensity_mean:   float
    intensity_std:    float
    physical_extent_mm: tuple[float, float, float]
```

### 6.5 `SliceScore` dataclass (Component 5 §7)

```python
@dataclass
class SliceScore:
    patient_id: str
    slice_y: int
    boxes:    np.ndarray   # (N, 4) post per-slice NMS
    scores:   np.ndarray   # (N,)
    aux_seg_max: float     # max sigmoid value of aux seg head, slice-level presence proxy
```

### 6.6 `LesionDataModule.inference_dataloader()` + holdout guard

```python
class LesionDataModule(pl.LightningDataModule):
    def inference_dataloader(self, patient_ids: list[str]) -> DataLoader:
        """Yields slices in (patient_id ASC, slice_y ASC) order.
           batch_size = self.batch_size, num_workers = self.num_workers,
           augment = None, shuffle = False, drop_last = False.
           Caller groups results by patient_id for WBF aggregation."""
        if not self.allow_holdout:
            holdout_overlap = self._holdout_pids & set(patient_ids)
            if holdout_overlap:
                raise HoldoutAccessError(
                    f"Refusing to load holdout patients {sorted(holdout_overlap)} "
                    f"with allow_holdout=False."
                )
        ...
```

The holdout guard fires here AND in `setup()`. Two-layer defense.

### 6.7 `inference_pass()` (Component 5 §7)

```python
def inference_pass(
    model: pl.LightningModule,
    datamodule: LesionDataModule,
    patient_ids: list[str],
    split: Literal["val", "train_negatives", "holdout"],
    batch_size: int = 16,
) -> dict[str, list[SliceScore]]:
    """Run model in eval mode over every valid slice of every patient.
       Returns {patient_id: [SliceScore for each slice_y in valid range]}.
       Caller groups for WBF + FROC."""
```

Single implementation, used by:
- `PeriodicDeepEvalCallback` (Component 5).
- `eval` subcommand (Component 7).
- `predict_holdout` subcommand (Component 7).
- `gru/feature_cache.py` (Component 6.5) calls a backbone-only sibling, not this primitive.

### 6.8 `RTMDetHead` public API (Component 6 §5.2)

```python
class RTMDetHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, feat_channels: int,
                 stacked_convs: int, strides: tuple[int, ...], share_conv: bool = False): ...

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """(cls_scores, bbox_preds) per FPN level."""

    def loss(self, cls_scores, bbox_preds,
             gt_boxes_per_image:  list[Tensor],   # (N_i, 4) (x1,z1,x2,z2)
             gt_labels_per_image: list[Tensor],   # (N_i,)
             image_size:          tuple[int, int]) -> dict[str, Tensor]:
        """{'loss_cls': ..., 'loss_bbox': ...}"""

    def predict(self, cls_scores, bbox_preds, image_size,
                score_threshold=0.05, nms_iou_threshold=0.5,
                max_per_image=100) -> list[dict]:
        """Per-image {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}."""
```

### 6.9 `compute_total_loss` (Component 6 §6)

```python
def compute_total_loss(
    det_losses: dict[str, torch.Tensor],   # from RTMDetHead.loss
    aux_seg_logits: torch.Tensor,           # (B, 1, 384, 384)
    aux_seg_target: torch.Tensor,           # (B, 384, 384) uint8
    aux_seg_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns (total_scalar_loss, components_dict_for_logging)."""
```

### 6.10 `rescore_detector_outputs` (Component 6.5 §8)

```python
def rescore_detector_outputs(
    gru_ckpt_path: Path,
    feature_cache_path: Path,
    detector_boxes_per_slice: dict[int, dict],  # {slice_y: {boxes, scores}}
) -> dict[int, dict]:
    """Multiplies each box's score by the GRU's per-slice probability.
       Returns same-shaped dict with rescored scores."""
```

---

## 7. Invariants — post-preprocessing

These must hold after Phase 1 (preprocessing) completes successfully. Most have automated checks in Component 1's acceptance gates and the contracts in §5.2.

| ID | Invariant | Where checked |
|---|---|---|
| I.7.1 | All 608 cohort volumes resampled to `(0.82, 1.5, 0.82) mm` voxels (target spacing pinned in `scripts/preprocess.py`'s `TARGET_SPACING` constant) | preprocess.py + cohort acceptance gate |
| I.7.2 | All cached arrays have shape `(408, 174, 408)`, dtype `float16` for volumes and `uint8` for masks | I.2.3, I.2.4 |
| I.7.3 | `gt_boxes.parquet` total `(patient_id, cc_id)` pair count = 197; total row count ≈ 1,365 | I.3.1, I.3.2 |
| I.7.4 | `lesion_vs_ring_z >= 0.121` for every positive (regression vs phase-1 §1.4) | I.2.7 |
| I.7.5 | Holdout patients have NO `border_bands/<pid>.npy` file; CV patients (positives + negatives) all have one | I.2.5 |
| I.7.6 | `data/cohort.json` and `cache/v1/preprocessed_manifest.jsonl` agree on every fold assignment for every patient | preprocess.py precheck |
| I.7.7 | `cache/v1/lesion_banks/current.pkl` contains only CCs from `cohort=='cross-validation' AND label=='positive'` patients (no holdout leak) | Component 2 acceptance gate, I.4.2 |
| I.7.8 | `cache/v1/runtime/connectivity_lock.json` written; subsequent components import the locked connectivity | I.5.1 |
| I.7.9 | Cache disk usage in `[30, 50] GB` | preprocess.py cohort gate |
| I.7.10 | Re-running `preprocess.py` with no source changes is a no-op (idempotency) | Component 1 §7 |

---

## 8. Invariants — at training time

These must hold during a `run_experiment train` invocation.

| ID | Invariant | Where checked |
|---|---|---|
| I.8.1 | DataModule constructed with `allow_holdout=False`; no holdout patient enters `train_dataloader` or `val_dataloader` | Component 3 §11 + integration tests |
| I.8.2 | `WeightedScheduledSampler.set_epoch(n)` called once per epoch boundary by Lightning | Lightning's automatic sampler.set_epoch hook + sentinel test |
| I.8.3 | `ScoreEMATracker.update()` called for every negative slice in every training batch | Component 6 LightningModule training_step + test M13 |
| I.8.4 | `PeriodicDeepEvalCallback` fires only when `epoch >= 10 AND epoch % 10 == 0` | callback `_should_run` + test S12 |
| I.8.5 | `EmaCallback` swaps to EMA weights for validation and for the deep-eval pass; restores live weights afterward | EmaCallback hooks + Component 5 §6 deep-eval EMA wiring |
| I.8.6 | `experiment.yaml` matches the live `experiment.py` byte-for-byte (drift guard) | run_experiment.py first-call check |
| I.8.7 | `provenance.json` `fold_status[f]` updated atomically: `pending → running → complete | failed` | run_experiment.py |
| I.8.8 | All training input tensors are `(B, 5, H=Z=384, W=X=384)` with boxes `(x1, z1, x2, z2) ≡ (W_min, H_min, W_max, H_max)` (no permutation between collate and head) | Component 4 §9 transpose + test T1.18, T1.19 |
| I.8.9 | EMA shadow buffer is fp32 (not bf16) | EmaCallback `ModelEmaV3` config + test M14 |
| I.8.10 | Vendored `DynamicSoftLabelAssigner` byte-equals MMDet's on a fixed-input parity test | Component 6 §5.3 + test M8 |

---

## 9. Invariants — at evaluation time

| ID | Invariant | Where checked |
|---|---|---|
| I.9.1 | `eval` subcommand appends rows to `eval_report.csv`; never overwrites existing rows | Component 7 §9.2 test E15 |
| I.9.2 | `eval --use-gru` produces an additional `rescored=true` row set; without flag rows are `rescored=false` only | Component 7 test E12 |
| I.9.3 | `predict_holdout` is the only caller that sets `allow_holdout=True` on the DataModule | run_experiment.py source review + integration test |
| I.9.4 | When `predict_holdout --ckpts all`, the ensemble loads exactly 5 detector ckpts; `--use-gru` adds 5 GRU ckpts | invocation.json + test E11-style |
| I.9.5 | CV-pooled WBF threshold from `eval_thresholds.json["ensemble_threshold"]` is the threshold applied during `predict_holdout` (unless caller overrides) | Component 7 §5.2 |
| I.9.6 | Bootstrap CIs computed at the **patient level**, 1,000 resamples by default, seed 42 | Component 7 §6.2 |
| I.9.7 | Stratified breakdowns produced for {scanner, variant, slice_thickness_bin}; bootstrap resampling restricted within stratum | Component 7 §7 |

---

## 10. End-to-end execution sequence

The phase plan. Phases 0a–0c were **executed by the planning agent**. Phase 0d onward is the **implementation agent's autonomous responsibility.**

### Phase 0 — Bootstrap

#### Phase 0a — Data unification ✅ EXECUTED 2026-04-28

Outputs:
- `data/manifest.jsonl` (608 rows, mnemonic-keyed, sha256[:16] = `b87d4bb40559426c`)
- `data/cohort.json`
- `data/_archive/anon_id_mapping.csv` (5,089 rows, forensic-only)
- Originals moved to `data/_legacy/`.

Verification (executed):
- 608 lines in manifest.jsonl ✅
- Label distribution: 108 positive, 500 negative ✅
- Cohort: 486 cross-validation, 122 holdout ✅
- Fold counts: {0:100, 1:99, 2:96, 3:96, 4:95} ✅ (matches phase-1 §1.1)
- Scanner variant: 495 A, 113 B ✅ (matches phase-1 §1.2)
- 0 positives without lesion_mask path ✅
- 0 negatives with lesion_mask path ✅
- Soft negatives: 57 ✅

Script: `scripts/build_unified_manifest.py`. Idempotent (re-run is safe and a no-op once `_legacy/` is populated).

#### Phase 0b — Python pin + dependencies + .gitignore ✅ EXECUTED 2026-04-28

- `.python-version` → `3.12`
- `pyproject.toml` updated: pinned ML stack (torch, lightning, timm, ensemble-boxes, picai-eval, wandb, pydantic, etc.); dev group includes mmdet/mmengine/mmcv for vendoring + parity test
- `.gitignore` extended: `cache/`, `runs/`, `outputs/`, `wandb/`, `.env`, `data/_legacy/`, `data/_archive/`

#### Phase 0c — `src/` skeleton ✅ EXECUTED 2026-04-28

Created the directory tree under `src/` and `tests/` with empty `__init__.py` files. The implementation agent fills in the files.

#### Phase 0d — `uv sync` + MMDet vendoring ⏳ PENDING (implementation agent's first task)

Steps:

1. **Create the new venv against Python 3.12 and sync.**
    ```bash
    rm -rf .venv
    uv venv --python 3.12
    uv sync
    ```
    If `mmcv` build fails (CUDA mismatch is the common failure), pin to a wheel matching the installed torch+CUDA combination, e.g.:
    ```bash
    uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
    ```
    Verify: `uv run python -c "import torch, lightning, timm, mmdet, mmengine, mmcv; print(torch.__version__, torch.cuda.is_available())"` — should print version + `True`.

2. **Locate the vendor sources in `.venv`:**
    ```bash
    find .venv -path '*mmdet/models/dense_heads/rtmdet_head.py'
    find .venv -path '*mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py'
    ```

3. **Copy + strip imports:**
    ```bash
    cp .venv/.../mmdet/models/dense_heads/rtmdet_head.py             src/model/rtmdet_head.py
    cp .venv/.../mmdet/models/task_modules/assigners/dynamic_soft_label_assigner.py  src/model/assigner.py
    ```
    Per Component 6 §5.1: strip `mmcv.cnn`, `mmengine.model`, `mmdet.registry`, `mmdet.utils`, `ConfigDict`. Replace `BaseDenseHead` inheritance with `nn.Module`. Replace `BBoxOverlaps2D` with `torchvision.ops.box_iou` or hand-rolled CIoU. Strip the `with_objectness` branch. Total post-strip ≈ 600 LOC across both files.

4. **Add a vendor-source header to each file:**
    ```python
    """Vendored from mmdet @ <commit_sha> on 2026-04-28.
       Modifications:
         - removed `mmcv.cnn` ConvModule → torch.nn.Conv2d + torch.nn.GroupNorm
         - removed `mmengine.model` BaseDenseHead → torch.nn.Module
         - inlined the focal_loss γ=1.5 setting (was 2.0 default)
         - …
    """
    ```

5. **Run the assigner-parity test (Component 6 §5.3, test M8):** must pass byte-for-byte before any training begins.

6. **Lock the parity test as a pre-commit hook** so future changes that drift from MMDet's reference are caught.

Acceptance for Phase 0d: `uv run pytest tests/model/test_assigner_parity.py -v` passes.

### Phase 1 — Cache construction (Component 1)

Pre-flight: Phase 0d done.

```bash
# Step 1.1 — One-time analysis to fix in-plane resample target
uv run python scripts/analyze_inplane_spacing.py \
    --manifest data/manifest.jsonl \
    --raw-root data/ \
    --output agent/complete_spec/analysis_inplane_spacing.txt

# Step 1.2 — Edit scripts/preprocess.py: paste the chosen TARGET_SPACING
#   constant from the analysis output. Default expectation: (0.82, 1.5, 0.82).

# Step 1.3 — Run preprocessing on all 608 patients (~3-4 min wall-clock with 16 workers)
uv run python scripts/preprocess.py \
    --manifest data/manifest.jsonl \
    --cohort   data/cohort.json \
    --raw-root data/ \
    --cache-root cache/v1/ \
    --workers 16

# Step 1.4 — Connectivity probe (writes cache/v1/runtime/connectivity_lock.json)
#   Runs scipy.ndimage.label with both 6- and 26-connectivity over the 108 positives;
#   pins whichever produces 197 CCs to the lock file (per §13 amendment A.3).
#   Emit a fail message if NEITHER produces 197.
uv run python scripts/preprocess.py --probe-connectivity \
    --cache-root cache/v1/

# Step 1.5 — Verify acceptance gates from Component 1 §9.3
uv run pytest tests/preprocessing/ -v
```

Acceptance: I.7.1 through I.7.10. Cache disk between 30 and 50 GB.

### Phase 2 — Lesion bank + paste QC (Components 2, 4 dev workflow)

```bash
# Step 2.1 — Build the lesion bank (~1 s with 8 workers)
uv run python scripts/build_lesion_bank.py \
    --cache-root cache/v1/ \
    --workers 8

# Step 2.2 — Paste augmentation QC (dev workflow)
#   Renders 30 paste composites + emits a prompt the user/agent feeds to a
#   Claude Code subagent via the Task tool. NO API calls.
uv run python scripts/qc_paste_review.py \
    --cache-root cache/v1/ \
    --output cache/v1/runtime/qc_paste_review.html

# Step 2.3 — Tier-1 + Tier-2 unit/integration tests
uv run pytest tests/lesion_bank/ tests/augmentation/ -v
```

Acceptance: I.4.1–I.4.4. The agentic + human visual review (Component 4 §11.3, §11.4) runs as a dev workflow but **does not gate** anything in code (per §13 amendment A.2).

### Phase 3 — Model + Lightning module + smoke test (Components 3, 4, 5, 6, 8.smoke)

```bash
# Step 3.1 — Implement Components 3, 4, 5, 6 modules + tests.
#   Run unit + integration tests at each component boundary.
uv run pytest tests/dataset/ tests/augmentation/ tests/sampler/ tests/model/ -v

# Step 3.2 — Smoke training run (the integration gate; ~5 min on L40S)
uv run python -m endo.cli.run_experiment smoke --experiment experiments/smoke.py
```

Acceptance: All unit tests green. Smoke run completes 2 epochs on a 5-volume subset; first-10-step mean loss > last-10-step mean loss; no NaN; val/slice_auroc logged.

### Phase 4 — Train all 5 folds (Component 6, with Component 5 callbacks active)

```bash
# Default mode: 6000 samples/epoch, ~3.5 GPU-h per fold including deep eval.
uv run python -m endo.cli.run_experiment train \
    --experiment experiments/baseline_rtmdet_p2.py \
    --folds all \
    --device 0
# 5-fold sequential total: ~17.5 GPU-h.
# Add --wandb to enable WandB logging once you've landed one successful 10-epoch run.
```

Acceptance: Each `runs/<exp>/fold{f}/` contains `ckpts/best.ckpt`, `ckpts/last.ckpt`, at least one `runtime/deep_eval/epoch{n}_val.npz`. `provenance.json.fold_status` shows `complete` for all 5 folds. Coarse `val_volume_auroc_coarse` shows monotone improvement for at least 3 epochs.

### Phase 5 — GRU rescorer (Component 6.5)

```bash
# Phase 5.1 — Extract backbone features per fold (~5 min/fold)
for f in 0 1 2 3 4; do
    uv run python -m endo.cli.run_experiment train_gru \
        --experiment experiments/baseline_rtmdet_p2.py \
        --fold $f --stage feature_cache --device 0
done

# Phase 5.2 — Train GRUs (~3 min/fold)
for f in 0 1 2 3 4; do
    uv run python -m endo.cli.run_experiment train_gru \
        --experiment experiments/baseline_rtmdet_p2.py \
        --fold $f --stage train --device 0
done

# Total Stage-2: ~40 min sequential.
uv run pytest tests/gru/ -v
```

Acceptance: 5 feature caches + 5 ckpts; per-fold `val_auroc >= 0.5`.

### Phase 6 — CV evaluation (Component 7)

```bash
# Eval without GRU
uv run python -m endo.cli.run_experiment eval \
    --experiment experiments/baseline_rtmdet_p2.py
# Eval with GRU rescoring (adds rescored=true rows)
uv run python -m endo.cli.run_experiment eval \
    --experiment experiments/baseline_rtmdet_p2.py --use-gru
```

Outputs: `runs/<exp>/eval/eval_report.csv` (cv_pooled + per_fold + stratified rows; with and without GRU), `eval_thresholds.json`.

Acceptance: I.9.1–I.9.7. Soft target acceptance: cv_pooled volume_auroc ≥ 0.80, sens_at_2fp ≥ 0.70 (per phase-1 §3 abstract targets — these are aspirational, not blocking).

### Phase 7 — Holdout inference (Component 7)

```bash
# Single-model holdout (sanity check)
uv run python -m endo.cli.run_experiment predict_holdout \
    --experiment experiments/baseline_rtmdet_p2.py \
    --ckpts 0

# Full 5-model ensemble + GRU rescoring (the abstract number)
uv run python -m endo.cli.run_experiment predict_holdout \
    --experiment experiments/baseline_rtmdet_p2.py \
    --ckpts all \
    --use-gru
```

Outputs: `runs/<exp>/holdout/run_<timestamp>_<uuid8>/eval_report.csv` per invocation.

Acceptance: invocation completes; `eval_report.csv` has `scope=holdout` rows with bootstrap CIs.

### Phase 8 — Visualization (Component 8.viz)

```bash
for f in 0 1 2 3 4; do
    uv run python -m endo.cli.run_experiment viz \
        --experiment experiments/baseline_rtmdet_p2.py --fold $f
done
```

Outputs: `runs/<exp>/fold{f}/viz/*.png` + `manifest.csv`.

---

## 11. Test invariants table

For every test in every component spec, this table states what invariant the test proves. Tests live under `tests/<component>/`. The implementation agent uses this as the test-implementation checklist.

### 11.1 Preprocessing (`tests/preprocessing/`)

| ID | Test | Invariant proved |
|---|---|---|
| P1.1 | test_resample_isotropic | `scipy.ndimage.zoom` with computed factors produces correct output shape and conserves voxel sum (volumes only). |
| P1.2 | test_resample_mask_nn | Mask resampling stays binary (no fractional values). |
| P1.3 | test_norm_stats_inside_roi | ROI-aware normalization stats are computed on `volume[liver_roi==1]` voxels only. |
| P1.4 | test_clip_zscore_roundtrip | Clip + z-score reproduces analytically computable expected output. |
| P1.5 | test_post_resample_bbox | Post-resample bbox is the outer foreground bbox of the resampled liver_roi. |
| P1.6 | test_crop_and_pad_centering | Crop + center-pad to target shape produces correct pad offsets and centered foreground. |
| P1.7 | test_crop_and_pad_oversized_bbox | Oversized bbox raises with informative error (regression guard for outlier livers). |
| P1.8 | test_derive_2d_boxes_single_cc | One CC spanning N slices produces N rows with identical (x1,z1,x2,z2). |
| P1.9 | test_derive_2d_boxes_disjoint_ccs | Disjoint CCs produce distinct cc_id rows. |
| P1.10 | test_border_band_right_side_only | Border-band coords all have `x > liver_centroid_x`. |
| P1.11 | test_idempotency_skip | Re-running with same `cache_key` is a no-op. |
| P1.INT.1 | test_real_two_volume_e2e | Pipeline runs end-to-end on a 2-volume fixture. |
| P1.INT.2 | test_real_volume_shape_correct | Both fixture volumes cached at `(408, 174, 408)`. |
| P1.INT.3 | test_real_no_liver_mask_in_cache | `liver_mask.npy` does NOT exist under `volumes/<pid>/` (it's discarded after border-band derivation). |
| P1.INT.4 | test_real_lesion_recoverable | Positive fixture has `n_lesion_ccs > 0` matching reference. |
| P1.INT.5 | test_real_lesion_vs_ring_z_above_floor | I.7.4 (≥ 0.121). |
| P1.INT.6 | test_real_gt_boxes_inside_cache | I.3.3, I.3.4. |
| P1.INT.7 | test_real_border_band_holdout_skipped | I.7.5 (holdout fixture has no border_band). |
| P1.GATE | Cohort acceptance gate | I.7.1, I.7.2, I.7.3, I.7.4, I.7.6, I.7.9, I.7.10, plus per-fold variant balance ±20%. |

### 11.2 Lesion bank (`tests/lesion_bank/`)

| ID | Test | Invariant proved |
|---|---|---|
| L1 | test_extract_single_cc_shape | One synthetic CC produces one entry with correct tight_mask shape and physical_extent_mm. |
| L2 | test_extract_disjoint_ccs | Two disjoint CCs → two entries. |
| L3 | test_centroid_offset_in_tight | Centroid in tight-bbox-local coords matches geometric mean. |
| L4 | test_intensity_stats_correctness | Intensity mean/std computed only over CC voxels. |
| L5 | test_shell_excludes_cc | `tight_shell_mask & tight_mask = 0` everywhere. |
| L6 | test_shell_thickness_anisotropic | Shell thickness reflects anisotropic 1mm sampling: thicker in X/Z than Y. |
| L7 | test_intensities_outside_cc_zero | `tight_intensities` non-zero only inside CC. |
| L8 | test_z_extent_correct | `z_extent_voxels` matches the actual axis-1 span of the CC. |
| L9 | test_idempotency_skip | Re-build is a no-op. |
| L.INT.1 | test_real_one_donor_extracts | Donor patient bank entry count matches `preprocessed_manifest.n_lesion_ccs`. |
| L.GATE | Cohort acceptance gate | I.4.1, I.4.2, I.4.3, I.4.4. |

### 11.3 Dataset + DataModule (`tests/dataset/`)

| ID | Test | Invariant proved |
|---|---|---|
| D1 | test_dataset_len_matches_slice_index | Slice-index length equals total valid (pid, slice_y) count. |
| D2 | test_dataset_returns_5ch_correct_shape | Sample volume_5ch is `(5, 384, 384)` (or test fixture equivalent). |
| D3 | test_dataset_5ch_center_alignment | Channel 2 of volume_5ch equals `volume[:, slice_y, :]` after centered crop. |
| D4 | test_dataset_boxes_match_lookup | Per-slice boxes match `gt_boxes.parquet` rows. |
| D5 | test_dataset_no_boxes_for_negative_slice | Negative slice yields `boxes.shape == (0, 4)`. |
| D6 | test_inference_path_no_full_arrays | At `augment=None`, `volume_full_cropped is None`. |
| D7 | test_training_path_includes_full_arrays | At `augment != None`, full cropped arrays populated. |
| D8 | test_jitter_centered_at_validation | Val sample uses centered offset `(12, 7, 12)`. |
| D9 | test_border_band_translated_correctly | Border-band coords reflect `-jitter` translation and stay in cropped frame. |
| D10 | test_collate_fn_lists_for_boxes | `Batch.boxes` is a `list[Tensor]` of length B with variable per-image N. |
| D11 | test_holdout_blocked_by_default | I.8.1 — DataModule with `allow_holdout=False` raises on holdout patient. |
| D12 | test_holdout_inference_dataloader_refuses | `inference_dataloader` raises when `allow_holdout=False` AND any holdout pid requested. |
| D13 | test_holdout_inference_dataloader_allows | With `allow_holdout=True`, returns valid dataloader (single legitimate caller: `predict_holdout`). |
| D.INT.1 | test_real_setup_ram_within_budget | RSS after `setup()` < 50 GB. |
| D.INT.2 | test_real_no_holdout_in_train_or_val | I.8.1 on real cache. |
| D.INT.3 | test_real_box_validity_in_val_pass | I.3.3 holds for every box yielded in val pass. |

### 11.4 Augmentation (`tests/augmentation/`)

| ID | Test | Invariant proved |
|---|---|---|
| T1.1 | test_sample_n_pastes_distribution | P(n=0)=0.5; conditional on n>0, mode is 1; max ≤ 7. |
| T1.2 | test_sample_n_pastes_seeded_reproducible | Seeded RNG produces identical sequences. |
| T1.3 | test_paste_site_inside_border_band | Every successful paste site is in `border_band_coords`. |
| T1.4 | test_paste_no_overlap_with_existing | Paste does not land on existing lesion mask voxels. |
| T1.5 | test_paste_no_overlap_between_pastes | Paste 5 → no two paste_masks overlap. |
| T1.6 | test_paste_intensity_match_local_stats | Pasted region's mean ≈ target_local_mean ± 0.1σ. |
| T1.7 | test_paste_soft_blend_continuity | At paste boundary, intensity jump < 1.5σ. |
| T1.11 | test_geometric_lockstep | Volume + lesion_mask aligned after affine. |
| T1.12 | test_geometric_in_plane_only | No voxel moves across Y axis under any geometric transform. |
| T1.13 | test_geometric_y_coherent | Elastic field at slice 10 == field at slice 100. |
| T1.16 | test_box_rederivation_matches_mask | Re-derived boxes match analytic CC bbox. |
| T1.17 | test_box_skip_subpixel_artifacts | 1-voxel-wide CC dropped with warning. |
| T1.18 | test_5ch_slice_extraction_shape | I.8.8 shape contract holds. |
| T1.19 | test_5ch_center_channel_alignment | Channel 2 alignment per Component 4 §9. |
| T2.1 | test_paste_centroid_near_liver_border | ≥95% of paste centroids within 3mm of a true border-band voxel. |
| T2.4 | test_paste_right_side_only | All paste centroids have `x > liver_centroid_x`. |
| T2.5 | test_no_paste_outside_volume_bounds | All updated lesion_mask voxels in `[0,384)×[0,160)×[0,384)`. |

(Tier 3 + Tier 4 are the dev workflow visual review; not coded as gates.)

### 11.5 Sampler + HNM (`tests/sampler/`)

| ID | Test | Invariant proved |
|---|---|---|
| S1 | test_sampler_p_pos_decay_schedule | At {0,10,20,30,60}: p_pos matches decay schedule. |
| S2 | test_sampler_mix_at_epoch_0 | Distribution: pos≈50%, neg-pos-vol≈25%, neg-neg-vol≈25%. |
| S3 | test_sampler_mix_at_epoch_30 | Distribution converges to 25/37.5/37.5. |
| S4 | test_sampler_seeded_reproducible | I.8.2 derivative — deterministic with seed+epoch. |
| S5 | test_sampler_hard_pool_substitution_off_pre_epoch_5 | Hard-pool draw rate is 0 before epoch 5. |
| S6 | test_sampler_hard_pool_substitution_on_post_epoch_5 | At epoch 10 with non-empty pool, hard-pool draw rate ≈ 30%. |
| S8 | test_loss_ema_initialization | First update sets value; subsequent uses EMA decay. |
| S9 | test_loss_ema_skips_positive_slices | I.8.3 — only negative slices tracked. |
| S10 | test_loss_ema_top_k | top_k returns correct dataset indices. |
| S12 | test_periodic_callback_skips_pre_start_epoch | I.8.4 — callback no-op at epoch 5. |
| S13 | test_periodic_callback_writes_hard_negatives_json | At epoch 10, writes the JSON with correct schema (under `runs/<exp>/fold{f}/runtime/`). |
| S14 | test_periodic_callback_writes_deep_eval_cache | At epoch 10, writes the npz with correct arrays. |
| S.INT.1 | test_real_callback_runs_with_lightning | Mini Lightning trainer + mock model, 11 epochs → callback fires once at epoch 10. |
| S.INT.2 | test_real_inference_pass_throughput | ≥ 50 slices/s on L40S with mock model. |
| S.INT.3 | test_real_deep_eval_npz_roundtrip | Write + load + reconstruct per-patient SliceScore lists. |

### 11.6 Model (`tests/model/`)

| ID | Test | Invariant proved |
|---|---|---|
| M1 | test_backbone_5ch_input | ConvNeXt-tiny accepts `(1, 5, 384, 384)`; 4 stage outputs at strides 4,8,16,32. |
| M2 | test_conv1_renormalization_matches_doc | timm's 5ch conv1 renormalization matches doc spec `pretrained.repeat * 3/5`. |
| M3 | test_fpn_output_shapes | FPN produces 4 outputs at correct strides + channels. |
| M4 | test_aux_seg_head_output_stride1 | Aux seg head outputs `(B, 1, 384, 384)`. |
| M5 | test_rtmdet_head_forward_shapes | Head produces (cls_scores, bbox_preds) lists of length 4 with correct shapes. |
| M6 | test_rtmdet_head_loss_smoke | Forward + loss returns finite, non-NaN losses. |
| M7 | test_rtmdet_head_predict_smoke | Predict returns valid boxes/scores/labels. |
| M8 | test_assigner_parity_with_mmdet | I.8.10 — vendored assigner output byte-equals MMDet's on fixed input. |
| M9 | test_dice_bce_loss_zero_for_perfect | Dice+BCE on perfect logits ≈ 0. |
| M10 | test_total_loss_aggregates_correctly | total = cls + bbox + 0.3×aux_seg. |
| M11 | test_lightning_module_training_step_smoke | Single training_step on synthetic batch returns scalar loss. |
| M12 | test_lightning_module_validation_step_smoke | val step + on_validation_epoch_end logs slice_auroc. |
| M13 | test_score_ema_tracker_updated_on_train_negatives | I.8.3 — tracker has entries only for negative slices. |
| M14 | test_ema_callback_swap_swap_back | I.8.5 — live weights restored after validation_epoch_end. |
| M15 | test_warmup_cosine_lr_schedule | LR follows linear warmup → cosine decay. |
| M.INT.1 | test_real_one_train_batch | Real fold-0 batch produces finite loss. |
| M.INT.2 | test_real_two_epoch_loss_decreases | 2-epoch run on 5-volume subset → loss decreases. |
| M.INT.3 | test_real_checkpoint_save_load | Save best.ckpt; reload; val numbers reproduce. |

### 11.7 GRU rescorer (`tests/gru/`)

| ID | Test | Invariant proved |
|---|---|---|
| G1 | test_gru_forward_shape | Input `(4, 50, 768)` → output `(4, 50)`. |
| G3 | test_volume_score_max_and_topk | Manual computation matches. |
| G4 | test_volume_score_respects_mask | Padding values don't influence max/top-k. |
| G6 | test_rescore_multiplies_scores | Constant p=0.5 → all scores halved. |
| G7 | test_rescore_handles_missing_slice | Slice not in cache → score unchanged. |
| G.INT.1 | test_extract_features_for_fold_real | Real ckpt + 3 patients produces correct .npz schema. |
| G.INT.2 | test_train_gru_synthetic_correlation | Synthetic correlated dataset → val AUROC > 0.7 in 5 epochs. |

### 11.8 Eval (`tests/eval/`)

| ID | Test | Invariant proved |
|---|---|---|
| E1 | test_wbf_aggregates_overlapping | 3 overlapping boxes fuse to 1; score is weighted mean. |
| E3 | test_wbf_box_size_threshold | Large box at 0.06 passes; small box at 0.06 (below small_thr) drops. |
| E5 | test_compute_volume_metrics_smoke | All metric keys present, no NaN. |
| E6 | test_bootstrap_ci_widens_with_fewer_patients | I.9.6 derivative — patient-level resampling. |
| E9 | test_threshold_grid_search_finds_optimum | Grid search recovers the analytically optimal threshold. |
| E10 | test_stratified_breakdown_filters | I.9.7 — stratified subsets contain only matching stratum_value. |
| E11 | test_eval_one_fold_e2e | Real fold-0 deep_eval cache → eval_report.csv with finite metrics. |
| E12 | test_eval_with_and_without_gru | I.9.2 — both row sets produced. |
| E15 | test_eval_csv_append_only | I.9.1 — second eval run preserves first run's rows. |

### 11.9 Visualization (`tests/viz/`)

| ID | Test | Invariant proved |
|---|---|---|
| V1 | test_event_tagging_tp | IoU≥0.3 → tagged tp. |
| V3 | test_event_tagging_fp_low_iou | IoU<0.3 → tagged fp. |
| V4 | test_event_tagging_fn | GT exists, no pred matches → tagged fn. |
| V5 | test_event_tagging_mixed_slice | Mixed slice → 3 PNGs (tp, fp, fn). |
| V.INT.1 | test_visualize_real_run | Real fold-0 best.ckpt → ≥100 PNGs + valid manifest.csv. |

### 11.10 Smoke (`tests/smoke/`)

| ID | Test | Invariant proved |
|---|---|---|
| SM1 | test_smoke_completes_in_5_min | End-to-end smoke in <5 min wall-clock. |
| SM2 | test_smoke_loss_decreases | first_10 mean > last_10 mean. |
| SM3 | test_smoke_no_nan | All step losses finite. |
| SM4 | test_smoke_val_auroc_logged | `val/slice_auroc` present in callback_metrics. |

---

## 12. Resource accounting

Validates the plan fits the L40S 46 GB GPU + 250 GB RAM + 20-core compute node.

### 12.1 Disk

| Item | Estimate |
|---|---|
| `data/raw/` etc. | 19 GB (existing, frozen) |
| `data/manifest.jsonl` + `cohort.json` + `_archive/` | 3 MB |
| `cache/v1/volumes/` (608 × 408×174×408 × 2 B fp16) | ~35 GB |
| `cache/v1/lesion_mask` (108 positives × 408×174×408 × 1 B uint8) | ~3 GB |
| `cache/v1/border_bands/` (486 CV pids × ~50 KB) | ~25 MB |
| `cache/v1/gt_boxes.parquet` | ~150 KB |
| `cache/v1/lesion_banks/` | ~5 MB |
| `cache/v1/runtime/` | <10 MB |
| **Cache total** | **~38 GB** |
| `runs/<exp>/` per experiment (5 ckpts × ~250 MB + deep_eval npz × 6 × 5 ≈ 900 MB + GRU × 5 × ~120 MB + viz × 600 MB) | ~3 GB / experiment |

Stays well within `/scratch` quota.

### 12.2 RAM (training-time)

| Item | Estimate |
|---|---|
| Eager-loaded cache (volumes + lesion_masks + border_bands) | ~38 GB |
| Lesion bank | <10 MB |
| `cohort_local_std` + manifest | <100 MB |
| Python + Lightning + 8 workers (CoW) | ~5 GB resident |
| **Peak per training process** | **~45 GB** |

Single L40S node has 250 GB RAM. Comfortable headroom even for multi-fold parallelism via `--devices`.

### 12.3 GPU memory

Per Component 6 §14: bf16, ConvNeXt-tiny + FPN + RTMDet + aux seg head + EMA shadow on fp32. Forward + backward at batch=8 input `(B, 5, 384, 384)`:

| Item | Estimate |
|---|---|
| Activations | ~12 GB |
| Weights + grads + optimizer state | ~5 GB |
| EMA shadow (fp32) | ~3 GB |
| CUDA workspace + buffers | ~3 GB |
| **Peak** | **~23 GB** (well under L40S 46 GB) |

Safety margin: ~20 GB. If batch=8 OOMs unexpectedly, drop to 6 (see Component 6 §13 fallback).

### 12.4 GPU-hours (Stage-1 + Stage-2)

| Phase | Cost |
|---|---|
| Stage-1 (Component 6 default mode, 5 folds × ~3.5 GPU-h) | ~17.5 GPU-h |
| Periodic deep-eval refresh (5 folds × 6 refreshes × 5 min) | included above |
| Stage-2 GRU (5 folds × ~8 min) | ~0.7 GPU-h |
| CV eval (`eval` × 2 modes) | ~10 min |
| Holdout inference (5-model ensemble, full + GRU) | ~15 min |
| Visualization (5 folds × ~5 min) | ~25 min |
| **Total** | **~19 GPU-h** |

Well under the L40S budget of ~168 GPU-h/week. Plenty of room for one ablation run (e.g., paste=0).

---

## 13. Open issues, spec amendments, deviations

This section is the single place where the PRD overrides one of the 8 specs or flags an unresolved item. Implementation agent: read this carefully — silent disagreement between specs and PRD always favors the PRD.

### 13.1 Spec amendments (PRD overrides spec)

| ID | Spec | Original | PRD override | Why |
|---|---|---|---|---|
| A.1 | Component 1 §4.2 | `preprocessed_manifest.csv` | `preprocessed_manifest.jsonl` | Unified JSONL convention with `data/manifest.jsonl`. CSV's flat-string-typing already hurt us once. |
| A.2 | Component 4 §11.4 + Component 6 §10 | Component 4 says `train.py` refuses without `qc_human_signoff.json`; Component 6 says no QC precheck. Direct contradiction. | **No QC signoff gate.** Tier 3/4 are dev workflow only. `run_experiment train` never checks for signoff. | Explicit user direction (planning Q6.3). |
| A.3 | Component 1 §5.1 step 7 vs Component 2 §5.1 vs Component 4 §8 | Component 1: 6-conn; Component 2: 26-conn (with reproduction probe); Component 4: "match Component 2." Three-way confusion. | **Lock connectivity once at preprocessing time** via a one-time probe that picks whichever value reproduces phase-1 §1.3's 197-CC count. Pin to `cache/v1/runtime/connectivity_lock.json` and import everywhere downstream. | Avoids three-way drift; turns a bug-prone implementation choice into a single explicit decision pinned in cache provenance. |
| A.4 | Components 5, 6.5, 7 | Runtime artifacts (`hard_negatives.json`, `deep_eval/*.npz`, `gru_features/`, `gru_ckpts/`) under `cache/v1/`. | **Move all model-dependent runtime artifacts to `runs/<exp>/fold{f}/runtime/` and `/gru/`.** Cache stays cache; runs own model state. | Multi-experiment correctness: cache is shared across experiments, but model-dependent artifacts are not. Otherwise, two experiments would clobber each other's hard-negative pools. |
| A.5 | Component 7 §5.2 | `--i-mean-it` flag + `holdout_touched_<run_id>.json` audit + global lockfile. | **Drop the flag and lockfile.** Each `predict_holdout` invocation produces a fresh `runs/<exp>/holdout/run_<timestamp_uuid8>/` subdir. The DataModule guard (`allow_holdout=False` default; `predict_holdout` is the sole caller toggling it) is the only enforcement. | Explicit user direction (planning Q1.4): treat each holdout pass as a unique run; "touch holdout once" is enforced by user discipline, not code. |
| A.6 | Component 4 §11.3 | Anthropic API call from `qc_paste_agentic_review.py`. | **Drop `anthropic` dep.** Tier 3 review is invoked via Claude Code's Task tool against rendered PNGs — a dev workflow, not a code path. | Explicit user direction (planning Q5.2). We are inside Claude Code; no API key needed. |
| A.7 | phase-1 doc §10 | "MONAI transforms" for augmentation. | **Hand-rolled scipy** for the augmentation backend (Component 4 §6.4 default). MONAI not added as dep. | Lockstep + Y-coherent elastic + paste-first ordering all need fine-grained control that MONAI handles awkwardly. |
| A.8 | phase-1 doc §7 | Best ckpt by best-val-FROC. | **Best ckpt by `val/slice_auroc`** (Component 6 §15). | Train-time validation = slice proxies only (Q6 in earlier planning); volume FROC only refreshed every 10 epochs via deep-eval, too sparse to drive ckpt selection. |
| A.9 | Components 5, 6 | Default WandB logging on. | **WandB OFF by default.** Opt in via `--wandb` flag once a successful 10-epoch run lands. | Explicit user direction (planning Q7.3): keeps the dashboard clean of smoke and aborted runs. |
| A.10 | Component 1 §4.1 | `gt_boxes.parquet` columns: `x1, z1, x2, z2, box_max_dim_mm`. | **Locked schema:** `(patient_id, slice_y, cc_id, x1, z1, x2, z2, box_max_dim_mm)`. | Original list omitted the join keys. Multiple downstream callers (Component 4 §8 box re-derivation, Component 7 FROC) need them explicitly. |
| A.11 | Multiple | Reference to `data/manifest.csv` and `data/splits.json`. | **`data/manifest.jsonl` and `data/cohort.json`** (Phase 0a). Field renames: `bucket → cohort`, `cohort → label`. | Phase-0 unification. Originals preserved under `data/_legacy/`. |

### 13.2 Open issues (not blockers, but on the radar)

| ID | Issue | Status |
|---|---|---|
| O.1 | TARGET_SPACING constant in `preprocess.py` not yet locked. | Pending one-time `scripts/analyze_inplane_spacing.py` run in Phase 1. Default expectation `(0.82, 1.5, 0.82)` per phase-1 §1.2 cohort median. |
| O.2 | mmcv installation may fail against specific torch+CUDA combos. | Phase 0d mitigation: pin to OpenMMLab wheel index URL if `uv sync` fails with build error. |
| O.3 | Two outlier patients (`glass_puma_glade`, `pine_wren_fjord`) have lesion voxels outside the 20mm liver_roi (phase-1 §1.5). | Mitigation: clip at training time. Already a soft invariant of `lesion_mask & liver_roi` overlap — does not affect the cache. |
| O.4 | `samples_per_epoch=6000` derivation: rough budget for ~3 GPU-h/fold. Should validate empirically after first fold completes. | If wall-clock per fold exceeds 4 GPU-h, reduce to 5000. If under 2.5 GPU-h, can raise to 8000. |
| O.5 | Variant-B (113 vols, 11 positives) representation in CV folds is small (≈ 2 pos/fold). Stratified breakdown CIs will be wide. | Reported transparently; no mitigation in v1. |
| O.6 | RTMDet head + DynamicSoftLabelAssigner vendoring may need iteration if MMDet's API has shifted since Component 6 §5.1 was written. | Phase 0d task; assigner-parity test M8 catches drift. |

### 13.3 Out of scope (explicit non-goals)

Per phase-1 doc §14 — these are **not** in the Week-1 build:
- 3D detectors, DETR-family, RAD-DINO/BiomedCLIP/MedSAM-2 backbones, SSL pretraining on negatives, RadImageNet ablation, multi-scale TTA, multi-arch ensembling, horizontal flip aug, mosaic/mixup/cutmix/cutout, 3-channel input ablation.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Cache** | `cache/v1/` — preprocessed, cohort-wide artifacts, experiment-independent. Keyed on `(preprocessing_code_sha, target_spacing, target_shape, raw_sha256)`. |
| **Cohort** | The top-level partition: `cross-validation` (486 patients, 86 pos + 400 neg) or `holdout` (122 patients, 22 pos + 100 neg). Note: in the legacy `manifest.csv` this was confusingly called `bucket`. |
| **CC** | Connected component (a 3D foreground region in a lesion mask). 197 across the 108 positives. |
| **Experiment** | A specific Pydantic config declared in `experiments/<name>.py`. Identified by `(name, uuid)`. |
| **Fold** | The validation partition index 0..4. Fold N = patients in `manifest.fold == N` are held out as val; the other 4 folds train. |
| **Fold-as-run** | Each `(experiment, fold)` pair is one run. Lives at `runs/<exp>/fold{f}/`. |
| **Holdout** | The 122-patient locked test set. Touched once or several times across an experiment campaign — each touch = one `runs/<exp>/holdout/run_<uuid>/` subdir. |
| **Label** | The patient-level binary label: `positive` (108) or `negative` (500). In legacy `manifest.csv` this was confusingly called `cohort`. |
| **Run** | One training instance = experiment × fold. Produces one `best.ckpt`. |
| **Variant A / B** | The two LAVA-Flex sequence variants: A = 1.5mm reconstructed slice spacing (495 vols, both scanners); B = 3.6mm slice spacing (113 vols, Explorer only). |
| **WBF** | Weighted Box Fusion; 3D version aggregates per-slice 2D boxes across the volume's slices. Used at inference for per-volume score derivation. |

---

**End of PRD.** When the implementation agent has executed Phase 0d through Phase 8 and produced the holdout report, this PRD is the document to consult for any cross-component question. The 8 component specs answer "how does Component N work internally?"; this PRD answers "how do components 1..8 fit together?"
