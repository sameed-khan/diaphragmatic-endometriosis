## Diaphragmatic Endometriosis

### Goal
Develop an object detection model that detects endometriosis lesions in the 
diaphragm

## Notes
1. Unless otherwise indicated, you are currently on a GPU compute node on the CWRU HPC
2. Storage space may run out if on the /home disk - you can check by using `quotagrp`
	1. If storage space is exceeded you can experiment with deleting unnecessary files on our home disk or looking at moving certain files / folders over to /scratch temporarily

### Overall Directives
1. All agent outputs such as plans, research and documents go in the agent/
folder unless otherwise instructed
2. Always use `uv` for all python related tasks rather than `python` or `pip`
    1. Instead of `pip install` always do `uv add`
    2. Instead of `python3 <script>` always do `uv run <script>` or `uv run -m <script>`
3. For tabular data analysis, always use `polars` instead of `pandas`
4. Always base your answers and advice on research from deep learning empirics on similar tasks;
clearly outline when something is a guess versus when something comes from hard knowledge you can point to
5. Always consider the overall goal of the project and push back on unnecessary or redundant steps if you believe they are inadvisable
6. Frequently ask for clarification if needed; ensure you have a perfectly clear understanding of my intent before executing any task

### Miscellaneous
- .env file contains WANDB_API_KEY and HF_TOKEN if needed to access wandb or huggingface

### Data + cache sync (fresh checkout)

The bulky binary artifacts (NIfTI volumes, masks, preprocessed cache) are **not in
git**. They live in a private Hugging Face Hub dataset repo:

  **`sameedkhan/diaphragmatic-endometriosis-data`** (private)

On a fresh checkout (Lambda Labs VM, CWRU HPC node, or anywhere else), after
`git clone` and `uv sync`:

```bash
# 1. Auth to HF (token in .env, see WANDB_API_KEY/HF_TOKEN above)
export $(grep ^HF_TOKEN= .env | xargs)
uv run hf auth login --token "$HF_TOKEN"

# 2. Pull the data + cache trees from HF into the repo root
uv run hf download sameedkhan/diaphragmatic-endometriosis-data \
  --repo-type dataset \
  --local-dir .

# 3. Verify
ls data/raw/cross-validation/positive | head -5    # expect mnemonic .nii.gz
ls cache/v1/volumes | head -5                      # expect preprocessed .npz
```

`hf download` is resumable and content-addressed (Xet/LFS dedup), so re-running
it is a no-op when files are already present. To pull only one subtree (e.g.
just the cache for an inference-only VM):

```bash
uv run hf download sameedkhan/diaphragmatic-endometriosis-data \
  --repo-type dataset --local-dir . \
  --include "cache/v1/**" --include "data/manifest.jsonl" --include "data/cohort.json"
```

Authoritative pointer: `data/manifest.jsonl` (committed to git) is the source of
truth for which patients exist. Any binary file referenced by manifest.jsonl
that's missing locally should be on HF; if not, that's a sync bug.

**What lives where:**

| Artifact | In git? | In HF dataset repo? | Notes |
|---|---|---|---|
| Code (`endo/`, `scripts/`, `experiments/`, `tests/`) | yes | no | source of truth for code is GitHub |
| Coordination files (`data/manifest.jsonl`, `data/cohort.json`, `data/README.md`) | yes | yes | mirrored for self-contained HF sync |
| `data/raw/`, `data/lesion_masks/`, `data/liver_masks/`, `data/liver_rois/` | no | yes | ~18 GB, the medical imaging payload |
| `cache/v1/volumes/`, `cache/v1/border_bands/`, `cache/v1/lesion_banks/`, `cache/v1/runtime/` | no | yes | ~36 GB, deterministic preprocessing output (regenerable from `data/raw/` via `scripts/preprocess.py` + `scripts/build_lesion_bank.py`, but cheaper to just download) |
| `runs/` (training checkpoints, eval outputs) | no | no | use W&B Artifacts (`upload_checkpoints=True` in the experiment file). Fresh-VM workflow: `wandb.Api().artifact("clevelandclinic/diaphragmatic-endometriosis/<artifact-name>:<alias>").download(...)` |
| `wandb/` local run dirs | no | no | regenerable from W&B server

**Versioning.** Each push to the HF dataset repo is a git commit on the HF
side, so version history is captured natively (browsable at
`huggingface.co/datasets/sameedkhan/diaphragmatic-endometriosis-data/commits/main`).
To pin a specific revision in code, pass `--revision <sha>` to `hf download`
or use `huggingface_hub.snapshot_download(... , revision=...)`.

**Why not DVC?** Considered but skipped — DVC's HF integration is currently
read-only (`dvc import-url` works, `dvc push` to HF is not yet supported as
of mid-2025). HF's own git-commit-per-upload model gives us versioning
natively without a second layer. If you later want hash-pinning of
specific files inside the git repo, `uv run dvc init` + `dvc add <file>` is
non-destructive and can be added without breaking the HF sync workflow.

### CLI flag contract — `endo.cli.run_experiment`

Logging + W&B are controlled by `LoggingConfig` inside the experiment file.
Three CLI flags can override that config without editing the file:

- `--wandb` / `--no-wandb` (mutually exclusive) — overrides `experiment.logging.wandb.enabled` for this invocation only. Never modifies `experiments/<name>.py` or `runs/<exp>/experiment.yaml`.
- `--wandb-mode {online,offline,disabled}` — overrides `experiment.logging.wandb.mode`.
- `-v` / `-vv` — overrides `experiment.logging.file.level_console` (and, with `-vv`, `level_file`) to `DEBUG`.

These flags propagate uniformly to all subcommands (`train`, `train_gru`, `eval`, `predict_holdout`, `viz`).

The whole `LoggingConfig` subtree is **drift-exempt** in `ExperimentConfig.diff(...)` — toggling
`logging.*` between resumes does NOT trip the drift guard. Per-fold file logs land at
`runs/<exp>/fold{N}/run.log` (rotating, 50 MB × 3 backups by default); the top-level run log is at `runs/<exp>/run.log`.
