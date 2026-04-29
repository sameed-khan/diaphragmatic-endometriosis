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
