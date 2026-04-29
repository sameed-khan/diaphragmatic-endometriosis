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
