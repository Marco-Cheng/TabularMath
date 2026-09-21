# TabularMath Artefacts

`datasets/` contains the released benchmark tables used as evaluation inputs.
Completed experiment reports, plots, and logs are not part of the repository.

1. **Curation** – copy the generated GSM8K + AIME parquet folders into `artifacts/datasets/`.
2. **Evaluation** – `evaluation/scripts/all_exps.py` writes per-model JSON reports into `artifacts/reports/raw/`, aggregated TSVs into `artifacts/reports/summaries/`, plots into `artifacts/plots_png/`, and logs into `artifacts/logs/`.

The evaluation output directories are ignored by Git and created on demand.
Keep `datasets/` when clearing local run outputs.
