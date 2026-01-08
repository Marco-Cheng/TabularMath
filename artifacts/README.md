# TabularMath Artefacts

This directory hosts all benchmark artefacts (datasets + evaluation outputs).  
The repository ships with the fully populated release, but you can delete it and regenerate everything as follows:

1. **Curation** – copy the generated GSM8K + AIME parquet folders into `artifacts/datasets/`.
2. **Evaluation** – `evaluation/scripts/all_exps.py` writes per-model JSON reports into `artifacts/reports/raw/`, aggregated TSVs into `artifacts/reports/summaries/`, plots into `artifacts/plots_png/`, and logs into `artifacts/logs/`.

You may safely delete and regenerate these contents at any time; the scripts will recreate the expected subdirectories on demand.
