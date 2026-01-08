# TabularMath Benchmark

Math-focused tabular regression benchmark derived from GSM8K and AIME problems.  
This repository now separates three concerns so a new reader can jump in easily:

- **`curation/`** – code + documentation for rebuilding the numeric tables from the released augmentation JSONL files.
- **`evaluation/`** – experiment drivers, manifests, and shared utilities.
- **`artifacts/`** – generated data (parquet tables) and the publication-ready outputs (JSON reports, TSV summaries, plots, logs).

`docs/` carries human-readable summaries; everything else is code.

## Directory layout
```
tabularmath/
├─ README.md
├─ artifacts/
│  ├─ datasets/          # Sanitized GSM8K + AIME tables (CSV + parquet)
│  ├─ reports/           # Raw model JSON + aggregated TSVs
│  ├─ plots_png/         # Rounded-consistency curves
│  └─ logs/              # Run logs, sanitization logs, matplotlib cache
├─ curation/             # Data curation pipeline (inputs, scripts, Makefile, requirements)
├─ evaluation/           # Experiment code (requirements, manifests, scripts, helpers)
└─ docs/                 # Publication notes
```

`artifacts/` is the shared output staging area.  
For convenience the repository ships with the fully populated release artefacts, but you can delete the directory at any point and regenerate it by following the curation + evaluation steps below.

## Workflow overview
1. **Curation** (`curation/`): run the augmentation pipeline to turn the raw GSM8K/AIME problems into per-problem tables, then copy the results into `artifacts/datasets/` (see `curation/README.md` for full instructions).
2. **Evaluation** (`evaluation/`): once `artifacts/datasets/` contains the parquet tables, run the experiment drivers to regenerate every JSON report, TSV summary, plot, and log under `artifacts/`.

## Evaluation environment
1. Python 3.10+ and pip.
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install --extra-index-url https://download.pytorch.org/whl/cpu -r evaluation/requirements.txt
   ```
3. Export your Prior Labs TabPFN token for the API baselines:
   ```bash
   export PRIORLAB_API_KEY="<your-tabpfn-token>"
   ```
4. Optional: point `TABMATH_ICL_LLM_CLIENT="module:function"` at a callable that accepts `(prompt: str, logid: str)` and returns the LLM response text (see “ICL hook” below).

Run everything from the repo root so relative paths match the commands below.

## Artifacts (data + reports)
Out of the box these directories already contain the published results. Re-running the steps below will regenerate the same layout:
- Sanitized GSM8K and AIME tables live at `artifacts/datasets/`. Raw 2,048-row tables are bundled; regenerate normalized or alternative row-cap variants via `curation/` when needed.
- Model reports land in `artifacts/reports/raw/` (one JSON per model/split/row-cap).
- Aggregated TSVs + ranking tables live in `artifacts/reports/summaries/`.
- Curves render to `artifacts/plots_png/`.
- Run logs/sanitization logs go under `artifacts/logs/` (created automatically on first run).

These folders are created automatically by the scripts, so you can safely delete `artifacts/` and rebuild everything from scratch whenever needed.

## Evaluation quick start
Use the smoke test (after curation has populated `artifacts/datasets/`) to confirm the toolchain before touching TabPFN credits:

```bash
PYTHONPATH=evaluation \
python evaluation/scripts/eval_generic_regression.py \
  --model random_forest \
  --standardize --random_split --max_rows 32 \
  --dataset gsm8k-000007:artifacts/datasets/gsm8k_problem_tables_2048/gsm8k-000007.parquet \
  --report_json artifacts/reports/raw/smoke_random_forest.json
```

ICL smoke test (uses the placeholder LLM response when `TABMATH_ICL_LLM_CLIENT` is unset):
```bash
PYTHONPATH=evaluation \
python evaluation/scripts/eval_icl_llm.py \
  --manifest evaluation/data_manifests/tmp_dataset_smoke.txt \
  --rowcap 32 --random_split \
  --out artifacts/reports/raw/icl_llm_smoke.json \
  --log_jsonl artifacts/reports/icl_llm_smoke_logs.jsonl
```

### Full pipeline
Run the orchestrator once you are ready to regenerate every result (TabPFN tokens + optional ICL hook required):
```bash
PYTHONPATH=evaluation python evaluation/scripts/all_exps.py \
  --manifest evaluation/data_manifests/tmp_datasets_rows2048.txt \
  --skip_icl              # drop this flag when an LLM client is configured
```

What it does for each split (`random`, `ood`) and row cap {32,64,128,256,512,1024,2048}:
- TabPFN v2 (`model_path=v2_default`) via the Prior Labs API
- TabPFN v2.5 (`model_path=v2.5_default`) via the Prior Labs API
- XGBoost
- Classical baselines: RandomForest, LightGBM, CatBoost, TabM, RealMLP, xRFM (skips models whose deps are missing)
- Optional ICL baseline for row caps {32,64,128} when `TABMATH_ICL_LLM_CLIENT` resolves to a callable

The driver auto-aggregates the TSVs and regenerates the rounded-consistency plots at the end.

### ICL hook reference
1. Implement a small wrapper:
   ```python
   # my_llm.py
   from some_llm_sdk import Client
   client = Client(...)
   def predict(prompt: str, logid: str) -> str:
       return client.complete(prompt)
   ```
2. Export the hook before running the pipeline:
   ```bash
   export TABMATH_ICL_LLM_CLIENT="my_llm:predict"
   ```
3. Optional: set `TABMATH_ICL_PLACEHOLDER="0"` if you want a deterministic fallback string for dry runs.  
   Without a hook, the driver returns `"<unknown>"`, which is fine for smoke tests but not for the published results.

## Curation pipeline
Everything needed to rebuild the parquet tables is under `curation/`. Follow `curation/README.md` for the exact commands, summarized here:

```bash
cd curation
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

make all      # builds GSM8K + AIME tables, normalizes them, and verifies uniqueness

# Copy into the artifacts tree inside the repo root:
rsync -a tabular/gsm8k_problem_tables_2048 ../artifacts/datasets/
rsync -a tabular/aime_problem_tables_2048 ../artifacts/datasets/
rsync -a tabular/gsm8k_problem_tables_2048_norm ../artifacts/datasets/
rsync -a tabular/aime_problem_tables_2048_norm ../artifacts/datasets/
```

The Makefile pins the exact row counts, random seeds, and AIME problem IDs used for the paper.  
Additional documentation inside `curation/` explains how the augmentation loop works (`curation/augmentation/`) and how verification is performed (`curation/check_tables.py`).

## Outputs
- `artifacts/reports/raw/*.json` – per-model metrics keyed by dataset names.
- `artifacts/reports/summaries/model_rowcap_metrics.tsv` – flattened per-dataset metrics (MSE, RMSE, MAE, R², rounded consistency).
- `artifacts/reports/summaries/model_rowcap_summary.tsv` – grouped statistics per dataset family and split.
- `artifacts/reports/summaries/rounded_consistency_mean_by_family_with_best.tsv` – table used to render the paper figures.
- `artifacts/plots_png/*.png` – final rounded-consistency plots for both splits.
- `artifacts/logs/all_exps_run.log` – orchestrator log (plus per-model logs for debugging API hiccups).

## Reproducibility notes
- Every evaluation script standardizes features (and target when `--standardize` is passed) per dataset before training; rounded consistency is always measured on the original target scale.
- Random split uses an 80/20 shuffle; OOD split uses the top 20% of `y` values as the query set.
- ICL evaluation only applies to row caps ≤128 (matching the published setup); set `--skip_icl` if you do not provide an LLM hook.
- The end-to-end smoke tests listed above are run after each structural change to guarantee that both curation and evaluation remain reproducible for fresh users.
