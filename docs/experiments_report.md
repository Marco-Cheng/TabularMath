# TabularMath Experiments Report

This document records the decisions and artefacts required to reproduce the TabularMath benchmark after dataset sanitization.

## 1. Benchmark specification
- **Problems.** 100 GSM8K and 14 AIME tasks converted into numeric tabular form; each table has 2,048 deduped rows.
- **Sanitization.** Columns `lang,len_chars,delta_num,delta_text,delta_total` are removed, and any column constant across all rows (except the target `y`) is dropped. See `artifacts/logs/dataset_sanitization_log.json`.
- **Row caps.** {32, 64, 128, 256, 512, 1,024, 2,048}; subsets sampled with `DataFrame.sample(..., random_state=seed)` before splitting.
- **Splits.**
  - `random`: 80/20 shuffle (seed 2025 for scikit-style models).
  - `ood`: top 20% of `y` values become the test/query set; the rest form the context/train set.
- **Manifests.** `data_manifests/tmp_datasets_rows2048.txt` lists every dataset via `--dataset name:path` and feeds all scripts.

## 2. Models and hyperparameters
All models use median-filled numeric features. With `--standardize`, features and targets are normalised per dataset; metrics are reported on the standardized scale while rounded consistency is measured on original targets.

| Model | Script | Key hyperparameters |
| --- | --- | --- |
| Random Forest | `eval_generic_regression.py` | `RandomForestRegressor(n_estimators=500, max_features=None, n_jobs=-1, random_state=seed)` |
| LightGBM | same | `LGBMRegressor(n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=seed)` |
| CatBoost | same | `CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function=RMSE, verbose=False, allow_writing_files=False)` |
| XGBoost | `eval_xgboost_regression.py` | `XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0)` |
| RealMLP (TD-S) | `eval_generic_regression.py` | `Standalone_RealMLP_TD_S_Regressor(device="cpu")` (requires `third_party.realmlp`) |
| TabM | `eval_generic_regression.py` | TabM encoder, `AdamW(lr=2e-3, weight_decay=3e-4)`, batch size = `min(256, N)`, 80 epochs |
| xRFM | `eval_generic_regression.py` | `xRFM(device="cpu", random_state=seed)` with 20% validation split |
| TabPFN v2 (API) | `eval_tabpfn_client.py` | Prior Labs API `TabPFNRegressor(model_path="v2_default", n_estimators=8, random_state=seed)` |
| TabPFN v2.5 (API) | `eval_tabpfn_client.py` | Prior Labs API `TabPFNRegressor(model_path="v2.5_default", n_estimators=8, random_state=seed)` |
| LLM ICL | `eval_icl_llm.py` | user-pluggable LLM hook (`TABMATH_ICL_LLM_CLIENT`); context = 80% rows, queries = 20%; retries up to 10 with regex parsing |

## 3. Execution pipeline
- Use `evaluation/scripts/all_exps.py` (cross-platform) from `tabularmath/`. It clears previous artefacts, runs every model/split/row-cap combination, aggregates reports, and plots curves. ICL runs are skipped automatically when `--skip_icl` is passed (default is enabled).
- `PRIORLAB_API_KEY` must be set for TabPFN v2.5.
- Outputs: per-model JSON in `artifacts/reports/raw/`, TSV summaries in `artifacts/reports/summaries/`, plots in `artifacts/plots_png/`.

## 4. Outputs and interpretation aids
- `artifacts/reports/summaries/model_rowcap_metrics.tsv`: per-problem metrics (MSE, RMSE, MAE, R2, rounded consistency) for every configuration.
- `artifacts/reports/summaries/model_rowcap_summary.tsv`: descriptive stats (mean/median/quantiles) grouped by dataset family and split.
- `artifacts/reports/summaries/rounded_consistency_mean_table.tsv` and `..._with_best.tsv`: mean rounded consistency overall and per-family with best-model annotations.
- `artifacts/plots_png/rounded_consistency_{random,ood}_{AIME,GSM8K}.png`: rounded-consistency curves for quick visual ranking.

## 5. Reproducibility checklist
- Use the sanitized datasets in `artifacts/datasets/` (already cleaned of metadata and constant columns).
- Respect the split flags (`--ood_split` / `--random_split`) and row caps to match the reported grid.
- Standardize features/targets (`--standardize`) for classical models and TabPFN runs to reproduce the reported scales.
- Ensure `PRIORLAB_API_KEY` is exported for TabPFN v2.5; provide an LLM client (or leave the default placeholder) if running ICL.
