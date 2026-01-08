#!/usr/bin/env python3
"""
Evaluate XGBoost regressor on per-problem parquet tables.

Usage mirrors the generic regression driver:
  python3 evaluation/scripts/eval_xgboost_regression.py \
      --dataset gsm8k-000027:artifacts/datasets/gsm8k_problem_tables_512/gsm8k-000027.parquet \
      ... \
      --report_json artifacts/reports/raw/xgboost_rows512_standardized_report_random.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import load_dataset


def _parse_dataset_arg(arg: str) -> Tuple[str, str]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError("Dataset argument must be NAME:PATH")
    name, path = arg.split(":", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("Dataset argument must be NAME:PATH with non-empty fields")
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Dataset path does not exist: {path}")
    return name, path


def _select_feature_columns(df: pd.DataFrame, max_features: int) -> List[str]:
    cols = [c for c in df.columns if c != "y"]
    if max_features and len(cols) > max_features:
        coverage = df[cols].notna().mean().sort_values(ascending=False)
        cols = list(coverage.index[:max_features])
    return cols


def _ood_split_indices(y: pd.Series, test_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    values = y.to_numpy()
    n = len(values)
    n_test = max(1, int(np.ceil(n * test_fraction)))
    sorted_idx = np.argsort(values)
    test_idx = sorted_idx[-n_test:]
    train_idx = sorted_idx[:-n_test] if n_test < n else np.array([], dtype=int)
    if train_idx.size == 0:
        train_idx = sorted_idx[:1]
        test_idx = sorted_idx[1:] if n > 1 else sorted_idx[:1]
    return train_idx, test_idx


def _split_indices(y: pd.Series, test_fraction: float, seed: int, ood_split: bool) -> Tuple[np.ndarray, np.ndarray]:
    if ood_split:
        return _ood_split_indices(y, test_fraction)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_fraction, random_state=seed)
    return train_idx, test_idx


def _evaluate_dataset(name: str, path: str, max_rows: int, max_features: int,
                      test_size: float, seed: int, device: str,
                      holdout_dir: str | None, standardize: bool,
                      ood_split: bool) -> Dict[str, float]:
    df, dropped_cols = load_dataset(Path(path), rowcap=max_rows or None, seed=seed, target_col="y")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "y" not in num_df.columns:
        raise ValueError(f"'y' column missing in {path} after sanitization")

    X_cols = _select_feature_columns(num_df, max_features)
    X = num_df[X_cols].fillna(num_df[X_cols].median()).astype(np.float32)
    y = num_df["y"].astype(np.float32)

    train_idx, test_idx = _split_indices(y, test_size, seed, ood_split)

    X_train = X.iloc[train_idx].values.astype(np.float32)
    y_train = y.iloc[train_idx].values.astype(np.float32)
    X_test = X.iloc[test_idx].values.astype(np.float32)
    y_test = y.iloc[test_idx].values.astype(np.float32)

    y_train_orig = y_train.copy()
    y_test_orig = y_test.copy()

    if standardize:
        feat_mean = X_train.mean(axis=0).astype(np.float32)
        feat_std = X_train.std(axis=0, ddof=0).astype(np.float32)
        feat_std[feat_std == 0] = 1.0
        X_train = (X_train - feat_mean) / feat_std
        X_test = (X_test - feat_mean) / feat_std

        y_mean = float(y_train.mean())
        y_std = float(y_train.std(ddof=0))
        if y_std == 0:
            y_std = 1.0
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        y_mean = 0.0
        y_std = 1.0

    booster = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        device=device if device != "cpu" else "cpu"
    )
    booster.fit(X_train, y_train)
    preds = booster.predict(X_test)

    if standardize:
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        preds_orig = preds * y_std + y_mean
        y_eval_orig = y_test_orig
    else:
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        preds_orig = preds
        y_eval_orig = y_test

    rounded_consistent = float(np.mean(np.rint(preds_orig) == np.rint(y_eval_orig)))

    if holdout_dir:
        out_dir = Path(holdout_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = num_df.copy()
        partial.loc[test_idx, "y"] = pd.NA
        partial_path = out_dir / f"{name}_partial.csv"
        partial.to_csv(partial_path, index=False)
        gold = pd.DataFrame({"row_index": test_idx, "y": y.iloc[test_idx].values})
        gold.to_csv(out_dir / f"{name}_test_targets.csv", index=False)

    return {
        "n_samples": int(len(num_df)),
        "n_features": int(len(X_cols)),
        "dropped_columns": dropped_cols,
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "rounded_consistency": rounded_consistent,
        "split": "ood" if ood_split else "random",
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate XGBoost regressor on parquet datasets.")
    ap.add_argument("--dataset", action="append", default=[], help="Dataset spec NAME:PATH", dest="datasets")
    ap.add_argument("--max_rows", type=int, default=0, help="Max rows sampled per dataset (0 = all)")
    ap.add_argument("--max_features", type=int, default=0, help="Max numeric feature columns (0 = all)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split proportion")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed")
    ap.add_argument("--device", type=str, default="cpu", help="Device hint (cpu, cuda, etc.)")
    ap.add_argument("--holdout_dir", type=str, default=None, help="Optional directory for partial tables")
    ap.add_argument("--report_json", type=str, default=None, help="Optional JSON report path")
    ap.add_argument("--standardize", action="store_true",
                    help="Standardize features and target per dataset before training; metrics reported on original scale.")
    ap.add_argument("--ood_split", dest="ood_split", action="store_true", default=True,
                    help="Use out-of-distribution split (top 20%% y values as test). (default)")
    ap.add_argument("--random_split", dest="ood_split", action="store_false",
                    help="Use random train/test split instead of OOD split.")
    args = ap.parse_args()

    if not args.datasets:
        ap.error("At least one --dataset NAME:PATH must be provided.")

    results: Dict[str, Dict[str, float]] = {}
    for arg in args.datasets:
        name, path = _parse_dataset_arg(arg)
        print(f"[INFO] Evaluating {name} ({path}) ...")
        metrics = _evaluate_dataset(
            name=name,
            path=path,
            max_rows=args.max_rows,
            max_features=args.max_features,
            test_size=args.test_size,
            seed=args.seed,
            device=args.device,
            holdout_dir=args.holdout_dir,
            standardize=args.standardize,
            ood_split=args.ood_split
        )
        results[name] = metrics
        print(f"  samples={metrics['n_samples']} features={metrics['n_features']} "
              f"RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} R2={metrics['r2']:.3f} "
              f"Consistent={metrics['rounded_consistency']:.3f}")

    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        meta = {
            "standardize_stats": "train_only" if args.standardize else "disabled",
        }
        payload = {"xgboost_regression": results, "_meta": meta}
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved report to {args.report_json}")


if __name__ == "__main__":
    main()
