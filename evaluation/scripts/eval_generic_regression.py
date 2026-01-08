#!/usr/bin/env python3
"""
Evaluate classical tabular regressors (RandomForest, LightGBM, CatBoost, etc.)
on per-problem parquet tables.

Usage mirrors the TabPFN client driver, with an extra --model flag:

  python3 evaluation/scripts/eval_generic_regression.py \
      --model lightgbm \
      --dataset gsm8k-000027:artifacts/datasets/gsm8k_problem_tables_512/gsm8k-000027.parquet \
      ... \
      --report_json artifacts/reports/raw/lightgbm_rows512_standardized_report_random.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

try:
    from xrfm import xRFM as XRFMLib
except Exception:
    XRFMLib = None
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import load_dataset

try:
    from third_party.realmlp import Standalone_RealMLP_TD_S_Regressor
except Exception:
    Standalone_RealMLP_TD_S_Regressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from tabm import TabM
except Exception:
    TabM = None

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


def _model_factory(model_name: str) -> Callable[[int], object]:
    name = model_name.lower()
    if name == "random_forest":
        def _rf(seed: int):
            return RandomForestRegressor(
                n_estimators=500,
                random_state=seed,
                n_jobs=-1,
                max_features=None,
            )
        return _rf
    if name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed.")
        def _lgbm(seed: int):
            return LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
            )
        return _lgbm
    if name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed.")
        def _cat(seed: int):
            return CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
        return _cat
    if name == "tabm":
        if TabM is None:
            raise RuntimeError("tabm is not installed. Run `pip install tabm` first.")
        def _tabm(seed: int):
            return None
        return _tabm
    if name == "realmlp":
        def _realmlp(seed: int):
            torch_seed = seed if seed is not None else 0
            try:
                import torch
                torch.manual_seed(torch_seed)
            except Exception:
                pass
            if Standalone_RealMLP_TD_S_Regressor is None:
                raise RuntimeError("RealMLP dependency missing (third_party.realmlp).")
            return Standalone_RealMLP_TD_S_Regressor(device="cpu")
        return _realmlp
    if name == "xrfm":
        if XRFMLib is None:
            raise RuntimeError("xRFM library not installed. Run `pip install xrfm`.")
        def _xrfm(seed: int):
            return XRFMLib(device="cpu", random_state=seed)
        return _xrfm
    raise ValueError(f"Unsupported model '{model_name}'. Choose from random_forest, lightgbm, catboost.")


def _train_tabm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    if TabM is None:
        raise RuntimeError("tabm is not installed. Run `pip install tabm` to enable it.")
    torch.manual_seed(seed if seed is not None else 0)
    device = torch.device("cpu")
    model = TabM.make(n_num_features=X_train.shape[1], d_out=1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=3e-4)
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    batch_size = min(256, len(train_ds))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    epochs = 80
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).squeeze(-1)
            loss = ((preds - yb.unsqueeze(-1)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        preds = model(test_tensor).mean(dim=1).squeeze(-1).cpu().numpy()
    return preds


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


def _evaluate_dataset(model_fn: Callable[[int], object],
                      name: str,
                      path: str,
                      max_rows: int,
                      max_features: int,
                      test_size: float,
                      seed: int,
                      holdout_dir: str | None,
                      standardize: bool,
                      model_name: str,
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

    model = None
    if model_name != "tabm":
        model = model_fn(seed)

    constant_target = np.allclose(y_train, y_train.mean())
    if model_name == "tabm":
        if constant_target:
            preds = np.full_like(y_test, y_train.mean())
        else:
            preds = _train_tabm_regressor(X_train, y_train, X_test, seed)
    else:
        if constant_target:
            preds = np.full_like(y_test, y_train.mean())
        else:
            if model_name == "xrfm":
                train_size = len(X_train)
                if train_size < 2:
                    raise ValueError("Not enough training samples for xRFM validation split.")
                rng = np.random.RandomState(seed)
                val_count = max(1, int(max(1, train_size * 0.2)))
                if train_size - val_count < 1:
                    val_count = max(1, train_size - 1)
                val_idx = rng.choice(train_size, size=val_count, replace=False)
                mask = np.ones(train_size, dtype=bool)
                mask[val_idx] = False
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                X_train = X_train[mask]
                y_train = y_train[mask]
                model.fit(X_train, y_train, X_val, y_val)
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_test)

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
    ap = argparse.ArgumentParser(description="Evaluate classic regressors on parquet datasets.")
    ap.add_argument("--model", required=True, help="Which model to run (random_forest, lightgbm, catboost).")
    ap.add_argument("--dataset", action="append", default=[], help="Dataset spec NAME:PATH", dest="datasets")
    ap.add_argument("--max_rows", type=int, default=0, help="Max rows sampled per dataset (0 = all)")
    ap.add_argument("--max_features", type=int, default=0, help="Max numeric feature columns (0 = all)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split proportion")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed")
    ap.add_argument("--holdout_dir", type=str, default=None, help="Optional directory for partial tables")
    ap.add_argument("--report_json", type=str, default=None, help="Optional JSON report path")
    ap.add_argument("--standardize", action="store_true",
                    help="Standardize features and target per dataset before training; metrics reported on standardized scale.")
    ap.add_argument("--ood_split", dest="ood_split", action="store_true", default=True,
                    help="Use out-of-distribution split (top 20%% y values as test). (default)")
    ap.add_argument("--random_split", dest="ood_split", action="store_false",
                    help="Use random train/test split instead of OOD split.")
    args = ap.parse_args()

    if not args.datasets:
        ap.error("At least one --dataset NAME:PATH must be provided.")

    model_fn = _model_factory(args.model)

    results: Dict[str, Dict[str, float]] = {}
    for arg in args.datasets:
        name, path = _parse_dataset_arg(arg)
        print(f"[INFO] Evaluating {args.model} on {name} ({path}) ...")
        metrics = _evaluate_dataset(
            model_fn=model_fn,
            name=name,
            path=path,
            max_rows=args.max_rows,
            max_features=args.max_features,
            test_size=args.test_size,
            seed=args.seed,
            holdout_dir=args.holdout_dir,
            standardize=args.standardize,
            model_name=args.model.lower(),
            ood_split=args.ood_split,
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
        payload = {f"{args.model}_regression": results, "_meta": meta}
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved report to {args.report_json}")


if __name__ == "__main__":
    main()
