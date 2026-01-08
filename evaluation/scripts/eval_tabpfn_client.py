#!/usr/bin/env python3
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

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import MANIFESTS_DIR

from data_utils import load_dataset
from tabpfn_client import TabPFNRegressor, init as tabpfn_init, set_access_token

def load_manifest(manifest: Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    lines = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i] == "--dataset" and i + 1 < len(lines):
            spec = lines[i + 1]
            if ":" in spec:
                name, path = spec.split(":", 1)
                entries.append((name.strip(), path.strip()))
            i += 2
        else:
            i += 1
    return entries

def standardize(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, float]:
    mean = train.mean(axis=0).values.astype(np.float32)
    std = train.std(axis=0, ddof=0).values.astype(np.float32)
    std[std == 0] = 1.0
    X_train = ((train.values - mean) / std).astype(np.float32)
    X_test = ((test.values - mean) / std).astype(np.float32)
    return X_train, X_test, mean, std

def standardize_target(y: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if y_std == 0:
        y_std = 1.0
    y_train = ((y.values - y_mean) / y_std).astype(np.float32)
    y_test_std = ((y_test.values - y_mean) / y_std).astype(np.float32)
    return y_train, y_test_std, y_mean, y_std

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


def evaluate_dataset(name: str, path: str, rowcap: int, seed: int, ood_split: bool, model_path: str) -> Dict[str, float]:
    df, dropped_cols = load_dataset(Path(path), rowcap=rowcap or None, seed=seed, target_col="y")
    if len(df) < 5:
        raise ValueError("Not enough rows in table after sanitization")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "y" not in num_df.columns:
        raise ValueError("y column missing after sanitization")
    cols = [c for c in num_df.columns if c != "y"]
    X = num_df[cols].fillna(num_df[cols].median()).astype(np.float32)
    y = num_df["y"].astype(np.float32)
    train_idx, test_idx = _split_indices(y, test_fraction=0.2, seed=seed, ood_split=ood_split)
    X_train_df = X.iloc[train_idx]
    X_test_df = X.iloc[test_idx]
    y_train_series = y.iloc[train_idx]
    y_test_series = y.iloc[test_idx]

    X_train, X_test, _, _ = standardize(X_train_df, X_test_df)
    y_train, y_test_std, y_mean, y_std = standardize_target(y_train_series, y_test_series)

    reg = TabPFNRegressor(model_path=model_path, n_estimators=8, random_state=seed)
    reg.fit(X_train, y_train)
    preds_std = reg.predict(X_test, output_type="mean")
    if isinstance(preds_std, dict):
        preds_std = preds_std.get("y_pred", preds_std)
    preds_std = np.asarray(preds_std, dtype=np.float32)
    preds = preds_std * y_std + y_mean
    y_eval = y_test_series.values

    mse = mean_squared_error(y_eval, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_eval, preds)
    r2 = r2_score(y_eval, preds)
    mse_norm = mean_squared_error(y_test_std, preds_std)
    rmse_norm = float(np.sqrt(mse_norm))
    mae_norm = mean_absolute_error(y_test_std, preds_std)
    r2_norm = r2_score(y_test_std, preds_std)
    rounded_consistency = float(np.mean(np.rint(preds) == np.rint(y_eval)))
    return {
        "n_samples": int(len(df)),
        "n_features": int(len(cols)),
        "dropped_columns": dropped_cols,
        "model_path": model_path,
        "mse": float(mse_norm),
        "rmse": float(rmse_norm),
        "mae": float(mae_norm),
        "r2": float(r2_norm),
        "mse_original": float(mse),
        "rmse_original": rmse,
        "mae_original": float(mae),
        "r2_original": float(r2),
        "rounded_consistency": rounded_consistency,
        "split": "ood" if ood_split else "random",
    }

def main():
    default_manifest = MANIFESTS_DIR / "tmp_datasets_rows2048.txt"
    ap = argparse.ArgumentParser(description="Evaluate TabPFN v2.5 via API")
    ap.add_argument("--manifest", default=str(default_manifest))
    ap.add_argument("--rowcap", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--run_name", type=str, default="tabpfn25", help="Prefix used in the output JSON key.")
    ap.add_argument("--model_path", type=str, default="v2.5_default", help="TabPFN model_path (e.g., v2_default, v2.5_default).")
    ap.add_argument("--access_token", type=str, default=None,
                    help="Prior Labs API token. Falls back to PRIORLAB_API_KEY env var.")
    ap.add_argument("--ood_split", dest="ood_split", action="store_true", default=True,
                    help="Use out-of-distribution split (top 20%% y values as test). (default)")
    ap.add_argument("--random_split", dest="ood_split", action="store_false",
                    help="Use random train/test split instead of OOD split.")
    args = ap.parse_args()

    api_key = args.access_token or os.environ.get("PRIORLAB_API_KEY")
    if not api_key:
        raise RuntimeError("PRIORLAB_API_KEY not set (and --access_token not provided)")
    set_access_token(api_key)
    tabpfn_init()

    entries = load_manifest(Path(args.manifest))
    tag = args.run_name
    results = {}
    for name, path in entries:
        try:
            metrics = evaluate_dataset(name, path, args.rowcap, args.seed, args.ood_split, args.model_path)
            results[name] = metrics
            print(f"[{tag}] {name} rows={metrics['n_samples']} r2={metrics['r2']:.3f}")
        except Exception as exc:
            print(f"[{tag}][FAIL] {name}: {exc}")
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({f"{args.run_name}_regression": results}, f, indent=2)
    print(f"Saved report to {out}")

if __name__ == "__main__":
    main()
