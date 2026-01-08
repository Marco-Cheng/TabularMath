#!/usr/bin/env python3
"""
Create normalized copies of per-problem parquet tables.

Usage:
  python3 curation/normalize_tables.py \
      --input_dir tabular/gsm8k_problem_tables_512 \
      --output_dir tabular/gsm8k_problem_tables_512_norm

All numeric feature columns (everything except 'y') are standardized to zero mean
and unit variance using statistics computed over the entire table. Columns with
zero variance are set to 0.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "y" and np.issubdtype(df[c].dtype, np.number)]
    for col in feature_cols:
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if std and std > 0:
            df[col] = (series - mean) / std
        else:
            df[col] = 0.0
    return df


def collect_parquets(input_dir: Path) -> Sequence[Path]:
    return sorted(input_dir.glob("*.parquet"))


def main():
    ap = argparse.ArgumentParser(description="Normalize per-problem parquet tables.")
    ap.add_argument("--input_dir", required=True, help="Directory containing source parquet files.")
    ap.add_argument("--output_dir", required=True, help="Directory to write normalized parquet files.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in collect_parquets(in_dir):
        df = pd.read_parquet(src)
        norm_df = normalize_table(df)
        dst = out_dir / src.name
        norm_df.to_parquet(dst, index=False)
        norm_df.to_csv(dst.with_suffix(".csv"), index=False)
        print(f"[OK] {src.name} -> {dst}")


if __name__ == "__main__":
    main()
