#!/usr/bin/env python3
"""
Shared helpers for loading and sanitizing TabularMath datasets.

All tables carry a numeric target column named 'y'. The first few metadata
columns (lang/len_chars/delta_*) are uninformative for modeling, and some
feature columns are constant across all rows. We drop those before any
training or logging to ensure only meaningful signals remain.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Explicit list of front-matter columns to strip from every table.
MEANINGLESS_PREFIX_COLUMNS: List[str] = ["lang", "len_chars", "delta_num", "delta_text", "delta_total"]


def sanitize_dataframe(df: pd.DataFrame, target_col: str = "y") -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove the known meaningless prefix columns and any columns that are
    constant across all rows (excluding the target). Returns the sanitized
    dataframe and the list of dropped columns.
    """
    drop_cols: List[str] = []
    drop_cols.extend([c for c in MEANINGLESS_PREFIX_COLUMNS if c in df.columns])

    for col in df.columns:
        if col == target_col:
            continue
        # Treat all-NaN and single-unique-value columns as meaningless.
        if df[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols


def load_dataset(
    path: Path,
    rowcap: int | None = None,
    seed: int = 2025,
    target_col: str = "y",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a parquet table, optionally subsample to `rowcap`, and sanitize its
    columns. Returns the dataframe and the list of dropped columns.
    """
    df = pd.read_parquet(path)
    if rowcap and len(df) > rowcap:
        df = df.sample(n=rowcap, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df, dropped = sanitize_dataframe(df, target_col=target_col)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in {path}")
    return df, dropped
