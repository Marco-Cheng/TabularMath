#!/usr/bin/env python3
"""Lightweight sanity checks for curated parquet tables."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def inspect_parquet(path: Path, expected_rows: int, require_unique: bool) -> Tuple[int, int, List[str], List[str]]:
    df = pd.read_parquet(path)
    rows = len(df)
    unique_rows = len(df.drop_duplicates())
    issues: List[str] = []
    warnings: List[str] = []

    if expected_rows and rows != expected_rows:
        issues.append(f"expected {expected_rows} rows, found {rows}")
    if require_unique and unique_rows != rows:
        issues.append(f"{rows - unique_rows} duplicate rows detected")
    if "y" not in df.columns:
        issues.append("missing target column 'y'")
    return rows, unique_rows, issues, warnings


def collect_parquets(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.parquet"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify curated parquet tables.")
    ap.add_argument("--dir", action="append", required=True, help="Directory containing per-problem parquet files.")
    ap.add_argument("--expected_rows", type=int, default=0, help="Expected number of rows per file (0 = skip check).")
    ap.add_argument("--require_unique", action="store_true", help="Fail if duplicate rows are found.")
    args = ap.parse_args()

    any_errors = False
    for dir_path in args.dir:
        directory = Path(dir_path)
        if not directory.exists():
            print(f"[ERR] Directory not found: {directory}")
            any_errors = True
            continue
        print(f"[SCAN] {directory}")
        files = collect_parquets(directory)
        if not files:
            print("  (no parquet files)")
            continue
        for parquet in files:
            rows, unique_rows, issues, _ = inspect_parquet(parquet, args.expected_rows, args.require_unique)
            if issues:
                any_errors = True
                issue_str = "; ".join(issues)
                print(f"  [FAIL] {parquet.name}: rows={rows} unique={unique_rows} :: {issue_str}")
            else:
                print(f"  [OK] {parquet.name}: rows={rows} unique={unique_rows}")

    if any_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
