#!/usr/bin/env python3
"""
Strip meaningless columns from every dataset and rewrite CSV/Parquet files.

Removals:
- Always drop the first four metadata columns (lang, len_chars, delta_num, delta_text, delta_total).
- Drop any additional columns that are constant across all rows (excluding the target y).

Outputs:
- Datasets are rewritten in place (both .parquet and matching .csv when present).
- A JSON log summarizing dropped columns per file is written to logs/dataset_sanitization_log.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import sanitize_dataframe
from path_utils import DATASETS_DIR, LOGS_DIR

DATASET_ROOT = DATASETS_DIR
LOG_PATH = LOGS_DIR / "dataset_sanitization_log.json"


def _sanitize_file(parquet_path: Path) -> Dict[str, object]:
    df = pd.read_parquet(parquet_path)
    original_cols = list(df.columns)
    sanitized, dropped = sanitize_dataframe(df, target_col="y")

    # Rewrite parquet and matching CSV (if it exists) with the cleaned columns.
    sanitized.to_parquet(parquet_path, index=False)
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        sanitized.to_csv(csv_path, index=False)

    return {
        "path": str(parquet_path),
        "rows": int(len(sanitized)),
        "cols_before": len(original_cols),
        "cols_after": sanitized.shape[1],
        "dropped": dropped,
    }


def main() -> None:
    results: List[Dict[str, object]] = []
    parquet_files = sorted(DATASET_ROOT.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {DATASET_ROOT}")

    for pq in parquet_files:
        info = _sanitize_file(pq)
        results.append(info)
        print(f"[sanitize] {pq.name}: dropped {len(info['dropped'])} cols -> {info['cols_after']} kept")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote sanitization log to {LOG_PATH}")


if __name__ == "__main__":
    main()
