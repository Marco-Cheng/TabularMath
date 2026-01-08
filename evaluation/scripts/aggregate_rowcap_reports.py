#!/usr/bin/env python3
"""
Aggregate per-dataset metrics for multiple row-cap experiments.

Outputs (configurable via CLI flags):
 - runs/model_rowcap_metrics.tsv
 - runs/model_rowcap_summary.tsv
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import RAW_REPORTS_DIR, SUMMARY_DIR
MEAN_TABLE_PATH = SUMMARY_DIR / "rounded_consistency_mean_table.tsv"
MEAN_FAMILY_BEST_PATH = SUMMARY_DIR / "rounded_consistency_mean_by_family_with_best.tsv"

ROWCAPS = [32, 64, 128, 256, 512, 1024, 2048]
ICL_ROWCAPS = [32, 64, 128]
ICL_SPLITS = ["random", "ood"]
REPORT_SUFFIXES = [
    ("standardized_report.json", None),
    ("standardized_report_random.json", "random"),
    ("standardized_report_ood.json", "ood"),
]
MODELS = {
    "tabpfn2": "tabpfn2_regression",
    "tabpfn25": "tabpfn25_regression",
    "xgboost": "xgboost_regression",
    "random_forest": "random_forest_regression",
    "lightgbm": "lightgbm_regression",
    "catboost": "catboost_regression",
    "realmlp": "realmlp_regression",
    "tabm": "tabm_regression",
    "xrfm": "xrfm_regression",
    "icl_llm": None,
}


def dataset_family(dataset: str) -> str:
    return "AIME" if dataset.startswith("202") else "GSM8K"


def read_report(path: Path, top_key: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if top_key not in data:
        raise KeyError(f"Report {path} missing top-level key '{top_key}'")
    data = data[top_key]
    df = (
        pd.DataFrame.from_dict(data, orient="index")
        .rename_axis("dataset")
        .reset_index()
    )
    return df

def read_icl_report(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["rowcap"] = data.get("rowcap")
    df["split"] = data.get("split") or "random"
    df["model"] = "icl_llm"
    df["n_samples"] = df.get("context_rows", 0) + df.get("query_rows", 0)
    for col in ["mse", "rmse", "mae"]:
        if col not in df.columns:
            df[col] = pd.NA
    if "n_features" not in df.columns:
        df["n_features"] = pd.NA
    return df

def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate row-cap experiment metrics.")
    ap.add_argument(
        "--metrics_out",
        default=str(SUMMARY_DIR / "model_rowcap_metrics.tsv"),
        help="Where to save the combined per-dataset metrics (TSV).",
    )
    ap.add_argument(
        "--summary_out",
        default=str(SUMMARY_DIR / "model_rowcap_summary.tsv"),
        help="Where to save the aggregated summary statistics (TSV).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    metrics_frames = []
    for model, top_key in MODELS.items():
        if model == "icl_llm":
            continue
        for rows in ROWCAPS:
            for suffix, default_split in REPORT_SUFFIXES:
                report_path = RAW_REPORTS_DIR / f"{model}_rows{rows}_{suffix}"
                if not report_path.exists():
                    continue
                df = read_report(report_path, top_key)
                if "split" not in df.columns:
                    df["split"] = default_split if default_split else "random"
                df["model"] = model
                df["rowcap"] = rows
                metrics_frames.append(df)
    # Add ICL reports
    for split in ICL_SPLITS:
        for rows in ICL_ROWCAPS:
            report_path = RAW_REPORTS_DIR / f"icl_llm_rows{rows}_{split}.json"
            if not report_path.exists():
                continue
            df = read_icl_report(report_path)
            if df.empty:
                continue
            metrics_frames.append(df)
    if not metrics_frames:
        raise SystemExit("No row-cap reports found in artifacts/reports/raw/")

    combined = pd.concat(metrics_frames, ignore_index=True)
    combined["family"] = combined["dataset"].apply(dataset_family)
    if "split" not in combined.columns:
        combined["split"] = "random"
    combined["split"] = combined["split"].fillna("random")
    for col in [
        "mse_original",
        "rmse_original",
        "mae_original",
        "r2_original",
    ]:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[
        [
            "dataset",
            "family",
            "split",
            "model",
            "rowcap",
            "n_samples",
            "n_features",
            "mse",
            "rmse",
            "mae",
            "r2",
            "mse_original",
            "rmse_original",
            "mae_original",
            "r2_original",
            "rounded_consistency",
        ]
    ]
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.metrics_out, sep="\t", index=False)

    summary_rows = []
    metrics = [
        "mse",
        "rmse",
        "mae",
        "r2",
        "mse_original",
        "rmse_original",
        "mae_original",
        "r2_original",
        "rounded_consistency",
    ]
    families = ["AIME", "GSM8K"]
    splits = sorted(combined["split"].dropna().unique())
    for split_name in splits:
        split_df = combined[combined["split"] == split_name]
        for family in families:
            fam_df = split_df[split_df["family"] == family]
            if fam_df.empty:
                continue
            for model in MODELS.keys():
                for rows in ROWCAPS:
                    subset = fam_df[(fam_df["model"] == model) & (fam_df["rowcap"] == rows)]
                    if subset.empty:
                        continue
                    for metric in metrics:
                        series = subset[metric].dropna()
                        if series.empty:
                            continue
                        summary_rows.append(
                            {
                                "split": split_name,
                                "family": family,
                                "model": model,
                                "rowcap": rows,
                                "metric": metric,
                                "mean": series.mean(),
                                "median": series.median(),
                                "std": series.std(ddof=0),
                                "pct10": series.quantile(0.10),
                                "pct25": series.quantile(0.25),
                                "pct75": series.quantile(0.75),
                                "pct90": series.quantile(0.90),
                                "min": series.min(),
                                "max": series.max(),
                            }
    )
    summary = pd.DataFrame(summary_rows)
    summary.sort_values(["split", "family", "model", "rowcap", "metric"], inplace=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, sep="\t", index=False)

    # Write rounded consistency helper tables for plotting/reporting.
    def _write_mean_tables() -> None:
        if combined.empty:
            return
        rounded = combined[["split", "family", "rowcap", "model", "rounded_consistency"]].copy()

        overall = (
            rounded.groupby(["split", "rowcap", "model"])["rounded_consistency"]
            .mean()
            .reset_index()
        )
        overall_pivot = (
            overall.pivot_table(index=["split", "rowcap"], columns="model", values="rounded_consistency")
            .reset_index()
        )
        MEAN_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        overall_pivot.to_csv(MEAN_TABLE_PATH, sep="\t", index=False)

        family_mean = (
            rounded.groupby(["split", "family", "rowcap", "model"])["rounded_consistency"]
            .mean()
            .reset_index()
        )
        family_pivot = (
            family_mean.pivot_table(index=["split", "family", "rowcap"], columns="model", values="rounded_consistency")
            .reset_index()
        )
        model_cols = [c for c in family_pivot.columns if c not in {"split", "family", "rowcap"}]

        def _best(row):
            series = row[model_cols].dropna()
            if series.empty:
                return pd.Series({"best_model": pd.NA, "best_mean_rounded_consistency": pd.NA})
            best_model = series.idxmax()
            return pd.Series({"best_model": best_model, "best_mean_rounded_consistency": series[best_model]})

        if model_cols:
            best_df = family_pivot.apply(_best, axis=1)
            family_pivot = pd.concat([family_pivot, best_df], axis=1)
        MEAN_FAMILY_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        family_pivot.to_csv(MEAN_FAMILY_BEST_PATH, sep="\t", index=False)

    _write_mean_tables()
    print(f"Row-cap metrics aggregated into {args.metrics_out} and {args.summary_out}")


if __name__ == "__main__":
    main()
