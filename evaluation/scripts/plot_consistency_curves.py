#!/usr/bin/env python3
"""Plot mean rounded consistency curves for each split/family."""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import LOGS_DIR, PLOTS_DIR, SUMMARY_DIR

MPL_CACHE_DIR = LOGS_DIR / "mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    data_path = SUMMARY_DIR / "rounded_consistency_mean_by_family_with_best.tsv"
    if not data_path.exists():
        raise SystemExit(f"Input TSV not found: {data_path}")

    df = pd.read_csv(data_path, sep="\t")
    model_cols = [
        c
        for c in df.columns
        if c
        not in {
            "split",
            "family",
            "rowcap",
            "best_model",
            "best_mean_rounded_consistency",
        }
    ]

    plot_dir = PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    for split in sorted(df["split"].unique()):
        for family in sorted(df["family"].unique()):
            subset = df[(df["split"] == split) & (df["family"] == family)]
            if subset.empty:
                continue
            subset = subset.sort_values("rowcap")

            plt.figure(figsize=(8, 5))
            for model in model_cols:
                plt.plot(
                    subset["rowcap"],
                    subset[model],
                    marker="o",
                    label=model,
                )

            plt.title(f"Mean Rounded Consistency ({split.upper()} • {family})")
            plt.xlabel("Row cap")
            plt.ylabel("Mean rounded consistency")
            plt.xscale("log", base=2)
            plt.grid(alpha=0.3)
            plt.legend(loc="upper left", fontsize="small", ncol=2, bbox_to_anchor=(1.02, 1), borderaxespad=0.)

            out_file = plot_dir / f"rounded_consistency_{split}_{family}.png"
            plt.savefig(out_file, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
