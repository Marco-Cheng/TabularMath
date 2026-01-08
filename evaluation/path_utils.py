"""Centralized paths for the TabularMath evaluation stack."""
from __future__ import annotations

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
SCRIPTS_DIR = EVAL_DIR / "scripts"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RAW_REPORTS_DIR = REPORTS_DIR / "raw"
SUMMARY_DIR = REPORTS_DIR / "summaries"
PLOTS_DIR = ARTIFACTS_DIR / "plots_png"
LOGS_DIR = ARTIFACTS_DIR / "logs"
MANIFESTS_DIR = EVAL_DIR / "data_manifests"


def ensure_sys_path() -> None:
    """Add evaluation + repo roots to sys.path for script execution."""
    import sys

    for path in (EVAL_DIR, REPO_ROOT):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
