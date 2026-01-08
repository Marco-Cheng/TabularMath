#!/usr/bin/env python3
"""
Cross-platform experiment driver for TabularMath.

Runs the full grid of models, splits, and row caps:
- TabPFN v2 and v2.5 via tabpfn-client
- XGBoost
- Classic models: RandomForest, LightGBM, CatBoost, RealMLP, TabM, xRFM
- (optional) ICL LLM baseline when an LLM hook is configured

Outputs:
- Raw JSON reports under artifacts/reports/raw/
- Aggregated TSVs under artifacts/reports/summaries/
- Rounded-consistency plots under artifacts/plots_png/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent

for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import LOGS_DIR, MANIFESTS_DIR, PLOTS_DIR, RAW_REPORTS_DIR, SUMMARY_DIR, SCRIPTS_DIR


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

def _load_report_entries(path: Path) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    if not path.exists():
        return {}, {}
    try:
        text = path.read_text().strip()
    except OSError:
        return {}, {}
    if not text:
        return {}, {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}, {}
    entries: Dict[str, Dict[str, object]] = {}
    meta: Dict[str, object] = {}
    if isinstance(data, dict):
        meta = data.get("_meta", {}) if isinstance(data.get("_meta"), dict) else {}
        for key, value in data.items():
            if key == "_meta" or not isinstance(value, dict):
                continue
            for dataset_name, metrics in value.items():
                if isinstance(metrics, dict):
                    entries[dataset_name] = metrics
    return entries, meta


def report_status(path: Path, expected_datasets: Set[str],
                  required_meta: Dict[str, object] | None = None) -> Tuple[bool, Set[str]]:
    entries, meta = _load_report_entries(path)
    if required_meta:
        for key, value in required_meta.items():
            if meta.get(key) != value:
                return False, set(expected_datasets)
    available = {name for name, metrics in entries.items() if metrics}
    has_entries = bool(available)
    missing = set(expected_datasets) - available
    if not has_entries and not entries:
        missing = set(expected_datasets)
    return has_entries, missing


def build_dataset_args(selected: Set[str], entries: List[Tuple[str, str]]) -> List[str]:
    args: List[str] = []
    if not selected:
        return args
    order = set(selected)
    for name, path in entries:
        if name in order:
            args.extend(["--dataset", f"{name}:{path}"])
    return args


def merge_report_files(target: Path, update: Path) -> None:
    def _merge(dst, src):
        for key, value in src.items():
            if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    base: Dict[str, object] = {}
    if target.exists():
        try:
            base = json.loads(target.read_text())
        except json.JSONDecodeError:
            base = {}
    try:
        new_data = json.loads(update.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse partial report {update}") from exc
    _merge(base, new_data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(base, indent=2))
    try:
        update.unlink()
    except FileNotFoundError:
        pass


def write_partial_manifest(selected: Set[str], entries: List[Tuple[str, str]], tmp_dir: Path) -> Path:
    if not selected:
        raise ValueError("Cannot write manifest with no datasets.")
    fd, tmp_path = tempfile.mkstemp(prefix="tabmath_manifest_", suffix=".txt", dir=str(tmp_dir))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for name, path in entries:
            if name in selected:
                handle.write("--dataset\n")
                handle.write(f"{name}:{path}\n")
    return Path(tmp_path)


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run(cmd: List[str], env, label: str, allow_fail: bool = False) -> None:
    print(f"[RUN] {label}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] {label} failed with code {exc.returncode}")
        if not allow_fail:
            raise


def module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full TabularMath experiment grid.")
    ap.add_argument("--manifest", default=str(MANIFESTS_DIR / "tmp_datasets_rows2048.txt"))
    ap.add_argument("--skip_icl", action="store_true", help="Skip ICL LLM runs (use when no LLM hook is configured).")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    entries = load_manifest(manifest)
    dataset_names = [name for name, _ in entries]
    dataset_set: Set[str] = set(dataset_names)
    total_datasets = len(dataset_set)
    dataset_args_all = build_dataset_args(dataset_set, entries)

    raw_dir = RAW_REPORTS_DIR
    sum_dir = SUMMARY_DIR
    plots_dir = PLOTS_DIR
    log_dir = LOGS_DIR
    mpl_cache_dir = log_dir / "mpl_cache"
    for d in (raw_dir, sum_dir, plots_dir, log_dir, mpl_cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{EVAL_DIR}"
    env["MPLCONFIGDIR"] = str(mpl_cache_dir)

    python_bin = sys.executable
    split_flags = {"random": "--random_split", "ood": "--ood_split"}
    rowcaps_full = [32, 64, 128, 256, 512, 1024, 2048]
    rowcaps_icl = [32, 64, 128]
    generic_models = ["random_forest", "lightgbm", "catboost", "realmlp", "tabm", "xrfm"]
    standardize_meta_requirement = {"standardize_stats": "train_only"}

    icl_available = not args.skip_icl

    # TabPFN v2.5 requires an access token.
    if not env.get("PRIORLAB_API_KEY"):
        print("[WARN] PRIORLAB_API_KEY is not set; TabPFN v2.5 runs will fail.")

    split_order = {name: idx for idx, name in enumerate(split_flags.keys())}
    tabpfn_tasks = []
    task_sequence = 0

    def queue_tabpfn(model_path: str, run_name: str, rows: int, split: str, split_flag: str) -> None:
        nonlocal task_sequence
        out = raw_dir / f"{run_name}_rows{rows}_standardized_report_{split}.json"
        label = f"{run_name} rows={rows} split={split}"
        has_entries, missing = report_status(out, dataset_set)
        if not has_entries:
            selected = set(dataset_set)
        else:
            selected = missing
        if not selected:
            print(f"[SKIP] {label}: existing report at {out}")
            return
        partial_run = has_entries and len(selected) < total_datasets
        manifest_path = manifest
        cleanup_manifest: Path | None = None
        report_path = out
        if partial_run:
            manifest_path = write_partial_manifest(selected, entries, log_dir)
            cleanup_manifest = manifest_path
            report_path = out.with_suffix(out.suffix + ".partial")
            print(f"[INFO] {label}: filling {len(selected)}/{total_datasets} missing datasets")
        cmd = [
            python_bin,
            str(SCRIPTS_DIR / "eval_tabpfn_client.py"),
            "--manifest",
            str(manifest_path),
            "--rowcap",
            str(rows),
            split_flag,
            "--model_path",
            model_path,
            "--run_name",
            run_name,
            "--report_json",
            str(report_path),
        ]
        tabpfn_tasks.append(
            (
                rows,
                split_order[split],
                task_sequence,
                cmd,
                label,
                partial_run,
                out,
                report_path if partial_run else None,
                cleanup_manifest,
            )
        )
        task_sequence += 1

    for split, split_flag in split_flags.items():
        for rows in rowcaps_full:
            queue_tabpfn("v2_default", "tabpfn2", rows, split, split_flag)
            queue_tabpfn("v2.5_default", "tabpfn25", rows, split, split_flag)

            out = raw_dir / f"xgboost_rows{rows}_standardized_report_{split}.json"
            label = f"xgboost rows={rows} split={split}"
            has_entries, missing = report_status(out, dataset_set, standardize_meta_requirement)
            if not has_entries:
                selected = set(dataset_set)
            else:
                selected = missing
            if not selected:
                print(f"[SKIP] {label}: existing report at {out}")
            else:
                partial_run = has_entries and len(selected) < total_datasets
                report_path = out if not partial_run else out.with_suffix(out.suffix + ".partial")
                ds_args = dataset_args_all if not partial_run else build_dataset_args(selected, entries)
                if partial_run:
                    print(f"[INFO] {label}: filling {len(selected)}/{total_datasets} missing datasets")
                try:
                    run(
                        [python_bin, str(SCRIPTS_DIR / "eval_xgboost_regression.py"),
                         "--standardize", split_flag, "--max_rows", str(rows),
                         "--report_json", str(report_path), *ds_args],
                        env, label=label
                    )
                except Exception:
                    if partial_run and report_path.exists():
                        safe_unlink(report_path)
                    raise
                if partial_run:
                    merge_report_files(out, report_path)

            for model in generic_models:
                out = raw_dir / f"{model}_rows{rows}_standardized_report_{split}.json"
                allow_fail = False
                if model == "realmlp" and not module_available("third_party.realmlp"):
                    print(f"[WARN] Skipping {model} (third_party.realmlp missing).")
                    continue
                if model == "tabm" and not module_available("tabm"):
                    print(f"[WARN] Skipping {model} (tabm not installed).")
                    continue
                if model == "xrfm" and not module_available("xrfm"):
                    print(f"[WARN] Skipping {model} (xrfm not installed).")
                    continue
                label = f"{model} rows={rows} split={split}"
                has_entries, missing = report_status(out, dataset_set, standardize_meta_requirement)
                if not has_entries:
                    selected = set(dataset_set)
                else:
                    selected = missing
                if not selected:
                    print(f"[SKIP] {label}: existing report at {out}")
                    continue
                partial_run = has_entries and len(selected) < total_datasets
                report_path = out if not partial_run else out.with_suffix(out.suffix + ".partial")
                ds_args = dataset_args_all if not partial_run else build_dataset_args(selected, entries)
                if partial_run:
                    print(f"[INFO] {label}: filling {len(selected)}/{total_datasets} missing datasets")
                try:
                    run(
                        [python_bin, str(SCRIPTS_DIR / "eval_generic_regression.py"),
                         "--model", model, "--standardize", split_flag, "--max_rows", str(rows),
                         "--report_json", str(report_path), *ds_args],
                        env, label=label, allow_fail=allow_fail
                    )
                except Exception:
                    if partial_run and report_path.exists():
                        safe_unlink(report_path)
                    raise
                if partial_run:
                    merge_report_files(out, report_path)

        if icl_available:
            for rows in rowcaps_icl:
                out = raw_dir / f"icl_llm_rows{rows}_{split}.json"
                log = raw_dir / f"icl_llm_rows{rows}_{split}_log.jsonl"
                run(
                    [python_bin, str(SCRIPTS_DIR / "eval_icl_llm.py"),
                     "--manifest", str(manifest), "--rowcap", str(rows), split_flag,
                     "--out", str(out), "--log_jsonl", str(log)],
                    env, label=f"icl rows={rows} split={split}"
                )

    if tabpfn_tasks:
        tabpfn_tasks.sort(key=lambda item: ((item[0] if item[0] != 2048 else 10_000), item[1], item[2]))
        for task in tabpfn_tasks:
            _, _, _, cmd, label, partial_run, final_path, temp_path, temp_manifest = task
            try:
                run(cmd, env, label=label, allow_fail=True)
            except Exception:
                if temp_path and temp_path.exists():
                    safe_unlink(temp_path)
                if temp_manifest and temp_manifest.exists():
                    safe_unlink(temp_manifest)
                raise
            if partial_run and temp_path:
                merge_report_files(final_path, temp_path)
            if temp_manifest and temp_manifest.exists():
                safe_unlink(temp_manifest)

    run(
        [python_bin, str(SCRIPTS_DIR / "aggregate_rowcap_reports.py"),
         "--metrics_out", str(sum_dir / "model_rowcap_metrics.tsv"),
         "--summary_out", str(sum_dir / "model_rowcap_summary.tsv")],
        env, label="aggregate reports"
    )

    run(
        [python_bin, str(SCRIPTS_DIR / "plot_consistency_curves.py")],
        env, label="plot curves"
    )

    print("[DONE] All experiments complete.")


if __name__ == "__main__":
    main()
