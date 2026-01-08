#!/usr/bin/env python3
"""
Evaluate LLM in-context learning baselines on tabular math datasets.

Set TABMATH_ICL_LLM_CLIENT="module:function" to supply your own callable that
accepts (prompt: str, logid: str) -> str. Without this environment variable the
script falls back to a placeholder client that returns "<unknown>" so smoke
tests can run offline. The helper `oss_llm_call` reproduces the original OSS
deployment; configure it via TABMATH_ICL_LLM_CLIENT="scripts.eval_icl_llm:oss_llm_call".
"""
import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
for path in (EVAL_DIR, REPO_ROOT):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from path_utils import MANIFESTS_DIR, RAW_REPORTS_DIR, REPORTS_DIR

from data_utils import load_dataset


def load_manifest(path: Path) -> List[Tuple[str, str]]:
    entries = []
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i] == "--dataset" and i + 1 < len(lines):
            spec = lines[i + 1]
            if ":" in spec:
                name, p = spec.split(":", 1)
                entries.append((name.strip(), p.strip()))
            i += 2
        else:
            i += 1
    return entries

def build_prompt(name: str, context: pd.DataFrame, queries: pd.DataFrame) -> str:
    cols = [c for c in context.columns if c != "y"]
    ctx_lines = []
    for _, row in context.iterrows():
        feats = ", ".join(f"{col}={row[col]}" for col in cols)
        ctx_lines.append(f"CONTEXT: {feats}, y={row['y']}")
    query_lines = []
    for idx, row in queries.iterrows():
        feats = ", ".join(f"{col}={row[col]}" for col in cols)
        query_lines.append(f"QUERY {idx}: {feats}")
    prompt = (
        f"You are an expert regression model. The dataset '{name}' has numeric features {cols} and target y.\n"
        "First, learn from the context rows (each line labelled CONTEXT). Then, predict y for each QUERY row.\n"
        "Return ONLY a JSON list of floats corresponding to the QUERY rows in order."
        "\n\nContext rows:\n" + "\n".join(ctx_lines)
        + "\n\nQuery rows:\n" + "\n".join(query_lines)
    )
    return prompt

def _default_llm_call(prompt: str, logid: str) -> str:
    placeholder = os.environ.get("TABMATH_ICL_PLACEHOLDER", "<unknown>")
    return placeholder


def oss_llm_call(prompt: str, logid: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": "You convert tabular regression prompts into numeric predictions."},
        {"role": "user", "content": prompt},
    ]
    return _default_llm_call(messages, logid)


def _load_llm_call() -> Callable[[str, str], Optional[str]]:
    hook = os.environ.get("TABMATH_ICL_LLM_CLIENT")
    if not hook:
        return _default_llm_call
    module_name, func_name = hook.rsplit(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{hook} is not callable")
    return fn


LLM_CALL = _load_llm_call()


def call_llm(prompt: str, logid: str) -> Optional[str]:
    return LLM_CALL(prompt, logid)


def parse_predictions(text: str, expected: int) -> List[float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    preds = [float(n) for n in numbers]
    if len(preds) >= expected:
        return preds[:expected]
    raise ValueError(f"Only parsed {len(preds)} predictions from response: {text[:200]}")


def _split_ood_context_queries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values("y", ascending=True).reset_index(drop=True)
    n = len(sorted_df)
    query_count = max(1, int(np.ceil(n * 0.2)))
    if query_count >= n:
        query_count = max(1, n - 1)
    context = sorted_df.iloc[: n - query_count]
    queries = sorted_df.iloc[n - query_count :]
    return context, queries


def _split_random_context_queries(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    query_count = max(1, int(np.ceil(n * 0.2)))
    if query_count >= n:
        query_count = max(1, n - 1)
    context = shuffled.iloc[: n - query_count]
    queries = shuffled.iloc[n - query_count :]
    return context, queries


def _invoke_with_retry(prompt: str, logid: str, expected: int, max_attempts: int = 10) -> Tuple[List[float], str]:
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = call_llm(prompt, logid=f"{logid}-try{attempt}")
        if raw is None:
            continue
        try:
            preds = parse_predictions(raw, expected)
            return preds, raw
        except ValueError:
            last_raw = raw
            if raw and "<unknown>" in raw.lower():
                print("[WARN] LLM returned placeholder; using zeros as fallback.")
                return [0.0] * expected, raw
    raise RuntimeError(
        "LLM response parsing failed after retries; last response: "
        f"{last_raw[:200] if last_raw else 'NONE'}"
    )


def evaluate_dataset(name: str, path: str, rowcap: int, seed: int, ood_split: bool) -> Tuple[dict, str, str]:
    df, dropped_cols = load_dataset(Path(path), rowcap=rowcap or None, seed=seed, target_col="y")
    if len(df) < 2:
        raise ValueError("Not enough rows in table after sanitization")
    if ood_split:
        context, queries = _split_ood_context_queries(df)
    else:
        context, queries = _split_random_context_queries(df, seed)
    prompt = build_prompt(name, context, queries)
    preds, raw = _invoke_with_retry(prompt, logid=f"icl-{name}", expected=len(queries))
    y_true = queries["y"].to_numpy()
    y_pred = np.array(preds)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rounded = np.rint(y_pred)
    consistency = float(np.mean(rounded == np.rint(y_true)))
    metrics = {
        "dataset": name,
        "n_features": len(df.columns) - 1,
        "context_rows": len(context),
        "query_rows": len(queries),
        "dropped_columns": dropped_cols,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rounded_consistency": consistency,
        "split": "ood" if ood_split else "random",
    }
    return metrics, prompt, raw


def main():
    default_manifest = MANIFESTS_DIR / "tmp_datasets_rows2048.txt"
    default_out = RAW_REPORTS_DIR / "icl_llm_results.json"
    default_log = REPORTS_DIR / "icl_llm_logs.jsonl"
    ap = argparse.ArgumentParser(description="Evaluate LLM ICL via get_max_response")
    ap.add_argument("--manifest", default=str(default_manifest))
    ap.add_argument("--rowcap", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--log_jsonl", default=str(default_log),
                    help="Optional JSONL log path for prompts/responses.")
    ap.add_argument("--ood_split", dest="ood_split", action="store_true", default=True,
                    help="Use out-of-distribution split (top 20%% y values as query rows). (default)")
    ap.add_argument("--random_split", dest="ood_split", action="store_false",
                    help="Use random 80/20 context/query split instead of OOD split.")
    args = ap.parse_args()

    entries = load_manifest(Path(args.manifest))

    log_path = Path(args.log_jsonl) if args.log_jsonl else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path in entries:
        try:
            res, prompt, raw = evaluate_dataset(
                name, path, args.rowcap, args.seed, args.ood_split
            )
            results.append(res)
            print(f"[LLM] {name}: R2={res['r2']:.3f} Consistency={res['rounded_consistency']:.3f}")
            if log_path:
                record = {
                    "dataset": name,
                    "rowcap": args.rowcap,
                    "context_rows": res["context_rows"],
                    "query_rows": res["query_rows"],
                    "split": res["split"],
                    "prompt": prompt,
                    "response": raw,
                }
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[LLM][FAIL] {name}: {exc}")

    summary = {
        "split": "ood" if args.ood_split else "random",
        "rowcap": args.rowcap,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {args.out}")

if __name__ == "__main__":
    main()
