#!/usr/bin/env python3
"""
Build per-problem tabular datasets from augmented math JSONL specs.

Each selected problem ID produces its own CSV and Parquet table containing
only feature columns (no metadata columns like dataset/id/template) plus target y.

Example:
  python3 curation/make_problem_tables.py \
    --aug_jsonl curation/data/augmented_gsm8k.jsonl \
    --out_dir tabular/gsm8k_tasks \
    --max_problems 10 \
    --rows_per_problem 256 \
    --seed 1337
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

# Allow running this script directly from the tabularmath package by ensuring
# the bundled augmentation module is on sys.path.
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from augmentation.dataset_io import load_jsonl as read_jsonl
from augmentation.tabular.make_tabular import (
    _build_row,
    _compose_generator_code,
)
from augmentation.utils import set_seed
from augmentation.verifier import RNGShim, compile_verifier, run_verifier

try:
    from augmentation.verifier import compile_generator
except Exception:
    compile_generator = None


def _choose_problem_ids(data: Sequence[Dict], requested: List[str],
                        max_problems: int, base_seed: int) -> List[str]:
    if max_problems and max_problems < 0:
        raise ValueError("max_problems must be >= 0")
    ids = [str(rec.get("id", "")).strip() for rec in data if rec.get("id")]
    if requested:
        chosen = [pid for pid in ids if pid in requested]
    else:
        chosen = ids[:]
    rnd = random.Random(base_seed)
    rnd.shuffle(chosen)
    if max_problems:
        chosen = chosen[:min(max_problems, len(chosen))]
    return chosen


def _unsafe_compile_generator(spec: Dict) -> callable:
    gen_spec = (spec or {}).get("generator") or {}
    code = gen_spec.get("code") or ""
    if not code:
        return None
    glob: Dict[str, any] = {"__builtins__": __builtins__, "math": math, "random": random}
    loc: Dict[str, any] = {}
    exec(code, glob, loc)
    fn = loc.get("generator")
    if not callable(fn):
        return None

    def _gen(rng: RNGShim):
        return fn(rng)

    return _gen


def _unsafe_compile_verifier(code: str):
    glob: Dict[str, any] = {"__builtins__": __builtins__, "math": math, "random": random}
    loc: Dict[str, any] = {}
    exec(code, glob, loc)
    fn = loc.get("verifier")
    if not callable(fn):
        raise ValueError("verifier function not defined in code")

    def _ver(assign: Dict[str, any]):
        res = fn(assign)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return bool(res[0]), res[1]
        return True, res

    return _ver


def _gather_rows_for_problem(
    problem_id: str,
    record: Dict,
    rows_per_problem: int,
    base_seed: int,
    drop_duplicates: bool,
    max_sampling_factor: int,
    unsafe_eval: bool,
    require_unique_rows: bool,
    time_limit_seconds: float,
) -> pd.DataFrame:
    spec = record.get("spec") or {}
    if not spec or "verifier" not in spec:
        raise ValueError(f"Problem {problem_id} missing verifier spec.")
    if "generator" not in spec:
        raise ValueError(f"Problem {problem_id} missing generator spec.")

    if unsafe_eval:
        ver = _unsafe_compile_verifier(spec["verifier"]["code"])
        gen = _unsafe_compile_generator(spec)
    else:
        ver = compile_verifier(spec["verifier"]["code"])
        gen_code = _compose_generator_code(spec)
        gen = compile_generator(gen_code) if compile_generator and gen_code else None

    cat_maps: Dict[str, Dict[str, int]] = {}
    rows: List[Dict] = []
    seen_rows: set = set()

    variants = record.get("variants") or []
    local_rng = random.Random(base_seed)
    variant_rows = 0
    generated_rows = 0

    start_time = time.time()

    def _try_add_row(row: Dict) -> bool:
        key = tuple(sorted(row.items()))
        if key in seen_rows:
            return False
        seen_rows.add(key)
        rows.append(row)
        return True

    # reuse existing variants first
    for v in variants:
        try:
            row = _build_row(
                dataset_tag=str(record.get("dataset_tag", "")),
                family_id=problem_id,
                item=v,
                spec=spec,
                assign=v.get("assignment") or {},
                y=v.get("y"),
                template_id=int(v.get("text_template_id", 0)),
                deltas={
                    "delta_num": v.get("delta_num", 0.0),
                    "delta_text": v.get("delta_text", 0.0),
                    "delta_total": v.get("delta_total", 0.0),
                },
                cat_maps=cat_maps,
                include_meta=False,
            )
            if _try_add_row(row):
                variant_rows += 1
            if len(rows) >= rows_per_problem:
                break
        except Exception:
            continue

    # sample new variants via generator if needed
    tries = 0
    max_tries = rows_per_problem * max(1, max_sampling_factor)
    text_count = max(1, len(spec.get("text_templates") or [None]))
    while len(rows) < rows_per_problem and tries < max_tries:
        tries += 1
        if gen is None:
            break
        if time_limit_seconds > 0 and (time.time() - start_time) >= time_limit_seconds:
            break
        try:
            seed_val = local_rng.getrandbits(30)
            if unsafe_eval:
                rng = RNGShim(seed_val)
                assign = gen(rng)
                ok, y = ver(assign)
            else:
                rng = RNGShim(seed_val)
                assign = gen(rng)
                ok, y = run_verifier(ver, assign)
            if not ok:
                continue
            tid = len(rows) % text_count
            row = _build_row(
                dataset_tag=str(record.get("dataset_tag", "")),
                family_id=problem_id,
                item={"x": ""},
                spec=spec,
                assign=assign,
                y=y,
                template_id=tid,
                deltas={"delta_num": 0.0, "delta_text": 0.0, "delta_total": 0.0},
                cat_maps=cat_maps,
                include_meta=False,
            )
            if _try_add_row(row):
                generated_rows += 1
        except Exception:
            continue

    if not rows:
        raise RuntimeError(f"No valid rows generated for problem {problem_id}")

    df = pd.DataFrame(rows)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    unique_rows = len(df)
    if require_unique_rows and unique_rows < rows_per_problem:
        raise RuntimeError(f"Only {unique_rows} unique rows (<{rows_per_problem})")
    df.attrs["variant_rows"] = variant_rows
    df.attrs["generated_rows"] = generated_rows
    df.attrs["unique_rows"] = unique_rows
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate per-problem tabular datasets.")
    ap.add_argument("--aug_jsonl", required=True, help="Path to augmented_* JSONL file.")
    ap.add_argument("--out_dir", required=True, help="Directory for per-problem outputs.")
    ap.add_argument("--rows_per_problem", type=int, default=256, help="Target rows per problem.")
    ap.add_argument("--max_problems", type=int, default=0, help="Max number of problems to export (0 = all).")
    ap.add_argument("--problem_ids", type=str, default="", help="Comma-separated list of problem IDs to include.")
    ap.add_argument("--seed", type=int, default=1337, help="Base random seed for generation.")
    ap.add_argument("--drop_duplicates", action="store_true", help="Drop duplicate rows before saving.")
    ap.add_argument("--min_rows_required", type=int, default=0,
                    help="Skip problems that produce fewer rows than this threshold (0 = allow any).")
    ap.add_argument("--target_count", type=int, default=0,
                    help="Stop after writing this many problem tables (0 = process all selected).")
    ap.add_argument("--max_sampling_factor", type=int, default=40,
                    help="Maximum multiple of rows_per_problem generator attempts (default 40).")
    ap.add_argument("--unsafe_eval", action="store_true",
                    help="Execute generator/verifier code in-process for speed (trusts the specs).")
    ap.add_argument("--require_unique_rows", action="store_true",
                    help="Ensure all rows within a table are unique; fail if not enough unique rows.")
    ap.add_argument("--time_limit_minutes", type=float, default=0.0,
                    help="Per-problem time limit while searching for unique rows (0 = unlimited).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    data = read_jsonl(args.aug_jsonl)
    requested_ids = [p.strip() for p in args.problem_ids.split(",") if p.strip()]
    chosen_ids = _choose_problem_ids(data, requested_ids, args.max_problems, args.seed)

    if not chosen_ids:
        ap.error("No problem IDs found to process.")

    record_map = {str(rec.get("id", "")).strip(): rec for rec in data}

    summary = []
    shortfalls = []
    for idx, pid in enumerate(chosen_ids):
        rec = record_map.get(pid)
        if not rec:
            print(f"[WARN] Problem {pid} not found; skipping.")
            continue
        seed_offset = args.seed + idx * 9973
        try:
            df = _gather_rows_for_problem(
                problem_id=pid,
                record=rec,
                rows_per_problem=args.rows_per_problem,
                base_seed=seed_offset,
                drop_duplicates=args.drop_duplicates,
                max_sampling_factor=args.max_sampling_factor,
                unsafe_eval=args.unsafe_eval,
                require_unique_rows=args.require_unique_rows,
                time_limit_seconds=args.time_limit_minutes * 60.0,
            )
        except Exception as exc:
            tag = "[DROP]" if args.require_unique_rows else "[WARN]"
            print(f"{tag} Failed to build table for {pid}: {exc}")
            continue

        if args.min_rows_required and len(df) < args.min_rows_required:
            print(f"[SKIP] {pid}: produced {len(df)} rows (<{args.min_rows_required}); skipping.")
            continue

        # ensure y is last column for readability
        cols = [c for c in df.columns if c != "y"] + ["y"]
        df = df[cols]

        out_csv = os.path.join(args.out_dir, f"{pid}.csv")
        out_parq = os.path.join(args.out_dir, f"{pid}.parquet")
        df.to_csv(out_csv, index=False)
        df.to_parquet(out_parq, index=False)
        variant_rows = df.attrs.get("variant_rows", 0)
        generated_rows = df.attrs.get("generated_rows", 0)
        unique_rows = df.attrs.get("unique_rows", len(df))
        summary.append((pid, len(df), len(df.columns), variant_rows, generated_rows, unique_rows))
        if unique_rows < args.rows_per_problem:
            shortfalls.append((pid, unique_rows))
        print(f"[OK] {pid}: rows={len(df)} cols={len(df.columns)} "
              f"(unique={unique_rows} variants={variant_rows} sampled={generated_rows})")

        if args.target_count and len(summary) >= args.target_count:
            break

    if summary:
        print("\nSummary:")
        for pid, rows, cols, var_rows, gen_rows, uniq in summary:
            print(f"  {pid}: rows={rows} cols={cols} (unique={uniq} variants={var_rows} sampled={gen_rows})")
        if shortfalls:
            print("\nShortfalls (unique rows < requested rows_per_problem):")
            for pid, uniq in shortfalls:
                print(f"  {pid}: unique_rows={uniq}")
    else:
        print("No tables generated.")


if __name__ == "__main__":
    main()
