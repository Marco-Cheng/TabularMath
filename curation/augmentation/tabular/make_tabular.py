#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random
from typing import Any, Dict, List
import pandas as pd

from augmentation.dataset_io import load_jsonl as read_jsonl
from augmentation.verifier import compile_verifier, run_verifier
try:
    from augmentation.verifier import compile_generator  # optional in some repos
except Exception:
    compile_generator = None
from augmentation.utils import set_seed

# ---------- text / simple stats ----------
def _lang_hint(text: str) -> str:
    s = text or ""
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):
        return "zh"
    return "lat"

def _num_feats(name: str, v: float) -> Dict[str, Any]:
    out = {name: v, f"{name}_abs_log1p": math.log1p(abs(float(v)))}
    out[f"{name}_sign"] = 0 if v == 0 else (1 if v > 0 else -1)
    return out

def _int_feats(name: str, v: int) -> Dict[str, Any]:
    out = _num_feats(name, int(v))
    out[f"{name}_is_even"] = int(v % 2 == 0)
    for m in (3, 5, 7, 10):
        out[f"{name}_mod{m}"] = int(v % m)
    return out

def _float_feats(name: str, v: float) -> Dict[str, Any]:
    out = _num_feats(name, float(v))
    frac = abs(v - int(v))
    out[f"{name}_frac"] = float(frac)
    s = f"{v}"
    dec = len(s.split(".")[1]) if "." in s else 0
    out[f"{name}_n_decimals"] = int(dec)
    return out

def _categorical_id(table: Dict[str,int], key: str) -> int:
    if key not in table:
        table[key] = len(table)
    return table[key]

def _pairwise_math_feats(int_vals: List[int]) -> Dict[str, Any]:
    import math
    feats = {}
    if not int_vals:
        feats.update({"gcd_all": 0, "n_multiple_pairs": 0})
        return feats
    g = 0
    for v in int_vals:
        g = math.gcd(g, abs(int(v)))
    feats["gcd_all"] = int(g)
    n_mult = 0
    for i in range(len(int_vals)):
        for j in range(i+1, len(int_vals)):
            a, b = abs(int_vals[i]), abs(int_vals[j])
            if a and b and (a % b == 0 or b % a == 0):
                n_mult += 1
    feats["n_multiple_pairs"] = int(n_mult)
    return feats

# ---------- row builder ----------
def _build_row(dataset_tag: str, family_id: str, item: Dict[str,Any],
               spec: Dict[str,Any], assign: Dict[str,Any], y: float,
               template_id: int, deltas: Dict[str,Any],
               cat_maps: Dict[str, Dict[str,int]], include_meta: bool = True) -> Dict[str,Any]:
    row: Dict[str,Any] = {}
    if include_meta:
        row["dataset"] = dataset_tag
        row["family_id"] = family_id
        row["template_id"] = template_id
    # Text / perturbation
    txt = item.get("x") or item.get("source", {}).get("question", "")
    row["lang"] = _lang_hint(txt)
    row["len_chars"] = len(txt)
    row["delta_num"]   = float(deltas.get("delta_num", 0.0) or 0.0)
    row["delta_text"]  = float(deltas.get("delta_text",0.0) or 0.0)
    row["delta_total"] = float(deltas.get("delta_total",0.0) or 0.0)

    # Slots → numeric/categorical
    int_vals = []
    for k, v in (assign or {}).items():
        if isinstance(v, bool):
            row[f"slot_{k}"] = int(v)
        elif isinstance(v, int):
            row.update(_int_feats(f"slot_{k}", v))
            int_vals.append(int(v))
        elif isinstance(v, float):
            row.update(_float_feats(f"slot_{k}", v))
        else:
            mid = _categorical_id(cat_maps.setdefault(k, {}), str(v))
            row[f"slot_{k}"] = int(mid)

    # Derived math features
    row.update(_pairwise_math_feats(int_vals))

    # Supervised column (regression target)
    row["y"] = float(round(y, 2))
    return row

def _compose_generator_code(spec: Dict[str, Any]) -> str:
    """Detect generator implementations that expect the verifier in scope."""
    gen = (spec or {}).get("generator") or {}
    code = gen.get("code") or ""
    if not code:
        return ""
    if "verifier" in code:
        ver_code = ((spec or {}).get("verifier") or {}).get("code") or ""
        if ver_code:
            return f"{ver_code}\n\n{code}"
    return code

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Turn augmented math JSONL into tabular CSV/Parquet.")
    ap.add_argument("--aug_jsonl", required=True, help="Path to augmented_*.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--rows_per_family", type=int, default=1000, help="Target rows per seed family")
    ap.add_argument("--dataset_tag", type=str, default=None, help="Optional dataset label to stamp in the table")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Guess dataset tag if not provided
    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        base = os.path.basename(args.aug_jsonl).lower()
        if "gsm8k" in base: dataset_tag = "gsm8k"
        elif "aime25" in base: dataset_tag = "aime25"
        else: dataset_tag = "math"

    data = read_jsonl(args.aug_jsonl)
    all_rows: List[Dict[str,Any]] = []
    cat_maps_global: Dict[str, Dict[str,int]] = {}

    for rec in data:
        # Skip error-only rows
        spec = rec.get("spec")
        if not spec or "verifier" not in spec or "generator" not in spec:
            continue

        fid = str(rec.get("id", ""))
        variants = rec.get("variants") or []

        # Compile once per family
        try:
            ver = compile_verifier(spec["verifier"]["code"])
            gen_code = _compose_generator_code(spec)
            gen = compile_generator(gen_code) if (compile_generator and gen_code) else None
        except Exception:
            # Bad spec: skip this family
            continue

        used = 0

        # 1) First, use existing variants (closest to your augmentation distribution)
        for v in variants:
            try:
                row = _build_row(
                    dataset_tag, fid, v, spec,
                    v.get("assignment") or {},
                    v.get("y"), int(v.get("text_template_id", 0)),
                    {
                        "delta_num": v.get("delta_num", 0.0),
                        "delta_text": v.get("delta_text", 0.0),
                        "delta_total": v.get("delta_total", 0.0)
                    },
                    cat_maps_global
                )
                all_rows.append(row); used += 1
                if used >= args.rows_per_family: break
            except Exception:
                continue

        # 2) Then, top up by sampling from generator+verifier
        tries = 0
        tcount = max(1, len(spec.get("text_templates") or [None]))
        while used < args.rows_per_family and tries < args.rows_per_family * 20:
            tries += 1
            try:
                if gen is None:
                    break
                assign = gen(random.randrange(1 << 30))
                ok, y = run_verifier(ver, assign)
                if not ok:
                    continue
                tid = used % tcount
                row = _build_row(
                    dataset_tag, fid, {"x": ""}, spec, assign, y, tid,
                    {"delta_num": 0.0, "delta_text": 0.0, "delta_total": 0.0},
                    cat_maps_global
                )
                all_rows.append(row); used += 1
            except Exception:
                continue

    # Save
    if not all_rows:
        print("No rows produced. Check your augmented file or specs.")
        return

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.out_dir, f"tabular_{dataset_tag}.csv")
    out_parq = os.path.join(args.out_dir, f"tabular_{dataset_tag}.parquet")
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False)

    meta = {
        "dataset": dataset_tag,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "families": int(df["family_id"].nunique()) if "family_id" in df else None,
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]
    }
    with open(os.path.join(args.out_dir, f"tabular_{dataset_tag}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {dataset_tag}: rows={len(df)} cols={df.shape[1]}")
    print("CSV   →", out_csv)
    print("PARQ  →", out_parq)

if __name__ == "__main__":
    main()
