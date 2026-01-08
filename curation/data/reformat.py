#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from decimal import Decimal, ROUND_HALF_UP

def _to_2dec_str(val):
    """
    Round numeric to 2 decimals; return a compact string (no trailing zeros/dot).
    Accepts int/float/str; if not numeric, return original as str.
    """
    try:
        dec = Decimal(str(val))
        q = dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        s = format(q.normalize(), "f")  # remove scientific; keep as plain
        # Limit to at most 2 decimals (normalize might produce many zeros)
        if "." in s:
            whole, frac = s.split(".", 1)
            s = whole + ("." + frac[:2] if frac else "")
            s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(val)

def reformat(in_path: str, out_path: str):
    n_in = n_out = n_bad = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            # skip rows with explicit error
            if obj.get("error"):
                n_bad += 1
                continue

            variants = obj.get("variants") or []
            for v in variants:
                x = v.get("x")
                y = v.get("y")
                if not x or y is None:
                    continue
                # numeric → 2 decimals; format GSM8K-style answer "#### <num>"
                ans: str
                if isinstance(y, (int, float)) or (isinstance(y, str) and y.strip().replace(".","",1).replace("-","",1).isdigit()):
                    y2 = _to_2dec_str(y)
                    ans = f"#### {y2}"
                else:
                    # non-numeric answer → keep as string (still prefixed to match extractor)
                    ans = f"#### {str(y)}"
                out_row = {"question": x, "answer": ans}
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Read augmented rows: {n_in}  Wrote generalized rows: {n_out}  Skipped: {n_bad}")

def main():
    ap = argparse.ArgumentParser(description="Convert augmented.jsonl to GSM8K-like flat JSONL (generalized_gsm8k.jsonl).")
    ap.add_argument("--augmented_jsonl", default="augmented.jsonl")
    ap.add_argument("--out_jsonl", default="generalized_gsm8k.jsonl")
    args = ap.parse_args()
    reformat(args.augmented_jsonl, args.out_jsonl)

if __name__ == "__main__":
    main()
