from __future__ import annotations
from typing import List, Dict
import json
import csv

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_gsm8k_jsonl(path: str) -> List[Dict]:
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("query") or r.get("input") or ""
        a = r.get("answer") or r.get("output") or r.get("label") or ""
        rid = r.get("id") or f"gsm8k-{i:06d}"
        out.append({"id": rid, "question": q, "answer": a, "raw": r})
    return out


def read_math_csv(path: str, answer_prefix: str = "#### ") -> List[Dict]:
    """Load simple CSV with id, question, answer columns."""
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rid = row.get("id") or f"item-{idx:05d}"
            question = row.get("question") or ""
            ans_raw = (row.get("answer") or "").strip()
            if ans_raw.startswith("####"):
                answer = ans_raw
            else:
                answer = f"{answer_prefix}{ans_raw}".rstrip()
            rows.append({
                "id": str(rid),
                "question": question,
                "answer": answer,
                "raw": row,
            })
    return rows
