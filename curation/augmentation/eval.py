from __future__ import annotations
from typing import List, Dict, Any, Optional
from .dataset_io import load_jsonl
from .utils import extract_numeric_answer, set_seed
from .oracles import StudentOracle
import random
from tqdm import tqdm

def _iter_variants(records: List[Dict[str, Any]]):
    for r in records:
        for v in r.get("variants") or []:
            yield r, v

def evaluate_student_on_augmented(path: str, student: StudentOracle, max_perturb: float, max_items: Optional[int] = None, seed: int = 42) -> Dict[str, Any]:
    set_seed(seed)
    rows = load_jsonl(path)
    pool = [(rec, var) for rec, var in _iter_variants(rows) if var.get("delta_total", 0.0) <= max_perturb]
    if max_items is not None:
        random.shuffle(pool); pool = pool[:max_items]

    total = 0; correct = 0; details = []
    for rec, var in tqdm(pool):
        q = var["x"]; gold = var["y"]; rid = rec.get("id","")
        total += 1
        pred_text = student.infer(q, rid)
        pred_num = extract_numeric_answer(pred_text)
        ok = False
        if isinstance(gold, (int,float)) and pred_num is not None:
            ok = abs(float(gold) - float(pred_num)) <= max(1e-6, 1e-6 * max(1.0, abs(float(gold))))
        else:
            ok = (str(pred_text).strip() == str(gold).strip())
        correct += int(ok)
        details.append({
            "id": rid, "ok": ok, "gold": gold, "pred_text": pred_text, "pred_num": pred_num,
            "delta_total": var.get("delta_total"), "delta_num": var.get("delta_num"), "delta_text": var.get("delta_text")
        })
    acc = correct / max(1, total)
    return {"accuracy": acc, "total": total, "correct": correct, "max_perturb": max_perturb, "items": details}
