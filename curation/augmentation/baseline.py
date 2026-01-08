from __future__ import annotations
from typing import List, Dict, Any, Optional
import random
from tqdm import tqdm

from .dataset_io import read_gsm8k_jsonl
from .utils import extract_numeric_answer, set_seed
from .oracles import StudentOracle

def evaluate_student_on_gsm8k_raw(
    path: str,
    student: StudentOracle,
    max_items: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a student model directly on the ORIGINAL GSM8K JSONL (question/answer fields).
    Gold answers are parsed with the same numeric extractor used for predictions.
    Returns a dict with accuracy, counts, and per-item details.
    """
    set_seed(seed)
    items = read_gsm8k_jsonl(path)
    if max_items is not None:
        random.shuffle(items)
        items = items[:max_items]

    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []

    for it in tqdm(items):
        rid = it.get("id", "")
        q = it.get("question", "")
        a = it.get("answer", "")

        pred_text = student.infer(q, rid)
        pred_num = extract_numeric_answer(pred_text)
        gold_num = extract_numeric_answer(a)

        ok = False
        if gold_num is not None and pred_num is not None:
            ok = abs(float(gold_num) - float(pred_num)) <= max(
                1e-6, 1e-6 * max(1.0, abs(float(gold_num)))
            )
        else:
            ok = str(pred_text).strip() == str(a).strip()

        total += 1
        correct += int(ok)
        details.append(
            {
                "id": rid,
                "ok": ok,
                "gold": gold_num if gold_num is not None else a,
                "pred_text": pred_text,
                "pred_num": pred_num,
            }
        )

    acc = correct / max(1, total)
    return {
        "accuracy": acc,
        "total": total,
        "correct": correct,
        "items": details,
    }
