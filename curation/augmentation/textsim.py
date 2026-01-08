from __future__ import annotations
from typing import Callable
import re

TextSim = Callable[[str,str], float]

def jaccard_similarity(a: str, b: str) -> float:
    def toks(s: str):
        return set([w.lower() for w in re.findall(r"[a-zA-Z]+", s)])
    A = toks(a); B = toks(b)
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def text_delta(carrier_ref: str, candidate: str, sim: TextSim = jaccard_similarity) -> float:
    return 1.0 - sim(carrier_ref, candidate)
