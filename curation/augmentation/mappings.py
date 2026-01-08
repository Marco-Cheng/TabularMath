from __future__ import annotations
from typing import Any, Dict, List
import math

def map_value(map_spec: Dict[str, Any], u: float) -> Any:
    kind = map_spec.get("kind")
    if kind == "int_range":
        lo = int(map_spec["lo"]); hi = int(map_spec["hi"]); step = int(map_spec.get("step", 1))
        width = max(1, (hi - lo) // step + 1)
        idx = min(width - 1, int(math.floor(u * width)))
        return lo + idx * step
    elif kind == "float_range":
        lo = float(map_spec["lo"]); hi = float(map_spec["hi"])
        val = lo + (hi - lo) * max(0.0, min(1.0, u))
        prec = map_spec.get("precision")
        if prec is not None:
            val = round(val, int(prec))
        return val
    elif kind == "choice":
        opts: List[Any] = map_spec["options"]
        if not opts:
            raise ValueError("choice options empty")
        idx = min(len(opts) - 1, int(math.floor(u * len(opts))))
        return opts[idx]
    elif kind == "custom_py":
        code = map_spec.get("code", "")
        loc: Dict[str, Any] = {}
        glob = {"math": math}
        exec(code, glob, loc)
        fn = loc.get("map_fn")
        if fn is None:
            raise ValueError("custom_py must define map_fn(u)")
        return fn(u)
    else:
        raise ValueError(f"Unknown map kind: {kind}")

def is_numeric_slot(slot_kind: str, map_kind: str) -> bool:
    if slot_kind in ("int","float"):
        return True
    if map_kind in ("int_range","float_range"):
        return True
    return False

def numeric_delta(slot_name: str, new_val: float, base_val: float, map_spec: Dict[str, Any] | None) -> float:
    if map_spec is None:
        denom = max(1.0, abs(float(base_val))) if base_val is not None else 1.0
        return abs(float(new_val) - float(base_val)) / denom
    mk = map_spec.get("kind", "")
    if mk in ("int_range","float_range"):
        lo = float(map_spec["lo"]); hi = float(map_spec["hi"])
        rng = max(1e-9, hi - lo)
        return abs(float(new_val) - float(base_val)) / rng
    denom = max(1.0, abs(float(base_val))) if base_val is not None else 1.0
    return abs(float(new_val) - float(base_val)) / denom
