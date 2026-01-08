from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re

from .data_models import SlotSpec, TemplateSpec, VerifierSpec, GeneratorSpec, AugmentedRecord, VariantRecord
from .mappings import is_numeric_slot, numeric_delta
from .verifier import compile_verifier, compile_generator, run_verifier, RNGShim
from .textsim import text_delta, jaccard_similarity

_TOKEN_BRACE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_TOKEN_BRACKET = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*)\]")

def _normalize_slots(slots_any) -> List[SlotSpec]:
    slots_list = []
    if isinstance(slots_any, dict):
        items = [(k, v) for k, v in slots_any.items()]
    else:
        items = [(s.get("name"), s) for s in slots_any]
    for name, s in items:
        if name is None: name = s.get("name")
        slots_list.append(SlotSpec(
            name=name,
            kind=s.get("kind", "int"),
            interval=tuple(s.get("interval", (0.0,1.0))),
            map=s.get("map"),
            weight=float(s.get("weight", 1.0)),
            base_value=s.get("base_value"),
            meta=s.get("meta", {})
        ))
    return slots_list

def _derive_carrier_text(example_text: str, slots: List[SlotSpec]) -> str:
    def repl(name: str) -> str:
        slot = next((s for s in slots if s.name == name), None)
        if slot is None: return "[VAR]"
        if slot.kind in ("int","float"): return "[#]"
        if slot.kind in ("entity","str","choice"): return "[NAME]"
        return "[VAR]"
    text = example_text
    for m in _TOKEN_BRACE.findall(text):
        text = text.replace("{"+m+"}", repl(m))
    for m in _TOKEN_BRACKET.findall(text):
        text = text.replace("["+m+"]", repl(m))
    return text

def _compose(template: str, assign: Dict[str, Any]) -> str:
    def br_sub(m): return str(assign.get(m.group(1), ""))
    def bk_sub(m): return str(assign.get(m.group(1), ""))
    text = _TOKEN_BRACE.sub(br_sub, template)
    text = _TOKEN_BRACKET.sub(bk_sub, text)
    return text

class TeacherConverter:
    def __init__(self, teacher_oracle, text_sim = jaccard_similarity):
        self.teacher = teacher_oracle
        self.text_sim = text_sim

    def convert_item(self, item: Dict[str, Any], n_text_templates: int = 3, feedback: str | None = None) -> TemplateSpec:
        q = item["question"]; a = item["answer"]; rid = item["id"]
        spec_d = self.teacher.convert_to_spec(q, a, rid, n_text_templates=n_text_templates, feedback=feedback)

        slots = _normalize_slots(spec_d["slots"])
        verifier = VerifierSpec(type=spec_d["verifier"]["type"], code=spec_d["verifier"]["code"], timeout_ms=spec_d["verifier"].get("timeout_ms", 1000))
        generator = GeneratorSpec(type=spec_d["generator"]["type"], code=spec_d["generator"]["code"]) if spec_d.get("generator") else None
        if generator is None:
            raise ValueError("Teacher must provide a generator per the simplified spec.")

        template = spec_d.get("template")
        text_templates = spec_d.get("text_templates", [])
        if not text_templates:
            raise ValueError("Teacher must provide text_templates.")

        ts = TemplateSpec(
            template=template,
            slots=slots,
            verifier=verifier,
            generator=generator,
            text_templates=text_templates,
            base_assignment=spec_d.get("base_assignment", {}),
            meta=spec_d.get("meta", {})
        )

        example_text = text_templates[0]
        ts.carrier_text = spec_d.get("carrier_text") or _derive_carrier_text(example_text, slots)
        ts.text_deltas = [ text_delta(ts.carrier_text, t, self.text_sim) for t in ts.text_templates ]
        return ts

class VariantGenerator:
    def __init__(self, lambda_text: float = 0.5):
        self.lambda_text = lambda_text
        self._compiled = {}

    def _compile(self, ts: TemplateSpec):
        key = id(ts)
        if key in self._compiled: return self._compiled[key]
        ver = compile_verifier(ts.verifier.code)
        gen = compile_generator(ts.generator.code)
        self._compiled[key] = (ver, gen)
        return ver, gen

    def sample_assignment(self, ts: TemplateSpec, seed: int, max_tries: int = 100) -> Dict[str, Any]:
        ver, gen = self._compile(ts)
        rng = RNGShim(seed)
        last_assign = None
        for _ in range(max_tries):
            try:
                assign = gen(rng)
                last_assign = assign
                ok, _ = ver(assign)
                if ok: return assign
            except Exception:
                continue
        if ts.base_assignment:
            ok, _ = ver(ts.base_assignment)
            if ok: return ts.base_assignment
        return last_assign or {}

    def compose(self, template: str, assign: Dict[str, Any]) -> str:
        return _compose(template, assign)

    def compute_num_delta(self, slots: List[SlotSpec], assign: Dict[str, Any]) -> Tuple[float, Dict[str,float]]:
        per = {}; num_terms = []
        for s in slots:
            if s.base_value is None: continue
            v_new = assign.get(s.name)
            if v_new is None: continue
            if is_numeric_slot(s.kind, (s.map or {}).get("kind","")):
                d = numeric_delta(s.name, v_new, s.base_value, s.map)
                per[s.name] = d * s.weight
            else:
                per[s.name] = 0.0 if v_new == s.base_value else 1.0 * s.weight
            num_terms.append(per[s.name])
        total = (sum(num_terms) / max(1, len(num_terms))) if num_terms else 0.0
        return total, per

def compute_deltas_for_variant(ts: TemplateSpec, vg: VariantGenerator, assign: Dict[str, Any], text_template_id: int) -> Tuple[float, float, float, Dict[str,float]]:
    dnum, per = vg.compute_num_delta(ts.slots, assign)
    dtext = 0.0
    if 0 <= text_template_id < len(ts.text_deltas):
        dtext = ts.text_deltas[text_template_id]
    dtotal = vg.lambda_text * dtext + (1.0 - vg.lambda_text) * dnum
    return dnum, dtext, dtotal, per
