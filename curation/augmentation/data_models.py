from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

MappingSpec = Dict[str, Any]

@dataclass
class SlotSpec:
    name: str
    kind: str
    interval: Tuple[float, float] = (0.0, 1.0)
    map: Optional[MappingSpec] = None
    weight: float = 1.0
    base_value: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerifierSpec:
    type: str   # 'python'
    code: str
    timeout_ms: Optional[int] = 1000

@dataclass
class GeneratorSpec:
    type: str   # 'python'
    code: str

@dataclass
class TemplateSpec:
    template: Optional[str]
    slots: List[SlotSpec]
    verifier: VerifierSpec
    generator: GeneratorSpec
    text_templates: List[str]
    carrier_text: Optional[str] = None
    base_assignment: Dict[str, Any] = field(default_factory=dict)
    text_deltas: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VariantRecord:
    seed: int
    assignment: Dict[str, Any]
    text_template_id: int
    x: str
    y: Any
    delta_num: float
    delta_text: float
    delta_total: float
    per_slot_delta: Dict[str, float]
    judge_status: str = "unknown"
    judge_category: Optional[str] = None
    judge_justification: Optional[str] = None

@dataclass
class RejectedRecord:
    id: str
    stage: str
    reason_code: str
    reason: str
    candidate: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "candidate": self.candidate,
            "meta": self.meta,
        }

@dataclass
class AugmentedRecord:
    id: str
    source: Dict[str, Any]
    spec: TemplateSpec
    variants: List[VariantRecord] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        def slot_to_dict(s: SlotSpec) -> Dict[str, Any]:
            d = {
                "name": s.name, "kind": s.kind,
                "interval": list(s.interval) if s.interval else [0,1],
                "map": s.map, "weight": s.weight, "base_value": s.base_value, "meta": s.meta
            }
            return d
        def spec_to_dict(sp: TemplateSpec) -> Dict[str, Any]:
            return {
                "template": sp.template,
                "slots": [slot_to_dict(s) for s in sp.slots],
                "verifier": {"type": sp.verifier.type, "code": sp.verifier.code, "timeout_ms": sp.verifier.timeout_ms},
                "generator": {"type": sp.generator.type, "code": sp.generator.code},
                "text_templates": sp.text_templates,
                "carrier_text": sp.carrier_text,
                "base_assignment": sp.base_assignment,
                "text_deltas": sp.text_deltas,
                "meta": sp.meta
            }
        def variant_to_dict(v: VariantRecord) -> Dict[str, Any]:
            return {
                "seed": v.seed, "assignment": v.assignment, "text_template_id": v.text_template_id,
                "x": v.x, "y": v.y,
                "delta_num": v.delta_num, "delta_text": v.delta_text, "delta_total": v.delta_total,
                "per_slot_delta": v.per_slot_delta,
                "judge_status": v.judge_status,
                "judge_category": v.judge_category,
                "judge_justification": v.judge_justification,
            }
        return {"id": self.id, "source": self.source, "spec": spec_to_dict(self.spec),
                "variants": [variant_to_dict(v) for v in self.variants]}
