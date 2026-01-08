from __future__ import annotations
from typing import Protocol, Dict, Any
import importlib, json, re
from .prompt_templates import teacher_messages, student_messages, judge_messages
from .oracle_llm_io import teacher_llm_call, student_llm_call

class TeacherOracle(Protocol):
    def convert_to_spec(self, question: str, answer: str, example_id: str, n_text_templates: int = 3, feedback: str | None = None,) -> Dict[str, Any]:
        ...

class StudentOracle(Protocol):
    def infer(self, prompt: str, example_id: str) -> str:
        ...

class JudgeOracle(Protocol):
    def review_candidate(
        self,
        candidate_text: str,
        answer: Any,
        example_id: str,
        source_question: str,
        source_answer: Any,
    ) -> Dict[str, Any]:
        ...

def load_impl(path: str):
    if ":" not in path:
        mod, cls = path, None
    else:
        mod, cls = path.split(":", 1)
    m = importlib.import_module(mod)
    if cls is None:
        return m
    T = getattr(m, cls)
    return T()

def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}\s*$", cleaned)
    s = cleaned if m is None else m.group(0)
    return json.loads(s)

class PromptTeacher:
    def __init__(self): ...
    def convert_to_spec(self, question: str, answer: str, example_id: str, n_text_templates: int = 3, feedback: str | None = None) -> Dict[str, Any]:
        msgs = teacher_messages(question, answer, example_id, n_text_templates=n_text_templates, feedback=feedback)
        raw = teacher_llm_call(msgs)
        data = _extract_json(raw)
        needed = ["slots","verifier","generator","text_templates","base_assignment"]
        for k in needed:
            if k not in data:
                raise ValueError(f"Teacher JSON missing key: {k}")
        if data.get("verifier",{}).get("type") != "python":
            raise ValueError("verifier.type must be 'python'")
        if data.get("generator",{}).get("type") != "python":
            raise ValueError("generator.type must be 'python'")
        data.setdefault("template", None)
        data.setdefault("carrier_text", None)
        data.setdefault("meta", {})
        data["meta"].setdefault("example_id", example_id)
        return data

class PromptStudent:
    def __init__(self): ...
    def infer(self, prompt: str, example_id: str) -> str:
        msgs = student_messages(prompt, example_id)
        return student_llm_call(msgs)


class PromptJudge:
    def __init__(self): ...

    def review_candidate(
        self,
        candidate_text: str,
        answer: Any,
        example_id: str,
        source_question: str,
        source_answer: Any,
    ) -> Dict[str, Any]:
        msgs = judge_messages(
            candidate_text=candidate_text,
            expected_answer=answer,
            example_id=example_id,
            source_question=source_question,
            source_answer=source_answer,
        )
        raw = teacher_llm_call(msgs)
        data = _extract_json(raw)
        decision = data.get("decision")
        category = data.get("category")
        justification = data.get("justification")
        if decision not in {"accept", "reject"}:
            raise ValueError("Judge JSON missing/invalid decision (expected 'accept' or 'reject')")
        if not isinstance(category, str) or not category:
            raise ValueError("Judge JSON missing category")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError("Judge JSON missing justification")
        return {
            "decision": decision,
            "category": category,
            "justification": justification.strip(),
            "raw": data,
        }
