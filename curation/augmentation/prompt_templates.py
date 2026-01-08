from __future__ import annotations
from typing import List, Dict, Optional, Any

JUDGE_DECISION_CATEGORIES = [
    "sound",           # statement is coherent and solvable
    "contradiction",   # contains logically conflicting facts or math
    "ambiguous",       # unclear wording or missing disambiguation
    "missing_info",    # lacks data needed to solve
    "math_error",      # incorrect operations, impossible arithmetic
    "format_issue",    # answer format/units mismatch the prompt
    "unsolvable",      # setup cannot be solved even with clarification
    "other"            # anything else (describe explicitly)
]

def teacher_messages(question: str, answer: str, example_id: str, n_text_templates: int = 6, feedback: Optional[str] = None) -> List[Dict[str,str]]:
    sys = (
        "You are an expert at parameterizing math word problems in GSM8K. "
        "You need to generalize a problem by parameterizing as many parts of it as possible without changing the core calculation logic."
        "Return ONE JSON OBJECT only. No code fences, no prose."
    )
    schema_example='''
OUTPUT SCHEMA (STRICT JSON, an example below)
---------------------------
{
  "text_templates": [
    "If [name1] has [a] apples and [name2] has [b], how many more does [name1] have?",
    "[name1] possesses [a] apples while [name2] has [b]; compute the difference.",
    "有[name1][a]个苹果，[name2]有[b]个。[name1]比[name2]多几个？",
    "Si [name1] a [a] pommes et [name2] en a [b], quelle est la différence ?",
    "[name1] has [a] marbles; [name2] has [b]. Find [a] - [b].",
    "Suppose [name1] bought [a] stickers and [name2] bought [b]. How many more did [name1] buy?"
  ],
  "slots": {
    "a": {"kind":"int","interval":[0,1], "map": {"kind":"int_range","lo":5,"hi":50,"step":1}, "base_value": 12},
    "b": {"kind":"int","interval":[0,1], "map": {"kind":"int_range","lo":1,"hi":50,"step":1}, "base_value": 3},
    "name1": {"kind":"entity","meta":{"names":["Alice","Xiao Ming","Jean","Lucia"]}, "base_value":"Alice"},
    "name2": {"kind":"entity","meta":{"names":["Bob","Xiao Hong","Marie","Diego"]}, "base_value":"Bob"}
  },
  "verifier": {"type":"python","code":"def verifier(assign):\n    a = int(assign['a']); b = int(assign['b'])\n    if a<=b or a<0 or b<0: return False, None\n    return True, a - b"},
  "generator": {"type":"python","code":"def generator(rng):\n    a = rng.randint(5,50)\n    b = rng.randint(1,a-1)\n    name1 = 'Alice'\n    name2 = 'Bob'\n    return {'a':a,'b':b,'name1':name1,'name2':name2}"},
  "base_assignment": {"a":12, "b":3, "name1":"Alice", "name2":"Bob"},
  "meta": {"source":"gsm8k","example_id":"<EXAMPLE_ID>"}
}
'''.rstrip()
    
    parts = [
        "SEED",
        "----",
        f"question_id: {example_id}",
        f"question: {question}",
        f"answer: {answer}",
        "",
        "GOAL",
        "----",
        "Output a SINGLE JSON object implementing a minimal, coherent augmentation spec of the given seed question above:",
        f"- text_templates: list[str] using [slot_name] placeholders (REQUIRED). Number of templates = {n_text_templates}.",
        "  * Be CREATIVE and DIFFERENT while keeping the same core math logic consistent and validated by the verifier/generator.",
        "  * Allowed changes (not necessarily all, no particular order, must keep slot count and names consistent across ALL templates):",
        "      1) Paraphrase.",
        "      2) Translate to other languages (Chinese, French, Spanish, etc.).",
        "      3) Change/add/remove background story or entities (names, places), as long as slot semantics are preserved.",
        "  * Do NOT leak the final answer y inside templates.",
        "- slots: either a dict {{\"slot_name\":{{...}} }} or a list of objects with fields:",
        "    name, kind in [\"int\",\"float\",\"choice\",\"str\",\"entity\",\"unit\"], optional interval/map, weight, base_value, meta.",
        "- verifier:  { \"type\":\"python\", \"code\": \"def verifier(assign): ... return True, y\" } - A verifier should validate whether the input assign is valid, and return the desired answer for a valid assignment  (if assign is invalid, return False, None).",
        "- generator: { \"type\":\"python\", \"code\": \"def generator(rng): ... return assign\" } - The generator should randomly generate an assign whose format is coherent with the verifier and can pass the verifier.",
        "- base_assignment: assignment corresponding to the original seed question; MUST pass the verifier.",
        "",
        "RULES",
        "-----",
        "- Use ONLY [slot_name] tokens in text_templates (no {{braces}}). Slot names must be ASCII snake_case.",
        "- The set of slot names in text_templates, slots, base_assignment, verifier inputs, and generator outputs MUST MATCH exactly.",
        "- Code restrictions: pure Python 3; no I/O; no imports; we provide 'math' and 'random' at runtime, but be sure to refer to them (e.g. math.gcd, math.lcm).",
        "- y should be numeric when the problem is numeric (usual GSM8K).",
        "- Your output should always guarantee that the output of generator MUST PASS the verifier and contributes to a valid new augmented task.",
        "",
        schema_example,
    ]

    if feedback:
        parts += [
            "",
            "RETRY_FEEDBACK",
            "--------------",
            "The previous attempt failed. Diagnose and fix the issues below.",
            "Ensure generator yields VALID assignments passing the verifier, and that verifier compiles and returns (bool, y).",
            "",
            "Issues to fix:",
            feedback,
        ]

    parts += [
        "",
        "OUTPUT",
        "------",
        "Return ONLY the JSON object.",
    ]

    user = "\n".join(parts)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def student_messages(prompt: str, example_id: str) -> List[Dict[str,str]]:
    sys = (
        "You are solving a math problem. "
        "Respond with ONLY the final numeric answer (retain at most 2 decimal points) in the format: '#### <number>'."
    )
    user = f"""question_id: {example_id}

Problem:
{prompt}

Output format (STRICT):
#### <number>
"""
    return [
        {"role":"system","content": sys},
        {"role":"user","content": user}
    ]


def judge_messages(
    candidate_text: str,
    expected_answer: Any,
    example_id: str,
    source_question: str,
    source_answer: Any,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    cats = categories or JUDGE_DECISION_CATEGORIES
    cats_str = ", ".join(cats)
    sys = (
        "You are a meticulous math competition judge. "
        "You must decide whether a generated math word problem statement is sound, "
        "non-contradictory, and solvable given its provided numeric answer. "
        "Respond in STRICT JSON."
    )

    schema = f"""
Return ONLY one JSON object with fields:
{{
  "decision": "accept" | "reject",
  "category": "<one of: {cats_str}>",
  "justification": "Short natural language explanation."
}}

Rules:
- A problem is ACCEPTABLE only if its text is self-consistent, unambiguous, and solvable with the given answer.
- Reject if there are contradictions, missing quantities, impossible operations, unit mismatches, or any ambiguity that blocks a confident solve.
- Use category "sound" only when you accept.
- When rejecting, pick the category that best matches the most critical flaw and describe it clearly.
""".strip()

    user_parts = [
        f"candidate_id: {example_id}",
        "SOURCE SEED",
        "------------",
        f"question: {source_question}",
        f"answer: {source_answer}",
        "",
        "CANDIDATE",
        "---------",
        candidate_text,
        f"Expected numeric answer: {expected_answer}",
        "",
        "EVALUATION",
        "----------",
        "Decide ACCEPT vs REJECT following the schema above.",
        schema,
    ]

    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
