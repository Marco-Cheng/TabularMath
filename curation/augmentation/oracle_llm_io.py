"""Environment-configured OpenAI-compatible teacher and student adapters."""
from __future__ import annotations

import json
import os
from typing import Dict, List
import requests


def _call(role: str, messages: List[Dict[str, str]]) -> str:
    def setting(name, default=""):
        return os.getenv(f"TABMATH_{role}_{name}") or os.getenv(f"TABMATH_{name}", default)
    key, model = setting("API_KEY"), setting("MODEL")
    if not key or not model:
        raise ValueError("Set TABMATH_API_KEY and TABMATH_MODEL, or the corresponding role overrides.")
    params = json.loads(setting("PARAMS", "{}"))
    if not isinstance(params, dict) or {"model", "messages", "stream"} & params.keys():
        raise ValueError("TABMATH_PARAMS must be a JSON object without model, messages or stream.")
    response = requests.post(
        setting("BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "stream": False, **params},
        timeout=float(setting("TIMEOUT", "180")),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{role} endpoint returned no text.")
    return content


def student_llm_call(messages: List[Dict[str, str]]) -> str:
    return _call("STUDENT", messages)


def teacher_llm_call(messages: List[Dict[str, str]]) -> str:
    return _call("TEACHER", messages)
