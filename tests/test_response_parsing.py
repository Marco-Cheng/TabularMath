import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ENTRY = Path(__file__).resolve().parents[1] / "evaluation/scripts/eval_icl_llm.py"
spec = importlib.util.spec_from_file_location("eval_icl_llm", ENTRY)
evaluation = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {}, clear=True):
    spec.loader.exec_module(evaluation)


class ResponseParsingTests(unittest.TestCase):
    def test_explicit_unknown_uses_documented_zero_fallback(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(evaluation, "call_llm", return_value="<unknown>") as call:
            predictions, raw = evaluation._invoke_with_retry("fixture", "fixture", 3)
        self.assertEqual(predictions, [0.0, 0.0, 0.0])
        self.assertEqual(raw, "<unknown>")
        call.assert_called_once()

    def test_malformed_response_retries_before_success(self):
        with patch.object(evaluation, "call_llm", side_effect=["No numeric prediction", "[3, 4]"]) as call:
            predictions, _ = evaluation._invoke_with_retry("fixture", "fixture", 2)
        self.assertEqual(predictions, [3.0, 4.0])
        self.assertEqual(call.call_count, 2)

    def test_exhausted_retries_are_not_scored_as_zero(self):
        with patch.object(evaluation, "call_llm", return_value="No numeric prediction"):
            with self.assertRaises(RuntimeError):
                evaluation._invoke_with_retry("fixture", "fixture", 2, max_attempts=2)


if __name__ == "__main__":
    unittest.main()
