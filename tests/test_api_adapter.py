import os
import unittest
from unittest.mock import Mock, patch

from curation.augmentation import oracle_llm_io


class AdapterTests(unittest.TestCase):
    def test_unconfigured_adapter_makes_no_call(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(oracle_llm_io.requests, "post") as post:
            with self.assertRaises(ValueError):
                oracle_llm_io.student_llm_call([])
            post.assert_not_called()

    def test_configured_adapter(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "[1, 2]"}}]}
        env = {"TABMATH_API_KEY": "fixture-key", "TABMATH_MODEL": "fixture",
               "TABMATH_STUDENT_MODEL": "student", "TABMATH_BASE_URL": "https://example.invalid/v1"}
        with patch.dict(os.environ, env, clear=True), patch.object(oracle_llm_io.requests, "post", return_value=response) as post:
            self.assertEqual(oracle_llm_io.student_llm_call([]), "[1, 2]")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "student")
            self.assertEqual(post.call_args.args[0], "https://example.invalid/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
