from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "20_prepare_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("prepare_benchmarks", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareBenchmarkTests(unittest.TestCase):
    def test_mmlu_record_has_stable_multiple_choice_prompt(self) -> None:
        record = PREPARE.mmlu_pro_record(
            {
                "question": "Which value equals two plus two?",
                "options": ["three", "four", "five"],
                "category": "math",
                "answer_index": 1,
            },
            0,
        )
        self.assertEqual(record["benchmark"], "mmlu_pro")
        self.assertIn("A. three", record["messages"][0]["content"])
        self.assertIn("B. four", record["messages"][0]["content"])
        self.assertEqual(record["metadata"]["option_count"], 3)

    def test_swebench_record_preserves_issue_context(self) -> None:
        record = PREPARE.swebench_lite_record(
            {
                "instance_id": "django__django-123",
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix the regression in the parser.",
            },
            2,
        )
        self.assertEqual(record["metadata"]["source_id"], "django__django-123")
        self.assertIn("Repository: django/django", record["messages"][1]["content"])

    def test_ruler_input_is_deterministic(self) -> None:
        first = PREPARE.ruler_niah_record(3, filler_words=64, seed=1024)
        second = PREPARE.ruler_niah_record(3, filler_words=64, seed=1024)
        self.assertEqual(first, second)
        self.assertIn(first["metadata"]["needle"], first["messages"][0]["content"])

    def test_parse_benchmarks_rejects_invalid_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            PREPARE.parse_benchmarks("mmlu_pro,unknown")


if __name__ == "__main__":
    unittest.main()
