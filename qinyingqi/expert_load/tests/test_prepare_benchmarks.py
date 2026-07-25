from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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

    def test_livecodebench_dates_are_json_serializable(self) -> None:
        record = PREPARE.livecodebench_record(
            {
                "question_id": "problem-1",
                "question_content": "Return the sum of two integers.",
                "contest_date": datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc),
                "difficulty": "easy",
            },
            0,
        )
        self.assertEqual(record["metadata"]["contest_date"], "2025-01-02T03:04:00+00:00")
        PREPARE.json.dumps(record)

        record = PREPARE.livecodebench_record(
            {
                "question_id": "problem-2",
                "question_content": "Return the product of two integers.",
                "contest_date": date(2025, 1, 3),
            },
            1,
        )
        self.assertEqual(record["metadata"]["contest_date"], "2025-01-03")

    def test_jsonl_write_does_not_leave_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.jsonl"
            with self.assertRaises(TypeError):
                PREPARE.write_jsonl(
                    output,
                    [{"ok": True}, {"not_json": object()}],
                )
            self.assertFalse(output.exists())
            self.assertFalse((Path(directory) / ".input.jsonl.tmp").exists())

    def test_parse_benchmarks_rejects_invalid_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            PREPARE.parse_benchmarks("mmlu_pro,unknown")

    def test_swebench_uses_only_parquet_files(self) -> None:
        data_format, files = PREPARE.select_dataset_files(
            "swebench_lite",
            "test",
            [
                "README.md",
                "legacy_loader.py",
                "data/dev-00000-of-00001.parquet",
                "data/test-00000-of-00002.parquet",
                "data/test-00001-of-00002.parquet",
            ],
        )
        self.assertEqual(data_format, "parquet")
        self.assertEqual(
            files,
            [
                "data/test-00000-of-00002.parquet",
                "data/test-00001-of-00002.parquet",
            ],
        )

    def test_livecodebench_uses_release_latest_jsonl_order(self) -> None:
        data_format, files = PREPARE.select_dataset_files(
            "livecodebench",
            "test",
            ["test6.jsonl", "test.jsonl", "code_generation_lite.py", "test2.jsonl"],
        )
        self.assertEqual(data_format, "json")
        self.assertEqual(files, ["test.jsonl", "test2.jsonl", "test6.jsonl"])

    def test_remote_loader_bypasses_dataset_repository_script(self) -> None:
        calls: dict[str, object] = {}

        class FakeApi:
            def dataset_info(self, **kwargs):
                calls["dataset_info"] = kwargs
                return SimpleNamespace(sha="a" * 40)

            def list_repo_files(self, **kwargs):
                calls["list_repo_files"] = kwargs
                return ["dataset_script.py", "data/test-00000-of-00001.parquet"]

        def fake_hf_hub_url(**kwargs) -> str:
            calls["hub_url"] = kwargs
            return "https://example.invalid/test.parquet"

        def fake_load_dataset(builder: str, **kwargs):
            calls["load_dataset"] = (builder, kwargs)
            return [{"instance_id": "one"}]

        with mock.patch.object(
            PREPARE,
            "import_dataset_dependencies",
            return_value=(fake_load_dataset, FakeApi, fake_hf_hub_url),
        ):
            rows, source = PREPARE.load_remote_rows(
                "swebench_lite", Path("/cache"), "main", None, limit=50
            )

        self.assertEqual(rows, [{"instance_id": "one"}])
        self.assertEqual(calls["load_dataset"][0], "parquet")
        self.assertEqual(
            calls["load_dataset"][1]["data_files"],
            {"test": ["https://example.invalid/test.parquet"]},
        )
        self.assertEqual(source["data_files"], ["data/test-00000-of-00001.parquet"])
        self.assertFalse(calls["load_dataset"][1]["streaming"])

    def test_limited_livecodebench_uses_streaming_loader(self) -> None:
        calls: dict[str, object] = {}

        class FakeApi:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha="b" * 40)

            def list_repo_files(self, **kwargs):
                return ["test.jsonl", "test2.jsonl"]

        def fake_hf_hub_url(**kwargs) -> str:
            return f"https://example.invalid/{kwargs['filename']}"

        def fake_load_dataset(builder: str, **kwargs):
            calls["load_dataset"] = (builder, kwargs)
            return iter([{"question_id": "one"}])

        with mock.patch.object(
            PREPARE,
            "import_dataset_dependencies",
            return_value=(fake_load_dataset, FakeApi, fake_hf_hub_url),
        ):
            rows, source = PREPARE.load_remote_rows(
                "livecodebench", Path("/cache"), "main", None, limit=50
            )

        self.assertEqual(list(rows), [{"question_id": "one"}])
        self.assertEqual(calls["load_dataset"][0], "json")
        self.assertTrue(calls["load_dataset"][1]["streaming"])
        self.assertTrue(source["streaming"])

    def test_livecodebench_can_reuse_non_streaming_cache(self) -> None:
        calls: dict[str, object] = {}

        class FakeApi:
            def dataset_info(self, **kwargs):
                return SimpleNamespace(sha="b" * 40)

            def list_repo_files(self, **kwargs):
                return ["test.jsonl", "test2.jsonl"]

        def fake_hf_hub_url(**kwargs) -> str:
            return f"https://example.invalid/{kwargs['filename']}"

        def fake_load_dataset(builder: str, **kwargs):
            calls["load_dataset"] = (builder, kwargs)
            return [{"question_id": "one"}]

        with mock.patch.object(
            PREPARE,
            "import_dataset_dependencies",
            return_value=(fake_load_dataset, FakeApi, fake_hf_hub_url),
        ):
            rows, source = PREPARE.load_remote_rows(
                "livecodebench",
                Path("/cache"),
                "main",
                None,
                limit=50,
                livecodebench_streaming=False,
            )

        self.assertEqual(rows, [{"question_id": "one"}])
        self.assertFalse(calls["load_dataset"][1]["streaming"])
        self.assertFalse(source["streaming"])


if __name__ == "__main__":
    unittest.main()
