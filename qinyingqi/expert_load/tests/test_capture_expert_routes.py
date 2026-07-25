from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT_PATH = SCRIPTS_DIR / "21_capture_expert_routes.py"
SPEC = importlib.util.spec_from_file_location("capture_expert_routes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def encoded_routes() -> str:
    prompt_tokens = 3
    output_tokens = 4
    routes = np.zeros((prompt_tokens + output_tokens - 1, 78, 8), dtype=np.uint8)
    for row in range(routes.shape[0]):
        for layer in range(3, 78):
            base = (row * 13 + layer * 7) % (256 - 8)
            routes[row, layer] = base + np.arange(8, dtype=np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, routes, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def response() -> dict:
    return {
        "id": "chatcmpl-capture-test",
        "model": "glm-52",
        "prompt_token_ids": [11, 12, 13],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "token_ids": [21, 22, 23, 24],
                "routed_experts": encoded_routes(),
            }
        ],
    }


class CaptureExpertRoutesTests(unittest.TestCase):
    def test_capture_writes_routes_and_aggregate_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "request_id": "mmlu-1",
                        "benchmark": "mmlu_pro",
                        "messages": [{"role": "user", "content": "Question"}],
                        "metadata": {"source_id": "one"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "capture"
            argv = [
                str(SCRIPT_PATH),
                "--input-jsonl",
                str(input_path),
                "--base-url",
                "http://127.0.0.1:7000/v1",
                "--model",
                "glm-52",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(CAPTURE, "post_json", return_value=response()):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(CAPTURE.main(), 0)
            with np.load(output_dir / "aggregate-counts.npz", allow_pickle=False) as archive:
                self.assertEqual(int(archive["request_count"][0]), 1)
                self.assertEqual(int(archive["counts"][0].sum()), 3 * 75 * 8)
                self.assertEqual(int(archive["counts"][1].sum()), 3 * 75 * 8)
            self.assertTrue(any((output_dir / "routes").glob("*.npy")))

    def test_loopback_is_distinguished_from_remote_urls(self) -> None:
        self.assertTrue(CAPTURE.is_loopback_url("http://127.0.0.1:7000/v1"))
        self.assertTrue(CAPTURE.is_loopback_url("http://localhost:7000/v1"))
        self.assertFalse(CAPTURE.is_loopback_url("http://10.0.0.1:7000/v1"))


if __name__ == "__main__":
    unittest.main()
