from __future__ import annotations

import base64
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "12_smoke_request.py"
SPEC = importlib.util.spec_from_file_location("smoke_request", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def valid_response() -> tuple[dict, np.ndarray]:
    prompt_token_ids = [11, 12, 13]
    output_token_ids = [21, 22, 23, 24]
    rows = len(prompt_token_ids) + len(output_token_ids) - 1
    routes = np.zeros((rows, 78, 8), dtype=np.uint8)
    for row in range(rows):
        for layer in range(3, 78):
            base = (row * 17 + layer * 11) % (256 - 8)
            routes[row, layer, :] = base + np.arange(8, dtype=np.uint8)
    response = {
        "id": "chatcmpl-test",
        "model": "glm-52",
        "prompt_token_ids": prompt_token_ids,
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "token_ids": output_token_ids,
                "routed_experts": encode_array(routes),
            }
        ],
    }
    return response, routes


class RouteValidationTests(unittest.TestCase):
    def test_loopback_request_bypasses_environment_proxy(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self) -> bytes:
                return b'{"id": "response"}'

        with mock.patch.object(SMOKE.urllib.request, "build_opener") as build_opener:
            with mock.patch.object(SMOKE.urllib.request, "urlopen") as urlopen:
                build_opener.return_value.open.return_value = FakeResponse()
                status, response = SMOKE.post_json(
                    "http://127.0.0.1:7000/v1/chat/completions",
                    {"model": "glm-52"},
                    "proxy-test",
                    "EMPTY",
                    1,
                )
        self.assertEqual(status, 200)
        self.assertEqual(response["id"], "response")
        build_opener.assert_called_once()
        urlopen.assert_not_called()

    def test_valid_routes_and_phase_boundary(self) -> None:
        response, expected = valid_response()
        routes, summary = SMOKE.validate_response(
            response, require_routes=True, expected_model="glm-52"
        )
        np.testing.assert_array_equal(routes, expected)
        self.assertEqual(summary["shape"], [6, 78, 8])
        self.assertEqual(summary["prefill_rows"], 3)
        self.assertEqual(summary["decode_rows"], 3)
        self.assertEqual(summary["prefill_assignments"], 3 * 75 * 8)
        self.assertEqual(summary["decode_assignments"], 3 * 75 * 8)

    def test_routes_may_be_absent_for_vendor_smoke(self) -> None:
        response, _ = valid_response()
        response["choices"][0]["routed_experts"] = None
        routes, summary = SMOKE.validate_response(response, require_routes=False)
        self.assertIsNone(routes)
        self.assertFalse(summary["routing_available"])

    def test_routes_are_required_for_capture_gate(self) -> None:
        response, _ = valid_response()
        response["choices"][0]["routed_experts"] = None
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "missing"):
            SMOKE.validate_response(response, require_routes=True)

    def test_invalid_base64_is_rejected(self) -> None:
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "valid base64"):
            SMOKE.decode_routed_experts("not base64!")

    def test_zero_filled_moe_capture_is_rejected(self) -> None:
        response, routes = valid_response()
        routes[:, 3:, :] = 0
        response["choices"][0]["routed_experts"] = encode_array(routes)
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "not unique"):
            SMOKE.validate_response(response, require_routes=True)

    def test_constant_but_unique_topk_is_rejected(self) -> None:
        response, routes = valid_response()
        routes[:, 3:, :] = np.arange(8, dtype=np.uint8)
        response["choices"][0]["routed_experts"] = encode_array(routes)
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "constant/stale"):
            SMOKE.validate_response(response, require_routes=True)

    def test_wrong_dtype_is_rejected(self) -> None:
        response, routes = valid_response()
        response["choices"][0]["routed_experts"] = encode_array(
            routes.astype(np.int32)
        )
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "must be uint8"):
            SMOKE.validate_response(response, require_routes=True)

    def test_usage_mismatch_is_rejected(self) -> None:
        response, _ = valid_response()
        response["usage"]["prompt_tokens"] = 999
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "usage.prompt_tokens"):
            SMOKE.validate_response(response, require_routes=True)

    def test_capture_gate_requires_decode_rows(self) -> None:
        response, routes = valid_response()
        response["choices"][0]["token_ids"] = [21]
        response["usage"] = {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
        response["choices"][0]["routed_experts"] = encode_array(routes[:3])
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "at least 4"):
            SMOKE.validate_response(response, require_routes=True)

    def test_dense_layer_pollution_is_rejected(self) -> None:
        response, routes = valid_response()
        routes[0, 2, 0] = 1
        response["choices"][0]["routed_experts"] = encode_array(routes)
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "dense layers"):
            SMOKE.validate_response(response, require_routes=True)

    def test_token_row_mismatch_is_rejected(self) -> None:
        response, routes = valid_response()
        response["choices"][0]["routed_experts"] = encode_array(routes[:-1])
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "shape mismatch"):
            SMOKE.validate_response(response, require_routes=True)

    def test_model_mismatch_is_rejected(self) -> None:
        response, _ = valid_response()
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "model mismatch"):
            SMOKE.validate_response(
                response, require_routes=True, expected_model="another-model"
            )

    def test_choice_index_mismatch_is_rejected(self) -> None:
        response, _ = valid_response()
        response["choices"][0]["index"] = 1
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "index"):
            SMOKE.validate_response(response, require_routes=True)

    def test_identical_repeat_matches_full_output_and_routes(self) -> None:
        response, routes = valid_response()
        result = SMOKE.validate_repeat_consistency(
            response, routes, json.loads(json.dumps(response)), routes.copy()
        )
        self.assertTrue(result["repeat_full_routes_match"])

    def test_repeat_output_token_change_is_rejected(self) -> None:
        response, routes = valid_response()
        repeat = json.loads(json.dumps(response))
        repeat["choices"][0]["token_ids"][-1] += 1
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "output token IDs"):
            SMOKE.validate_repeat_consistency(
                response, routes, repeat, routes.copy()
            )

    def test_one_token_request_proves_phase_boundary(self) -> None:
        response, routes = valid_response()
        one_token = json.loads(json.dumps(response))
        one_token["choices"][0]["token_ids"] = [21]
        one_token["usage"] = {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
        one_token["choices"][0]["routed_experts"] = encode_array(routes[:3])
        one_routes, _ = SMOKE.validate_response(one_token, require_routes=False)
        assert one_routes is not None
        result = SMOKE.validate_phase_boundary(
            response, routes, one_token, one_routes
        )
        self.assertTrue(result["prefill_boundary_match"])
        self.assertEqual(result["long_decode_route_rows"], 3)

    def test_boundary_prefill_mismatch_is_rejected(self) -> None:
        response, routes = valid_response()
        one_token = json.loads(json.dumps(response))
        one_token["choices"][0]["token_ids"] = [21]
        one_token["usage"] = {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
        changed = routes[:3].copy()
        changed[0, 3:, :] = (
            changed[0, 3:, :].astype(np.uint16) + 31
        ).astype(np.uint8)
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "prefill rows"):
            SMOKE.validate_phase_boundary(response, routes, one_token, changed)

    def test_different_prompt_must_change_routes(self) -> None:
        response, routes = valid_response()
        contrast = json.loads(json.dumps(response))
        contrast["prompt_token_ids"] = [11, 99, 13]
        changed = routes.copy()
        changed[:3, 3:, :] = (
            changed[:3, 3:, :].astype(np.uint16) + 31
        ).astype(np.uint8)
        result = SMOKE.validate_prompt_sensitivity(
            response, routes, contrast, changed
        )
        self.assertGreater(result["contrast_changed_moe_rows"], 0)

    def test_input_independent_routes_are_rejected(self) -> None:
        response, routes = valid_response()
        contrast = json.loads(json.dumps(response))
        contrast["prompt_token_ids"] = [11, 99, 13]
        with self.assertRaisesRegex(SMOKE.RouteValidationError, "input-independent"):
            SMOKE.validate_prompt_sensitivity(
                response, routes, contrast, routes.copy()
            )

    def test_initial_transport_failure_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with mock.patch.object(
                SMOKE, "post_json", side_effect=RuntimeError("offline")
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    SMOKE.execute_request(
                        endpoint="http://127.0.0.1:1/v1/chat/completions",
                        payload={"model": "glm-52"},
                        request_id="transport-test",
                        output_dir=output_dir,
                        api_key="EMPTY",
                        timeout_seconds=1,
                    )
            self.assertTrue((output_dir / "transport-test.request.json").is_file())
            self.assertTrue(
                (output_dir / "transport-test.transport-error.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
