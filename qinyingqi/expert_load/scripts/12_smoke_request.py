#!/usr/bin/env python3
"""Send deterministic chat requests and validate routed-expert output."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np


NUM_LAYERS = 78
NUM_DENSE_LAYERS = 3
NUM_MOE_LAYERS = 75
NUM_LOGICAL_EXPERTS = 256
TOP_K = 8


class RouteValidationError(ValueError):
    """The response violates the GLM-5.2 route-capture contract."""


def decode_routed_experts(encoded: str) -> np.ndarray:
    if not isinstance(encoded, str) or not encoded:
        raise RouteValidationError("routed_experts is not a non-empty string")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RouteValidationError("routed_experts is not valid base64") from exc
    try:
        routes = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RouteValidationError("routed_experts is not a valid non-pickle .npy") from exc
    if not isinstance(routes, np.ndarray):
        raise RouteValidationError("decoded routed_experts is not an ndarray")
    return routes


def _require_token_ids(value: Any, field: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise RouteValidationError(f"{field} must be a list of integer token IDs")
    if not value and not allow_empty:
        raise RouteValidationError(f"{field} must not be empty")
    return value


def validate_response(
    response: dict[str, Any],
    require_routes: bool = False,
    expected_model: str | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RouteValidationError("expected exactly one response choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RouteValidationError("choices[0] must be an object")
    if choice.get("index") != 0:
        raise RouteValidationError("choices[0].index must be 0")
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RouteValidationError("choices[0].finish_reason must be a non-empty string")
    if expected_model is not None and response.get("model") != expected_model:
        raise RouteValidationError(
            f"response model mismatch: expected {expected_model!r}, got {response.get('model')!r}"
        )

    prompt_token_ids = _require_token_ids(
        response.get("prompt_token_ids"), "prompt_token_ids"
    )
    output_token_ids = _require_token_ids(choice.get("token_ids"), "choices[0].token_ids")
    prompt_tokens = len(prompt_token_ids)
    output_tokens = len(output_token_ids)
    expected_rows = prompt_tokens + output_tokens - 1
    if require_routes and output_tokens < 4:
        raise RouteValidationError(
            f"capture gate requires at least 4 output tokens, got {output_tokens}"
        )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RouteValidationError("usage must be an object")
    expected_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    for field, expected in expected_usage.items():
        if usage.get(field) != expected:
            raise RouteValidationError(
                f"usage.{field} mismatch: expected {expected}, got {usage.get(field)!r}"
            )

    summary: dict[str, Any] = {
        "valid": True,
        "routing_available": False,
        "response_id": response.get("id"),
        "model": response.get("model"),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_route_rows": expected_rows,
        "prefill_rows": prompt_tokens,
        "decode_rows": max(output_tokens - 1, 0),
        "usage": usage,
    }

    encoded = choice.get("routed_experts")
    if encoded in (None, ""):
        if require_routes:
            raise RouteValidationError(
                "routed_experts is missing; server flag or W8A8 capture hook is not active"
            )
        return None, summary

    routes = decode_routed_experts(encoded)
    if routes.dtype != np.dtype(np.uint8):
        raise RouteValidationError(f"route dtype must be uint8, got {routes.dtype}")
    expected_shape = (expected_rows, NUM_LAYERS, TOP_K)
    if routes.shape != expected_shape:
        raise RouteValidationError(
            f"route shape mismatch: expected {expected_shape}, got {routes.shape}"
        )
    if not np.all(routes[:, :NUM_DENSE_LAYERS, :] == 0):
        raise RouteValidationError("dense layers 0..2 contain non-zero route data")

    moe_routes = routes[:, NUM_DENSE_LAYERS:, :]
    if moe_routes.size == 0:
        raise RouteValidationError("MoE route tensor is empty")
    min_expert = int(moe_routes.min())
    max_expert = int(moe_routes.max())
    if min_expert < 0 or max_expert >= NUM_LOGICAL_EXPERTS:
        raise RouteValidationError(
            f"logical expert IDs must be in [0, 255], got [{min_expert}, {max_expert}]"
        )
    sorted_topk = np.sort(moe_routes, axis=-1)
    if not np.all(np.diff(sorted_topk, axis=-1) > 0):
        raise RouteValidationError(
            "top-8 experts are not unique for every token/layer; capture may be zero-filled"
        )
    covered_experts = int(np.unique(moe_routes).size)
    unique_route_tuples = int(
        np.unique(moe_routes.reshape(-1, TOP_K), axis=0).shape[0]
    )
    if covered_experts <= TOP_K or unique_route_tuples <= 1:
        raise RouteValidationError(
            "MoE routes are constant/stale across tokens and layers"
        )

    prefill = routes[:prompt_tokens]
    decode = routes[prompt_tokens:]
    prefill_assignments = int(prefill.shape[0] * NUM_MOE_LAYERS * TOP_K)
    decode_assignments = int(decode.shape[0] * NUM_MOE_LAYERS * TOP_K)
    summary.update(
        {
            "routing_available": True,
            "shape": list(routes.shape),
            "dtype": str(routes.dtype),
            "dense_layers_zero": True,
            "topk_unique": True,
            "covered_logical_experts": covered_experts,
            "unique_route_tuples": unique_route_tuples,
            "min_logical_expert_id": min_expert,
            "max_logical_expert_id": max_expert,
            "prefill_assignments": prefill_assignments,
            "decode_assignments": decode_assignments,
            "total_assignments": prefill_assignments + decode_assignments,
        }
    )
    return routes, summary


def validate_repeat_consistency(
    primary_response: dict[str, Any],
    primary_routes: np.ndarray,
    repeat_response: dict[str, Any],
    repeat_routes: np.ndarray,
) -> dict[str, Any]:
    primary_prompt = _require_token_ids(
        primary_response.get("prompt_token_ids"), "primary.prompt_token_ids"
    )
    repeat_prompt = _require_token_ids(
        repeat_response.get("prompt_token_ids"), "repeat.prompt_token_ids"
    )
    primary_output = _require_token_ids(
        primary_response["choices"][0].get("token_ids"),
        "primary.choices[0].token_ids",
    )
    repeat_output = _require_token_ids(
        repeat_response["choices"][0].get("token_ids"),
        "repeat.choices[0].token_ids",
    )
    if primary_prompt != repeat_prompt:
        raise RouteValidationError("repeat request returned different prompt token IDs")
    if primary_output != repeat_output:
        raise RouteValidationError("repeat request returned different output token IDs")
    if not np.array_equal(primary_routes, repeat_routes):
        raise RouteValidationError(
            "identical deterministic requests produced different full route tensors"
        )
    return {
        "repeat_prompt_token_ids_match": True,
        "repeat_output_token_ids_match": True,
        "repeat_full_routes_match": True,
    }


def validate_phase_boundary(
    long_response: dict[str, Any],
    long_routes: np.ndarray,
    one_token_response: dict[str, Any],
    one_token_routes: np.ndarray,
) -> dict[str, Any]:
    long_prompt = _require_token_ids(
        long_response.get("prompt_token_ids"), "long.prompt_token_ids"
    )
    short_prompt = _require_token_ids(
        one_token_response.get("prompt_token_ids"), "one_token.prompt_token_ids"
    )
    long_output = _require_token_ids(
        long_response["choices"][0].get("token_ids"),
        "long.choices[0].token_ids",
    )
    short_output = _require_token_ids(
        one_token_response["choices"][0].get("token_ids"),
        "one_token.choices[0].token_ids",
    )
    if long_prompt != short_prompt:
        raise RouteValidationError(
            "max_tokens=1 and long requests returned different prompt token IDs"
        )
    if len(short_output) != 1:
        raise RouteValidationError(
            f"max_tokens=1 request returned {len(short_output)} output tokens"
        )
    if len(long_output) < 4:
        raise RouteValidationError("long boundary request needs at least 4 output tokens")
    if short_output[0] != long_output[0]:
        raise RouteValidationError(
            "max_tokens=1 and long requests produced different first output tokens"
        )
    prompt_rows = len(long_prompt)
    if one_token_routes.shape[0] != prompt_rows:
        raise RouteValidationError(
            "max_tokens=1 route tensor does not contain exactly P rows"
        )
    if long_routes.shape[0] - prompt_rows != len(long_output) - 1:
        raise RouteValidationError(
            "long route tensor does not contain exactly G-1 decode rows"
        )
    if not np.array_equal(one_token_routes, long_routes[:prompt_rows]):
        raise RouteValidationError(
            "max_tokens=1 routes do not match the long request's P prefill rows"
        )
    return {
        "one_token_output_count": 1,
        "one_token_route_rows": prompt_rows,
        "long_decode_route_rows": len(long_output) - 1,
        "first_output_token_match": True,
        "prefill_boundary_match": True,
    }


def validate_prompt_sensitivity(
    primary_response: dict[str, Any],
    primary_routes: np.ndarray,
    contrast_response: dict[str, Any],
    contrast_routes: np.ndarray,
) -> dict[str, Any]:
    primary_prompt = _require_token_ids(
        primary_response.get("prompt_token_ids"), "primary.prompt_token_ids"
    )
    contrast_prompt = _require_token_ids(
        contrast_response.get("prompt_token_ids"), "contrast.prompt_token_ids"
    )
    if primary_prompt == contrast_prompt:
        raise RouteValidationError("contrast prompt produced identical prompt token IDs")
    overlap_rows = min(len(primary_prompt), len(contrast_prompt))
    if overlap_rows < 2:
        raise RouteValidationError("contrast prompts have too few comparable prefill rows")
    primary_overlap = primary_routes[
        :overlap_rows, NUM_DENSE_LAYERS:, :
    ]
    contrast_overlap = contrast_routes[
        :overlap_rows, NUM_DENSE_LAYERS:, :
    ]
    changed_rows = int(
        np.count_nonzero(
            np.any(primary_overlap != contrast_overlap, axis=(1, 2))
        )
    )
    if changed_rows == 0:
        raise RouteValidationError(
            "different prompts produced identical overlapping MoE prefill routes; "
            "capture may be input-independent or stale"
        )
    return {
        "contrast_prompt_token_ids_differ": True,
        "contrast_overlap_rows": overlap_rows,
        "contrast_changed_moe_rows": changed_rows,
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    request_id: str,
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"server returned non-JSON body: {body[:500]!r}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("server response JSON is not an object")
    return status, decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL ending in /v1")
    parser.add_argument("--model", default="glm-52")
    parser.add_argument("--prompt", default="Reply with exactly: OK")
    parser.add_argument(
        "--contrast-prompt",
        default="Explain why two plus two equals four in one sentence.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id")
    parser.add_argument("--require-routes", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="Defaults to VLLM_API_KEY or EMPTY; never written to artifacts.",
    )
    return parser.parse_args()


def execute_request(
    *,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    output_dir: Path,
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, dict[str, Any], float, Path]:
    request_path = output_dir / f"{request_id}.request.json"
    request_path.write_text(
        json.dumps(
            {
                "client_request_id": request_id,
                "endpoint": endpoint,
                "payload": payload,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        status, response = post_json(
            endpoint, payload, request_id, api_key, timeout_seconds
        )
    except RuntimeError as exc:
        latency_seconds = time.perf_counter() - started
        error_path = output_dir / f"{request_id}.transport-error.json"
        error_path.write_text(
            json.dumps(
                {
                    "client_request_id": request_id,
                    "latency_seconds": latency_seconds,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{request_id}: {exc}") from exc
    latency_seconds = time.perf_counter() - started
    response_path = output_dir / f"{request_id}.response.json"
    response_path.write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return status, response, latency_seconds, response_path


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.require_routes and (args.max_tokens < 4 or not args.ignore_eos):
        raise ValueError(
            "--require-routes requires --ignore-eos and --max-tokens >= 4 "
            "so decode capture is exercised"
        )
    if args.require_routes and args.prompt == args.contrast_prompt:
        raise ValueError("--contrast-prompt must differ from --prompt")
    request_id = args.request_id or f"route-smoke-{uuid.uuid4().hex}"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "n": 1,
        "stream": False,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    artifacts: dict[str, Any] = {}
    try:
        status, response, latency_seconds, response_path = execute_request(
            endpoint=endpoint,
            payload=payload,
            request_id=request_id,
            output_dir=output_dir,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
        )
        artifacts["primary_response_path"] = str(response_path)
        if status != 200:
            raise RouteValidationError(f"expected HTTP 200, got {status}")
        routes, summary = validate_response(
            response,
            require_routes=args.require_routes,
            expected_model=args.model,
        )

        if args.require_routes:
            assert routes is not None
            repeat_request_id = f"{request_id}-repeat"
            repeat_status, repeat_response, repeat_latency, repeat_response_path = (
                execute_request(
                    endpoint=endpoint,
                    payload=payload,
                    request_id=repeat_request_id,
                    output_dir=output_dir,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            artifacts["repeat_response_path"] = str(repeat_response_path)
            if repeat_status != 200:
                raise RouteValidationError(
                    f"repeat request expected HTTP 200, got {repeat_status}"
                )
            repeat_routes, repeat_summary = validate_response(
                repeat_response,
                require_routes=True,
                expected_model=args.model,
            )
            assert repeat_routes is not None
            repeat_consistency = validate_repeat_consistency(
                response, routes, repeat_response, repeat_routes
            )
            repeat_route_path = output_dir / f"{repeat_request_id}.routes.npy"
            np.save(repeat_route_path, repeat_routes, allow_pickle=False)
            artifacts["repeat_route_path"] = str(repeat_route_path)

            one_token_request_id = f"{request_id}-one-token"
            one_token_payload = dict(payload)
            one_token_payload["max_tokens"] = 1
            one_status, one_response, one_latency, one_response_path = execute_request(
                endpoint=endpoint,
                payload=one_token_payload,
                request_id=one_token_request_id,
                output_dir=output_dir,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
            )
            artifacts["one_token_response_path"] = str(one_response_path)
            if one_status != 200:
                raise RouteValidationError(
                    f"max_tokens=1 request expected HTTP 200, got {one_status}"
                )
            one_routes, one_summary = validate_response(
                one_response, require_routes=False, expected_model=args.model
            )
            if one_routes is None:
                raise RouteValidationError(
                    "max_tokens=1 request is missing routed_experts"
                )
            boundary_summary = validate_phase_boundary(
                response, routes, one_response, one_routes
            )
            one_route_path = output_dir / f"{one_token_request_id}.routes.npy"
            np.save(one_route_path, one_routes, allow_pickle=False)
            artifacts["one_token_route_path"] = str(one_route_path)

            contrast_request_id = f"{request_id}-contrast"
            contrast_payload = dict(one_token_payload)
            contrast_payload["messages"] = [
                {"role": "user", "content": args.contrast_prompt}
            ]
            contrast_status, contrast_response, contrast_latency, contrast_response_path = (
                execute_request(
                    endpoint=endpoint,
                    payload=contrast_payload,
                    request_id=contrast_request_id,
                    output_dir=output_dir,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            artifacts["contrast_response_path"] = str(contrast_response_path)
            if contrast_status != 200:
                raise RouteValidationError(
                    f"contrast request expected HTTP 200, got {contrast_status}"
                )
            contrast_routes, contrast_summary = validate_response(
                contrast_response, require_routes=False, expected_model=args.model
            )
            if contrast_routes is None:
                raise RouteValidationError("contrast request is missing routed_experts")
            sensitivity_summary = validate_prompt_sensitivity(
                response, routes, contrast_response, contrast_routes
            )
            contrast_route_path = output_dir / f"{contrast_request_id}.routes.npy"
            np.save(contrast_route_path, contrast_routes, allow_pickle=False)
            artifacts["contrast_route_path"] = str(contrast_route_path)

            summary.update(
                {
                    "repeat_request_id": repeat_request_id,
                    "repeat_http_status": repeat_status,
                    "repeat_latency_seconds": repeat_latency,
                    "repeat_response_path": str(repeat_response_path),
                    "repeat_route_path": str(repeat_route_path),
                    "repeat_route_sha256": hashlib.sha256(
                        repeat_route_path.read_bytes()
                    ).hexdigest(),
                    **repeat_consistency,
                    "repeat_summary": repeat_summary,
                    "one_token_request_id": one_token_request_id,
                    "one_token_http_status": one_status,
                    "one_token_latency_seconds": one_latency,
                    "one_token_summary": one_summary,
                    **boundary_summary,
                    "contrast_request_id": contrast_request_id,
                    "contrast_http_status": contrast_status,
                    "contrast_latency_seconds": contrast_latency,
                    "contrast_summary": contrast_summary,
                    **sensitivity_summary,
                }
            )
    except (RouteValidationError, RuntimeError) as exc:
        failure = {
            "valid": False,
            "client_request_id": request_id,
            "artifacts": artifacts,
            "error": str(exc),
        }
        (output_dir / f"{request_id}.summary.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2

    summary.update(
        {
            "client_request_id": request_id,
            "http_status": status,
            "latency_seconds": latency_seconds,
            "response_path": str(response_path),
            "artifacts": artifacts,
        }
    )
    if routes is not None:
        route_path = output_dir / f"{request_id}.routes.npy"
        np.save(route_path, routes, allow_pickle=False)
        summary["route_path"] = str(route_path)
        summary["route_sha256"] = hashlib.sha256(route_path.read_bytes()).hexdigest()

    summary_path = output_dir / f"{request_id}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
