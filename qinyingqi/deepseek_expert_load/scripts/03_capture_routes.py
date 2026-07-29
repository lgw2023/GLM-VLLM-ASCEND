#!/usr/bin/env python3
"""Send deterministic requests and persist validated DeepSeek expert routes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from deepseek_common import (
    ModelTopology,
    count_assignments,
    decode_routed_experts,
    load_topology,
    sha256_file,
    validate_routes,
)


PHASE_NAMES = ("total", "prefill", "decode")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_inputs(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"input record is not an object at line {line_number}")
            request_id = record.get("request_id")
            benchmark = record.get("benchmark")
            messages = record.get("messages")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"invalid request_id at line {line_number}")
            if not isinstance(benchmark, str) or not benchmark:
                raise ValueError(f"invalid benchmark at line {line_number}")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"invalid messages at line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"input JSONL has no records: {path}")
    return records


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    parsed = urllib.parse.urlparse(url)
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("server response is not a JSON object")
    return decoded


def token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(type(item) is int for item in value):
        raise ValueError(f"{label} must be a non-empty list of integer token IDs")
    return value


def extract_response(
    response: dict[str, Any],
    expected_model: str,
    topology: ModelTopology,
    *,
    require_unique_topk: bool = True,
    unique_scope: str = "all",
) -> tuple[np.ndarray, dict[str, Any]]:
    if response.get("model") != expected_model:
        raise ValueError(
            f"response model mismatch: expected {expected_model!r}, got {response.get('model')!r}"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    prompt_ids = token_ids(response.get("prompt_token_ids"), "prompt_token_ids")
    completion_ids = token_ids(choice.get("token_ids"), "choices[0].token_ids")
    usage = response.get("usage")
    expected_usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "total_tokens": len(prompt_ids) + len(completion_ids),
    }
    if not isinstance(usage, dict):
        raise ValueError("response.usage must be an object")
    for key, expected in expected_usage.items():
        if usage.get(key) != expected:
            raise ValueError(f"usage.{key}: expected {expected}, got {usage.get(key)!r}")

    routes = decode_routed_experts(choice.get("routed_experts"))
    summary = validate_routes(
        routes,
        topology,
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion_ids),
        require_unique_topk=require_unique_topk,
        unique_scope=unique_scope,
    )
    if summary["covered_experts"] <= topology.top_k or summary["unique_route_tuples"] <= 1:
        raise ValueError("routes are constant or stale across the request")
    summary.update(
        {
            "response_id": response.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
        }
    )
    return routes, summary


def safe_stem(request_id: str, index: int) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", request_id).strip("-")
    return f"{index:06d}-{cleaned[:100]}"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"invalid record at {path}:{line_number}")
            completed[request_id] = record
    return completed


def save_aggregate(
    path: Path,
    counts: np.ndarray,
    topology: ModelTopology,
    benchmark: str,
    request_count: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            schema_version=np.array([1], dtype=np.int64),
            benchmark=np.array([benchmark]),
            phase_names=np.array(PHASE_NAMES),
            counts=counts,
            moe_layer_indices=np.array(topology.moe_layer_indices, dtype=np.int64),
            num_hidden_layers=np.array([topology.num_hidden_layers], dtype=np.int64),
            num_experts=np.array([topology.num_experts], dtype=np.int64),
            top_k=np.array([topology.top_k], dtype=np.int64),
            request_count=np.array([request_count], dtype=np.int64),
            prompt_tokens=np.array([prompt_tokens], dtype=np.int64),
            completion_tokens=np.array([completion_tokens], dtype=np.int64),
        )
    os.replace(temporary, path)


def rebuild_aggregate(
    output_dir: Path,
    completed: dict[str, dict[str, Any]],
    topology: ModelTopology,
    *,
    require_unique_topk: bool = True,
    unique_scope: str = "all",
) -> tuple[np.ndarray, int, int]:
    counts = np.zeros((3, topology.num_moe_layers, topology.num_experts), dtype=np.int64)
    prompt_tokens = 0
    completion_tokens = 0
    for record in completed.values():
        route_path = output_dir / record["route_path"]
        if not route_path.is_file():
            raise FileNotFoundError(f"recorded route file is missing: {route_path}")
        if sha256_file(route_path) != record["route_sha256"]:
            raise RuntimeError(f"recorded route file digest changed: {route_path}")
        routes = np.load(route_path, allow_pickle=False)
        prompt = int(record["prompt_tokens"])
        completion = int(record["completion_tokens"])
        validate_routes(
            routes,
            topology,
            prompt,
            completion,
            require_unique_topk=require_unique_topk,
            unique_scope=unique_scope,
        )
        counts += count_assignments(routes, topology, prompt)
        prompt_tokens += prompt
        completion_tokens += completion
    return counts, prompt_tokens, completion_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--prompt")
    parser.add_argument("--benchmark", default="smoke")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--unique-scope",
        choices=("all", "decode", "none"),
        default="decode",
        help=(
            "Which phases must have unique top-k IDs. Default decode: Ascend "
            "All2All TP-split often leaves prefill zeros even with FLASHCOMM1=0."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-topk",
        action="store_true",
        help=(
            "Keep collecting when top-k IDs are not unique. Saves routes and "
            "quality diagnostics so benchmarks can finish before fixing capture."
        ),
    )
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if args.max_tokens < 4:
        raise ValueError("--max-tokens must be at least 4 for the capture gate")
    if args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative")
    topology = load_topology(args.model_path.expanduser().resolve())
    if not topology.model_type.startswith("deepseek"):
        raise ValueError(f"expected a DeepSeek model, got model_type={topology.model_type!r}")

    input_sha256: str | None = None
    if args.input_jsonl:
        input_path = args.input_jsonl.expanduser().resolve()
        records = load_inputs(input_path)
        input_sha256 = sha256_file(input_path)
    else:
        input_path = None
        records = [
            {
                "request_id": "deepseek-route-smoke-000001",
                "benchmark": args.benchmark,
                "messages": [{"role": "user", "content": args.prompt}],
                "metadata": {"synthetic": True},
            }
        ]
    if args.max_requests:
        records = records[: args.max_requests]
    benchmarks = sorted({str(record["benchmark"]) for record in records})
    if len(benchmarks) != 1:
        raise ValueError(f"one capture directory must contain one benchmark, got {benchmarks}")
    benchmark = benchmarks[0]

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "capture-manifest.json"
    records_path = output_dir / "records.jsonl"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"capture output exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("requests", "responses", "routes"):
        (output_dir / name).mkdir(exist_ok=True)

    expected_manifest = {
        "schema_version": 1,
        "benchmark": benchmark,
        "input_jsonl": str(input_path) if input_path else None,
        "input_sha256": input_sha256,
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "model_path": str(args.model_path.expanduser().resolve()),
        "topology": topology.to_dict(),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "unique_scope": (
            "none" if args.allow_duplicate_topk else args.unique_scope
        ),
        "allow_duplicate_topk": bool(args.allow_duplicate_topk),
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in expected_manifest.items():
            if existing_manifest.get(key) != expected:
                raise RuntimeError(f"resume manifest mismatch for {key}")
    elif args.resume:
        raise FileNotFoundError(f"cannot resume without manifest: {manifest_path}")
    else:
        expected_manifest["created_at"] = utc_now()
        manifest_path.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    unique_scope = "none" if args.allow_duplicate_topk else args.unique_scope
    completed = load_completed(records_path)
    counts, total_prompt_tokens, total_completion_tokens = rebuild_aggregate(
        output_dir,
        completed,
        topology,
        require_unique_topk=not args.allow_duplicate_topk,
        unique_scope=unique_scope,
    )
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    failures_path = output_dir / "failures.jsonl"
    imperfect_routes = 0

    for index, record in enumerate(records):
        request_id = record["request_id"]
        if request_id in completed:
            if not completed[request_id].get("route_summary", {}).get("unique_topk", True):
                imperfect_routes += 1
            continue
        stem = safe_stem(request_id, index)
        payload = {
            "model": args.model,
            "messages": record["messages"],
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": int(record.get("max_tokens", args.max_tokens)),
            "n": 1,
            "stream": False,
            "ignore_eos": True,
            "return_token_ids": True,
            "return_prompt_text": True,
        }
        request_path = output_dir / "requests" / f"{stem}.json"
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        try:
            response = post_json(endpoint, payload, args.timeout_seconds)
            routes, route_summary = extract_response(
                response,
                args.model,
                topology,
                require_unique_topk=not args.allow_duplicate_topk,
                unique_scope=unique_scope,
            )
        except Exception as exc:
            append_jsonl(
                failures_path,
                {
                    "failed_at": utc_now(),
                    "request_id": request_id,
                    "benchmark": benchmark,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        route_path = output_dir / "routes" / f"{stem}.npy"
        with route_path.open("wb") as output:
            np.save(output, routes, allow_pickle=False)
        response_copy = json.loads(json.dumps(response))
        response_copy["choices"][0].pop("routed_experts", None)
        response_path = output_dir / "responses" / f"{stem}.json"
        response_path.write_text(
            json.dumps(response_copy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        prompt_count = int(route_summary["prompt_tokens"])
        completion_count = int(route_summary["completion_tokens"])
        if not route_summary.get("unique_topk", True):
            imperfect_routes += 1
        completed_record = {
            "completed_at": utc_now(),
            "request_id": request_id,
            "benchmark": benchmark,
            "metadata": record.get("metadata", {}),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "prompt_tokens": prompt_count,
            "completion_tokens": completion_count,
            "route_path": str(route_path.relative_to(output_dir)),
            "route_sha256": sha256_file(route_path),
            "response_path": str(response_path.relative_to(output_dir)),
            "route_summary": route_summary,
        }
        append_jsonl(records_path, completed_record)
        completed[request_id] = completed_record
        counts += count_assignments(routes, topology, prompt_count)
        total_prompt_tokens += prompt_count
        total_completion_tokens += completion_count
        save_aggregate(
            output_dir / "aggregate-counts.npz",
            counts,
            topology,
            benchmark,
            len(completed),
            total_prompt_tokens,
            total_completion_tokens,
        )
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "completed": len(completed),
                    "prompt_tokens": prompt_count,
                    "completion_tokens": completion_count,
                    "covered_experts": route_summary["covered_experts"],
                    "unique_topk": route_summary.get("unique_topk", True),
                    "unique_topk_decode": route_summary.get("unique_topk_decode", True),
                    "unique_scope": unique_scope,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    quality = {
        "schema_version": 1,
        "benchmark": benchmark,
        "request_count": len(completed),
        "imperfect_route_requests": imperfect_routes,
        "unique_scope": unique_scope,
        "allow_duplicate_topk": bool(args.allow_duplicate_topk),
        "routes_trusted_for_full_load_analysis": imperfect_routes == 0,
        "routes_trusted_for_decode_load_analysis": all(
            record.get("route_summary", {}).get("unique_topk_decode", True)
            for record in completed.values()
        ),
    }
    (output_dir / "route-quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if imperfect_routes == 0:
        status = "CAPTURE_OK"
    elif unique_scope == "decode" and quality["routes_trusted_for_decode_load_analysis"]:
        status = "CAPTURE_OK_DECODE_TRUSTED"
    else:
        status = "CAPTURE_OK_WITH_IMPERFECT_ROUTES"
    print(
        f"{status} benchmark={benchmark} requests={len(completed)} "
        f"imperfect_routes={imperfect_routes} unique_scope={unique_scope} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
