#!/usr/bin/env python3
"""Send sequential benchmark prompts and preserve GLM-5.2 expert routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from expert_load_common import (
    NUM_LOGICAL_EXPERTS,
    NUM_MOE_LAYERS,
    PHASE_NAMES,
    RouteCaptureError,
    count_route_assignments,
    extract_validated_routes,
)


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_stem(request_id: str, index: int) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:06d}-{digest}"


def require_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{message_index}].role must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"messages[{message_index}].content must be a non-empty string"
            )
        messages.append({"role": role, "content": content})
    return messages


def load_inputs(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"input record at line {line_number} must be an object")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"input record at line {line_number} lacks request_id")
            if request_id in seen_ids:
                raise ValueError(f"duplicate request_id in input: {request_id}")
            seen_ids.add(request_id)
            record["messages"] = require_messages(record.get("messages"))
            benchmark = record.get("benchmark")
            if not isinstance(benchmark, str) or not benchmark:
                raise ValueError(f"input record {request_id} lacks benchmark")
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"input record {request_id} metadata must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"input JSONL has no records: {path}")
    return records


def is_loopback_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    return hostname in {"127.0.0.1", "localhost", "::1"}


def post_json(url: str, payload: dict[str, Any], timeout_seconds: int, api_key: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        if is_loopback_url(url):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            response_context = opener.open(request, timeout=timeout_seconds)
        else:
            response_context = urllib.request.urlopen(request, timeout=timeout_seconds)
        with response_context as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"expected HTTP 200 from {url}, got {status}")
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"server returned non-JSON body: {body[:500]!r}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("server response JSON is not an object")
    return decoded


def validate_usage(response: dict[str, Any], prompt_tokens: int, output_tokens: int) -> None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RouteCaptureError("response usage must be an object")
    expected = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    for field, value in expected.items():
        if usage.get(field) != value:
            raise RouteCaptureError(
                f"usage.{field} mismatch: expected {value}, got {usage.get(field)!r}"
            )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_completed_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(f"invalid completed record at {path}:{line_number}")
            completed[request_id] = record
    return completed


def rebuild_aggregate(output_dir: Path, completed: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = np.zeros((3, NUM_MOE_LAYERS, NUM_LOGICAL_EXPERTS), dtype=np.int64)
    request_count = 0
    prompt_tokens = 0
    decode_tokens = 0
    for record in completed:
        route_path = output_dir / record["route_path"]
        if not route_path.is_file():
            raise FileNotFoundError(f"recorded route artifact is missing: {route_path}")
        if sha256_file(route_path) != record["route_sha256"]:
            raise RuntimeError(f"route artifact digest changed: {route_path}")
        routes = np.load(route_path, allow_pickle=False)
        prompt_count = int(record["prompt_tokens"])
        output_count = int(record["output_tokens"])
        counts += count_route_assignments(routes, prompt_count)
        request_count += 1
        prompt_tokens += prompt_count
        decode_tokens += max(output_count - 1, 0)
    np.savez_compressed(
        output_dir / "aggregate-counts.npz",
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int64),
        phase_names=np.array(PHASE_NAMES),
        counts=counts,
        request_count=np.array([request_count], dtype=np.int64),
        prompt_tokens=np.array([prompt_tokens], dtype=np.int64),
        decode_tokens=np.array([decode_tokens], dtype=np.int64),
    )
    return {
        "request_count": request_count,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--min-output-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.min_output_tokens < 1 or args.min_output_tokens > args.max_tokens:
        raise ValueError("--min-output-tokens must be in 1..--max-tokens")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    if args.max_requests < 0 or args.sleep_seconds < 0:
        raise ValueError("--max-requests and --sleep-seconds must be non-negative")

    input_path = args.input_jsonl.expanduser().resolve()
    inputs = load_inputs(input_path)
    if args.max_requests:
        inputs = inputs[: args.max_requests]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("requests", "responses", "routes"):
        (output_dir / subdirectory).mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "capture-manifest.json"
    input_sha256 = sha256_file(input_path)

    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"capture output already exists: {output_dir}; use --resume to continue"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != input_sha256:
            raise RuntimeError("--resume input JSONL differs from the existing capture")
        if manifest.get("model") != args.model or manifest.get("base_url") != args.base_url:
            raise RuntimeError("--resume model or base URL differs from the existing capture")
    elif args.resume:
        raise FileNotFoundError(f"cannot resume without {manifest_path}")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "input_jsonl": str(input_path),
            "input_sha256": input_sha256,
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "min_output_tokens": args.min_output_tokens,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    completed = load_completed_records(records_path)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    failures_path = output_dir / "failures.jsonl"

    for index, record in enumerate(inputs):
        request_id = record["request_id"]
        if request_id in completed:
            continue
        stem = artifact_stem(request_id, index)
        payload = {
            "model": args.model,
            "messages": record["messages"],
            "temperature": 0,
            "seed": args.seed,
            "max_tokens": int(record.get("max_tokens", args.max_tokens)),
            "n": 1,
            "stream": False,
            "return_token_ids": True,
            "return_prompt_text": True,
            "ignore_eos": args.ignore_eos,
        }
        request_path = output_dir / "requests" / f"{stem}.json"
        request_path.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "benchmark": record["benchmark"],
                    "metadata": record.get("metadata", {}),
                    "endpoint": endpoint,
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        started = time.perf_counter()
        try:
            response = post_json(endpoint, payload, args.timeout_seconds, args.api_key)
            latency_seconds = time.perf_counter() - started
            response_path = output_dir / "responses" / f"{stem}.json"
            response_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            routes, prompt_count, output_count = extract_validated_routes(
                response, expected_model=args.model
            )
            validate_usage(response, prompt_count, output_count)
            if output_count < args.min_output_tokens:
                raise RouteCaptureError(
                    f"only {output_count} output tokens; require at least "
                    f"{args.min_output_tokens} for decode analysis"
                )
            route_path = output_dir / "routes" / f"{stem}.npy"
            np.save(route_path, routes, allow_pickle=False)
            completed_record = {
                "request_id": request_id,
                "benchmark": record["benchmark"],
                "metadata": record.get("metadata", {}),
                "route_path": str(route_path.relative_to(output_dir)),
                "route_sha256": sha256_file(route_path),
                "request_path": str(request_path.relative_to(output_dir)),
                "response_path": str(response_path.relative_to(output_dir)),
                "response_id": response.get("id"),
                "prompt_tokens": prompt_count,
                "output_tokens": output_count,
                "latency_seconds": latency_seconds,
            }
            append_jsonl(records_path, completed_record)
            completed[request_id] = completed_record
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "request_id": request_id,
                        "prompt_tokens": prompt_count,
                        "output_tokens": output_count,
                        "latency_seconds": round(latency_seconds, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except (RouteCaptureError, RuntimeError, urllib.error.URLError) as exc:
            failure = {
                "failed_at": utc_now(),
                "request_id": request_id,
                "benchmark": record["benchmark"],
                "error": str(exc),
                "request_path": str(request_path.relative_to(output_dir)),
            }
            append_jsonl(failures_path, failure)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
            return 2
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    aggregate = rebuild_aggregate(output_dir, completed.values())
    manifest.update({"completed_at": utc_now(), **aggregate})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"capture_dir": str(output_dir), **aggregate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
