#!/usr/bin/env python3
"""Run exactly one prepared LiveCodeBench request and validate GLM-5.1 routes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from glm51_common import decode_routes, load_topology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def read_record(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("index must be non-negative")
    current = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if current == index:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"record at line {line_number} is not an object")
                if not isinstance(record.get("messages"), list) or not record["messages"]:
                    raise ValueError(f"record at line {line_number} has no messages")
                return record
            current += 1
    raise IndexError(f"input has no non-empty record at index {index}")


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
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:4000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("server response is not a JSON object")
    return value


def integer_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(type(item) is int for item in value):
        raise ValueError(f"{label} must be a non-empty integer list")
    return value


def route_counts(routes: np.ndarray, moe_layers: tuple[int, ...], prompt_tokens: int, experts: int) -> np.ndarray:
    counts = np.zeros((3, len(moe_layers), experts), dtype=np.int64)
    phases = (routes, routes[:prompt_tokens], routes[prompt_tokens:])
    for phase_index, phase in enumerate(phases):
        selected = phase[:, moe_layers, :]
        for layer_position in range(len(moe_layers)):
            np.add.at(counts[phase_index, layer_position], selected[:, layer_position, :].reshape(-1), 1)
    return counts


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_tokens <= 256:
        raise ValueError("max-tokens must be in 1..256 for this smoke run")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = load_topology(args.model_path.resolve())
    record = read_record(args.input_jsonl, args.index)
    request_id = record.get("request_id", f"livecodebench-index-{args.index}")
    payload = {
        "model": args.model,
        "messages": record["messages"],
        "temperature": 0.0,
        "seed": 1024,
        "max_tokens": args.max_tokens,
        "n": 1,
        "stream": False,
        "ignore_eos": True,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    (output_dir / "input-record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    response = post_json(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
        args.timeout_seconds,
    )
    choices = response.get("choices")
    if response.get("model") != args.model:
        raise ValueError(f"response model mismatch: {response.get('model')!r}")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    prompt_ids = integer_ids(response.get("prompt_token_ids"), "prompt_token_ids")
    completion_ids = integer_ids(choice.get("token_ids"), "choices[0].token_ids")
    routes = decode_routes(choice.get("routed_experts"))

    # Preserve the raw payload before validation so a failed first run remains debuggable.
    with (output_dir / "routes.raw.npy").open("wb") as output:
        np.save(output, routes, allow_pickle=False)
    response_copy = json.loads(json.dumps(response))
    response_copy["choices"][0].pop("routed_experts", None)
    (output_dir / "response.json").write_text(
        json.dumps(response_copy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    usage = response.get("usage")
    expected_usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "total_tokens": len(prompt_ids) + len(completion_ids),
    }
    if not isinstance(usage, dict):
        raise ValueError("response.usage is missing")
    for key, expected in expected_usage.items():
        if usage.get(key) != expected:
            raise ValueError(f"usage.{key}: expected {expected}, got {usage.get(key)!r}")

    expected_shape = (
        len(prompt_ids) + len(completion_ids) - 1,
        topology.num_hidden_layers,
        topology.top_k,
    )
    if routes.shape != expected_shape:
        raise ValueError(f"route shape mismatch: expected {expected_shape}, got {routes.shape}")
    if not np.issubdtype(routes.dtype, np.integer):
        raise ValueError(f"route dtype is not integer: {routes.dtype}")
    if topology.dense_layer_indices and not np.all(routes[:, topology.dense_layer_indices, :] == 0):
        raise ValueError(f"dense layers contain route IDs: {topology.dense_layer_indices}")
    moe_routes = routes[:, topology.moe_layer_indices, :]
    minimum = int(moe_routes.min())
    maximum = int(moe_routes.max())
    if minimum < 0 or maximum >= topology.num_experts:
        raise ValueError(f"expert IDs out of range: min={minimum} max={maximum}")
    sorted_topk = np.sort(moe_routes, axis=-1)
    duplicate_rows = int(np.any(np.diff(sorted_topk, axis=-1) <= 0, axis=-1).sum())
    if duplicate_rows:
        raise ValueError(f"top-k expert IDs are duplicated in {duplicate_rows} token/layer rows")
    covered = int(np.unique(moe_routes).size)
    if covered <= topology.top_k:
        raise ValueError(f"routes are constant or stale: covered_experts={covered}")

    counts = route_counts(routes, topology.moe_layer_indices, len(prompt_ids), topology.num_experts)
    np.savez_compressed(
        output_dir / "aggregate-counts.npz",
        phase_names=np.array(["total", "prefill", "decode"]),
        counts=counts,
        moe_layer_indices=np.array(topology.moe_layer_indices),
        num_experts=np.array([topology.num_experts]),
        top_k=np.array([topology.top_k]),
        prompt_tokens=np.array([len(prompt_ids)]),
        completion_tokens=np.array([len(completion_ids)]),
    )
    global_counts = counts[0].sum(axis=0)
    order = np.argsort(-global_counts, kind="stable")[:20]
    summary = {
        "schema_version": 1,
        "request_id": request_id,
        "input_index": args.index,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "model": args.model,
        "topology": topology.to_dict(),
        "route_shape": list(routes.shape),
        "route_dtype": str(routes.dtype),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "covered_experts": covered,
        "duplicate_token_layer_rows": duplicate_rows,
        "top20_experts_global": [
            {"expert_id": int(expert), "assignments": int(global_counts[expert])}
            for expert in order
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"LIVECODEBENCH_ONE_OK output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
