#!/usr/bin/env python3
"""Audit a local DeepSeek checkpoint without loading model tensors."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from deepseek_common import load_json, topology_from_config


QUANT_MARKER = re.compile(r"w\d+a\d+(?:c\d+)?", re.IGNORECASE)


def walk_scalars(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_scalars(child)
    elif value is not None:
        yield str(value)


def quantization_summary(model_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    description_path = model_path / "quant_model_description.json"
    sources: list[tuple[str, Any]] = []
    config_quant = config.get("quantization_config")
    if config_quant is not None:
        sources.append(("config.json:quantization_config", config_quant))
    if description_path.is_file():
        sources.append((description_path.name, load_json(description_path)))

    marker_counts: Counter[str] = Counter()
    for _, source in sources:
        for scalar in walk_scalars(source):
            for marker in QUANT_MARKER.findall(scalar):
                marker_counts[marker.upper()] += 1
    return {
        "sources": [name for name, _ in sources],
        "markers": dict(sorted(marker_counts.items())),
        "w4a8_detected": marker_counts["W4A8"] > 0,
        "quant_model_description_present": description_path.is_file(),
    }


def weight_summary(model_path: Path) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    shards = sorted(model_path.glob("*.safetensors"))
    zero_size = [path.name for path in shards if path.stat().st_size == 0]
    if not shards:
        problems.append("no .safetensors files found")
    if zero_size:
        problems.append(f"zero-size safetensors files: {zero_size}")

    index_path = model_path / "model.safetensors.index.json"
    referenced_files: set[str] = set()
    tensor_count = 0
    if index_path.is_file():
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            problems.append("model.safetensors.index.json has no non-empty weight_map")
        else:
            tensor_count = len(weight_map)
            referenced_files = {
                value for value in weight_map.values() if isinstance(value, str)
            }
            missing = sorted(name for name in referenced_files if not (model_path / name).is_file())
            if missing:
                problems.append(f"index references missing shards: {missing}")
    elif len(shards) > 1:
        problems.append("multiple safetensors shards found without model.safetensors.index.json")

    return (
        {
            "shard_count": len(shards),
            "total_shard_bytes": sum(path.stat().st_size for path in shards),
            "total_shard_gib": round(
                sum(path.stat().st_size for path in shards) / 1024**3, 3
            ),
            "index_present": index_path.is_file(),
            "indexed_tensor_count": tensor_count,
            "referenced_shard_count": len(referenced_files),
        },
        problems,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-model-type", default="")
    parser.add_argument("--require-w4a8", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    problems: list[str] = []
    config_path = model_path / "config.json"
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")

    config = load_json(config_path)
    topology = topology_from_config(config)
    weights, weight_problems = weight_summary(model_path)
    problems.extend(weight_problems)
    quantization = quantization_summary(model_path, config)

    if args.require_model_type and topology.model_type != args.require_model_type:
        problems.append(
            f"model_type must be {args.require_model_type!r}, got {topology.model_type!r}"
        )
    if args.require_w4a8 and not quantization["w4a8_detected"]:
        problems.append(
            "W4A8 was not proven by config.json or quant_model_description.json"
        )

    report = {
        "schema_version": 1,
        "model_path": str(model_path),
        "topology": topology.to_dict(),
        "quantization": quantization,
        "weights": weights,
        "problems": problems,
        "compatible": not problems,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
