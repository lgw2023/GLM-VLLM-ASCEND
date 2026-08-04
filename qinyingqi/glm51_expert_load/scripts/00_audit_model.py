#!/usr/bin/env python3
"""Audit a GLM-5.1 ModelSlim W4A8 checkpoint without loading tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from glm51_common import load_json, topology_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-shard-gib", type=float, default=480.0)
    return parser.parse_args()


def audit(model_path: Path, max_shard_gib: float) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {
            "schema_version": 1,
            "compatible": False,
            "model_path": str(model_path),
            "problems": ["config.json is missing"],
            "warnings": [],
        }

    config = load_json(config_path)
    try:
        topology = topology_from_config(config)
    except ValueError as exc:
        topology = None
        problems.append(str(exc))
    if topology is not None:
        expected = {
            "num_hidden_layers": 78,
            "num_experts": 256,
            "top_k": 8,
            "first_k_dense_replace": 3,
            "num_moe_layers": 75,
        }
        actual = {
            "num_hidden_layers": topology.num_hidden_layers,
            "num_experts": topology.num_experts,
            "top_k": topology.top_k,
            "first_k_dense_replace": topology.first_k_dense_replace,
            "num_moe_layers": topology.num_moe_layers,
        }
        mismatches = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        if mismatches:
            problems.append(f"unexpected GLM-5.1 MoE topology: {mismatches}")

    description_path = model_path / "quant_model_description.json"
    description_text = ""
    if description_path.is_file():
        description_text = description_path.read_text(encoding="utf-8", errors="replace").upper()
    else:
        problems.append("quant_model_description.json is missing")
    w4a8_detected = "W4A8" in description_text
    w8a8_detected = "W8A8" in description_text
    if not w4a8_detected:
        problems.append("W4A8 was not proven by quant_model_description.json")
    if w8a8_detected and not w4a8_detected:
        problems.append("W8A8 is not a single-node 8-card A2 baseline")

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        problems.append("no top-level .safetensors shards found")
    zero_size = [path.name for path in shards if path.stat().st_size == 0]
    if zero_size:
        problems.append(f"zero-size safetensors shards: {zero_size[:20]}")
    shard_bytes = sum(path.stat().st_size for path in shards)
    shard_gib = shard_bytes / 1024**3
    if shard_gib > max_shard_gib:
        warnings.append(
            f"checkpoint shards use {shard_gib:.3f} GiB, above the "
            f"{max_shard_gib:.3f} GiB caution threshold"
        )

    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        problems.append(f"multiple safetensors index files: {[p.name for p in index_files]}")
    referenced_shards: set[str] = set()
    indexed_tensors = 0
    if len(index_files) == 1:
        index = load_json(index_files[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            problems.append(f"{index_files[0].name} has no non-empty weight_map")
        else:
            indexed_tensors = len(weight_map)
            referenced_shards = {
                value for value in weight_map.values() if isinstance(value, str)
            }
            missing = sorted(referenced_shards - {path.name for path in shards})
            if missing:
                problems.append(f"index references missing shards: {missing[:20]}")
    elif len(shards) > 1:
        warnings.append(
            "multiple shards have no root index; vLLM may load them by glob, "
            "but tensor-to-shard completeness is not proven"
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "compatible": not problems,
        "model_path": str(model_path),
        "target_profile": "Atlas 800 A2, 8x Ascend 910B1, TP8+EP, W4A8",
        "quantization": {
            "required": "w4a8",
            "w4a8_detected": w4a8_detected,
            "w8a8_detected": w8a8_detected,
            "quant_model_description_present": description_path.is_file(),
        },
        "weights": {
            "shard_count": len(shards),
            "total_shard_bytes": shard_bytes,
            "total_shard_gib": round(shard_gib, 3),
            "index_present": len(index_files) == 1,
            "indexed_tensor_count": indexed_tensors,
            "referenced_shard_count": len(referenced_shards),
        },
        "topology": topology.to_dict() if topology is not None else None,
        "problems": problems,
        "warnings": warnings,
    }
    return result


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    report = audit(model_path, args.max_shard_gib)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
