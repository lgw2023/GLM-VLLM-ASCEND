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


def quantization_summary(
    model_path: Path,
    config: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    description_path = model_path / "quant_model_description.json"
    sources: list[tuple[str, Any]] = []
    config_quant = config.get("quantization_config")
    text_config = config.get("text_config")
    if config_quant is None and isinstance(text_config, dict):
        config_quant = text_config.get("quantization_config")
    if config_quant is not None:
        sources.append(("config.json:quantization_config", config_quant))
    if description_path.is_file():
        sources.append((description_path.name, load_json(description_path)))

    marker_counts: Counter[str] = Counter()
    for _, source in sources:
        for scalar in walk_scalars(source):
            for marker in QUANT_MARKER.findall(scalar):
                marker_counts[marker.upper()] += 1

    quant_method: str | None = None
    if isinstance(config_quant, dict):
        value = config_quant.get("quant_method")
        if isinstance(value, str) and value.strip():
            quant_method = value.strip().lower()

    expert_dtype: str | None = None
    expert_dtype_source: str | None = None
    for source_name, source in (
        ("config.json", config),
        ("config.json:text_config", text_config),
    ):
        if not isinstance(source, dict):
            continue
        value = source.get("expert_dtype")
        if isinstance(value, str) and value.strip():
            expert_dtype = value.strip().lower()
            expert_dtype_source = source_name
            break

    native_fp8 = model_type == "deepseek_v4" and quant_method in {
        "fp8",
        "deepseek_v4_fp8",
    }
    if native_fp8 and expert_dtype is None:
        # vLLM 0.22.1's DeepseekV4FP8Config defaults a missing expert_dtype
        # to fp4. Keep that default explicit in the audit contract.
        expert_dtype = "fp4"
        expert_dtype_source = "vllm-0.22.1-default"

    explicit_w4a8 = marker_counts["W4A8"] > 0
    explicit_w8a8 = marker_counts["W8A8"] > 0
    native_fp8_fp4 = native_fp8 and expert_dtype == "fp4"
    if description_path.is_file() and explicit_w4a8:
        deployment_profile = "modelslim_w4a8"
        recommended_quantization = "ascend"
        expert_quantization = "w4a8"
    elif description_path.is_file() and explicit_w8a8:
        deployment_profile = "modelslim_w8a8"
        recommended_quantization = "ascend"
        expert_quantization = "w8a8"
    elif native_fp8_fp4:
        deployment_profile = "deepseek_v4_native_fp8_fp4"
        # Match config.json exactly. vLLM-Ascend registers both names to its
        # FP8 config, which selects the W4A8 MoE scheme for FusedMoE layers.
        recommended_quantization = quant_method
        expert_quantization = "w4a8_mxfp4"
    else:
        deployment_profile = "unsupported_or_unproven"
        recommended_quantization = None
        expert_quantization = None

    evidence: list[str] = []
    if explicit_w4a8:
        evidence.append("explicit_w4a8_marker")
    if explicit_w8a8:
        evidence.append("explicit_w8a8_marker")
    if native_fp8_fp4:
        evidence.append("deepseek_v4_fp8_linear_plus_fp4_experts")
    return {
        "sources": [name for name, _ in sources],
        "markers": dict(sorted(marker_counts.items())),
        "quant_method": quant_method,
        "expert_dtype": expert_dtype,
        "expert_dtype_source": expert_dtype_source,
        "explicit_w4a8_marker_detected": explicit_w4a8,
        "explicit_w8a8_marker_detected": explicit_w8a8,
        "native_fp8_fp4_detected": native_fp8_fp4,
        "w4a8_detected": explicit_w4a8 or native_fp8_fp4,
        "w8a8_detected": explicit_w8a8,
        "w4a8_evidence": evidence,
        "expert_quantization": expert_quantization,
        "deployment_profile": deployment_profile,
        "recommended_vllm_quantization": recommended_quantization,
        "quant_model_description_present": description_path.is_file(),
    }


def weight_summary(
    model_path: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    shards = sorted(model_path.rglob("*.safetensors"))
    shard_by_name = {
        path.relative_to(model_path).as_posix(): path
        for path in shards
    }
    zero_size = [
        path.relative_to(model_path).as_posix()
        for path in shards
        if path.stat().st_size == 0
    ]
    if not shards:
        problems.append("no .safetensors files found")
    if zero_size:
        problems.append(f"zero-size safetensors files: {zero_size}")

    index_paths = sorted(model_path.glob("*.safetensors.index.json"))
    index_path: Path | None = None
    if len(index_paths) == 1:
        index_path = index_paths[0]
    elif len(index_paths) > 1:
        names = [path.name for path in index_paths]
        problems.append(f"multiple safetensors index files found: {names}")

    referenced_files: set[str] = set()
    tensor_count = 0
    if index_path is not None:
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            problems.append(f"{index_path.name} has no non-empty weight_map")
        else:
            tensor_count = len(weight_map)
            referenced_files = {
                value for value in weight_map.values() if isinstance(value, str)
            }
            invalid_values = sum(
                1 for value in weight_map.values() if not isinstance(value, str)
            )
            if invalid_values:
                problems.append(
                    f"index weight_map has {invalid_values} non-string shard values"
                )
            missing = sorted(referenced_files - set(shard_by_name))
            if missing:
                problems.append(f"index references missing shards: {missing}")
    elif len(shards) > 1:
        warnings.append(
            "multiple safetensors files found without a root "
            "*.safetensors.index.json; vLLM can load top-level files by glob, "
            "but the audit cannot prove a tensor-to-shard manifest"
        )

    if referenced_files:
        active_files = referenced_files & set(shard_by_name)
        unreferenced_files = set(shard_by_name) - referenced_files
    else:
        active_files = set(shard_by_name)
        unreferenced_files = set()
    if unreferenced_files:
        warnings.append(
            f"{len(unreferenced_files)} safetensors shards are not referenced by "
            f"{index_path.name if index_path else 'an index'}; top-level duplicates "
            "are filtered by vLLM and nested optional files are not globbed by its "
            "default loader; do not delete them"
        )

    total_shard_bytes = sum(path.stat().st_size for path in shards)
    active_shard_bytes = sum(
        shard_by_name[name].stat().st_size for name in active_files
    )
    unreferenced_shard_bytes = sum(
        shard_by_name[name].stat().st_size for name in unreferenced_files
    )

    return (
        {
            "shard_count": len(shards),
            "total_shard_bytes": total_shard_bytes,
            "total_shard_gib": round(total_shard_bytes / 1024**3, 3),
            "index_present": index_path is not None,
            "index_name": index_path.name if index_path else None,
            "index_candidates": [path.name for path in index_paths],
            "indexed_tensor_count": tensor_count,
            "referenced_shard_count": len(referenced_files),
            "active_shard_count": len(active_files),
            "active_shard_bytes": active_shard_bytes,
            "active_shard_gib": round(active_shard_bytes / 1024**3, 3),
            "unreferenced_shard_count": len(unreferenced_files),
            "unreferenced_shard_bytes": unreferenced_shard_bytes,
            "unreferenced_shard_gib": round(
                unreferenced_shard_bytes / 1024**3,
                3,
            ),
            "unreferenced_shards_first20": sorted(unreferenced_files)[:20],
            "nested_shard_count": sum(
                1 for path in shards if path.parent != model_path
            ),
        },
        problems,
        warnings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-model-type", default="")
    parser.add_argument("--require-w4a8", action="store_true")
    parser.add_argument(
        "--require-expert-quantization",
        choices=("w4a8", "w8a8"),
        default="",
    )
    parser.add_argument("--target-soc", default="")
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
    weights, weight_problems, warnings = weight_summary(model_path)
    problems.extend(weight_problems)
    quantization = quantization_summary(model_path, config, topology.model_type)

    required_expert_quantization = args.require_expert_quantization
    if args.require_w4a8:
        if required_expert_quantization not in ("", "w4a8"):
            raise ValueError(
                "--require-w4a8 conflicts with "
                f"--require-expert-quantization={required_expert_quantization}"
            )
        required_expert_quantization = "w4a8"

    target_soc = re.sub(r"[^A-Z0-9]", "", args.target_soc.upper())
    soc_compatible: bool | None = None
    soc_evidence: list[str] = []
    if target_soc:
        if (
            quantization["deployment_profile"]
            == "deepseek_v4_native_fp8_fp4"
            and "910B" in target_soc
        ):
            soc_compatible = False
            soc_evidence.append(
                "native MXFP4 uses float4_e2m1fn_x2 customize_dtype, which "
                "Ascend 910B does not support"
            )
        elif (
            quantization["deployment_profile"] == "modelslim_w8a8"
            and "910B" in target_soc
        ):
            soc_compatible = True
            soc_evidence.append(
                "vLLM-Ascend documents DeepSeek-V4-Flash W8A8 on one "
                "8-card Atlas 800 A2 node"
            )

    if args.require_model_type and topology.model_type != args.require_model_type:
        problems.append(
            f"model_type must be {args.require_model_type!r}, got {topology.model_type!r}"
        )
    actual_expert_quantization = quantization["expert_quantization"]
    if required_expert_quantization == "w4a8" and actual_expert_quantization not in {
        "w4a8",
        "w4a8_mxfp4",
    }:
        problems.append(
            "W4A8 Expert execution was not proven by an explicit ModelSlim "
            "marker or a DeepSeek-V4 FP8+FP4 configuration"
        )
    if (
        required_expert_quantization == "w8a8"
        and actual_expert_quantization != "w8a8"
    ):
        problems.append(
            "W8A8 Expert execution was not proven by quant_model_description.json"
        )
    if soc_compatible is False:
        problems.append(
            f"deployment profile {quantization['deployment_profile']!r} is "
            f"incompatible with target SoC {args.target_soc!r}"
        )

    report = {
        "schema_version": 4,
        "model_path": str(model_path),
        "topology": topology.to_dict(),
        "quantization": quantization,
        "hardware": {
            "target_soc": args.target_soc or None,
            "normalized_target_soc": target_soc or None,
            "soc_compatible": soc_compatible,
            "evidence": soc_evidence,
        },
        "weights": weights,
        "warnings": warnings,
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
