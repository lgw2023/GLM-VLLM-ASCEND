#!/usr/bin/env python3
"""Validate the pinned Eco-Tech GLM-5.2 W8A8 ModelScope snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CONFIG: dict[str, Any] = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
}
PINNED_REVISION = "edd93687ef1c3417d0b92e2cd01cf67e9e9c0039"
CONFIG_SHA256 = "817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073"
QUANT_DESCRIPTION_FILE = "quant_model_description.json"
QUANT_DESCRIPTION_SHA256 = "3386f968cd7049fe95f896c1a1aeacaa5c1c0659ac2ed9a42cd783cc48ef29ba"
INDEX_FILE = "quant_model_weights.safetensors.index.json"
INDEX_SHA256 = "dfa97fa50b5e675ff6cea6ddeae3110795b6b7e971e6dc9cf565a4005fcb079c"
EXPECTED_SHARD_COUNT = 182
INDEX_TOTAL_EXCLUDED_SHARDS = frozenset({"rot.safetensors"})
EXPECTED_INDEX_TOTAL_SIZE = 773_778_904_680
EXPECTED_INDEX_TOTAL_EXCLUDED_TENSOR_BYTES = 75_497_472
EXPECTED_TOTAL_TENSOR_BYTES = (
    EXPECTED_INDEX_TOTAL_SIZE + EXPECTED_INDEX_TOTAL_EXCLUDED_TENSOR_BYTES
)
EXPECTED_TOTAL_SHARD_BYTES = 773_876_016_944
MAX_SAFETENSORS_HEADER_BYTES = 512 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_safetensors_file(
    path: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    file_size = path.stat().st_size
    if file_size < 16:
        raise ValueError(f"safetensors file is truncated: {path.name}")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
        if (
            header_length < 2
            or header_length > MAX_SAFETENSORS_HEADER_BYTES
            or header_length > file_size - 8
        ):
            raise ValueError(
                f"invalid safetensors header length in {path.name}: {header_length}"
            )
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header JSON in {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path.name}")

    data_bytes = file_size - 8 - header_length
    tensor_sizes: dict[str, int] = {}
    tensor_ranges: list[tuple[int, int, str]] = []
    for tensor_name, tensor in header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid tensor metadata in {path.name}: {tensor_name}")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(value) is int for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_bytes
        ):
            raise ValueError(f"invalid tensor offsets/schema in {path.name}: {tensor_name}")
        start, end = offsets
        tensor_sizes[tensor_name] = end - start
        tensor_ranges.append((start, end, tensor_name))
    if not tensor_sizes:
        raise ValueError(f"safetensors file has no tensor entries: {path.name}")

    previous_end = 0
    for start, end, tensor_name in sorted(tensor_ranges):
        if start != previous_end:
            relation = "overlap" if start < previous_end else "gap"
            raise ValueError(
                f"safetensors tensor data has a {relation} in {path.name}: "
                f"{tensor_name} starts at {start}, expected {previous_end}"
            )
        previous_end = end
    if previous_end != data_bytes:
        raise ValueError(
            f"safetensors tensor data has a trailing gap in {path.name}: "
            f"last tensor ends at {previous_end}, payload has {data_bytes} bytes"
        )

    summary = {
        "file_bytes": file_size,
        "header_bytes": header_length,
        "tensor_data_bytes": data_bytes,
        "tensor_count": len(tensor_sizes),
    }
    return summary, tensor_sizes


def validate_model(
    model_path: Path,
    *,
    enforce_pinned_metadata: bool = True,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    quant_path = model_path / QUANT_DESCRIPTION_FILE
    index_path = model_path / INDEX_FILE
    for required_path in (config_path, quant_path, index_path):
        if not required_path.is_file():
            raise ValueError(f"missing {required_path}")

    config = load_json_object(config_path)
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"GLM-5.2 config mismatch: {mismatches}")

    metadata_hashes = {
        "config.json": sha256_file(config_path),
        QUANT_DESCRIPTION_FILE: sha256_file(quant_path),
        INDEX_FILE: sha256_file(index_path),
    }
    if enforce_pinned_metadata:
        expected_hashes = {
            "config.json": CONFIG_SHA256,
            QUANT_DESCRIPTION_FILE: QUANT_DESCRIPTION_SHA256,
            INDEX_FILE: INDEX_SHA256,
        }
        if metadata_hashes != expected_hashes:
            raise ValueError(
                f"metadata does not match pinned ModelScope revision {PINNED_REVISION}: "
                f"expected={expected_hashes}, actual={metadata_hashes}"
            )

    quant_description = load_json_object(quant_path)
    quant_counts = Counter(
        value for value in quant_description.values() if isinstance(value, str)
    )
    if quant_counts["W8A8_DYNAMIC"] == 0 or quant_counts["W8A8"] == 0:
        raise ValueError(
            "quant_model_description.json does not contain both W8A8_DYNAMIC and W8A8 entries"
        )

    index = load_json_object(index_path)
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no non-empty weight_map")
    if not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int:
        raise ValueError("safetensors index has no integer metadata.total_size")

    shard_names: list[str] = []
    indexed_tensors_by_shard: dict[str, set[str]] = {}
    for tensor_name, value in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"safetensors index contains an invalid tensor name: {tensor_name!r}")
        if not isinstance(value, str) or not value.endswith(".safetensors"):
            raise ValueError(f"safetensors index contains an invalid shard name: {value!r}")
        relative = Path(value)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != value:
            raise ValueError(f"safetensors shard must stay inside model root: {value!r}")
        shard_names.append(value)
        indexed_tensors_by_shard.setdefault(value, set()).add(tensor_name)
    unique_shards = sorted(set(shard_names))
    if enforce_pinned_metadata and len(unique_shards) != EXPECTED_SHARD_COUNT:
        raise ValueError(
            f"pinned shard count mismatch: expected {EXPECTED_SHARD_COUNT}, got {len(unique_shards)}"
        )

    shard_summaries: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    total_shard_bytes = 0
    total_tensors = 0
    total_header_bytes = 0
    total_tensor_bytes = 0
    for shard_name in unique_shards:
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            missing.append(shard_name)
            continue
        summary, tensor_sizes = validate_safetensors_file(shard_path)
        indexed_tensor_names = indexed_tensors_by_shard[shard_name]
        actual_tensor_names = set(tensor_sizes)
        missing_tensors = sorted(indexed_tensor_names - actual_tensor_names)
        unexpected_tensors = sorted(actual_tensor_names - indexed_tensor_names)
        if missing_tensors or unexpected_tensors:
            raise ValueError(
                f"safetensors index/header mismatch in {shard_name}: "
                f"missing={missing_tensors[:10]}, unexpected={unexpected_tensors[:10]}"
            )
        shard_summaries[shard_name] = summary
        total_shard_bytes += summary["file_bytes"]
        total_header_bytes += summary["header_bytes"]
        total_tensor_bytes += summary["tensor_data_bytes"]
        total_tensors += summary["tensor_count"]
    if missing:
        raise ValueError(f"indexed safetensors shards are missing: {missing}")

    excluded_shards = sorted(INDEX_TOTAL_EXCLUDED_SHARDS.intersection(shard_summaries))
    index_total_excluded_tensor_bytes = sum(
        shard_summaries[shard_name]["tensor_data_bytes"]
        for shard_name in excluded_shards
    )
    index_total_covered_tensor_bytes = (
        total_tensor_bytes - index_total_excluded_tensor_bytes
    )
    if index_total_covered_tensor_bytes != metadata["total_size"]:
        raise ValueError(
            "index metadata tensor byte total mismatch: "
            f"index={metadata['total_size']}, "
            f"actual={index_total_covered_tensor_bytes}, "
            f"excluded_shards={excluded_shards}"
        )
    if enforce_pinned_metadata:
        if metadata["total_size"] != EXPECTED_INDEX_TOTAL_SIZE:
            raise ValueError(
                "pinned index metadata total_size mismatch: "
                f"expected {EXPECTED_INDEX_TOTAL_SIZE}, got {metadata['total_size']}"
            )
        if set(excluded_shards) != INDEX_TOTAL_EXCLUDED_SHARDS:
            raise ValueError(
                "pinned index-total-excluded shard mismatch: "
                f"expected {sorted(INDEX_TOTAL_EXCLUDED_SHARDS)}, got {excluded_shards}"
            )
        if (
            index_total_excluded_tensor_bytes
            != EXPECTED_INDEX_TOTAL_EXCLUDED_TENSOR_BYTES
        ):
            raise ValueError(
                "pinned index-total-excluded tensor byte total mismatch: "
                f"expected {EXPECTED_INDEX_TOTAL_EXCLUDED_TENSOR_BYTES}, "
                f"got {index_total_excluded_tensor_bytes}"
            )
        if total_tensor_bytes != EXPECTED_TOTAL_TENSOR_BYTES:
            raise ValueError(
                "pinned tensor byte total mismatch: "
                f"expected {EXPECTED_TOTAL_TENSOR_BYTES}, got {total_tensor_bytes}"
            )
        if total_shard_bytes != EXPECTED_TOTAL_SHARD_BYTES:
            raise ValueError(
                "pinned shard file byte total mismatch: "
                f"expected {EXPECTED_TOTAL_SHARD_BYTES}, got {total_shard_bytes}"
            )

    return {
        "valid": True,
        "model_path": str(model_path),
        "pinned_revision": PINNED_REVISION,
        "pinned_metadata_enforced": enforce_pinned_metadata,
        "expected_config": EXPECTED_CONFIG,
        "metadata_sha256": metadata_hashes,
        "quant_type_counts": {
            key: quant_counts[key] for key in ("W8A8", "W8A8_DYNAMIC", "FLOAT")
        },
        "tensor_count_from_index": len(weight_map),
        "safetensors_tensor_count": total_tensors,
        "shard_count": len(unique_shards),
        "index_metadata_total_size": metadata["total_size"],
        "index_total_covered_tensor_bytes": index_total_covered_tensor_bytes,
        "index_total_excluded_shards": excluded_shards,
        "index_total_excluded_tensor_bytes": index_total_excluded_tensor_bytes,
        "total_tensor_bytes": total_tensor_bytes,
        "total_shard_bytes": total_shard_bytes,
        "total_safetensors_header_bytes": total_header_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    result = validate_model(args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
