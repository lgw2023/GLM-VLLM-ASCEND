#!/usr/bin/env python3
"""Shared model-topology and routed-expert helpers for DeepSeek experiments."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ModelTopology:
    model_type: str
    num_hidden_layers: int
    num_experts: int
    top_k: int
    first_k_dense_replace: int
    moe_layer_freq: int
    moe_layer_indices: tuple[int, ...]

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    @property
    def dense_layer_indices(self) -> tuple[int, ...]:
        moe = set(self.moe_layer_indices)
        return tuple(index for index in range(self.num_hidden_layers) if index not in moe)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["moe_layer_indices"] = list(self.moe_layer_indices)
        value["dense_layer_indices"] = list(self.dense_layer_indices)
        value["num_moe_layers"] = self.num_moe_layers
        return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _positive_int(config: dict[str, Any], keys: Iterable[str], label: str) -> int:
    for key in keys:
        value = config.get(key)
        if type(value) is int and value > 0:
            return value
    raise ValueError(f"cannot resolve positive integer {label} from keys {tuple(keys)}")


def topology_from_config(config: dict[str, Any]) -> ModelTopology:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("config.text_config must be an object when present")

    model_type = text_config.get("model_type", config.get("model_type"))
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("cannot resolve model_type")
    num_layers = _positive_int(text_config, ("num_hidden_layers",), "num_hidden_layers")
    num_experts = _positive_int(
        text_config,
        ("n_routed_experts", "num_experts", "num_local_experts"),
        "routed expert count",
    )
    top_k = _positive_int(
        text_config,
        ("num_experts_per_tok", "num_experts_per_token", "experts_per_token"),
        "experts per token",
    )
    first_dense = text_config.get("first_k_dense_replace", 0)
    frequency = text_config.get("moe_layer_freq", 1)
    if type(first_dense) is not int or first_dense < 0 or first_dense > num_layers:
        raise ValueError(f"invalid first_k_dense_replace={first_dense!r}")
    if type(frequency) is not int or frequency < 1:
        raise ValueError(f"invalid moe_layer_freq={frequency!r}")

    # This is the same layer predicate used by vLLM-Ascend's MoE helpers.
    moe_layers = tuple(
        index
        for index in range(num_layers)
        if index >= first_dense and index % frequency == 0
    )
    if not moe_layers:
        raise ValueError("model configuration resolves to zero MoE layers")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    return ModelTopology(
        model_type=model_type,
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        first_k_dense_replace=first_dense,
        moe_layer_freq=frequency,
        moe_layer_indices=moe_layers,
    )


def load_topology(model_path: Path) -> ModelTopology:
    return topology_from_config(load_json(model_path / "config.json"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_routed_experts(encoded: str) -> "np.ndarray":
    import numpy as np

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("routed_experts is missing or empty")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("routed_experts is not valid base64") from exc
    try:
        routes = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("routed_experts is not a valid non-pickle NPY payload") from exc
    if not isinstance(routes, np.ndarray):
        raise ValueError("decoded routed_experts payload is not an ndarray")
    return routes


def _phase_duplicate_stats(
    moe_routes: "np.ndarray",
    top_k: int,
) -> dict[str, Any]:
    import numpy as np

    if moe_routes.size == 0 or top_k <= 1:
        return {
            "rows": 0,
            "duplicate_cells": 0,
            "duplicate_cell_fraction": 0.0,
            "all_zero_cells": 0,
            "all_zero_cell_fraction": 0.0,
        }
    sorted_topk = np.sort(moe_routes, axis=-1)
    duplicate_mask = np.any(np.diff(sorted_topk, axis=-1) <= 0, axis=-1)
    all_zero_mask = np.all(moe_routes == 0, axis=-1)
    cells = int(duplicate_mask.size)
    duplicate_cells = int(duplicate_mask.sum())
    all_zero_cells = int(all_zero_mask.sum())
    return {
        "rows": int(moe_routes.shape[0]),
        "duplicate_cells": duplicate_cells,
        "duplicate_cell_fraction": duplicate_cells / cells if cells else 0.0,
        "all_zero_cells": all_zero_cells,
        "all_zero_cell_fraction": all_zero_cells / cells if cells else 0.0,
    }


def align_routes_to_usage(
    routes: "np.ndarray",
    topology: ModelTopology,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    unique_scope: str = "all",
) -> tuple["np.ndarray", dict[str, Any]]:
    """Normalize routed-expert rows to prompt+completion-1.

    Ascend All2All TP-split often returns a shorter tensor than the chat
    usage counts imply: only a prefill shard plus decode rows survive. In
    decode-tolerant modes, left-pad missing prefill rows with zeros and keep
    the trailing decode rows.
    """
    import numpy as np

    expected_rows = prompt_tokens + completion_tokens - 1
    expected_tail = (topology.num_hidden_layers, topology.top_k)
    if routes.ndim != 3 or tuple(routes.shape[1:]) != expected_tail:
        raise ValueError(
            f"route trailing shape mismatch: expected (*, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {routes.shape}"
        )
    if not np.issubdtype(routes.dtype, np.integer):
        raise ValueError(f"route dtype must be integer, got {routes.dtype}")

    repair: dict[str, Any] = {
        "expected_rows": expected_rows,
        "raw_rows": int(routes.shape[0]),
        "repaired": False,
        "prefill_trusted": True,
        "decode_trusted": True,
    }
    if routes.shape[0] == expected_rows:
        return routes, repair

    decode_rows = completion_tokens - 1
    if unique_scope == "all":
        raise ValueError(
            f"route shape mismatch: expected {(expected_rows, *expected_tail)}, "
            f"got {routes.shape}. Prefill row loss under Ascend TP-split is "
            "common; use --unique-scope decode or rebuild the v6 capture image."
        )
    if routes.shape[0] < decode_rows:
        raise ValueError(
            f"route tensor too short for decode rows: need >= {decode_rows}, "
            f"got {routes.shape[0]}"
        )
    if routes.shape[0] > expected_rows:
        raise ValueError(
            f"route tensor longer than usage implies: expected {expected_rows}, "
            f"got {routes.shape[0]}"
        )

    # Short tensor: keep trailing decode rows, left-pad prefill with zeros.
    aligned = np.zeros((expected_rows, *expected_tail), dtype=routes.dtype)
    raw_prefill = int(routes.shape[0] - decode_rows)
    if raw_prefill > 0:
        aligned[prompt_tokens - raw_prefill : prompt_tokens] = routes[:raw_prefill]
    if decode_rows > 0:
        aligned[prompt_tokens:] = routes[-decode_rows:]
    repair.update(
        {
            "repaired": True,
            "prefill_trusted": False,
            "raw_prefill_rows": raw_prefill,
            "padded_prefill_rows": prompt_tokens - raw_prefill,
        }
    )
    return aligned, repair


def validate_routes(
    routes: "np.ndarray",
    topology: ModelTopology,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    require_unique_topk: bool = True,
    unique_scope: str = "all",
) -> dict[str, Any]:
    import numpy as np

    if prompt_tokens < 1 or completion_tokens < 1:
        raise ValueError("prompt and completion token counts must be positive")
    if unique_scope not in {"all", "decode", "none"}:
        raise ValueError("unique_scope must be one of: all, decode, none")

    routes, repair = align_routes_to_usage(
        routes,
        topology,
        prompt_tokens,
        completion_tokens,
        unique_scope=unique_scope,
    )
    expected_rows = prompt_tokens + completion_tokens - 1
    expected_shape = (expected_rows, topology.num_hidden_layers, topology.top_k)
    if routes.shape != expected_shape:
        raise ValueError(f"route shape mismatch after align: expected {expected_shape}, got {routes.shape}")

    dense_layers = topology.dense_layer_indices
    if dense_layers and not np.all(routes[:, dense_layers, :] == 0):
        raise ValueError(f"dense layers contain non-zero route data: {dense_layers}")

    moe_routes = routes[:, topology.moe_layer_indices, :]
    if moe_routes.size == 0:
        raise ValueError("MoE route tensor is empty")
    minimum = int(moe_routes.min())
    maximum = int(moe_routes.max())
    if minimum < 0 or maximum >= topology.num_experts:
        raise ValueError(
            f"expert IDs must be in [0, {topology.num_experts - 1}], "
            f"got [{minimum}, {maximum}]"
        )

    prefill = moe_routes[:prompt_tokens]
    decode = moe_routes[prompt_tokens:]
    uniqueness = {
        "unique_topk": True,
        "unique_scope": unique_scope,
        "total": _phase_duplicate_stats(moe_routes, topology.top_k),
        "prefill": _phase_duplicate_stats(prefill, topology.top_k),
        "decode": _phase_duplicate_stats(decode, topology.top_k),
    }
    if topology.top_k > 1:
        sorted_all = np.sort(moe_routes, axis=-1)
        uniqueness["unique_topk"] = bool(np.all(np.diff(sorted_all, axis=-1) > 0))
        if decode.size:
            sorted_decode = np.sort(decode, axis=-1)
            uniqueness["unique_topk_decode"] = bool(np.all(np.diff(sorted_decode, axis=-1) > 0))
        else:
            uniqueness["unique_topk_decode"] = True

        enforce = require_unique_topk and unique_scope != "none"
        if enforce:
            if unique_scope == "decode":
                ok = bool(uniqueness["unique_topk_decode"])
                label = "decode"
            else:
                ok = bool(uniqueness["unique_topk"])
                label = "all phases"
            if not ok:
                raise ValueError(
                    "top-k expert IDs are not unique for every token/layer "
                    f"(scope={label}); "
                    f"duplicate_cell_fraction="
                    f"{uniqueness['total']['duplicate_cell_fraction']:.4f}, "
                    f"all_zero_cell_fraction="
                    f"{uniqueness['total']['all_zero_cell_fraction']:.4f}, "
                    f"prefill_dup={uniqueness['prefill']['duplicate_cell_fraction']:.4f}, "
                    f"decode_dup={uniqueness['decode']['duplicate_cell_fraction']:.4f}. "
                    "Prefill zeros/short tensors with clean decode usually mean "
                    "Ascend All2All TP-split. Rebuild the v8 capture image, or "
                    "pass --unique-scope decode / --allow-duplicate-topk."
                )

    # For decode-trusted mode, diversity checks must look at decode only when
    # prefill was zero-padded or truncated by TP-split.
    diversity_routes = decode if unique_scope == "decode" and decode.size else moe_routes
    covered_experts = int(np.unique(diversity_routes).size)
    unique_route_tuples = int(
        np.unique(diversity_routes.reshape(-1, topology.top_k), axis=0).shape[0]
    )

    return {
        "shape": list(routes.shape),
        "dtype": str(routes.dtype),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_rows": prompt_tokens,
        "decode_rows": completion_tokens - 1,
        "minimum_expert_id": minimum,
        "maximum_expert_id": maximum,
        "covered_experts": covered_experts,
        "unique_route_tuples": unique_route_tuples,
        "unique_topk": uniqueness["unique_topk"],
        "unique_topk_decode": uniqueness.get("unique_topk_decode", uniqueness["unique_topk"]),
        "uniqueness": uniqueness,
        "shape_repair": repair,
        "aligned_routes": routes,
    }


def count_assignments(
    routes: "np.ndarray",
    topology: ModelTopology,
    prompt_tokens: int,
) -> "np.ndarray":
    """Return counts with shape [phase(total,prefill,decode), moe_layer, expert]."""
    import numpy as np

    moe_routes = routes[:, topology.moe_layer_indices, :]
    phase_routes = (moe_routes, moe_routes[:prompt_tokens], moe_routes[prompt_tokens:])
    counts = np.zeros((3, topology.num_moe_layers, topology.num_experts), dtype=np.int64)
    for phase_index, values in enumerate(phase_routes):
        for layer_index in range(topology.num_moe_layers):
            counts[phase_index, layer_index] = np.bincount(
                values[:, layer_index, :].reshape(-1),
                minlength=topology.num_experts,
            )
    return counts
