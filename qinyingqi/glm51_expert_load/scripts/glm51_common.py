#!/usr/bin/env python3
"""Small topology and route helpers for the GLM-5.1 experiment."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    def dense_layer_indices(self) -> tuple[int, ...]:
        moe = set(self.moe_layer_indices)
        return tuple(i for i in range(self.num_hidden_layers) if i not in moe)

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["moe_layer_indices"] = list(self.moe_layer_indices)
        result["dense_layer_indices"] = list(self.dense_layer_indices)
        result["num_moe_layers"] = self.num_moe_layers
        return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def positive_int(config: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = config.get(key)
        if type(value) is int and value > 0:
            return value
    raise ValueError(f"cannot resolve a positive integer from {keys}")


def topology_from_config(config: dict[str, Any]) -> ModelTopology:
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("config.text_config must be an object")
    model_type = text.get("model_type", config.get("model_type"))
    if model_type != "glm_moe_dsa":
        raise ValueError(f"expected model_type='glm_moe_dsa', got {model_type!r}")
    layers = positive_int(text, "num_hidden_layers")
    experts = positive_int(text, "n_routed_experts", "num_experts")
    top_k = positive_int(text, "num_experts_per_tok", "num_experts_per_token")
    first_dense = text.get("first_k_dense_replace", 0)
    frequency = text.get("moe_layer_freq", 1)
    if type(first_dense) is not int or not 0 <= first_dense <= layers:
        raise ValueError(f"invalid first_k_dense_replace={first_dense!r}")
    if type(frequency) is not int or frequency < 1:
        raise ValueError(f"invalid moe_layer_freq={frequency!r}")
    moe_layers = tuple(
        index
        for index in range(layers)
        if index >= first_dense and index % frequency == 0
    )
    if not moe_layers or top_k > experts:
        raise ValueError("invalid MoE topology")
    return ModelTopology(
        model_type=model_type,
        num_hidden_layers=layers,
        num_experts=experts,
        top_k=top_k,
        first_k_dense_replace=first_dense,
        moe_layer_freq=frequency,
        moe_layer_indices=moe_layers,
    )


def load_topology(model_path: Path) -> ModelTopology:
    return topology_from_config(load_json(model_path / "config.json"))


def decode_routes(encoded: str):
    import numpy as np

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("routed_experts is missing")
    try:
        payload = base64.b64decode(encoded, validate=True)
        routes = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ValueError("routed_experts is not a valid base64 NPY payload") from exc
    if not isinstance(routes, np.ndarray):
        raise ValueError("decoded routed_experts is not an ndarray")
    return routes

