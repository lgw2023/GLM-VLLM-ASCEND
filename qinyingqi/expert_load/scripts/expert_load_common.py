#!/usr/bin/env python3
"""Shared validation and statistics for GLM-5.2 routed-expert captures."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import numpy as np


NUM_LAYERS = 78
NUM_DENSE_LAYERS = 3
NUM_MOE_LAYERS = 75
NUM_LOGICAL_EXPERTS = 256
TOP_K = 8
STRICT_TOP_20_PERCENT_EXPERTS = 51
ROUNDING_TOP_20_PERCENT_EXPERTS = 52
PHASE_NAMES = ("prefill", "decode", "combined")


class RouteCaptureError(ValueError):
    """A response does not meet the GLM-5.2 route-capture contract."""


def _require_token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RouteCaptureError(f"{field} must be a non-empty list")
    if not all(type(token_id) is int for token_id in value):
        raise RouteCaptureError(f"{field} must contain only integer token IDs")
    return value


def decode_routed_experts(encoded: Any) -> np.ndarray:
    """Decode the base64 NumPy route tensor returned by vLLM."""
    if not isinstance(encoded, str) or not encoded:
        raise RouteCaptureError("routed_experts is missing or empty")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RouteCaptureError("routed_experts is not valid base64") from exc
    try:
        routes = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RouteCaptureError("routed_experts is not a valid non-pickle .npy") from exc
    if not isinstance(routes, np.ndarray):
        raise RouteCaptureError("decoded routed_experts is not an ndarray")
    return routes


def extract_validated_routes(
    response: dict[str, Any], expected_model: str | None = None
) -> tuple[np.ndarray, int, int]:
    """Validate one non-streaming response and return routes and token counts."""
    if expected_model is not None and response.get("model") != expected_model:
        raise RouteCaptureError(
            f"response model mismatch: expected {expected_model!r}, "
            f"got {response.get('model')!r}"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RouteCaptureError("expected exactly one response choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise RouteCaptureError("choices[0] must be the index-0 response object")

    prompt_token_ids = _require_token_ids(
        response.get("prompt_token_ids"), "prompt_token_ids"
    )
    output_token_ids = _require_token_ids(choice.get("token_ids"), "choices[0].token_ids")
    prompt_tokens = len(prompt_token_ids)
    output_tokens = len(output_token_ids)
    expected_rows = prompt_tokens + output_tokens - 1
    if expected_rows <= 0:
        raise RouteCaptureError("route tensor must contain at least one token row")

    routes = decode_routed_experts(choice.get("routed_experts"))
    if routes.dtype != np.dtype(np.uint8):
        raise RouteCaptureError(f"route dtype must be uint8, got {routes.dtype}")
    expected_shape = (expected_rows, NUM_LAYERS, TOP_K)
    if routes.shape != expected_shape:
        raise RouteCaptureError(
            f"route shape mismatch: expected {expected_shape}, got {routes.shape}"
        )
    if not np.all(routes[:, :NUM_DENSE_LAYERS, :] == 0):
        raise RouteCaptureError("dense layers 0..2 contain routed-expert data")

    moe_routes = routes[:, NUM_DENSE_LAYERS:, :]
    if moe_routes.size == 0:
        raise RouteCaptureError("MoE route tensor is empty")
    min_expert = int(moe_routes.min())
    max_expert = int(moe_routes.max())
    if min_expert < 0 or max_expert >= NUM_LOGICAL_EXPERTS:
        raise RouteCaptureError(
            "logical expert IDs must be in [0, 255], got "
            f"[{min_expert}, {max_expert}]"
        )
    sorted_topk = np.sort(moe_routes, axis=-1)
    if not np.all(np.diff(sorted_topk, axis=-1) > 0):
        raise RouteCaptureError("top-8 routed experts are not unique per token and layer")
    return routes, prompt_tokens, output_tokens


def count_route_assignments(routes: np.ndarray, prompt_tokens: int) -> np.ndarray:
    """Return prefill, decode, and combined assignment counts by layer/expert.

    The result has shape ``(3, 75, 256)``. Index 0 is prefill, index 1 is
    decode, and index 2 is their sum. A count is a token-expert assignment,
    not a token that touched at least one member of an expert set.
    """
    if routes.ndim != 3 or routes.shape[1:] != (NUM_LAYERS, TOP_K):
        raise RouteCaptureError(
            "routes must have shape (tokens, 78, 8), got " f"{routes.shape}"
        )
    if not 0 < prompt_tokens <= routes.shape[0]:
        raise RouteCaptureError(
            "prompt token count must be within route rows, got "
            f"{prompt_tokens} for {routes.shape[0]} rows"
        )

    counts = np.zeros((3, NUM_MOE_LAYERS, NUM_LOGICAL_EXPERTS), dtype=np.int64)
    phase_rows = (routes[:prompt_tokens], routes[prompt_tokens:])
    for phase_index, phase_routes in enumerate(phase_rows):
        if phase_routes.size == 0:
            continue
        for moe_layer_index in range(NUM_MOE_LAYERS):
            expert_ids = phase_routes[
                :, NUM_DENSE_LAYERS + moe_layer_index, :
            ].reshape(-1)
            counts[phase_index, moe_layer_index] = np.bincount(
                expert_ids, minlength=NUM_LOGICAL_EXPERTS
            )
    counts[2] = counts[0] + counts[1]
    return counts


def ranked_expert_ids(counts: np.ndarray) -> np.ndarray:
    """Return logical expert IDs sorted by decreasing assignment count."""
    if counts.shape != (NUM_LOGICAL_EXPERTS,):
        raise ValueError(f"expected 256 counts, got {counts.shape}")
    return np.argsort(-counts, kind="stable")


def distribution_metrics(counts: np.ndarray) -> dict[str, float | int | bool | None]:
    """Compute the assignment-based 20%-of-experts summary for one scope."""
    if counts.shape != (NUM_LOGICAL_EXPERTS,):
        raise ValueError(f"expected 256 counts, got {counts.shape}")
    total = int(counts.sum())
    if total == 0:
        return {
            "assignment_count": 0,
            "active_experts": 0,
            "top1_assignment_share": None,
            "top51_assignment_share": None,
            "top52_assignment_share": None,
            "k90": None,
            "k90_within_top51": False,
            "normalized_entropy": None,
            "gini": None,
        }

    ranked_counts = counts[ranked_expert_ids(counts)]
    cumulative = np.cumsum(ranked_counts, dtype=np.int64)
    k90 = int(np.flatnonzero(cumulative * 10 >= total * 9)[0] + 1)
    probabilities = ranked_counts[ranked_counts > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / float(np.log(NUM_LOGICAL_EXPERTS))
    ascending_counts = np.sort(counts)
    ranks = np.arange(1, NUM_LOGICAL_EXPERTS + 1, dtype=np.int64)
    gini = float(
        (2.0 * np.dot(ranks, ascending_counts) / (NUM_LOGICAL_EXPERTS * total))
        - (NUM_LOGICAL_EXPERTS + 1.0) / NUM_LOGICAL_EXPERTS
    )
    return {
        "assignment_count": total,
        "active_experts": int(np.count_nonzero(counts)),
        "top1_assignment_share": float(ranked_counts[0] / total),
        "top51_assignment_share": float(
            ranked_counts[:STRICT_TOP_20_PERCENT_EXPERTS].sum() / total
        ),
        "top52_assignment_share": float(
            ranked_counts[:ROUNDING_TOP_20_PERCENT_EXPERTS].sum() / total
        ),
        "k90": k90,
        "k90_within_top51": k90 <= STRICT_TOP_20_PERCENT_EXPERTS,
        "normalized_entropy": normalized_entropy,
        "gini": gini,
    }


def hot_expert_set(
    counts: np.ndarray, budget: int = STRICT_TOP_20_PERCENT_EXPERTS
) -> set[int]:
    """Return the hottest logical expert IDs for a positive assignment vector."""
    if not 0 < budget <= NUM_LOGICAL_EXPERTS:
        raise ValueError(f"expert budget must be in 1..256, got {budget}")
    if int(counts.sum()) == 0:
        return set()
    return set(int(expert_id) for expert_id in ranked_expert_ids(counts)[:budget])
