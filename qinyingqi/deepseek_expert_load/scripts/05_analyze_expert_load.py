#!/usr/bin/env python3
"""Analyze DeepSeek route aggregates and test expert-load concentration."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Aggregate:
    source_dir: Path
    benchmark: str
    phase_names: tuple[str, ...]
    counts: np.ndarray
    moe_layer_indices: np.ndarray
    num_hidden_layers: int
    num_experts: int
    top_k: int
    request_count: int
    prompt_tokens: int
    completion_tokens: int


def scalar(data: Any, name: str) -> int:
    value = np.asarray(data[name]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{name} must contain one value")
    return int(value[0])


def load_aggregate(source_dir: Path) -> Aggregate:
    path = source_dir / "aggregate-counts.npz"
    if not path.is_file():
        raise FileNotFoundError(f"aggregate not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        phase_names = tuple(str(value) for value in data["phase_names"].tolist())
        counts = np.asarray(data["counts"], dtype=np.int64)
        moe_layers = np.asarray(data["moe_layer_indices"], dtype=np.int64)
        benchmark = str(np.asarray(data["benchmark"]).reshape(-1)[0])
        result = Aggregate(
            source_dir=source_dir,
            benchmark=benchmark,
            phase_names=phase_names,
            counts=counts,
            moe_layer_indices=moe_layers,
            num_hidden_layers=scalar(data, "num_hidden_layers"),
            num_experts=scalar(data, "num_experts"),
            top_k=scalar(data, "top_k"),
            request_count=scalar(data, "request_count"),
            prompt_tokens=scalar(data, "prompt_tokens"),
            completion_tokens=scalar(data, "completion_tokens"),
        )
    expected_shape = (len(result.phase_names), len(result.moe_layer_indices), result.num_experts)
    if result.counts.shape != expected_shape:
        raise ValueError(f"{path}: expected counts shape {expected_shape}, got {result.counts.shape}")
    if np.any(result.counts < 0):
        raise ValueError(f"{path}: counts contain negative values")
    return result


def share_of_top(values: np.ndarray, count: int) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total == 0:
        return float("nan")
    count = min(max(int(count), 1), flat.size)
    return float(np.partition(flat, flat.size - count)[-count:].sum() / total)


def count_for_share(values: np.ndarray, target: float = 0.9) -> int:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total == 0:
        return 0
    cumulative = np.cumsum(np.sort(flat)[::-1])
    return int(np.searchsorted(cumulative, target * total, side="left") + 1)


def gini(values: np.ndarray) -> float:
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(flat.sum())
    if total == 0:
        return float("nan")
    indices = np.arange(1, flat.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(indices, flat) / (flat.size * total)) - (flat.size + 1) / flat.size)


def normalized_entropy(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total == 0 or flat.size <= 1:
        return float("nan")
    probability = flat[flat > 0] / total
    entropy = float(-np.sum(probability * np.log2(probability)))
    return entropy / math.log2(flat.size)


def coefficient_of_variation(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(flat.mean())
    return float(flat.std() / mean) if mean else float("nan")


def top_indices(values: np.ndarray, count: int) -> set[int]:
    flat = np.asarray(values).reshape(-1)
    count = min(max(int(count), 1), flat.size)
    return set(np.argsort(flat, kind="stable")[-count:].tolist())


def jensen_shannon(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("JSD inputs have different shapes")
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    p = a / a.sum()
    q = b / b.sum()
    midpoint = 0.5 * (p + q)

    def divergence(left: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / midpoint[mask])))

    return 0.5 * divergence(p) + 0.5 * divergence(q)


def finite_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def analyze_aggregate(aggregate: Aggregate) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    experts = aggregate.num_experts
    floor_count = max(1, math.floor(experts * 0.2))
    ceil_count = max(1, math.ceil(experts * 0.2))

    for phase_index, phase in enumerate(aggregate.phase_names):
        counts = aggregate.counts[phase_index]
        layer_rows: list[dict[str, Any]] = []
        for local_layer, model_layer in enumerate(aggregate.moe_layer_indices.tolist()):
            values = counts[local_layer]
            total = int(values.sum())
            needed_90 = count_for_share(values, 0.9)
            row = {
                "benchmark": aggregate.benchmark,
                "phase": phase,
                "model_layer": int(model_layer),
                "assignments": total,
                "top20_floor_experts": floor_count,
                "top20_floor_share": share_of_top(values, floor_count),
                "top20_ceil_experts": ceil_count,
                "top20_ceil_share": share_of_top(values, ceil_count),
                "experts_for_90pct": needed_90,
                "experts_for_90pct_fraction": needed_90 / experts if needed_90 else float("nan"),
                "gini": gini(values),
                "normalized_entropy": normalized_entropy(values),
                "coefficient_of_variation": coefficient_of_variation(values),
            }
            layer_rows.append(row)
            per_layer.append(row)

        layer_top20 = np.array([row["top20_ceil_share"] for row in layer_rows], dtype=np.float64)
        layer_need90 = np.array(
            [row["experts_for_90pct_fraction"] for row in layer_rows], dtype=np.float64
        )
        flat = counts.reshape(-1)
        pooled_ids = counts.sum(axis=0)
        global_pair_count = max(1, math.ceil(flat.size * 0.2))
        global_need90 = count_for_share(flat, 0.9)
        pooled_need90 = count_for_share(pooled_ids, 0.9)
        summaries.append(
            {
                "benchmark": aggregate.benchmark,
                "phase": phase,
                "source_dir": str(aggregate.source_dir),
                "request_count": aggregate.request_count,
                "prompt_tokens": aggregate.prompt_tokens,
                "completion_tokens": aggregate.completion_tokens,
                "assignments": int(flat.sum()),
                "num_moe_layers": len(aggregate.moe_layer_indices),
                "num_experts_per_layer": experts,
                "top_k": aggregate.top_k,
                "per_layer_top20_experts": ceil_count,
                "per_layer_top20_share_mean": finite_mean(layer_top20.tolist()),
                "per_layer_top20_share_median": float(np.nanmedian(layer_top20)),
                "per_layer_top20_share_min": float(np.nanmin(layer_top20)),
                "per_layer_top20_share_max": float(np.nanmax(layer_top20)),
                "layers_top20_ge_90_fraction": float(np.nanmean(layer_top20 >= 0.9)),
                "per_layer_experts_for_90pct_fraction_mean": finite_mean(layer_need90.tolist()),
                "global_layer_expert_top20_pairs": global_pair_count,
                "global_layer_expert_top20_share": share_of_top(flat, global_pair_count),
                "global_layer_expert_pairs_for_90pct": global_need90,
                "global_layer_expert_pairs_for_90pct_fraction": global_need90 / flat.size,
                "pooled_expert_id_top20_share": share_of_top(pooled_ids, ceil_count),
                "pooled_expert_ids_for_90pct": pooled_need90,
                "pooled_expert_ids_for_90pct_fraction": pooled_need90 / experts,
                "global_gini": gini(flat),
                "global_normalized_entropy": normalized_entropy(flat),
                "global_coefficient_of_variation": coefficient_of_variation(flat),
                "global_top20_ge_90": bool(share_of_top(flat, global_pair_count) >= 0.9),
                "mean_layer_top20_ge_90": bool(finite_mean(layer_top20.tolist()) >= 0.9),
                "all_layers_top20_ge_90": bool(np.all(layer_top20 >= 0.9)),
            }
        )
    return summaries, per_layer


def compare_aggregates(aggregates: list[Aggregate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(aggregates, 2):
        if (
            left.phase_names != right.phase_names
            or left.counts.shape != right.counts.shape
            or not np.array_equal(left.moe_layer_indices, right.moe_layer_indices)
            or left.num_experts != right.num_experts
        ):
            raise ValueError(
                f"cannot compare incompatible aggregates: {left.benchmark}, {right.benchmark}"
            )
        expert_top20 = max(1, math.ceil(left.num_experts * 0.2))
        for phase_index, phase in enumerate(left.phase_names):
            a = left.counts[phase_index]
            b = right.counts[phase_index]
            layer_jsd: list[float] = []
            layer_jaccard: list[float] = []
            for layer_index in range(a.shape[0]):
                layer_jsd.append(jensen_shannon(a[layer_index], b[layer_index]))
                top_a = top_indices(a[layer_index], expert_top20)
                top_b = top_indices(b[layer_index], expert_top20)
                layer_jaccard.append(len(top_a & top_b) / len(top_a | top_b))
            flat_count = max(1, math.ceil(a.size * 0.2))
            flat_a = a.reshape(-1)
            flat_b = b.reshape(-1)
            flat_top_a = top_indices(flat_a, flat_count)
            flat_top_b = top_indices(flat_b, flat_count)
            rows.append(
                {
                    "benchmark_a": left.benchmark,
                    "benchmark_b": right.benchmark,
                    "phase": phase,
                    "layer_jsd_mean": finite_mean(layer_jsd),
                    "layer_jsd_max": float(np.nanmax(layer_jsd)),
                    "pooled_layer_expert_jsd": jensen_shannon(flat_a, flat_b),
                    "pooled_expert_id_jsd": jensen_shannon(a.sum(axis=0), b.sum(axis=0)),
                    "per_layer_top20_jaccard_mean": finite_mean(layer_jaccard),
                    "global_layer_expert_top20_jaccard": len(flat_top_a & flat_top_b)
                    / len(flat_top_a | flat_top_b),
                }
            )
    return rows


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_value(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> None:
    lines = [
        "# DeepSeek Expert Load Report",
        "",
        "## Concentration summary",
        "",
        "| Benchmark | Phase | Assignments | Mean per-layer top-20% share | Global layer-expert top-20% share | Mean expert fraction for 90% | Layers reaching 90% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {benchmark} | {phase} | {assignments} | {layer_share} | {global_share} | "
            "{need90} | {layers90} |".format(
                benchmark=row["benchmark"],
                phase=row["phase"],
                assignments=row["assignments"],
                layer_share=percentage(row["per_layer_top20_share_mean"]),
                global_share=percentage(row["global_layer_expert_top20_share"]),
                need90=percentage(row["per_layer_experts_for_90pct_fraction_mean"]),
                layers90=percentage(row["layers_top20_ge_90_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Benchmark differences",
            "",
            "| Benchmark A | Benchmark B | Phase | Mean per-layer JSD | Global JSD | Mean top-20% Jaccard |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in pairwise:
        lines.append(
            "| {benchmark_a} | {benchmark_b} | {phase} | {layer_jsd_mean:.6f} | "
            "{pooled_layer_expert_jsd:.6f} | {per_layer_top20_jaccard_mean:.6f} |".format(**row)
        )
    if not pairwise:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- The primary HBM-placement metric treats every layer's experts as separate weights.",
            "- Per-layer top-20% uses ceil(0.2 * experts_per_layer) experts in each MoE layer.",
            "- Global layer-expert top-20% allows the hot budget to move between layers.",
            "- Pooled expert-ID metrics are diagnostic only: expert ID 7 in two layers is not the same weight.",
            "- JSD is in bits and ranges from 0 (identical) to 1 (maximally different).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    os.umask(0o077)
    args = parse_args()
    aggregates = [load_aggregate(path.expanduser().resolve()) for path in args.capture_dirs]
    benchmark_names = [aggregate.benchmark for aggregate in aggregates]
    if len(set(benchmark_names)) != len(benchmark_names):
        raise ValueError(f"benchmark names must be unique, got {benchmark_names}")

    summaries: list[dict[str, Any]] = []
    per_layer: list[dict[str, Any]] = []
    for aggregate in aggregates:
        aggregate_summary, aggregate_layers = analyze_aggregate(aggregate)
        summaries.extend(aggregate_summary)
        per_layer.extend(aggregate_layers)
    pairwise = compare_aggregates(aggregates)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "per-layer.csv", per_layer)
    write_csv(output_dir / "pairwise.csv", pairwise)
    payload = {
        "schema_version": 1,
        "summaries": summaries,
        "per_layer": per_layer,
        "pairwise": pairwise,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summaries, pairwise)
    print(f"ANALYSIS_OK benchmarks={','.join(benchmark_names)} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
