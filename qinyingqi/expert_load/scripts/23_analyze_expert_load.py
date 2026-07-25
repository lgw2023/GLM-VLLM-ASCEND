#!/usr/bin/env python3
"""Analyze GLM-5.2 routed-expert assignment counts across workloads."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from expert_load_common import (
    NUM_DENSE_LAYERS,
    NUM_LOGICAL_EXPERTS,
    NUM_MOE_LAYERS,
    PHASE_NAMES,
    STRICT_TOP_20_PERCENT_EXPERTS,
    distribution_metrics,
    hot_expert_set,
    ranked_expert_ids,
)


def parse_capture_spec(specification: str) -> tuple[str, Path]:
    if "=" in specification:
        name, raw_path = specification.split("=", 1)
        name = name.strip()
    else:
        raw_path = specification
        name = Path(raw_path).name
    if not name:
        raise ValueError(f"capture name is empty: {specification!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {path}")
    return name, path


def load_counts(capture_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    aggregate_path = capture_dir / "aggregate-counts.npz"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"missing aggregate counts: {aggregate_path}")
    with np.load(aggregate_path, allow_pickle=False) as archive:
        counts = archive["counts"]
        phase_names = tuple(str(value) for value in archive["phase_names"])
        if phase_names != PHASE_NAMES:
            raise ValueError(
                f"unexpected phase names in {aggregate_path}: {phase_names}; "
                f"expected {PHASE_NAMES}"
            )
        expected_shape = (len(PHASE_NAMES), NUM_MOE_LAYERS, NUM_LOGICAL_EXPERTS)
        if counts.shape != expected_shape:
            raise ValueError(
                f"unexpected count shape in {aggregate_path}: {counts.shape}; "
                f"expected {expected_shape}"
            )
        metadata = {
            "request_count": int(archive["request_count"][0]),
            "prompt_tokens": int(archive["prompt_tokens"][0]),
            "decode_tokens": int(archive["decode_tokens"][0]),
        }
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError(f"aggregate counts must be non-negative integers: {aggregate_path}")
    return counts.astype(np.int64, copy=False), metadata


def layer_metric_rows(workload: str, counts: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(PHASE_NAMES):
        phase_counts = counts[phase_index]
        for moe_layer_index in range(NUM_MOE_LAYERS):
            metrics = distribution_metrics(phase_counts[moe_layer_index])
            rows.append(
                {
                    "workload": workload,
                    "phase": phase,
                    "scope": "layer",
                    "moe_layer_index": moe_layer_index,
                    "transformer_layer": NUM_DENSE_LAYERS + moe_layer_index,
                    **metrics,
                }
            )
        global_metrics = distribution_metrics(phase_counts.sum(axis=0))
        rows.append(
            {
                "workload": workload,
                "phase": phase,
                "scope": "all_moe_layers",
                "moe_layer_index": None,
                "transformer_layer": None,
                **global_metrics,
            }
        )
    return rows


def expert_ranking_rows(workload: str, counts: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(PHASE_NAMES):
        phase_counts = counts[phase_index]
        for moe_layer_index in range(NUM_MOE_LAYERS):
            layer_counts = phase_counts[moe_layer_index]
            total = int(layer_counts.sum())
            if total == 0:
                continue
            for rank, expert_id in enumerate(ranked_expert_ids(layer_counts), start=1):
                assignment_count = int(layer_counts[expert_id])
                rows.append(
                    {
                        "workload": workload,
                        "phase": phase,
                        "moe_layer_index": moe_layer_index,
                        "transformer_layer": NUM_DENSE_LAYERS + moe_layer_index,
                        "logical_expert_id": int(expert_id),
                        "rank": rank,
                        "assignment_count": assignment_count,
                        "assignment_share": assignment_count / total,
                        "in_top51": rank <= STRICT_TOP_20_PERCENT_EXPERTS,
                    }
                )
    return rows


def workload_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for workload, phase in sorted(
        {(row["workload"], row["phase"]) for row in metric_rows}
    ):
        per_layer = [
            row
            for row in metric_rows
            if row["workload"] == workload
            and row["phase"] == phase
            and row["scope"] == "layer"
            and row["assignment_count"]
        ]
        if not per_layer:
            continue
        top51 = np.array([row["top51_assignment_share"] for row in per_layer])
        k90 = np.array([row["k90"] for row in per_layer])
        summaries.append(
            {
                "workload": workload,
                "phase": phase,
                "layers_with_assignments": len(per_layer),
                "mean_top51_assignment_share": float(np.mean(top51)),
                "median_top51_assignment_share": float(np.median(top51)),
                "min_top51_assignment_share": float(np.min(top51)),
                "max_top51_assignment_share": float(np.max(top51)),
                "mean_k90": float(np.mean(k90)),
                "median_k90": float(np.median(k90)),
                "fraction_layers_k90_within_top51": float(
                    np.mean(k90 <= STRICT_TOP_20_PERCENT_EXPERTS)
                ),
            }
        )
    return summaries


def overlap_rows(captures: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(sorted(captures), 2):
        left = captures[left_name][PHASE_NAMES.index("combined")]
        right = captures[right_name][PHASE_NAMES.index("combined")]
        layer_jaccard: list[float] = []
        for moe_layer_index in range(NUM_MOE_LAYERS):
            left_hot = hot_expert_set(left[moe_layer_index])
            right_hot = hot_expert_set(right[moe_layer_index])
            union = left_hot | right_hot
            if not union:
                continue
            intersection = left_hot & right_hot
            jaccard = len(intersection) / len(union)
            layer_jaccard.append(jaccard)
            rows.append(
                {
                    "left_workload": left_name,
                    "right_workload": right_name,
                    "scope": "layer",
                    "moe_layer_index": moe_layer_index,
                    "transformer_layer": NUM_DENSE_LAYERS + moe_layer_index,
                    "top51_intersection": len(intersection),
                    "top51_union": len(union),
                    "top51_jaccard": jaccard,
                }
            )
        if layer_jaccard:
            rows.append(
                {
                    "left_workload": left_name,
                    "right_workload": right_name,
                    "scope": "mean_across_layers",
                    "moe_layer_index": None,
                    "transformer_layer": None,
                    "top51_intersection": None,
                    "top51_union": None,
                    "top51_jaccard": float(np.mean(layer_jaccard)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        action="append",
        required=True,
        help="Capture directory, optionally NAME=/path; repeat for each workload",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / "analysis-summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"analysis already exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    captures: dict[str, np.ndarray] = {}
    capture_metadata: dict[str, dict[str, Any]] = {}
    for specification in args.capture_dir:
        name, capture_dir = parse_capture_spec(specification)
        if name in captures:
            raise ValueError(f"duplicate capture name: {name}")
        captures[name], capture_metadata[name] = load_counts(capture_dir)
        capture_metadata[name]["capture_dir"] = str(capture_dir)

    metric_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for workload, counts in sorted(captures.items()):
        metric_rows.extend(layer_metric_rows(workload, counts))
        ranking_rows.extend(expert_ranking_rows(workload, counts))
    summary_rows = workload_summary_rows(metric_rows)
    hot_overlap = overlap_rows(captures)

    write_csv(
        output_dir / "per-layer-metrics.csv",
        metric_rows,
        [
            "workload",
            "phase",
            "scope",
            "moe_layer_index",
            "transformer_layer",
            "assignment_count",
            "active_experts",
            "top1_assignment_share",
            "top51_assignment_share",
            "top52_assignment_share",
            "k90",
            "k90_within_top51",
            "normalized_entropy",
            "gini",
        ],
    )
    write_csv(
        output_dir / "expert-rankings.csv",
        ranking_rows,
        [
            "workload",
            "phase",
            "moe_layer_index",
            "transformer_layer",
            "logical_expert_id",
            "rank",
            "assignment_count",
            "assignment_share",
            "in_top51",
        ],
    )
    write_csv(
        output_dir / "workload-summary.csv",
        summary_rows,
        [
            "workload",
            "phase",
            "layers_with_assignments",
            "mean_top51_assignment_share",
            "median_top51_assignment_share",
            "min_top51_assignment_share",
            "max_top51_assignment_share",
            "mean_k90",
            "median_k90",
            "fraction_layers_k90_within_top51",
        ],
    )
    write_csv(
        output_dir / "hot-set-overlap.csv",
        hot_overlap,
        [
            "left_workload",
            "right_workload",
            "scope",
            "moe_layer_index",
            "transformer_layer",
            "top51_intersection",
            "top51_union",
            "top51_jaccard",
        ],
    )

    summary = {
        "logical_experts_per_layer": NUM_LOGICAL_EXPERTS,
        "moe_layers": NUM_MOE_LAYERS,
        "strict_hot_expert_budget": STRICT_TOP_20_PERCENT_EXPERTS,
        "counting_unit": "token-expert assignment",
        "capture_metadata": capture_metadata,
        "workload_summary": summary_rows,
        "outputs": {
            "per_layer_metrics": "per-layer-metrics.csv",
            "expert_rankings": "expert-rankings.csv",
            "workload_summary": "workload-summary.csv",
            "hot_set_overlap": "hot-set-overlap.csv",
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
