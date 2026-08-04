#!/usr/bin/env python3
"""Verify the runtime EP map is 8 disjoint ranks with 32 routed experts each."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from glm51_common import load_topology


PLACEMENT = re.compile(
    r"Expert parallelism is enabled\.\s+"
    r"ep_rank=(\d+)/(\d+),\s+"
    r"local_num_experts=(\d+),\s+"
    r"global_num_experts=(\d+),\s+"
    r"expert_map=(.*)$"
)
PAIR = re.compile(r"(\d+)->(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    topology = load_topology(args.model_path.resolve())
    if topology.num_experts % args.ep_size:
        raise ValueError(
            f"{topology.num_experts} experts cannot be evenly divided by EP={args.ep_size}"
        )
    expected_local = topology.num_experts // args.ep_size
    text = args.log.read_text(encoding="utf-8", errors="replace")
    by_rank: dict[int, set[int]] = {}
    for line in text.splitlines():
        match = PLACEMENT.search(line)
        if not match:
            continue
        rank, ep_size, local_count, global_count = map(int, match.groups()[:4])
        if ep_size != args.ep_size:
            raise ValueError(f"runtime EP size is {ep_size}, expected {args.ep_size}")
        if local_count != expected_local or global_count != topology.num_experts:
            raise ValueError(
                f"rank {rank} reports local/global={local_count}/{global_count}, "
                f"expected {expected_local}/{topology.num_experts}"
            )
        pairs = [(int(local), int(global_id)) for local, global_id in PAIR.findall(match.group(5))]
        local_ids = {local for local, _ in pairs}
        global_ids = {global_id for _, global_id in pairs}
        if local_ids != set(range(expected_local)) or len(global_ids) != expected_local:
            raise ValueError(f"rank {rank} has an incomplete or duplicated expert map")
        if rank in by_rank and by_rank[rank] != global_ids:
            raise ValueError(f"rank {rank} emitted inconsistent expert maps")
        by_rank[rank] = global_ids

    expected_ranks = set(range(args.ep_size))
    if set(by_rank) != expected_ranks:
        raise ValueError(
            f"runtime logs contain EP ranks {sorted(by_rank)}, expected {sorted(expected_ranks)}"
        )
    flattened = [expert for rank in sorted(by_rank) for expert in sorted(by_rank[rank])]
    if len(flattened) != len(set(flattened)):
        raise ValueError("the runtime expert maps overlap across EP ranks")
    if set(flattened) != set(range(topology.num_experts)):
        raise ValueError("the runtime expert maps do not cover experts 0..255 exactly once")

    if not re.search(r"Dynamic EPLB is False", text, re.IGNORECASE):
        raise ValueError("runtime log did not prove Dynamic EPLB is disabled")
    if not re.search(r"number of redundant experts is 0", text, re.IGNORECASE):
        raise ValueError("runtime log did not prove redundant experts=0")

    report = {
        "schema_version": 1,
        "verified": True,
        "ep_size": args.ep_size,
        "num_routed_experts": topology.num_experts,
        "experts_per_rank": expected_local,
        "dynamic_eplb": False,
        "num_redundant_experts": 0,
        "rank_to_global_experts": {
            str(rank): sorted(by_rank[rank]) for rank in sorted(by_rank)
        },
        "coverage": {
            "unique_experts": len(set(flattened)),
            "duplicates_across_ranks": 0,
            "missing_experts": [],
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "EXPERT_PLACEMENT_OK "
        f"ep_size={args.ep_size} experts_per_rank={expected_local} "
        f"unique_experts={len(set(flattened))} duplicates=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

