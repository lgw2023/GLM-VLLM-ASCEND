#!/usr/bin/env python3
"""Run one HCCL all-reduce and all-to-all across the full 2x8 topology."""

from __future__ import annotations

import datetime
import json
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the NPU backend


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 16:
        raise RuntimeError(f"expected WORLD_SIZE=16, got {world_size}")
    if not 0 <= local_rank < 8:
        raise RuntimeError(f"expected LOCAL_RANK in 0..7, got {local_rank}")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    timeout_seconds = int(os.environ.get("HCCL_TEST_TIMEOUT_SECONDS", "900"))
    initialized = False
    try:
        dist.init_process_group(
            backend="hccl",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        initialized = True

        reduced = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if float(reduced.cpu()[0]) != expected_sum:
            raise RuntimeError(
                f"rank {rank} all_reduce mismatch: {float(reduced.cpu()[0])} != {expected_sum}"
            )

        send = torch.arange(world_size, dtype=torch.float32, device=device)
        send.add_(rank * world_size)
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send)
        expected = torch.tensor(
            [source * world_size + rank for source in range(world_size)],
            dtype=torch.float32,
            device=device,
        )
        if not torch.equal(received, expected):
            raise RuntimeError(
                f"rank {rank} all_to_all mismatch: {received.cpu().tolist()}"
            )

        dist.barrier()
        torch.npu.synchronize()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "valid": True,
                        "backend": "hccl",
                        "world_size": world_size,
                        "all_reduce_sum": expected_sum,
                        "all_to_all_values_per_rank": world_size,
                    }
                ),
                flush=True,
            )
        return 0
    except Exception as exc:
        print(f"HCCL_COLLECTIVE_FAILED rank={rank}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

