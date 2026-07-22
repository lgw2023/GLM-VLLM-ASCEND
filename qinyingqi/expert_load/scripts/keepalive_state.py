#!/usr/bin/env python3
"""Snapshot and validate the server's official per-card keep-alive processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MARKERS = [f"#{index}#" for index in range(8)]


def read_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw = (entry / "cmdline").read_bytes()
            command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
            rows.append(
                {
                    "pid": pid,
                    "pgid": os.getpgid(pid),
                    "exe": os.readlink(entry / "exe"),
                    "markers": [marker for marker in MARKERS if marker in command],
                    "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return rows


def snapshot() -> dict[str, Any]:
    rows = read_processes()
    marker_rows = [row for row in rows if row["markers"]]
    marker_pgids = {row["pgid"] for row in marker_rows}
    pgid_members = [row for row in rows if row["pgid"] in marker_pgids]
    counts = {marker: 0 for marker in MARKERS}
    for row in marker_rows:
        for marker in row["markers"]:
            counts[marker] += 1

    normalized_groups = []
    for pgid in marker_pgids:
        members = [row for row in pgid_members if row["pgid"] == pgid]
        group_markers = sorted(
            {marker for row in members for marker in row["markers"]}
        )
        identities = sorted(
            (
                {
                    "exe": row["exe"],
                    "markers": sorted(row["markers"]),
                    "cmdline_sha256": row["cmdline_sha256"],
                }
                for row in members
            ),
            key=lambda value: json.dumps(value, sort_keys=True),
        )
        normalized_groups.append({"markers": group_markers, "members": identities})
    normalized_groups.sort(key=lambda value: json.dumps(value, sort_keys=True))

    return {
        "marker_process_count": len(marker_rows),
        "marker_counts": counts,
        "marker_processes": marker_rows,
        "marker_pgid_members": pgid_members,
        "normalized_marker_groups": normalized_groups,
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"snapshot is not a JSON object: {path}")
    return value


def validate_state(value: dict[str, Any], state: str) -> None:
    counts = value.get("marker_counts")
    if not isinstance(counts, dict):
        raise ValueError("snapshot has no marker_counts object")
    if state == "running":
        expected = {marker: 2 for marker in MARKERS}
        if value.get("marker_process_count") != 16 or counts != expected:
            raise ValueError(
                f"keep-alive running contract failed: count={value.get('marker_process_count')} markers={counts}"
            )
    elif state == "stopped":
        expected = {marker: 0 for marker in MARKERS}
        if value.get("marker_process_count") != 0 or counts != expected:
            raise ValueError(
                f"keep-alive stopped contract failed: count={value.get('marker_process_count')} markers={counts}"
            )
    else:
        raise ValueError(f"unknown expected state: {state}")


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in ("marker_counts", "normalized_marker_groups"):
        if before.get(key) != after.get(key):
            raise ValueError(f"restored keep-alive differs from pre-run snapshot: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--expected", choices=("running", "stopped"), required=True)
    validate_parser.add_argument("snapshot", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    args = parser.parse_args()
    os.umask(0o077)

    if args.command == "snapshot":
        value = snapshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(value, indent=2))
    elif args.command == "validate":
        validate_state(load_snapshot(args.snapshot), args.expected)
        print(f"KEEPALIVE_STATE_OK expected={args.expected} snapshot={args.snapshot}")
    else:
        compare_snapshots(load_snapshot(args.before), load_snapshot(args.after))
        print(f"KEEPALIVE_RESTORE_MATCH before={args.before} after={args.after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
