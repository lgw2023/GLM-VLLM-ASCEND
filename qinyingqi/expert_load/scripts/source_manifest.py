#!/usr/bin/env python3
"""Generate, verify, and bundle the Gitless expert-load source identity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
MANIFEST_NAME = "SOURCE_MANIFEST.json"
PACKAGE_ARCHIVE_PREFIX = Path("qinyingqi/expert_load")
GENERATED_SOURCE_FILES = frozenset({"glm52-expert-load-source.tar.gz"})
PINNED_VLLM_COMMIT = "0decac0d96c42b49572498019f0a0e3600f50398"
PINNED_VLLM_ASCEND_COMMIT = "5f6faa0cb8830f667266f3b8121cd1383606f2a1"
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SOURCE_FILES = frozenset(
    {
        "README.md",
        "scripts/00_preflight.sh",
        "scripts/10_launch_node.sh",
        "scripts/lib/common.sh",
        "scripts/source_manifest.py",
    }
)

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PACKAGE_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_REPO_ROOT = DEFAULT_PACKAGE_ROOT.parents[1]
DEFAULT_MANIFEST_PATH = DEFAULT_PACKAGE_ROOT / MANIFEST_NAME


class ManifestError(ValueError):
    """Raised when source identity generation or verification fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256.fullmatch(value) is None:
        raise ManifestError(f"{label} must be a lowercase SHA-256: {value!r}")
    return value


def validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"manifest path must be a non-empty string: {value!r}")
    if "\\" in value or "\x00" in value:
        raise ManifestError(f"manifest path contains a forbidden character: {value!r}")
    parts = value.split("/")
    if value.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ManifestError(f"manifest path must stay inside the package: {value!r}")
    if value == MANIFEST_NAME:
        raise ManifestError(f"manifest must not include itself: {value}")
    return value


def resolve_source_file(package_root: Path, relative_path: str) -> Path:
    candidate = package_root.joinpath(*relative_path.split("/"))
    current = package_root
    for part in relative_path.split("/"):
        current = current / part
        if current.is_symlink():
            raise ManifestError(f"source manifest does not allow symlinks: {relative_path}")
    try:
        candidate.resolve().relative_to(package_root.resolve())
    except ValueError as exc:
        raise ManifestError(
            f"manifest path escapes the package root: {relative_path}"
        ) from exc
    if not candidate.is_file():
        raise ManifestError(f"manifest source file is missing: {relative_path}")
    return candidate


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid source manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("source manifest root must be an object")
    expected_keys = {"schema_version", "hash_algorithm", "locks", "files"}
    if set(value) != expected_keys:
        raise ManifestError(
            "source manifest has unexpected top-level fields: "
            f"expected={sorted(expected_keys)}, actual={sorted(value)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported source manifest schema: {value['schema_version']!r}"
        )
    if value["hash_algorithm"] != HASH_ALGORITHM:
        raise ManifestError(
            f"unsupported source manifest hash: {value['hash_algorithm']!r}"
        )
    return value


def verify_manifest(
    package_root: Path,
    manifest_path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestError(f"source manifest is missing or is a symlink: {manifest_path}")

    manifest_sha256 = sha256_file(manifest_path)
    if expected_sha256 is not None:
        expected_sha256 = validate_sha256(
            expected_sha256, "expected source manifest SHA-256"
        )
        if manifest_sha256 != expected_sha256:
            raise ManifestError(
                "source manifest SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={manifest_sha256}"
            )

    manifest = load_manifest(manifest_path)
    locks = manifest["locks"]
    expected_locks = {
        "vllm_commit": PINNED_VLLM_COMMIT,
        "vllm_ascend_commit": PINNED_VLLM_ASCEND_COMMIT,
    }
    if locks != expected_locks:
        raise ManifestError(
            f"source lock mismatch: expected={expected_locks}, actual={locks}"
        )

    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ManifestError("source manifest files must be a non-empty list")

    seen_paths: set[str] = set()
    total_bytes = 0
    verified_entries: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ManifestError(f"invalid source manifest file entry: {entry!r}")
        relative_path = validate_relative_path(entry["path"])
        if relative_path in seen_paths:
            raise ManifestError(f"duplicate source manifest path: {relative_path}")
        seen_paths.add(relative_path)
        expected_bytes = entry["bytes"]
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ManifestError(
                f"invalid byte count for {relative_path}: {expected_bytes!r}"
            )
        expected_file_sha256 = validate_sha256(
            entry["sha256"], f"source file SHA-256 for {relative_path}"
        )
        source_path = resolve_source_file(package_root, relative_path)
        actual_bytes = source_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ManifestError(
                f"source file byte count mismatch for {relative_path}: "
                f"expected={expected_bytes}, actual={actual_bytes}"
            )
        actual_file_sha256 = sha256_file(source_path)
        if actual_file_sha256 != expected_file_sha256:
            raise ManifestError(
                f"source file SHA-256 mismatch for {relative_path}: "
                f"expected={expected_file_sha256}, actual={actual_file_sha256}"
            )
        total_bytes += actual_bytes
        verified_entries.append(
            {
                "path": relative_path,
                "bytes": actual_bytes,
                "sha256": actual_file_sha256,
            }
        )

    missing_required = sorted(REQUIRED_SOURCE_FILES - seen_paths)
    if missing_required:
        raise ManifestError(
            f"source manifest omits required runtime files: {missing_required}"
        )

    return {
        "source_id": f"sha256:{manifest_sha256}",
        "manifest_sha256": manifest_sha256,
        "file_count": len(verified_entries),
        "total_bytes": total_bytes,
        "locks": locks,
        "files": verified_entries,
    }


def run_git(repo_root: Path, *args: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ManifestError("git is required only for local manifest generation") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(f"git {' '.join(args)} failed: {stderr}")
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def collect_git_source_paths(repo_root: Path, package_root: Path) -> list[str]:
    repo_root = repo_root.resolve()
    package_root = package_root.resolve()
    actual_repo_root = Path(str(run_git(repo_root, "rev-parse", "--show-toplevel"))).resolve()
    if actual_repo_root != repo_root:
        raise ManifestError(
            f"repo root mismatch: expected={repo_root}, actual={actual_repo_root}"
        )
    try:
        package_relative = package_root.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ManifestError("package root must be inside the Git repository") from exc

    submodule_locks = {
        "upstream/vllm": PINNED_VLLM_COMMIT,
        "upstream/vllm-ascend": PINNED_VLLM_ASCEND_COMMIT,
    }
    for relative_path, expected_commit in submodule_locks.items():
        submodule_path = repo_root / relative_path
        actual_commit = str(run_git(submodule_path, "rev-parse", "HEAD"))
        if actual_commit != expected_commit:
            raise ManifestError(
                f"{relative_path} is not at the pinned commit: "
                f"expected={expected_commit}, actual={actual_commit}"
            )
        dirty = str(run_git(submodule_path, "status", "--porcelain"))
        if dirty:
            raise ManifestError(f"{relative_path} has uncommitted changes")

    raw_paths = bytes(
        run_git(
            repo_root,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            package_relative,
            binary=True,
        )
    )
    deleted_raw_paths = bytes(
        run_git(
            repo_root,
            "ls-files",
            "-z",
            "--deleted",
            "--",
            package_relative,
            binary=True,
        )
    )
    deleted_paths = {
        Path(os.fsdecode(raw_path))
        for raw_path in deleted_raw_paths.split(b"\0")
        if raw_path
    }
    relative_paths: list[str] = []
    for raw_path in raw_paths.split(b"\0"):
        if not raw_path:
            continue
        repo_relative = Path(os.fsdecode(raw_path))
        if repo_relative in deleted_paths:
            continue
        try:
            package_file = repo_relative.relative_to(Path(package_relative))
        except ValueError as exc:
            raise ManifestError(f"Git returned a path outside the package: {repo_relative}") from exc
        relative_path = package_file.as_posix()
        if relative_path == MANIFEST_NAME or relative_path in GENERATED_SOURCE_FILES:
            continue
        validate_relative_path(relative_path)
        resolve_source_file(package_root, relative_path)
        relative_paths.append(relative_path)

    relative_paths = sorted(set(relative_paths))
    missing_required = sorted(REQUIRED_SOURCE_FILES - set(relative_paths))
    if missing_required:
        raise ManifestError(f"local source set omits required files: {missing_required}")
    return relative_paths


def build_manifest(package_root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative_path in sorted(relative_paths):
        relative_path = validate_relative_path(relative_path)
        source_path = resolve_source_file(package_root, relative_path)
        files.append(
            {
                "path": relative_path,
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "locks": {
            "vllm_commit": PINNED_VLLM_COMMIT,
            "vllm_ascend_commit": PINNED_VLLM_ASCEND_COMMIT,
        },
        "files": files,
    }


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=manifest_path.parent, prefix=".source-manifest.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_manifest_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def generate_manifest(
    repo_root: Path,
    package_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    relative_paths = collect_git_source_paths(repo_root, package_root)
    manifest = build_manifest(package_root, relative_paths)
    write_manifest(manifest_path, manifest)
    return verify_manifest(package_root, manifest_path)


def normalized_tar_info(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    tar_info.uid = 0
    tar_info.gid = 0
    tar_info.uname = "root"
    tar_info.gname = "root"
    tar_info.mtime = 0
    return tar_info


def build_bundle(
    package_root: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(package_root)
    except ValueError:
        pass
    else:
        raise ManifestError("bundle output must be outside the source package")

    summary = verify_manifest(
        package_root, manifest_path, expected_sha256=expected_sha256
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=".source-bundle.", delete=False
        ) as raw_handle:
            temporary_path = Path(raw_handle.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gzip_handle:
                with tarfile.open(
                    mode="w", fileobj=gzip_handle, format=tarfile.PAX_FORMAT
                ) as archive:
                    archive_files = list(summary["files"]) + [
                        {
                            "path": MANIFEST_NAME,
                            "bytes": manifest_path.stat().st_size,
                            "sha256": summary["manifest_sha256"],
                        }
                    ]
                    for entry in sorted(archive_files, key=lambda item: item["path"]):
                        relative_path = entry["path"]
                        source_path = (
                            manifest_path
                            if relative_path == MANIFEST_NAME
                            else resolve_source_file(package_root, relative_path)
                        )
                        archive_name = (PACKAGE_ARCHIVE_PREFIX / relative_path).as_posix()
                        tar_info = normalized_tar_info(
                            archive.gettarinfo(str(source_path), arcname=archive_name)
                        )
                        if not tar_info.isfile():
                            raise ManifestError(
                                f"bundle source is not a regular file: {relative_path}"
                            )
                        with source_path.open("rb") as source_handle:
                            archive.addfile(tar_info, source_handle)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        **summary,
        "bundle_path": str(output_path),
        "bundle_sha256": sha256_file(output_path),
        "bundle_bytes": output_path.stat().st_size,
    }


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)


def print_summary(prefix: str, summary: dict[str, Any]) -> None:
    print(
        f"{prefix} source_id={summary['source_id']} "
        f"files={summary['file_count']} bytes={summary['total_bytes']} "
        f"vllm={summary['locks']['vllm_commit']} "
        f"vllm_ascend={summary['locks']['vllm_ascend_commit']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="generate the manifest from a trusted local Git worktree"
    )
    add_common_paths(generate_parser)
    generate_parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a source tree without requiring Git metadata"
    )
    add_common_paths(verify_parser)
    verify_parser.add_argument("--expected-sha256")
    verify_parser.add_argument("--quiet", action="store_true")

    digest_parser = subparsers.add_parser(
        "digest", help="verify the source tree and print only the manifest SHA-256"
    )
    add_common_paths(digest_parser)

    bundle_parser = subparsers.add_parser(
        "bundle", help="build a deterministic source-only tar.gz without .git"
    )
    add_common_paths(bundle_parser)
    bundle_parser.add_argument("--expected-sha256")
    bundle_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "generate":
            summary = generate_manifest(
                args.repo_root, args.package_root, args.manifest
            )
            print_summary("SOURCE_MANIFEST_GENERATED", summary)
        elif args.command == "verify":
            summary = verify_manifest(
                args.package_root,
                args.manifest,
                expected_sha256=args.expected_sha256,
            )
            if not args.quiet:
                print_summary("SOURCE_MANIFEST_OK", summary)
        elif args.command == "digest":
            summary = verify_manifest(args.package_root, args.manifest)
            print(summary["manifest_sha256"])
        else:
            summary = build_bundle(
                args.package_root,
                args.manifest,
                args.output,
                expected_sha256=args.expected_sha256,
            )
            print_summary("SOURCE_BUNDLE_OK", summary)
            print(
                f"bundle={summary['bundle_path']} "
                f"bundle_sha256={summary['bundle_sha256']} "
                f"bundle_bytes={summary['bundle_bytes']}"
            )
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
