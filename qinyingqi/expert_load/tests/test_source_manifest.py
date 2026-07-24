from __future__ import annotations

import io
import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "source_manifest.py"
SPEC = importlib.util.spec_from_file_location("source_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


class SourceManifestTests(unittest.TestCase):
    def make_package(self, root: Path) -> Path:
        files = {
            "README.md": "test package\n",
            "scripts/00_preflight.sh": "#!/usr/bin/env bash\n",
            "scripts/10_launch_node.sh": "#!/usr/bin/env bash\n",
            "scripts/lib/common.sh": "#!/usr/bin/env bash\n",
            "scripts/source_manifest.py": "# test verifier\n",
        }
        for relative_path, content in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        manifest = MANIFEST.build_manifest(root, files)
        MANIFEST.write_manifest(root / MANIFEST.MANIFEST_NAME, manifest)
        return root / MANIFEST.MANIFEST_NAME

    def test_complete_manifest_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_package(root)
            expected_sha256 = MANIFEST.sha256_file(manifest_path)

            result = MANIFEST.verify_manifest(
                root, manifest_path, expected_sha256=expected_sha256
            )

            self.assertEqual(result["source_id"], f"sha256:{expected_sha256}")
            self.assertEqual(result["file_count"], 5)

    def test_modified_source_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_package(root)
            (root / "scripts/00_preflight.sh").write_text(
                "modified\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                MANIFEST.ManifestError, "byte count mismatch|SHA-256 mismatch"
            ):
                MANIFEST.verify_manifest(root, manifest_path)

    def test_wrong_expected_manifest_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_package(root)

            with self.assertRaisesRegex(MANIFEST.ManifestError, "manifest SHA-256"):
                MANIFEST.verify_manifest(
                    root, manifest_path, expected_sha256="0" * 64
                )

    def test_digest_command_prints_only_verified_manifest_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_package(root)
            expected_sha256 = MANIFEST.sha256_file(manifest_path)
            output = io.StringIO()
            arguments = [
                str(SCRIPT_PATH),
                "digest",
                "--package-root",
                str(root),
                "--manifest",
                str(manifest_path),
            ]

            with patch.object(sys, "argv", arguments), redirect_stdout(output):
                exit_status = MANIFEST.main()

            self.assertEqual(exit_status, 0)
            self.assertEqual(output.getvalue(), f"{expected_sha256}\n")

    def test_parent_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.make_package(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][0]["path"] = "../outside"
            MANIFEST.write_manifest(manifest_path, manifest)

            with self.assertRaisesRegex(MANIFEST.ManifestError, "inside the package"):
                MANIFEST.verify_manifest(root, manifest_path)

    def test_bundle_contains_only_manifest_managed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            package_root = temporary_root / "package"
            package_root.mkdir()
            manifest_path = self.make_package(package_root)
            (package_root / "configs").mkdir()
            (package_root / "configs/node.env").write_text(
                "LOCAL_IP=secret\n", encoding="utf-8"
            )
            bundle_path = temporary_root / "bundle.tar.gz"

            MANIFEST.build_bundle(package_root, manifest_path, bundle_path)

            with tarfile.open(bundle_path, "r:gz") as archive:
                names = set(archive.getnames())
            prefix = MANIFEST.PACKAGE_ARCHIVE_PREFIX.as_posix()
            self.assertIn(f"{prefix}/{MANIFEST.MANIFEST_NAME}", names)
            self.assertIn(f"{prefix}/scripts/00_preflight.sh", names)
            self.assertNotIn(f"{prefix}/configs/node.env", names)

    def test_deleted_tracked_file_is_omitted_from_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            package_root = repo_root / "qinyingqi" / "expert_load"
            package_root.mkdir(parents=True)
            self.make_package(package_root)
            tracked_paths = [
                f"qinyingqi/expert_load/{relative_path}"
                for relative_path in sorted(MANIFEST.REQUIRED_SOURCE_FILES)
            ]
            generated_archive = (
                "qinyingqi/expert_load/glm52-expert-load-source.tar.gz"
            )
            deleted_path = "qinyingqi/expert_load/scripts/legacy_stop.py"

            def fake_run_git(
                command_root: Path, *args: str, binary: bool = False
            ) -> str | bytes:
                if args[:2] == ("rev-parse", "--show-toplevel"):
                    return str(repo_root)
                if args[:2] == ("rev-parse", "HEAD"):
                    if command_root.name == "vllm":
                        return MANIFEST.PINNED_VLLM_COMMIT
                    return MANIFEST.PINNED_VLLM_ASCEND_COMMIT
                if args[:2] == ("status", "--porcelain"):
                    return ""
                if args[:3] == ("ls-files", "-z", "--deleted"):
                    return f"{deleted_path}\0".encode()
                if args[:2] == ("ls-files", "-z"):
                    return (
                        "\0".join(
                            [*tracked_paths, generated_archive, deleted_path]
                        )
                        + "\0"
                    ).encode()
                self.fail(f"unexpected Git invocation: {command_root} {args}")

            with patch.object(MANIFEST, "run_git", side_effect=fake_run_git):
                paths = MANIFEST.collect_git_source_paths(repo_root, package_root)

            self.assertEqual(paths, sorted(MANIFEST.REQUIRED_SOURCE_FILES))


if __name__ == "__main__":
    unittest.main()
