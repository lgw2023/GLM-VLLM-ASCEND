from __future__ import annotations

import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
