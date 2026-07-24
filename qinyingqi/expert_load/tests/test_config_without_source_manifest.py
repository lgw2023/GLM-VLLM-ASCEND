from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
COMMON_SCRIPT = PACKAGE_ROOT / "scripts" / "lib" / "common.sh"


class ConfigWithoutSourceManifestTests(unittest.TestCase):
    def test_cluster_config_loads_without_source_manifest_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            cluster_config = temporary_root / "cluster.env"
            node_config = temporary_root / "node.env"
            cluster_config.write_text(
                (PACKAGE_ROOT / "configs" / "cluster.env.example")
                .read_text(encoding="utf-8")
                .replace(
                    "NODE0_COORDINATOR_IP=REPLACE_ME",
                    "NODE0_COORDINATOR_IP=10.0.0.1",
                ),
                encoding="utf-8",
            )
            node_config.write_text(
                (PACKAGE_ROOT / "configs" / "node0.env.example")
                .read_text(encoding="utf-8")
                .replace("LOCAL_IP=REPLACE_ME", "LOCAL_IP=10.0.0.1")
                .replace("PEER_IP=REPLACE_ME", "PEER_IP=10.0.0.2")
                .replace("LOCAL_NIC=REPLACE_ME", "LOCAL_NIC=eth0"),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; load_configs "$2" "$3"; '
                    '[[ -z "${SOURCE_MANIFEST_SHA256+x}" ]]',
                    "_",
                    str(COMMON_SCRIPT),
                    str(cluster_config),
                    str(node_config),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
