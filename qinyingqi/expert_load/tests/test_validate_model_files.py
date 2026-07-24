from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_model_files.py"
SPEC = importlib.util.spec_from_file_location("validate_model_files", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def safetensors_bytes(tensor_name: str) -> bytes:
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, byteorder="little") + header + b"\x00" * 4


class ModelValidationTests(unittest.TestCase):
    def make_model(self, root: Path) -> list[Path]:
        config = dict(VALIDATOR.EXPECTED_CONFIG)
        quant = {
            "model.a.weight": "W8A8",
            "model.b.weight": "W8A8_DYNAMIC",
            "model.norm.weight": "FLOAT",
        }
        shard_names = [
            "quant_model_weights-00001-of-00002.safetensors",
            "quant_model_weights-00002-of-00002.safetensors",
        ]
        shard_paths = [root / name for name in shard_names]
        shard_paths[0].write_bytes(safetensors_bytes("model.a"))
        shard_paths[1].write_bytes(safetensors_bytes("model.b"))
        index = {
            "metadata": {
                "total_size": 4 * len(shard_paths)
            },
            "weight_map": {
                "model.a": shard_names[0],
                "model.b": shard_names[1],
            },
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / VALIDATOR.QUANT_DESCRIPTION_FILE).write_text(
            json.dumps(quant), encoding="utf-8"
        )
        (root / VALIDATOR.INDEX_FILE).write_text(
            json.dumps(index), encoding="utf-8"
        )
        return shard_paths

    def validate_fixture(self, root: Path) -> dict:
        return VALIDATOR.validate_model(root, enforce_pinned_metadata=False)

    def test_complete_model_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths = self.make_model(root)
            result = self.validate_fixture(root)
            self.assertTrue(result["valid"])
            self.assertEqual(result["shard_count"], 2)
            self.assertEqual(
                result["total_shard_bytes"],
                sum(path.stat().st_size for path in shard_paths),
            )
            self.assertEqual(result["total_tensor_bytes"], 8)
            self.assertEqual(result["safetensors_tensor_count"], 2)

    def test_index_total_size_excludes_rot_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            rot_path = root / "rot.safetensors"
            rot_path.write_bytes(safetensors_bytes("model.rot"))
            index_path = root / VALIDATOR.INDEX_FILE
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.rot"] = rot_path.name
            index_path.write_text(json.dumps(index), encoding="utf-8")

            result = self.validate_fixture(root)

            self.assertEqual(result["index_metadata_total_size"], 8)
            self.assertEqual(result["index_total_covered_tensor_bytes"], 8)
            self.assertEqual(result["index_total_excluded_shards"], [rot_path.name])
            self.assertEqual(result["index_total_excluded_tensor_bytes"], 4)
            self.assertEqual(result["total_tensor_bytes"], 12)

    def test_default_mode_enforces_pinned_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            with self.assertRaisesRegex(ValueError, "pinned ModelScope revision"):
                VALIDATOR.validate_model(root)

    def test_missing_indexed_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths = self.make_model(root)
            shard_paths[1].unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                self.validate_fixture(root)

    def test_index_header_tensor_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            index_path = root / VALIDATOR.INDEX_FILE
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_name = index["weight_map"].pop("model.a")
            index["weight_map"]["model.not_a"] = shard_name
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "index/header mismatch"):
                self.validate_fixture(root)

    def test_wrong_glm_topology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["n_routed_experts"] = 128
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                self.validate_fixture(root)

    def test_wrong_quantization_description_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            quant_path = root / VALIDATOR.QUANT_DESCRIPTION_FILE
            quant_path.write_text(
                json.dumps({"model.a.weight": "FLOAT"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "W8A8_DYNAMIC and W8A8"):
                self.validate_fixture(root)

    def test_parent_path_in_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            index_path = root / VALIDATOR.INDEX_FILE
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.a"] = "../outside.safetensors"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inside model root"):
                self.validate_fixture(root)

    def test_truncated_safetensors_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths = self.make_model(root)
            shard_paths[0].write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "truncated"):
                self.validate_fixture(root)

    def test_index_total_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            index_path = root / VALIDATOR.INDEX_FILE
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["metadata"]["total_size"] += 1
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tensor byte total mismatch"):
                self.validate_fixture(root)


if __name__ == "__main__":
    unittest.main()
