from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "patches" / "apply_w8a8_route_capture.py"
SPEC = importlib.util.spec_from_file_location("apply_w8a8_route_capture", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def original_w8a8_source() -> str:
    return (
        "def apply(layer, topk_ids, topk_weights):\n"
        + PATCHER.W8A8_ANCHOR
        + "        if zero_expert_num > 0:\n"
        + "            pass\n"
        + "        if enable_force_load_balance:\n"
        + "            pass\n"
    )


def original_capture_source() -> str:
    return (
        "class Capturer:\n"
        "    def capture(self, layer_id, topk_ids):\n"
        + PATCHER.CAPTURE_ANCHOR
        + "            pass\n"
        + "        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[start_loc:end_loc, :]\n"
    )


class W8A8RouteCapturePatchTests(unittest.TestCase):
    def test_target_path_supports_editable_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ascend_root = Path(directory) / "vllm_ascend"
            vllm_root = Path(directory) / "vllm"
            w8a8_target = ascend_root / PATCHER.W8A8_PACKAGE_RELATIVE_PATH
            capture_target = vllm_root / PATCHER.CAPTURE_PACKAGE_RELATIVE_PATH
            w8a8_target.parent.mkdir(parents=True)
            capture_target.parent.mkdir(parents=True)
            w8a8_target.write_text("# installed W8A8 source\n", encoding="utf-8")
            capture_target.write_text("# installed capture source\n", encoding="utf-8")

            def fake_find_spec(name: str):
                root = ascend_root if name == "vllm_ascend" else vllm_root
                return SimpleNamespace(submodule_search_locations=[str(root)])

            with mock.patch.object(PATCHER, "find_spec", side_effect=fake_find_spec):
                self.assertEqual(
                    PATCHER.target_path(PATCHER.W8A8_PACKAGE, PATCHER.W8A8_PACKAGE_RELATIVE_PATH),
                    w8a8_target,
                )
                self.assertEqual(
                    PATCHER.target_path(PATCHER.CAPTURE_PACKAGE, PATCHER.CAPTURE_PACKAGE_RELATIVE_PATH),
                    capture_target,
                )

    def test_release_gate_accepts_only_local_build_suffixes(self) -> None:
        self.assertTrue(PATCHER.matches_release("0.22.1", "0.22.1"))
        self.assertTrue(PATCHER.matches_release("0.22.1+empty", "0.22.1"))
        self.assertFalse(PATCHER.matches_release("0.22.2", "0.22.1"))
        self.assertFalse(PATCHER.matches_release("0.22.1rc1", "0.22.1"))

    def test_patches_active_vllm_capture_before_buffer_write(self) -> None:
        w8a8 = PATCHER.patch_w8a8_source(original_w8a8_source())
        capture = PATCHER.patch_capture_source(original_capture_source())
        PATCHER.verify_w8a8_source(w8a8)
        PATCHER.verify_capture_source(capture)

        self.assertEqual(w8a8.count(PATCHER.W8A8_PATCH_MARKER), 1)
        self.assertIn(
            "capturer.capture(layer_id=layer.layer_id, topk_ids=capture_ids)",
            w8a8,
        )
        self.assertIn("prepare_finalize", w8a8)
        self.assertIn("orig_tokens", w8a8)
        self.assertIn("torch.tensor_split(gathered, tp_size, dim=0)", w8a8)
        self.assertEqual(capture.count(PATCHER.CAPTURE_PATCH_MARKER), 1)
        self.assertIn("already be full-length", capture)
        self.assertLess(
            capture.index(PATCHER.CAPTURE_PATCH_MARKER),
            capture.index("        self.device_buffer[:token_num_per_dp, layer_id, :] ="),
        )

    def test_patches_reject_already_patched_source(self) -> None:
        w8a8 = PATCHER.patch_w8a8_source(original_w8a8_source())
        capture = PATCHER.patch_capture_source(original_capture_source())
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_w8a8_source(w8a8)
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_capture_source(capture)

    def test_patches_reject_unexpected_source_layout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            PATCHER.patch_w8a8_source("def apply():\n    pass\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            PATCHER.patch_capture_source("def capture():\n    pass\n")


if __name__ == "__main__":
    unittest.main()
