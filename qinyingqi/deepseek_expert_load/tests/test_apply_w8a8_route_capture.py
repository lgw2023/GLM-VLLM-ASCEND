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


def original_fused_moe_source() -> str:
    return (
        "from vllm.forward_context import get_forward_context\n"
        "from vllm_ascend.flash_common3_context import get_flash_common3_context\n"
        "from vllm_ascend.ops.fused_moe.experts_selector import select_experts\n"
        "class AscendFusedMoE:\n"
        "    def forward_impl(self):\n"
        + PATCHER.FUSED_MOE_ANCHOR
        + "        return prepare_output\n"
    )


def original_capture_source() -> str:
    return (
        "class Capturer:\n"
        "    def capture(self, layer_id, topk_ids):\n"
        + PATCHER.CAPTURE_ANCHOR
        + "            pass\n"
        + PATCHER.CAPTURE_BUFFER_ANCHOR
    )


class W8A8RouteCapturePatchTests(unittest.TestCase):
    def test_target_path_supports_editable_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ascend_root = Path(directory) / "vllm_ascend"
            vllm_root = Path(directory) / "vllm"
            w8a8_target = ascend_root / PATCHER.W8A8_PACKAGE_RELATIVE_PATH
            fused_target = ascend_root / PATCHER.FUSED_MOE_PACKAGE_RELATIVE_PATH
            capture_target = vllm_root / PATCHER.CAPTURE_PACKAGE_RELATIVE_PATH
            w8a8_target.parent.mkdir(parents=True)
            fused_target.parent.mkdir(parents=True)
            capture_target.parent.mkdir(parents=True)
            w8a8_target.write_text("# installed W8A8 source\n", encoding="utf-8")
            fused_target.write_text("# installed fused_moe source\n", encoding="utf-8")
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
                    PATCHER.target_path(PATCHER.FUSED_MOE_PACKAGE, PATCHER.FUSED_MOE_PACKAGE_RELATIVE_PATH),
                    fused_target,
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

    def test_patches_capture_before_prepare_and_skips_w8a8_post_split(self) -> None:
        w8a8 = PATCHER.patch_w8a8_source(original_w8a8_source())
        fused = PATCHER.patch_fused_moe_source(original_fused_moe_source())
        capture = PATCHER.patch_capture_source(original_capture_source())
        PATCHER.verify_w8a8_source(w8a8)
        PATCHER.verify_fused_moe_source(fused)
        PATCHER.verify_capture_source(capture)

        self.assertEqual(w8a8.count(PATCHER.W8A8_PATCH_MARKER), 1)
        self.assertIn("Skipping post-split capture", w8a8)
        self.assertNotIn("capturer.capture(layer_id=layer.layer_id, topk_ids=", w8a8)

        self.assertEqual(fused.count(PATCHER.FUSED_MOE_PATCH_MARKER), 1)
        self.assertIn("_route_capturer.capture(", fused)
        self.assertIn("DEEPSEEK_ROUTE_CAPTURE_DIAG pre_prepare", fused)
        self.assertLess(
            fused.index(PATCHER.FUSED_MOE_PATCH_MARKER),
            fused.index("prepare_output = _EXTRA_CTX.moe_comm_method.prepare("),
        )

        self.assertEqual(capture.count(PATCHER.CAPTURE_PATCH_MARKER), 1)
        self.assertIn("pre-prepare capture", capture)
        self.assertIn("DEEPSEEK_ROUTE_CAPTURE_DIAG capturer_write", capture)

    def test_patches_reject_already_patched_source(self) -> None:
        w8a8 = PATCHER.patch_w8a8_source(original_w8a8_source())
        fused = PATCHER.patch_fused_moe_source(original_fused_moe_source())
        capture = PATCHER.patch_capture_source(original_capture_source())
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_w8a8_source(w8a8)
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_fused_moe_source(fused)
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_capture_source(capture)

    def test_patches_reject_unexpected_source_layout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            PATCHER.patch_w8a8_source("def apply():\n    pass\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            PATCHER.patch_fused_moe_source("def forward():\n    pass\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            PATCHER.patch_capture_source("def capture():\n    pass\n")


if __name__ == "__main__":
    unittest.main()
