from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "patches" / "apply_w8a8_route_capture.py"
SPEC = importlib.util.spec_from_file_location("apply_w8a8_route_capture", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def original_source() -> str:
    return (
        "def apply(layer, topk_ids, topk_weights):\n"
        + PATCHER.ANCHOR
        + "        if zero_expert_num > 0:\n"
        + "            pass\n"
        + "        if enable_force_load_balance:\n"
        + "            pass\n"
    )


class W8A8RouteCapturePatchTests(unittest.TestCase):
    def test_patch_inserts_logical_topk_hook_before_remapping(self) -> None:
        patched = PATCHER.patch_source(original_source())
        PATCHER.verify_source(patched)
        self.assertEqual(patched.count(PATCHER.PATCH_MARKER), 1)
        self.assertIn('getattr(layer, "router", None)', patched)
        self.assertIn("capture_fn(topk_ids)", patched)
        self.assertLess(
            patched.index(PATCHER.PATCH_MARKER),
            patched.index("        if zero_expert_num > 0"),
        )

    def test_patch_rejects_already_patched_source(self) -> None:
        patched = PATCHER.patch_source(original_source())
        with self.assertRaisesRegex(RuntimeError, "already present"):
            PATCHER.patch_source(patched)

    def test_patch_rejects_unexpected_source_layout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            PATCHER.patch_source("def apply():\n    pass\n")


if __name__ == "__main__":
    unittest.main()
