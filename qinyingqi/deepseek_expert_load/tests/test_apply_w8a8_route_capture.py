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
    def test_target_path_supports_editable_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory) / "vllm_ascend"
            target = package_root / PATCHER.PACKAGE_RELATIVE_PATH
            target.parent.mkdir(parents=True)
            target.write_text("# installed source\n", encoding="utf-8")
            spec = SimpleNamespace(submodule_search_locations=[str(package_root)])
            with mock.patch.object(PATCHER, "find_spec", return_value=spec):
                self.assertEqual(PATCHER.target_path(), target)

    def test_release_gate_accepts_only_local_build_suffixes(self) -> None:
        self.assertTrue(PATCHER.matches_release("0.22.1", "0.22.1"))
        self.assertTrue(PATCHER.matches_release("0.22.1+empty", "0.22.1"))
        self.assertFalse(PATCHER.matches_release("0.22.2", "0.22.1"))
        self.assertFalse(PATCHER.matches_release("0.22.1rc1", "0.22.1"))

    def test_patch_uses_ascend_capturer_before_remapping(self) -> None:
        patched = PATCHER.patch_source(original_source())
        PATCHER.verify_source(patched)
        self.assertEqual(patched.count(PATCHER.PATCH_MARKER), 1)
        self.assertIn('getattr(layer, "_ascend_routed_experts_capturer", None)', patched)
        self.assertIn(
            "capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)",
            patched,
        )
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
