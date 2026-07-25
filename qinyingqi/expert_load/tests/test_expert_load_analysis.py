from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT_PATH = SCRIPTS_DIR / "23_analyze_expert_load.py"
SPEC = importlib.util.spec_from_file_location("analyze_expert_load", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class ExpertLoadAnalysisTests(unittest.TestCase):
    def test_load_counts_and_layer_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture_dir = Path(directory)
            counts = np.zeros((3, 75, 256), dtype=np.int64)
            counts[0, :, 0] = 90
            counts[0, :, 1] = 10
            counts[2] = counts[0]
            np.savez_compressed(
                capture_dir / "aggregate-counts.npz",
                schema_version=np.array([1]),
                phase_names=np.array(["prefill", "decode", "combined"]),
                counts=counts,
                request_count=np.array([1]),
                prompt_tokens=np.array([3]),
                decode_tokens=np.array([0]),
            )
            loaded, metadata = ANALYZE.load_counts(capture_dir)
            rows = ANALYZE.layer_metric_rows("general", loaded)
            first_prefill_layer = next(
                row
                for row in rows
                if row["phase"] == "prefill" and row["scope"] == "layer"
            )
            self.assertEqual(metadata["request_count"], 1)
            self.assertEqual(first_prefill_layer["k90"], 1)
            self.assertTrue(first_prefill_layer["k90_within_top51"])

    def test_overlap_uses_top51_logical_experts(self) -> None:
        left = np.zeros((3, 75, 256), dtype=np.int64)
        right = np.zeros((3, 75, 256), dtype=np.int64)
        left[2, :, :51] = 1
        right[2, :, :51] = 1
        rows = ANALYZE.overlap_rows({"left": left, "right": right})
        first_layer = next(row for row in rows if row["scope"] == "layer")
        self.assertEqual(first_layer["top51_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
