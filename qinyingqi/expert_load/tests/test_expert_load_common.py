from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from expert_load_common import (  # noqa: E402
    NUM_DENSE_LAYERS,
    NUM_LAYERS,
    NUM_LOGICAL_EXPERTS,
    NUM_MOE_LAYERS,
    TOP_K,
    count_route_assignments,
    distribution_metrics,
    hot_expert_set,
)


def routes_with_rows(rows: int) -> np.ndarray:
    routes = np.zeros((rows, NUM_LAYERS, TOP_K), dtype=np.uint8)
    for row in range(rows):
        for layer in range(NUM_DENSE_LAYERS, NUM_LAYERS):
            base = (row + layer * TOP_K) % (NUM_LOGICAL_EXPERTS - TOP_K)
            routes[row, layer] = base + np.arange(TOP_K, dtype=np.uint8)
    return routes


class ExpertLoadCommonTests(unittest.TestCase):
    def test_count_route_assignments_separates_prefill_and_decode(self) -> None:
        counts = count_route_assignments(routes_with_rows(4), prompt_tokens=3)
        self.assertEqual(counts.shape, (3, NUM_MOE_LAYERS, NUM_LOGICAL_EXPERTS))
        self.assertEqual(int(counts[0].sum()), 3 * NUM_MOE_LAYERS * TOP_K)
        self.assertEqual(int(counts[1].sum()), NUM_MOE_LAYERS * TOP_K)
        self.assertEqual(int(counts[2].sum()), 4 * NUM_MOE_LAYERS * TOP_K)
        np.testing.assert_array_equal(counts[2], counts[0] + counts[1])

    def test_metrics_use_assignment_based_top51_and_k90(self) -> None:
        counts = np.zeros(NUM_LOGICAL_EXPERTS, dtype=np.int64)
        counts[7] = 90
        counts[8] = 10
        metrics = distribution_metrics(counts)
        self.assertEqual(metrics["assignment_count"], 100)
        self.assertEqual(metrics["k90"], 1)
        self.assertTrue(metrics["k90_within_top51"])
        self.assertEqual(metrics["top51_assignment_share"], 1.0)

    def test_hot_set_is_deterministic_for_equal_counts(self) -> None:
        counts = np.ones(NUM_LOGICAL_EXPERTS, dtype=np.int64)
        self.assertEqual(hot_expert_set(counts), set(range(51)))


if __name__ == "__main__":
    unittest.main()
