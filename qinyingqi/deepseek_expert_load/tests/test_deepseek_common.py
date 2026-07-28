from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from deepseek_common import count_assignments, topology_from_config, validate_routes  # noqa: E402


class DeepSeekCommonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = topology_from_config(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 6,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 2,
                "moe_layer_freq": 1,
            }
        )

    def test_topology(self) -> None:
        self.assertEqual(self.topology.moe_layer_indices, (2, 3, 4, 5))
        self.assertEqual(self.topology.dense_layer_indices, (0, 1))
        self.assertEqual(self.topology.num_experts, 8)
        self.assertEqual(self.topology.top_k, 2)

    def test_validate_and_count_routes(self) -> None:
        routes = np.zeros((5, 6, 2), dtype=np.uint8)
        for token in range(5):
            for layer in self.topology.moe_layer_indices:
                first = (token + layer) % 8
                routes[token, layer] = [first, (first + 1) % 8]
        summary = validate_routes(routes, self.topology, prompt_tokens=3, completion_tokens=3)
        self.assertEqual(summary["prefill_rows"], 3)
        self.assertEqual(summary["decode_rows"], 2)
        counts = count_assignments(routes, self.topology, prompt_tokens=3)
        self.assertEqual(counts.shape, (3, 4, 8))
        self.assertEqual(int(counts[0].sum()), 5 * 4 * 2)
        self.assertEqual(int(counts[1].sum()), 3 * 4 * 2)
        self.assertEqual(int(counts[2].sum()), 2 * 4 * 2)

    def test_dense_layer_nonzero_is_rejected(self) -> None:
        routes = np.zeros((1, 6, 2), dtype=np.uint8)
        routes[0, 0] = [1, 2]
        for layer in self.topology.moe_layer_indices:
            routes[0, layer] = [1, 2]
        with self.assertRaisesRegex(ValueError, "dense layers"):
            validate_routes(routes, self.topology, prompt_tokens=1, completion_tokens=1)


if __name__ == "__main__":
    unittest.main()
