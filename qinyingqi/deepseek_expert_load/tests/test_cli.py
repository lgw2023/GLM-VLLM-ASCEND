from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_python(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = str(Path(tempfile.gettempdir()) / "dsv4-test-pycache")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )


class AuditCliTests(unittest.TestCase):
    def test_w4a8_model_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_hidden_layers": 4,
                        "n_routed_experts": 16,
                        "num_experts_per_tok": 2,
                        "first_k_dense_replace": 1,
                        "moe_layer_freq": 1,
                    }
                ),
                encoding="utf-8",
            )
            (model / "quant_model_description.json").write_text(
                json.dumps({"model.layers.1.mlp.experts.weight": "W4A8_DYNAMIC"}),
                encoding="utf-8",
            )
            (model / "model-00001-of-00002.safetensors").write_bytes(b"a")
            (model / "model-00002-of-00002.safetensors").write_bytes(b"bb")
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.a": "model-00001-of-00002.safetensors",
                            "model.b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = Path(temporary) / "audit.json"
            result = run_python(
                "00_audit_model.py",
                "--model-path",
                str(model),
                "--require-model-type",
                "deepseek_v4",
                "--require-w4a8",
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["compatible"])
            self.assertTrue(report["quantization"]["w4a8_detected"])
            self.assertEqual(
                report["quantization"]["deployment_profile"],
                "modelslim_w4a8",
            )
            self.assertEqual(
                report["quantization"]["recommended_vllm_quantization"],
                "ascend",
            )
            self.assertEqual(
                report["quantization"]["expert_quantization"],
                "w4a8",
            )
            self.assertEqual(report["weights"]["shard_count"], 2)

    def test_w8a8_model_audit_on_910b1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_hidden_layers": 4,
                        "n_routed_experts": 16,
                        "num_experts_per_tok": 2,
                    }
                ),
                encoding="utf-8",
            )
            (model / "quant_model_description.json").write_text(
                json.dumps(
                    {"model.layers.0.mlp.experts.weight": "W8A8_DYNAMIC"}
                ),
                encoding="utf-8",
            )
            first = "quant_model_weights-00001-of-00002.safetensors"
            second = "quant_model_weights-00002-of-00002.safetensors"
            (model / first).write_bytes(b"weights-1")
            (model / second).write_bytes(b"weights-2")
            optional = model / "optional"
            optional.mkdir()
            (optional / "quarot.safetensors").write_bytes(b"optional")
            (model / "quant_model_weights.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.a": first,
                            "model.b": second,
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = run_python(
                "00_audit_model.py",
                "--model-path",
                str(model),
                "--require-model-type",
                "deepseek_v4",
                "--require-expert-quantization",
                "w8a8",
                "--target-soc",
                "ASCEND910B1",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            quantization = report["quantization"]
            self.assertTrue(report["compatible"])
            self.assertTrue(quantization["w8a8_detected"])
            self.assertEqual(
                quantization["deployment_profile"],
                "modelslim_w8a8",
            )
            self.assertEqual(quantization["expert_quantization"], "w8a8")
            self.assertEqual(
                quantization["recommended_vllm_quantization"],
                "ascend",
            )
            self.assertTrue(report["hardware"]["soc_compatible"])
            weights = report["weights"]
            self.assertTrue(weights["index_present"])
            self.assertEqual(
                weights["index_name"],
                "quant_model_weights.safetensors.index.json",
            )
            self.assertEqual(weights["active_shard_count"], 2)
            self.assertEqual(weights["unreferenced_shard_count"], 1)
            self.assertEqual(weights["nested_shard_count"], 1)
            self.assertEqual(len(report["warnings"]), 1)

    def test_native_fp8_fp4_model_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_hidden_layers": 4,
                        "n_routed_experts": 16,
                        "num_experts_per_tok": 2,
                        "quantization_config": {"quant_method": "fp8"},
                    }
                ),
                encoding="utf-8",
            )
            shard_payloads = {
                "model-00001-of-00002.safetensors": b"a",
                "model-00002-of-00002.safetensors": b"bb",
                "stale-00001-of-00002.safetensors": b"ccc",
                "stale-00002-of-00002.safetensors": b"dddd",
            }
            for name, payload in shard_payloads.items():
                (model / name).write_bytes(payload)
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.a": "model-00001-of-00002.safetensors",
                            "model.b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = run_python(
                "00_audit_model.py",
                "--model-path",
                str(model),
                "--require-model-type",
                "deepseek_v4",
                "--require-w4a8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            quantization = report["quantization"]
            self.assertTrue(report["compatible"])
            self.assertFalse(quantization["explicit_w4a8_marker_detected"])
            self.assertTrue(quantization["native_fp8_fp4_detected"])
            self.assertTrue(quantization["w4a8_detected"])
            self.assertEqual(quantization["expert_dtype"], "fp4")
            self.assertEqual(
                quantization["expert_dtype_source"],
                "vllm-0.22.1-default",
            )
            self.assertEqual(
                quantization["deployment_profile"],
                "deepseek_v4_native_fp8_fp4",
            )
            self.assertEqual(
                quantization["recommended_vllm_quantization"],
                "fp8",
            )
            self.assertEqual(report["weights"]["active_shard_count"], 2)
            self.assertEqual(report["weights"]["active_shard_bytes"], 3)
            self.assertEqual(report["weights"]["unreferenced_shard_count"], 2)
            self.assertEqual(report["weights"]["unreferenced_shard_bytes"], 7)
            self.assertEqual(len(report["warnings"]), 1)

            incompatible = run_python(
                "00_audit_model.py",
                "--model-path",
                str(model),
                "--require-model-type",
                "deepseek_v4",
                "--require-expert-quantization",
                "w4a8",
                "--target-soc",
                "ASCEND910B1",
            )
            self.assertEqual(incompatible.returncode, 2)
            incompatible_report = json.loads(incompatible.stdout)
            self.assertFalse(incompatible_report["compatible"])
            self.assertFalse(incompatible_report["hardware"]["soc_compatible"])
            self.assertTrue(
                any(
                    "incompatible with target SoC" in problem
                    for problem in incompatible_report["problems"]
                )
            )

    def test_native_fp8_experts_do_not_pass_w4a8_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v4",
                        "num_hidden_layers": 2,
                        "n_routed_experts": 8,
                        "num_experts_per_tok": 2,
                        "expert_dtype": "fp8",
                        "quantization_config": {"quant_method": "fp8"},
                    }
                ),
                encoding="utf-8",
            )
            (model / "model.safetensors").write_bytes(b"weights")
            result = run_python(
                "00_audit_model.py",
                "--model-path",
                str(model),
                "--require-model-type",
                "deepseek_v4",
                "--require-w4a8",
            )
            self.assertEqual(result.returncode, 2)
            report = json.loads(result.stdout)
            self.assertFalse(report["compatible"])
            self.assertFalse(report["quantization"]["w4a8_detected"])
            self.assertIn("W4A8 Expert execution was not proven", report["problems"][0])


class AnalysisCliTests(unittest.TestCase):
    @staticmethod
    def write_aggregate(path: Path, benchmark: str, hot_expert: int) -> None:
        import numpy as np

        path.mkdir()
        counts = np.zeros((3, 2, 10), dtype=np.int64)
        for phase in range(3):
            for layer in range(2):
                counts[phase, layer, hot_expert] = 90
                counts[phase, layer, (hot_expert + 1) % 10] = 10
        np.savez_compressed(
            path / "aggregate-counts.npz",
            schema_version=np.array([1]),
            benchmark=np.array([benchmark]),
            phase_names=np.array(["total", "prefill", "decode"]),
            counts=counts,
            moe_layer_indices=np.array([1, 2]),
            num_hidden_layers=np.array([3]),
            num_experts=np.array([10]),
            top_k=np.array([2]),
            request_count=np.array([2]),
            prompt_tokens=np.array([20]),
            completion_tokens=np.array([10]),
        )

    def test_analysis_and_pairwise_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "bench-a"
            second = root / "bench-b"
            output = root / "analysis"
            self.write_aggregate(first, "bench_a", 0)
            self.write_aggregate(second, "bench_b", 8)
            result = run_python(
                "05_analyze_expert_load.py",
                str(first),
                str(second),
                "--output-dir",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["summaries"]), 6)
            self.assertEqual(len(report["pairwise"]), 3)
            self.assertGreater(report["pairwise"][0]["pooled_layer_expert_jsd"], 0.5)
            self.assertTrue(report["summaries"][0]["global_top20_ge_90"])
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
