#!/usr/bin/env python3
"""Patch vLLM-Ascend 0.22.1rc1 to capture GLM-5.1 routes before TP split."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution, version
from importlib.util import find_spec
from pathlib import Path


EXPECTED_VLLM_VERSION = "0.22.1"
EXPECTED_ASCEND_VERSION = "0.22.1rc1"

FUSED_RELATIVE = "ops/fused_moe/fused_moe.py"
FUSED_MARKER = "# GLM51_FUSED_MOE_CAPTURE_BEFORE_PREPARE_V2"
FUSED_ANCHOR = (
    "        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(\n"
    "            hidden_states=hidden_states,\n"
    "            router_logits=router_logits,\n"
    "            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,\n"
    "            enable_shared_expert_dp=self.enable_shared_expert_dp,\n"
    "            quant_type=self.quant_type,\n"
    "        )\n"
)
FUSED_CAPTURE = (
    "        # GLM51_FUSED_MOE_CAPTURE_BEFORE_PREPARE_V2\n"
    "        # All2All prepare() splits tokens across TP ranks. Record logical\n"
    "        # expert IDs from the full router logits before that split.\n"
    "        _glm51_capturer = getattr(\n"
    "            self, \"_ascend_routed_experts_capturer\", None\n"
    "        )\n"
    "        if _glm51_capturer is None:\n"
    "            _glm51_capturer = getattr(\n"
    "                type(self), \"_glm51_global_route_capturer\", None\n"
    "            )\n"
    "        _glm51_enabled = (\n"
    "            self.vllm_config.model_config is not None\n"
    "            and self.vllm_config.model_config.enable_return_routed_experts\n"
    "        )\n"
    "        if _glm51_capturer is not None and _glm51_enabled:\n"
    "            _, _glm51_topk_ids = select_experts(\n"
    "                hidden_states=hidden_states,\n"
    "                router_logits=router_logits,\n"
    "                top_k=self.top_k,\n"
    "                use_grouped_topk=self.use_grouped_topk,\n"
    "                renormalize=self.renormalize,\n"
    "                topk_group=self.topk_group,\n"
    "                num_expert_group=self.num_expert_group,\n"
    "                custom_routing_function=self.custom_routing_function,\n"
    "                scoring_func=self.scoring_func,\n"
    "                routed_scaling_factor=self._original_routed_scaling_factor,\n"
    "                e_score_correction_bias=self.e_score_correction_bias,\n"
    "                num_experts=self.moe_config.num_experts,\n"
    "                input_ids=getattr(get_forward_context(), \"input_ids\", None),\n"
    "                tid2eid=self.tid2eid,\n"
    "            )\n"
    "            _glm51_capturer.capture(\n"
    "                layer_id=self.layer_id, topk_ids=_glm51_topk_ids\n"
    "            )\n"
    "\n"
)

RUNNER_RELATIVE = "worker/model_runner_v1.py"
RUNNER_MARKER = "# GLM51_MODEL_RUNNER_BIND_CAPTURE_V2"
RUNNER_ANCHOR = (
    "    def _bind_routed_experts_capturer(self, capturer=None) -> None:\n"
    "        # Upstream binds via ``module.router.set_capture_fn(...)`` on\n"
    "        # FusedMoE layers whose router is a ``BaseRouter``. Ascend's\n"
    "        # ``select_experts`` does not go through ``BaseRouter``, so the\n"
    "        # upstream hook never fires. Instead, stash the capturer as a\n"
    "        # plain attribute on every FusedMoE layer; ``apply()`` reads it\n"
    "        # back on the hot path.\n"
    "        from vllm.model_executor.layers.fused_moe.layer import FusedMoE\n"
    "        for module in self.compilation_config.static_forward_context.values():\n"
    "            if isinstance(module, FusedMoE):\n"
    "                module._ascend_routed_experts_capturer = capturer\n"
)
RUNNER_REPLACEMENT = (
    "    def _bind_routed_experts_capturer(self, capturer=None) -> None:\n"
    "        # GLM51_MODEL_RUNNER_BIND_CAPTURE_V2\n"
    "        # Bind both registered static modules and the loaded model tree.\n"
    "        from vllm.model_executor.layers.fused_moe.layer import FusedMoE\n"
    "        from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE\n"
    "\n"
    "        AscendFusedMoE._glm51_global_route_capturer = capturer\n"
    "        bound: set[int] = set()\n"
    "\n"
    "        def bind(module) -> None:\n"
    "            if not isinstance(module, (FusedMoE, AscendFusedMoE)):\n"
    "                return\n"
    "            if id(module) in bound:\n"
    "                return\n"
    "            module._ascend_routed_experts_capturer = capturer\n"
    "            bound.add(id(module))\n"
    "\n"
    "        for module in self.compilation_config.static_forward_context.values():\n"
    "            bind(module)\n"
    "        model = getattr(self, \"model\", None)\n"
    "        if model is not None:\n"
    "            for module in model.modules():\n"
    "                bind(module)\n"
    "        print(\n"
    "            f\"GLM51_ROUTE_CAPTURE_BIND bound={len(bound)} \"\n"
    "            f\"capturer={'set' if capturer is not None else 'none'}\",\n"
    "            flush=True,\n"
    "        )\n"
)

CAPTURE_RELATIVE = "model_executor/layers/fused_moe/routed_experts_capturer.py"
CAPTURE_MARKER = "# GLM51_VLLM_TP8_CAPTURE_GATHER_V2"
CAPTURE_ANCHOR = (
    "        ctx = get_forward_context()\n"
    "        if ctx.dp_metadata is None:  # single dp\n"
    "            start_loc = 0\n"
    "            end_loc = topk_ids.shape[0]\n"
    "            token_num_per_dp = topk_ids.shape[0]\n"
    "        else:  # multi dp\n"
)
CAPTURE_REPLACEMENT = (
    "        ctx = get_forward_context()\n"
    "        # GLM51_VLLM_TP8_CAPTURE_GATHER_V2\n"
    "        # Ascend sequence parallelism can pass one TP-local token shard\n"
    "        # even for DP=1. Reconstruct the complete step before writing the\n"
    "        # route buffer; otherwise most prefill rows remain zero.\n"
    "        if ctx.dp_metadata is None:  # single dp\n"
    "            n = topk_ids.shape[0]\n"
    "            start_loc = 0\n"
    "            end_loc = n\n"
    "            token_num_per_dp = n\n"
    "            expected_tokens = (\n"
    "                ctx.batch_descriptor.num_tokens\n"
    "                if ctx.batch_descriptor is not None\n"
    "                else None\n"
    "            )\n"
    "            if (\n"
    "                expected_tokens is not None\n"
    "                and self.tp_size > 1\n"
    "                and n < expected_tokens\n"
    "            ):\n"
    "                expected_shard = (\n"
    "                    expected_tokens + self.tp_size - 1\n"
    "                ) // self.tp_size\n"
    "                if n not in {expected_shard, expected_tokens // self.tp_size}:\n"
    "                    raise AssertionError(\n"
    "                        'GLM51 route capture got an unexpected TP shard: '\n"
    "                        f'local={n}, total={expected_tokens}, tp={self.tp_size}'\n"
    "                    )\n"
    "                import torch.distributed as dist\n"
    "\n"
    "                gathered = torch.empty(\n"
    "                    (max(expected_tokens, self.tp_size), topk_ids.shape[1]),\n"
    "                    dtype=topk_ids.dtype,\n"
    "                    device=topk_ids.device,\n"
    "                )\n"
    "                outputs = list(torch.tensor_split(gathered, self.tp_size, dim=0))\n"
    "                dist.all_gather(outputs, topk_ids, get_tp_group().device_group)\n"
    "                topk_ids = gathered\n"
    "                start_loc = 0\n"
    "                end_loc = expected_tokens\n"
    "                token_num_per_dp = expected_tokens\n"
    "                diagnostic_count = getattr(type(self), '_glm51_gather_diag', 0)\n"
    "                if diagnostic_count < 8:\n"
    "                    type(self)._glm51_gather_diag = diagnostic_count + 1\n"
    "                    print(\n"
    "                        'GLM51_ROUTE_CAPTURE_GATHER '\n"
    "                        f'layer={layer_id} local={n} total={expected_tokens} '\n"
    "                        f'tp={self.tp_size}',\n"
    "                        flush=True,\n"
    "                    )\n"
    "        else:  # multi dp\n"
)


def matches_release(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(expected + "+")


def assert_versions() -> None:
    actual_vllm = version("vllm")
    actual_ascend = version("vllm-ascend")
    if not matches_release(actual_vllm, EXPECTED_VLLM_VERSION):
        raise RuntimeError(f"expected vllm={EXPECTED_VLLM_VERSION}, got {actual_vllm}")
    if not matches_release(actual_ascend, EXPECTED_ASCEND_VERSION):
        raise RuntimeError(
            f"expected vllm-ascend={EXPECTED_ASCEND_VERSION}, got {actual_ascend}"
        )


def installed_source(
    import_name: str,
    distribution_name: str,
    package_prefix: str,
    relative: str,
) -> Path:
    spec = find_spec(import_name)
    if spec is not None and spec.submodule_search_locations is not None:
        paths = [Path(root) / relative for root in spec.submodule_search_locations]
        existing = [path for path in paths if path.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise RuntimeError(f"multiple installed sources for {relative}: {existing}")
    metadata_path = (
        Path(distribution(distribution_name).locate_file(package_prefix)) / relative
    )
    if not metadata_path.is_file():
        raise RuntimeError(f"installed source not found: {metadata_path}")
    return metadata_path


def patch_fused(source: str) -> str:
    if FUSED_MARKER in source:
        raise RuntimeError("fused-MoE capture patch is already present")
    if source.count(FUSED_ANCHOR) != 1:
        raise RuntimeError("unexpected fused_moe.py layout")
    patched = source.replace(FUSED_ANCHOR, FUSED_CAPTURE + FUSED_ANCHOR, 1)
    compile(patched, FUSED_RELATIVE, "exec")
    return patched


def patch_runner(source: str) -> str:
    if RUNNER_MARKER in source:
        raise RuntimeError("model-runner capture patch is already present")
    if source.count(RUNNER_ANCHOR) != 1:
        raise RuntimeError("unexpected model_runner_v1.py layout")
    patched = source.replace(RUNNER_ANCHOR, RUNNER_REPLACEMENT, 1)
    compile(patched, RUNNER_RELATIVE, "exec")
    return patched


def patch_capture(source: str) -> str:
    if CAPTURE_MARKER in source:
        raise RuntimeError("vLLM TP8 capture-gather patch is already present")
    if source.count(CAPTURE_ANCHOR) != 1:
        raise RuntimeError("unexpected routed_experts_capturer.py layout")
    patched = source.replace(CAPTURE_ANCHOR, CAPTURE_REPLACEMENT, 1)
    compile(patched, CAPTURE_RELATIVE, "exec")
    return patched


def verify_fused(source: str) -> None:
    if source.count(FUSED_MARKER) != 1:
        raise RuntimeError("fused-MoE capture marker is missing or duplicated")
    if source.index(FUSED_MARKER) > source.index(
        "        prepare_output = _EXTRA_CTX.moe_comm_method.prepare("
    ):
        raise RuntimeError("route capture is after communication prepare()")
    if "_glm51_capturer.capture(" not in source:
        raise RuntimeError("route capture call is missing")
    compile(source, FUSED_RELATIVE, "exec")


def verify_runner(source: str) -> None:
    if source.count(RUNNER_MARKER) != 1:
        raise RuntimeError("model-runner capture marker is missing or duplicated")
    if "AscendFusedMoE._glm51_global_route_capturer = capturer" not in source:
        raise RuntimeError("global capturer binding is missing")
    compile(source, RUNNER_RELATIVE, "exec")


def verify_capture(source: str) -> None:
    if source.count(CAPTURE_MARKER) != 1:
        raise RuntimeError("vLLM TP8 capture-gather marker is missing or duplicated")
    marker_index = source.index(CAPTURE_MARKER)
    multi_dp_index = source.index("        else:  # multi dp", marker_index)
    buffer_index = source.index(
        "        self.device_buffer[:token_num_per_dp, layer_id, :]"
    )
    if not marker_index < multi_dp_index < buffer_index:
        raise RuntimeError("TP8 gather is not before the route-buffer write")
    if "dist.all_gather(outputs, topk_ids, get_tp_group().device_group)" not in source:
        raise RuntimeError("TP8 gather collective is missing")
    compile(source, CAPTURE_RELATIVE, "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert_versions()
    fused_path = installed_source(
        "vllm_ascend", "vllm-ascend", "vllm_ascend", FUSED_RELATIVE
    )
    runner_path = installed_source(
        "vllm_ascend", "vllm-ascend", "vllm_ascend", RUNNER_RELATIVE
    )
    capture_path = installed_source("vllm", "vllm", "vllm", CAPTURE_RELATIVE)
    fused_source = fused_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8")
    capture_source = capture_path.read_text(encoding="utf-8")
    if args.verify:
        verify_fused(fused_source)
        verify_runner(runner_source)
        verify_capture(capture_source)
        print(
            "GLM51_ROUTE_CAPTURE_PATCH_OK "
            f"fused={fused_path} runner={runner_path} capture={capture_path}"
        )
        return 0
    fused_path.write_text(patch_fused(fused_source), encoding="utf-8")
    runner_path.write_text(patch_runner(runner_source), encoding="utf-8")
    capture_path.write_text(patch_capture(capture_source), encoding="utf-8")
    print(
        "GLM51_ROUTE_CAPTURE_PATCH_APPLIED "
        f"fused={fused_path} runner={runner_path} capture={capture_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
