#!/usr/bin/env python3
"""Patch DeepSeek-V4 W8A8 routed-expert capture for Ascend TP=8."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution, version
from importlib.util import find_spec
from pathlib import Path


EXPECTED_VLLM_VERSION = "0.22.1"
EXPECTED_VLLM_ASCEND_VERSION = "0.22.1rc1"

W8A8_PACKAGE = "vllm_ascend"
W8A8_TARGET_RELATIVE_PATH = "vllm_ascend/quantization/methods/w8a8_dynamic.py"
W8A8_PACKAGE_RELATIVE_PATH = "quantization/methods/w8a8_dynamic.py"
W8A8_PATCH_MARKER = "# DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V9"
W8A8_ANCHOR = (
    "        assert topk_ids is not None\n"
    "        assert topk_weights is not None\n"
)
# Do NOT capture the post-prepare local shard here. v6-v8 gathered or
# captured inside apply() after All2All tensor_split; that left most
# prefill rows as zeros. Capture now happens in AscendFusedMoE.forward
# before prepare().
W8A8_CAPTURE_BLOCK = (
    "        # DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V9\n"
    "        # Capture is done in AscendFusedMoE.forward before prepare()\n"
    "        # (full router logits). Skipping post-split capture here avoids\n"
    "        # overwriting the capturer buffer with a TP-local shard.\n"
)

FUSED_MOE_PACKAGE = "vllm_ascend"
FUSED_MOE_TARGET_RELATIVE_PATH = "vllm_ascend/ops/fused_moe/fused_moe.py"
FUSED_MOE_PACKAGE_RELATIVE_PATH = "ops/fused_moe/fused_moe.py"
FUSED_MOE_PATCH_MARKER = "# DEEPSEEK_V4_FUSED_MOE_CAPTURE_BEFORE_PREPARE_V9"
FUSED_MOE_ANCHOR = (
    "        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(\n"
    "            hidden_states=hidden_states,\n"
    "            router_logits=router_logits,\n"
    "            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,\n"
    "            enable_shared_expert_dp=self.enable_shared_expert_dp,\n"
    "            quant_type=self.quant_type,\n"
    "        )\n"
)
FUSED_MOE_CAPTURE_BLOCK = (
    "        # DEEPSEEK_V4_FUSED_MOE_CAPTURE_BEFORE_PREPARE_V9\n"
    "        # All2All/MC2 prepare() tensor_splits tokens across TP. Capture\n"
    "        # logical routes on the full batch before that split.\n"
    "        # DEEPSEEK_ROUTE_CAPTURE_DIAG: temporary shape dump (rate-limited).\n"
    "        _route_capturer = getattr(self, \"_ascend_routed_experts_capturer\", None)\n"
    "        if _route_capturer is None:\n"
    "            _route_capturer = getattr(\n"
    "                type(self), \"_ds_global_route_capturer\", None\n"
    "            )\n"
    "        _route_enabled = (\n"
    "            self.vllm_config.model_config is not None\n"
    "            and self.vllm_config.model_config.enable_return_routed_experts\n"
    "        )\n"
    "        if _route_capturer is not None and _route_enabled:\n"
    "            _capture_ids = None\n"
    "            _capture_src = \"select_experts\"\n"
    "            if self.multistream_overlap_gate:\n"
    "                _fc3 = get_flash_common3_context()\n"
    "                if _fc3 is not None and getattr(_fc3, \"topk_ids\", None) is not None:\n"
    "                    _capture_ids = _fc3.topk_ids\n"
    "                    _capture_src = \"flash_common3\"\n"
    "            if _capture_ids is None:\n"
    "                _capture_weights, _capture_ids = select_experts(\n"
    "                    hidden_states=hidden_states,\n"
    "                    router_logits=router_logits,\n"
    "                    top_k=self.top_k,\n"
    "                    use_grouped_topk=self.use_grouped_topk,\n"
    "                    renormalize=self.renormalize,\n"
    "                    topk_group=self.topk_group,\n"
    "                    num_expert_group=self.num_expert_group,\n"
    "                    custom_routing_function=self.custom_routing_function,\n"
    "                    scoring_func=self.scoring_func,\n"
    "                    routed_scaling_factor=self._original_routed_scaling_factor,\n"
    "                    e_score_correction_bias=self.e_score_correction_bias,\n"
    "                    num_experts=self.moe_config.num_experts,\n"
    "                    input_ids=getattr(get_forward_context(), \"input_ids\", None),\n"
    "                    tid2eid=self.tid2eid,\n"
    "                )\n"
    "            _diag_n = getattr(type(self), \"_ds_route_diag_n\", 0)\n"
    "            if _diag_n < 16 and self.layer_id in (0, 1, 2, 3, 4, 20, 40):\n"
    "                type(self)._ds_route_diag_n = _diag_n + 1\n"
    "                print(\n"
    "                    \"DEEPSEEK_ROUTE_CAPTURE_DIAG pre_prepare \"\n"
    "                    f\"layer={self.layer_id} src={_capture_src} \"\n"
    "                    f\"hidden={tuple(hidden_states.shape)} \"\n"
    "                    f\"router={tuple(router_logits.shape)} \"\n"
    "                    f\"topk={tuple(_capture_ids.shape)} \"\n"
    "                    f\"tp={getattr(self.moe_config, 'tp_size', '?')} \"\n"
    "                    f\"flashcomm1={getattr(_EXTRA_CTX, 'flash_comm_v1_enabled', '?')}\",\n"
    "                    flush=True,\n"
    "                )\n"
    "            _route_capturer.capture(\n"
    "                layer_id=self.layer_id, topk_ids=_capture_ids\n"
    "            )\n"
    "        else:\n"
    "            _miss_n = getattr(type(self), \"_ds_route_diag_miss\", 0)\n"
    "            if _miss_n < 8 and self.layer_id in (0, 1, 2, 3, 4):\n"
    "                type(self)._ds_route_diag_miss = _miss_n + 1\n"
    "                print(\n"
    "                    \"DEEPSEEK_ROUTE_CAPTURE_DIAG pre_prepare_skip \"\n"
    "                    f\"layer={self.layer_id} capturer={_route_capturer is not None} \"\n"
    "                    f\"enabled={_route_enabled} \"\n"
    "                    f\"hidden={tuple(hidden_states.shape)} \"\n"
    "                    f\"router={tuple(router_logits.shape)}\",\n"
    "                    flush=True,\n"
    "                )\n"
    "\n"
)

MODEL_RUNNER_PACKAGE = "vllm_ascend"
MODEL_RUNNER_TARGET_RELATIVE_PATH = "vllm_ascend/worker/model_runner_v1.py"
MODEL_RUNNER_PACKAGE_RELATIVE_PATH = "worker/model_runner_v1.py"
MODEL_RUNNER_PATCH_MARKER = "# DEEPSEEK_V4_MODEL_RUNNER_BIND_CAPTURE_V9"
MODEL_RUNNER_BIND_ANCHOR = (
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
MODEL_RUNNER_BIND_REPLACEMENT = (
    "    def _bind_routed_experts_capturer(self, capturer=None) -> None:\n"
    "        # DEEPSEEK_V4_MODEL_RUNNER_BIND_CAPTURE_V9\n"
    "        # Ascend W8A8 capture reads ``_ascend_routed_experts_capturer`` on\n"
    "        # the live MoE layer. DeepSeek-V4 may not match the old\n"
    "        # ``isinstance(..., FusedMoE)`` scan over static_forward_context,\n"
    "        # so bind both the registry and the loaded model tree, and keep a\n"
    "        # class-level fallback for the custom-op hot path.\n"
    "        from vllm.model_executor.layers.fused_moe.layer import FusedMoE\n"
    "        from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE\n"
    "\n"
    "        AscendFusedMoE._ds_global_route_capturer = capturer\n"
    "        bound_ids: set[int] = set()\n"
    "        static_fused = 0\n"
    "        model_fused = 0\n"
    "\n"
    "        def _maybe_bind(module) -> None:\n"
    "            nonlocal static_fused, model_fused\n"
    "            is_fused = isinstance(module, FusedMoE)\n"
    "            is_ascend = isinstance(module, AscendFusedMoE)\n"
    "            is_duck = (\n"
    "                not is_fused\n"
    "                and hasattr(module, \"moe_config\")\n"
    "                and hasattr(module, \"layer_name\")\n"
    "                and hasattr(module, \"forward_impl\")\n"
    "            )\n"
    "            if not (is_fused or is_ascend or is_duck):\n"
    "                return\n"
    "            module_id = id(module)\n"
    "            if module_id in bound_ids:\n"
    "                return\n"
    "            module._ascend_routed_experts_capturer = capturer\n"
    "            bound_ids.add(module_id)\n"
    "\n"
    "        for module in self.compilation_config.static_forward_context.values():\n"
    "            before = len(bound_ids)\n"
    "            _maybe_bind(module)\n"
    "            if len(bound_ids) > before:\n"
    "                static_fused += 1\n"
    "\n"
    "        model = getattr(self, \"model\", None)\n"
    "        if model is not None:\n"
    "            for module in model.modules():\n"
    "                before = len(bound_ids)\n"
    "                _maybe_bind(module)\n"
    "                if len(bound_ids) > before:\n"
    "                    model_fused += 1\n"
    "\n"
    "        print(\n"
    "            \"DEEPSEEK_ROUTE_CAPTURE_DIAG bind \"\n"
    "            f\"capturer={'set' if capturer is not None else 'none'} \"\n"
    "            f\"static_ctx={len(self.compilation_config.static_forward_context)} \"\n"
    "            f\"static_bound={static_fused} model_bound={model_fused} \"\n"
    "            f\"total_bound={len(bound_ids)}\",\n"
    "            flush=True,\n"
    "        )\n"
)

MODEL_RUNNER_EARLY_INIT_ANCHOR = (
    "        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:\n"
    "            self._start_dump_data()\n"
    "\n"
    "    def _start_dump_data(self) -> None:\n"
)
MODEL_RUNNER_EARLY_INIT_REPLACEMENT = (
    "        if (\n"
    "            self.model_config.enable_return_routed_experts\n"
    "            and not getattr(self, \"routed_experts_initialized\", False)\n"
    "        ):\n"
    "            self.init_routed_experts_capturer()\n"
    "\n"
    "        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:\n"
    "            self._start_dump_data()\n"
    "\n"
    "    def _start_dump_data(self) -> None:\n"
)

MODEL_RUNNER_REMOVE_LATE_INIT_ANCHOR = (
    "        if has_kv_transfer_group() and not is_profiling:\n"
    "            get_kv_transfer_group().register_kv_caches(kv_caches)\n"
    "\n"
    "        if self.model_config.enable_return_routed_experts:\n"
    "            self.init_routed_experts_capturer()\n"
    "\n"
)
MODEL_RUNNER_REMOVE_LATE_INIT_REPLACEMENT = (
    "        if has_kv_transfer_group() and not is_profiling:\n"
    "            get_kv_transfer_group().register_kv_caches(kv_caches)\n"
    "\n"
)

# vLLM-Ascend's worker patch is not imported when vLLM is exactly 0.22.1.
CAPTURE_PACKAGE = "vllm"
CAPTURE_TARGET_RELATIVE_PATH = (
    "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"
)
CAPTURE_PACKAGE_RELATIVE_PATH = "model_executor/layers/fused_moe/routed_experts_capturer.py"
CAPTURE_PATCH_MARKER = "# DEEPSEEK_V4_VLLM_TP8_CAPTURE_GATHER_V9"
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
    "        # DEEPSEEK_V4_VLLM_TP8_CAPTURE_GATHER_V9\n"
    "        # Pre-prepare capture may still pass TP-local shards when DP=1.\n"
    "        # DEEPSEEK_ROUTE_CAPTURE_DIAG: see buffer-write dump below.\n"
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
    "            tp_shard = (\n"
    "                expected_tokens is not None\n"
    "                and self.tp_size > 1\n"
    "                and n < expected_tokens\n"
    "                and (\n"
    "                    n == (expected_tokens + self.tp_size - 1) // self.tp_size\n"
    "                    or n == expected_tokens // self.tp_size\n"
    "                )\n"
    "            )\n"
    "            if tp_shard:\n"
    "                import torch.distributed as dist\n"
    "                from vllm.distributed.parallel_state import get_tp_group\n"
    "\n"
    "                gather_topk_ids = torch.empty(\n"
    "                    (expected_tokens, topk_ids.shape[1]),\n"
    "                    dtype=topk_ids.dtype,\n"
    "                    device=topk_ids.device,\n"
    "                )\n"
    "                split_topk_ids = torch.tensor_split(\n"
    "                    gather_topk_ids, self.tp_size, dim=0\n"
    "                )\n"
    "                dist.all_gather(\n"
    "                    list(split_topk_ids), topk_ids, get_tp_group().device_group\n"
    "                )\n"
    "                topk_ids = gather_topk_ids\n"
    "                token_num_per_dp = expected_tokens\n"
    "                start_loc = 0\n"
    "                end_loc = expected_tokens\n"
    "                _gather_n = getattr(type(self), \"_ds_route_diag_gather\", 0)\n"
    "                if _gather_n < 8 and layer_id in (0, 1, 2, 3, 4):\n"
    "                    type(self)._ds_route_diag_gather = _gather_n + 1\n"
    "                    print(\n"
    "                        \"DEEPSEEK_ROUTE_CAPTURE_DIAG capturer_gather \"\n"
    "                        f\"layer={layer_id} local_n={n} \"\n"
    "                        f\"expected={expected_tokens} gathered={tuple(topk_ids.shape)}\",\n"
    "                        flush=True,\n"
    "                    )\n"
    "        else:  # multi dp\n"
)

# Injected immediately before the device_buffer write.
CAPTURE_BUFFER_ANCHOR = (
    "        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[\n"
    "            start_loc:end_loc, :\n"
    "        ]\n"
)
CAPTURE_BUFFER_REPLACEMENT = (
    "        # DEEPSEEK_ROUTE_CAPTURE_DIAG temporary buffer-write dump.\n"
    "        _diag_n = getattr(type(self), \"_ds_route_diag_n\", 0)\n"
    "        if _diag_n < 16 and layer_id in (0, 1, 2, 3, 4, 20, 40):\n"
    "            type(self)._ds_route_diag_n = _diag_n + 1\n"
    "            _slice = topk_ids[start_loc:end_loc]\n"
    "            _nz = (\n"
    "                int((_slice != 0).any(dim=-1).sum().item())\n"
    "                if _slice.numel() > 0\n"
    "                else 0\n"
    "            )\n"
    "            print(\n"
    "                \"DEEPSEEK_ROUTE_CAPTURE_DIAG capturer_write \"\n"
    "                f\"layer={layer_id} in_shape={tuple(topk_ids.shape)} \"\n"
    "                f\"write_rows={token_num_per_dp} start={start_loc} end={end_loc} \"\n"
    "                f\"nonzero_rows={_nz} tp_size={self.tp_size} dp_rank={self.dp_rank}\",\n"
    "                flush=True,\n"
    "            )\n"
    "        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[\n"
    "            start_loc:end_loc, :\n"
    "        ]\n"
)


def package_source_path(
    import_name: str,
    package_prefix: str,
    package_relative_path: str,
    distribution_name: str,
) -> Path:
    """Resolve a source file in an editable or wheel installation."""
    spec = find_spec(import_name)
    if spec is not None and spec.submodule_search_locations is not None:
        candidates = [
            Path(location) / package_relative_path
            for location in spec.submodule_search_locations
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise RuntimeError(
                f"multiple installed {package_prefix} source files found: "
                + ", ".join(str(path) for path in existing)
            )

    package = distribution(distribution_name)
    path = Path(package.locate_file(f"{package_prefix}/{package_relative_path}"))
    if not path.is_file():
        raise RuntimeError(
            f"installed {package_prefix} source file not found via Python import "
            f"path or distribution metadata: {path}"
        )
    return path


def target_path(package: str, package_relative_path: str) -> Path:
    if package in {W8A8_PACKAGE, FUSED_MOE_PACKAGE, MODEL_RUNNER_PACKAGE}:
        return package_source_path(
            "vllm_ascend",
            "vllm_ascend",
            package_relative_path,
            "vllm-ascend",
        )
    if package == CAPTURE_PACKAGE:
        return package_source_path("vllm", "vllm", package_relative_path, "vllm")
    raise ValueError(f"unsupported package: {package}")


def matches_release(actual: str, expected: str) -> bool:
    """Accept PEP 440 local build metadata without weakening the release gate."""
    return actual == expected or actual.startswith(f"{expected}+")


def assert_package_versions() -> None:
    actual_vllm = version("vllm")
    actual_ascend = version("vllm-ascend")
    if not matches_release(actual_vllm, EXPECTED_VLLM_VERSION):
        raise RuntimeError(f"expected vllm={EXPECTED_VLLM_VERSION}, got {actual_vllm}")
    if not matches_release(actual_ascend, EXPECTED_VLLM_ASCEND_VERSION):
        raise RuntimeError(
            "expected vllm-ascend="
            f"{EXPECTED_VLLM_ASCEND_VERSION}, got {actual_ascend}"
        )


def patch_w8a8_source(source: str) -> str:
    if W8A8_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek W8A8 route-capture patch is already present")
    if source.count(W8A8_ANCHOR) != 1:
        raise RuntimeError("unexpected W8A8 source layout; capture anchor is not unique")
    patched = source.replace(W8A8_ANCHOR, W8A8_ANCHOR + W8A8_CAPTURE_BLOCK, 1)
    compile(patched, W8A8_TARGET_RELATIVE_PATH, "exec")
    return patched


def patch_fused_moe_source(source: str) -> str:
    if FUSED_MOE_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek fused-moe pre-prepare capture patch is already present")
    if source.count(FUSED_MOE_ANCHOR) != 1:
        raise RuntimeError("unexpected fused_moe source layout; prepare anchor is not unique")
    # Ensure helpers used by the injected block are imported.
    if "get_flash_common3_context" not in source:
        raise RuntimeError("fused_moe.py is missing get_flash_common3_context import")
    if "from vllm.forward_context import get_forward_context" not in source and (
        "get_forward_context" not in source
    ):
        raise RuntimeError("fused_moe.py is missing get_forward_context")
    patched = source.replace(FUSED_MOE_ANCHOR, FUSED_MOE_CAPTURE_BLOCK + FUSED_MOE_ANCHOR, 1)
    compile(patched, FUSED_MOE_TARGET_RELATIVE_PATH, "exec")
    return patched


def patch_model_runner_source(source: str) -> str:
    if MODEL_RUNNER_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek model-runner bind patch is already present")
    if source.count(MODEL_RUNNER_BIND_ANCHOR) != 1:
        raise RuntimeError("unexpected model_runner bind anchor is not unique")
    if source.count(MODEL_RUNNER_EARLY_INIT_ANCHOR) != 1:
        raise RuntimeError("unexpected model_runner load_model anchor is not unique")
    patched = source.replace(MODEL_RUNNER_BIND_ANCHOR, MODEL_RUNNER_BIND_REPLACEMENT, 1)
    patched = patched.replace(MODEL_RUNNER_EARLY_INIT_ANCHOR, MODEL_RUNNER_EARLY_INIT_REPLACEMENT, 1)
    if MODEL_RUNNER_REMOVE_LATE_INIT_ANCHOR in patched:
        patched = patched.replace(MODEL_RUNNER_REMOVE_LATE_INIT_ANCHOR, MODEL_RUNNER_REMOVE_LATE_INIT_REPLACEMENT, 1)
    compile(patched, MODEL_RUNNER_TARGET_RELATIVE_PATH, "exec")
    return patched


def patch_capture_source(source: str) -> str:
    if CAPTURE_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek vLLM TP8 capture-gather patch is already present")
    if source.count(CAPTURE_ANCHOR) != 1:
        raise RuntimeError("unexpected vLLM routed-experts capture source layout")
    if source.count(CAPTURE_BUFFER_ANCHOR) != 1:
        raise RuntimeError("unexpected vLLM routed-experts buffer-write layout")
    patched = source.replace(CAPTURE_ANCHOR, CAPTURE_REPLACEMENT, 1)
    patched = patched.replace(CAPTURE_BUFFER_ANCHOR, CAPTURE_BUFFER_REPLACEMENT, 1)
    compile(patched, CAPTURE_TARGET_RELATIVE_PATH, "exec")
    return patched


def verify_w8a8_source(source: str) -> None:
    if source.count(W8A8_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek W8A8 route-capture marker is missing or duplicated")
    if "Skipping post-split capture" not in source:
        raise RuntimeError("W8A8 v9 skip-post-split capture note is absent")
    # Must not call capturer with post-split topk_ids anymore.
    if "capturer.capture(layer_id=layer.layer_id, topk_ids=" in source:
        raise RuntimeError("W8A8 still captures post-split topk_ids")
    compile(source, W8A8_TARGET_RELATIVE_PATH, "exec")


def verify_fused_moe_source(source: str) -> None:
    if source.count(FUSED_MOE_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek fused-moe pre-prepare marker is missing or duplicated")
    marker_index = source.index(FUSED_MOE_PATCH_MARKER)
    prepare_index = source.index(
        "        prepare_output = _EXTRA_CTX.moe_comm_method.prepare("
    )
    capture_index = source.index("_route_capturer.capture(")
    if not marker_index < capture_index < prepare_index:
        raise RuntimeError("fused-moe capture is not before prepare()")
    if "DEEPSEEK_ROUTE_CAPTURE_DIAG pre_prepare" not in source:
        raise RuntimeError("fused-moe temporary capture diag log is absent")
    if "_ds_global_route_capturer" not in source:
        raise RuntimeError("fused-moe global capturer fallback is absent")
    compile(source, FUSED_MOE_TARGET_RELATIVE_PATH, "exec")


def verify_model_runner_source(source: str) -> None:
    if source.count(MODEL_RUNNER_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek model-runner bind marker is missing or duplicated")
    if "DEEPSEEK_ROUTE_CAPTURE_DIAG bind" not in source:
        raise RuntimeError("model-runner bind diagnostic log is absent")
    if "AscendFusedMoE._ds_global_route_capturer = capturer" not in source:
        raise RuntimeError("model-runner global capturer fallback is absent")
    if "init_routed_experts_capturer()" not in source.split("def load_model", 1)[1].split("def _start_dump_data", 1)[0]:
        raise RuntimeError("model-runner early capturer init is absent")
    if (
        "        if has_kv_transfer_group() and not is_profiling:\n"
        "            get_kv_transfer_group().register_kv_caches(kv_caches)\n"
        "\n"
        "        if self.model_config.enable_return_routed_experts:\n"
        "            self.init_routed_experts_capturer()\n"
        in source
    ):
        raise RuntimeError("model-runner still initializes capturer at end of kv_cache init")
    compile(source, MODEL_RUNNER_TARGET_RELATIVE_PATH, "exec")


def verify_capture_source(source: str) -> None:
    if source.count(CAPTURE_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek vLLM TP8 capture-gather marker is missing or duplicated")
    marker_index = source.index(CAPTURE_PATCH_MARKER)
    multi_dp_index = source.index("        else:  # multi dp")
    buffer_write_index = source.index("        self.device_buffer[:token_num_per_dp, layer_id, :]")
    if not marker_index < multi_dp_index < buffer_write_index:
        raise RuntimeError("TP8 gather marker is not before the routed-experts buffer write")
    if "Pre-prepare capture" not in source and "pre-prepare capture" not in source:
        raise RuntimeError("TP8 capturer marker missing pre-prepare contract note")
    if "capturer_gather" not in source:
        raise RuntimeError("capturer TP gather diagnostic log is absent")
    if "dist.all_gather" not in source:
        raise RuntimeError("capturer TP gather logic is absent")
    if "DEEPSEEK_ROUTE_CAPTURE_DIAG capturer_write" not in source:
        raise RuntimeError("capturer temporary buffer-write diag log is absent")
    compile(source, CAPTURE_TARGET_RELATIVE_PATH, "exec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify already patched installed source files instead of modifying them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_package_versions()
    w8a8_path = target_path(W8A8_PACKAGE, W8A8_PACKAGE_RELATIVE_PATH)
    fused_moe_path = target_path(FUSED_MOE_PACKAGE, FUSED_MOE_PACKAGE_RELATIVE_PATH)
    model_runner_path = target_path(MODEL_RUNNER_PACKAGE, MODEL_RUNNER_PACKAGE_RELATIVE_PATH)
    capture_path = target_path(CAPTURE_PACKAGE, CAPTURE_PACKAGE_RELATIVE_PATH)
    w8a8_source = w8a8_path.read_text(encoding="utf-8")
    fused_moe_source = fused_moe_path.read_text(encoding="utf-8")
    model_runner_source = model_runner_path.read_text(encoding="utf-8")
    capture_source = capture_path.read_text(encoding="utf-8")

    if args.verify:
        verify_w8a8_source(w8a8_source)
        verify_fused_moe_source(fused_moe_source)
        verify_model_runner_source(model_runner_source)
        verify_capture_source(capture_source)
        print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_OK path={w8a8_path}")
        print(f"DEEPSEEK_FUSED_MOE_CAPTURE_PATCH_OK path={fused_moe_path}")
        print(f"DEEPSEEK_MODEL_RUNNER_BIND_PATCH_OK path={model_runner_path}")
        print(f"DEEPSEEK_VLLM_TP8_CAPTURE_GATHER_PATCH_OK path={capture_path}")
        return 0

    w8a8_path.write_text(patch_w8a8_source(w8a8_source), encoding="utf-8")
    fused_moe_path.write_text(patch_fused_moe_source(fused_moe_source), encoding="utf-8")
    model_runner_path.write_text(patch_model_runner_source(model_runner_source), encoding="utf-8")
    capture_path.write_text(patch_capture_source(capture_source), encoding="utf-8")
    print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_APPLIED path={w8a8_path}")
    print(f"DEEPSEEK_FUSED_MOE_CAPTURE_PATCH_APPLIED path={fused_moe_path}")
    print(f"DEEPSEEK_MODEL_RUNNER_BIND_PATCH_APPLIED path={model_runner_path}")
    print(f"DEEPSEEK_VLLM_TP8_CAPTURE_GATHER_PATCH_APPLIED path={capture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
