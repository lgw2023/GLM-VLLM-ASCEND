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
W8A8_PATCH_MARKER = "# DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V6"
W8A8_ANCHOR = (
    "        assert topk_ids is not None\n"
    "        assert topk_weights is not None\n"
)
W8A8_CAPTURE_BLOCK = (
    "        # DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V6\n"
    "        # Ascend W8A8 does not call vLLM's router capture hook. Bind the\n"
    "        # capturer directly on the FusedMoE layer before any remapping.\n"
    "        capturer = getattr(layer, \"_ascend_routed_experts_capturer\", None)\n"
    "        if capturer is not None:\n"
    "            capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)\n"
    "        else:\n"
    "            capture_fn = getattr(getattr(layer, \"router\", None), \"capture_fn\", None)\n"
    "            if capture_fn is not None:\n"
    "                capture_fn(topk_ids)\n"
    "\n"
)

# vLLM-Ascend's worker patch is not imported when vLLM is exactly 0.22.1.
# Patch the vLLM class that the release actually instantiates instead.
CAPTURE_PACKAGE = "vllm"
CAPTURE_TARGET_RELATIVE_PATH = (
    "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"
)
CAPTURE_PACKAGE_RELATIVE_PATH = "model_executor/layers/fused_moe/routed_experts_capturer.py"
CAPTURE_PATCH_MARKER = "# DEEPSEEK_V4_VLLM_TP8_CAPTURE_GATHER_V6"
CAPTURE_ANCHOR = (
    "        ctx = get_forward_context()\n"
    "        if ctx.dp_metadata is None:  # single dp\n"
    "            start_loc = 0\n"
    "            end_loc = topk_ids.shape[0]\n"
    "            token_num_per_dp = topk_ids.shape[0]\n"
    "        else:  # multi dp\n"
)
# Ascend PrepareAndFinalizeWithAll2All.prepare() tensor_splits tokens across TP
# even when FLASHCOMM1/SP is disabled. Local topk_ids therefore cover only one
# shard; reconstruct with an uneven-safe gather, then trim pad rows.
CAPTURE_REPLACEMENT = (
    "        ctx = get_forward_context()\n"
    "        # DEEPSEEK_V4_VLLM_TP8_CAPTURE_GATHER_V6\n"
    "        if ctx.dp_metadata is None:  # single DP; Ascend may still TP-split\n"
    "            n = int(topk_ids.shape[0])\n"
    "            metadata = ctx.attn_metadata\n"
    "            if isinstance(metadata, list):\n"
    "                metadata = next((item for item in metadata if item), {})\n"
    "            if isinstance(metadata, dict) and metadata:\n"
    "                first_metadata = next(\n"
    "                    (item for item in metadata.values() if item is not None),\n"
    "                    None,\n"
    "                )\n"
    "            else:\n"
    "                first_metadata = metadata\n"
    "            actual_tokens = int(getattr(first_metadata, \"num_actual_tokens\", 0) or 0)\n"
    "            fallback_tokens = int(getattr(ctx, \"num_tokens\", 0) or 0)\n"
    "            hinted_tokens = actual_tokens or fallback_tokens\n"
    "            if self.tp_size > 1:\n"
    "                local_size = torch.tensor(\n"
    "                    [n], dtype=torch.int64, device=topk_ids.device\n"
    "                )\n"
    "                size_tensors = [\n"
    "                    torch.empty(1, dtype=torch.int64, device=topk_ids.device)\n"
    "                    for _ in range(self.tp_size)\n"
    "                ]\n"
    "                torch.distributed.all_gather(\n"
    "                    size_tensors,\n"
    "                    local_size,\n"
    "                    group=get_tp_group().device_group,\n"
    "                )\n"
    "                sizes = [int(value.item()) for value in size_tensors]\n"
    "                gathered_tokens = int(sum(sizes))\n"
    "                if hinted_tokens and hinted_tokens <= gathered_tokens:\n"
    "                    token_num_per_dp = hinted_tokens\n"
    "                elif (\n"
    "                    gathered_tokens == self.tp_size\n"
    "                    and min(sizes) == max(sizes) == 1\n"
    "                ):\n"
    "                    # N < TP pad-to-TP path without a reliable hint: keep one\n"
    "                    # real row (typical single-token decode).\n"
    "                    token_num_per_dp = 1\n"
    "                else:\n"
    "                    token_num_per_dp = gathered_tokens\n"
    "                if gathered_tokens != n:\n"
    "                    max_n = max(sizes)\n"
    "                    if n < max_n:\n"
    "                        pad = topk_ids.new_zeros((max_n - n, topk_ids.shape[1]))\n"
    "                        shard = torch.cat([topk_ids, pad], dim=0)\n"
    "                    else:\n"
    "                        shard = topk_ids\n"
    "                    gathered = get_tp_group().all_gather(shard, dim=0)\n"
    "                    pieces = []\n"
    "                    for rank, size in enumerate(sizes):\n"
    "                        begin = rank * max_n\n"
    "                        pieces.append(gathered[begin : begin + size])\n"
    "                    topk_ids = torch.cat(pieces, dim=0)\n"
    "                start_loc = 0\n"
    "                end_loc = token_num_per_dp\n"
    "                if int(topk_ids.shape[0]) < end_loc:\n"
    "                    raise AssertionError(\n"
    "                        \"RoutedExpertsCapturer: reconstructed routes shorter \"\n"
    "                        f\"than token_num_per_dp={token_num_per_dp}, got \"\n"
    "                        f\"{int(topk_ids.shape[0])}\"\n"
    "                    )\n"
    "            else:\n"
    "                start_loc = 0\n"
    "                end_loc = n\n"
    "                token_num_per_dp = n\n"
    "        else:  # multi dp\n"
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
    if package == W8A8_PACKAGE:
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


def patch_capture_source(source: str) -> str:
    if CAPTURE_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek vLLM TP8 capture-gather patch is already present")
    if source.count(CAPTURE_ANCHOR) != 1:
        raise RuntimeError("unexpected vLLM routed-experts capture source layout")
    patched = source.replace(CAPTURE_ANCHOR, CAPTURE_REPLACEMENT, 1)
    compile(patched, CAPTURE_TARGET_RELATIVE_PATH, "exec")
    return patched


def verify_w8a8_source(source: str) -> None:
    if source.count(W8A8_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek W8A8 route-capture marker is missing or duplicated")
    marker_index = source.index(W8A8_PATCH_MARKER)
    capture_index = source.index("capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)")
    zero_expert_index = source.index("        if zero_expert_num > 0")
    force_balance_index = source.index("        if enable_force_load_balance:")
    if not marker_index < capture_index < zero_expert_index < force_balance_index:
        raise RuntimeError("W8A8 capture hook is not before remapping/load balancing")
    compile(source, W8A8_TARGET_RELATIVE_PATH, "exec")


def verify_capture_source(source: str) -> None:
    if source.count(CAPTURE_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek vLLM TP8 capture-gather marker is missing or duplicated")
    marker_index = source.index(CAPTURE_PATCH_MARKER)
    gather_index = source.index("get_tp_group().all_gather(shard, dim=0)")
    multi_dp_index = source.index("        else:  # multi dp")
    buffer_write_index = source.index("        self.device_buffer[:token_num_per_dp, layer_id, :]")
    if not marker_index < gather_index < multi_dp_index < buffer_write_index:
        raise RuntimeError("TP8 gather hook is not before the routed-experts buffer write")
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
    capture_path = target_path(CAPTURE_PACKAGE, CAPTURE_PACKAGE_RELATIVE_PATH)
    w8a8_source = w8a8_path.read_text(encoding="utf-8")
    capture_source = capture_path.read_text(encoding="utf-8")

    if args.verify:
        verify_w8a8_source(w8a8_source)
        verify_capture_source(capture_source)
        print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_OK path={w8a8_path}")
        print(f"DEEPSEEK_VLLM_TP8_CAPTURE_GATHER_PATCH_OK path={capture_path}")
        return 0

    w8a8_path.write_text(patch_w8a8_source(w8a8_source), encoding="utf-8")
    capture_path.write_text(patch_capture_source(capture_source), encoding="utf-8")
    print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_APPLIED path={w8a8_path}")
    print(f"DEEPSEEK_VLLM_TP8_CAPTURE_GATHER_PATCH_APPLIED path={capture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
