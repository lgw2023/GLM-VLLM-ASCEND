#!/usr/bin/env python3
"""Patch DeepSeek-V4 W8A8 routed-expert capture for Ascend TP=8."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distribution, version
from importlib.util import find_spec
from pathlib import Path


EXPECTED_VLLM_VERSION = "0.22.1"
EXPECTED_VLLM_ASCEND_VERSION = "0.22.1rc1"
W8A8_TARGET_RELATIVE_PATH = "vllm_ascend/quantization/methods/w8a8_dynamic.py"
W8A8_PACKAGE_RELATIVE_PATH = "quantization/methods/w8a8_dynamic.py"
W8A8_EXPECTED_SOURCE_SHA256 = "1dd59f6f8114e19824d559b99cc4a22fed04e54ff0ecd9e853aa3b6a574699e2"
W8A8_PATCH_MARKER = "# DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V3"
W8A8_ANCHOR = (
    "        assert topk_ids is not None\n"
    "        assert topk_weights is not None\n"
)
W8A8_CAPTURE_BLOCK = (
    "        # DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V3\n"
    "        # Keep logical top-k IDs before zero-expert handling, load balancing,\n"
    "        # or logical-to-physical expert mapping can change their meaning.\n"
    "        # Ascend's model runner binds the capturer on the FusedMoE layer,\n"
    "        # not on layer.router.capture_fn.\n"
    "        capturer = getattr(layer, \"_ascend_routed_experts_capturer\", None)\n"
    "        if capturer is not None:\n"
    "            capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)\n"
    "        else:\n"
    "            capture_fn = getattr(getattr(layer, \"router\", None), \"capture_fn\", None)\n"
    "            if capture_fn is not None:\n"
    "                capture_fn(topk_ids)\n"
    "\n"
)

CAPTURE_TARGET_RELATIVE_PATH = "vllm_ascend/patch/worker/patch_routed_experts_capture.py"
CAPTURE_PACKAGE_RELATIVE_PATH = "patch/worker/patch_routed_experts_capture.py"
CAPTURE_EXPECTED_SOURCE_SHA256 = "4358e31c9945c1f3a5ad6a0123c40b244424f8a1b2ce54e747638af3b3e1484c"
CAPTURE_PATCH_MARKER = "# DEEPSEEK_V4_TP8_CAPTURE_GATHER_V3"
CAPTURE_ANCHOR = (
    "    ctx = get_forward_context()\n"
    "    if ctx.dp_metadata is None:  # single dp\n"
    "        start_loc = 0\n"
    "        end_loc = topk_ids.shape[0]\n"
    "        token_num_per_dp = topk_ids.shape[0]\n"
    "    else:  # multi dp\n"
)
CAPTURE_REPLACEMENT = (
    "    ctx = get_forward_context()\n"
    "    # DEEPSEEK_V4_TP8_CAPTURE_GATHER_V3\n"
    "    if ctx.dp_metadata is None:  # single DP, possibly sequence-parallel\n"
    "        metadata = ctx.attn_metadata\n"
    "        if isinstance(metadata, list):\n"
    "            metadata = next((item for item in metadata if item), {})\n"
    "        if isinstance(metadata, dict) and metadata:\n"
    "            first_metadata = next(iter(metadata.values()))\n"
    "            actual_tokens = getattr(first_metadata, \"num_actual_tokens\", 0)\n"
    "        else:\n"
    "            actual_tokens = 0\n"
    "        token_num_per_dp = int(actual_tokens or getattr(ctx, \"num_tokens\", topk_ids.shape[0]))\n"
    "        token_num_per_dp = max(token_num_per_dp, topk_ids.shape[0])\n"
    "\n"
    "        # With DP=1 and TP>1, Ascend sequence parallelism shards prefill\n"
    "        # tokens across TP ranks. The original capture path returned only\n"
    "        # rank 0's shard, leaving the remaining request slots as zeros.\n"
    "        if self.tp_size > 1 and topk_ids.shape[0] < token_num_per_dp:\n"
    "            gathered_rows = topk_ids.shape[0] * self.tp_size\n"
    "            if gathered_rows < token_num_per_dp:\n"
    "                raise AssertionError(\n"
    "                    \"RoutedExpertsCapturer: TP gather is shorter than the \"\n"
    "                    f\"actual token count: gathered={gathered_rows}, actual={token_num_per_dp}\"\n"
    "                )\n"
    "            gathered_topk_ids = torch.empty(\n"
    "                (gathered_rows, topk_ids.shape[1]),\n"
    "                dtype=topk_ids.dtype,\n"
    "                device=topk_ids.device,\n"
    "            )\n"
    "            gathered_splits = torch.tensor_split(gathered_topk_ids, self.tp_size, dim=0)\n"
    "            dist.all_gather(list(gathered_splits), topk_ids, get_tp_group().device_group)\n"
    "            topk_ids = gathered_topk_ids\n"
    "        start_loc = 0\n"
    "        end_loc = token_num_per_dp\n"
    "    else:  # multi dp\n"
)


def target_path(package_relative_path: str) -> Path:
    """Resolve an installed vllm-ascend source file, including editable images."""
    spec = find_spec("vllm_ascend")
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
                "multiple installed vllm-ascend source files found: "
                + ", ".join(str(path) for path in existing)
            )

    package = distribution("vllm-ascend")
    path = Path(package.locate_file(f"vllm_ascend/{package_relative_path}"))
    if not path.is_file():
        raise RuntimeError(
            "installed vllm-ascend source file not found via Python import path "
            f"or distribution metadata: {path}"
        )
    return path


def matches_release(actual: str, expected: str) -> bool:
    """Accept PEP 440 local build metadata without weakening the release gate."""
    return actual == expected or actual.startswith(f"{expected}+")


def assert_package_versions() -> None:
    actual_vllm = version("vllm")
    actual_ascend = version("vllm-ascend")
    if not matches_release(actual_vllm, EXPECTED_VLLM_VERSION):
        raise RuntimeError(
            f"expected vllm={EXPECTED_VLLM_VERSION}, got {actual_vllm}"
        )
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


def patch_tp_capture_source(source: str) -> str:
    if CAPTURE_PATCH_MARKER in source:
        raise RuntimeError("DeepSeek TP8 capture-gather patch is already present")
    if source.count(CAPTURE_ANCHOR) != 1:
        raise RuntimeError("unexpected routed-experts capture source layout")
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


def verify_tp_capture_source(source: str) -> None:
    if source.count(CAPTURE_PATCH_MARKER) != 1:
        raise RuntimeError("DeepSeek TP8 capture-gather marker is missing or duplicated")
    marker_index = source.index(CAPTURE_PATCH_MARKER)
    gather_index = source.index("dist.all_gather(list(gathered_splits), topk_ids, get_tp_group().device_group)")
    multi_dp_index = source.index("    else:  # multi dp")
    buffer_write_index = source.index("    self.device_buffer[:token_num_per_dp, layer_id, :] =")
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
    w8a8_path = target_path(W8A8_PACKAGE_RELATIVE_PATH)
    capture_path = target_path(CAPTURE_PACKAGE_RELATIVE_PATH)
    w8a8_source = w8a8_path.read_text(encoding="utf-8")
    capture_source = capture_path.read_text(encoding="utf-8")

    if args.verify:
        verify_w8a8_source(w8a8_source)
        verify_tp_capture_source(capture_source)
        print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_OK path={w8a8_path}")
        print(f"DEEPSEEK_TP8_CAPTURE_GATHER_PATCH_OK path={capture_path}")
        return 0

    w8a8_sha256 = hashlib.sha256(w8a8_source.encode("utf-8")).hexdigest()
    if w8a8_sha256 != W8A8_EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "unexpected v0.22.1rc1 W8A8 source SHA-256: "
            f"expected {W8A8_EXPECTED_SOURCE_SHA256}, got {w8a8_sha256}"
        )
    capture_sha256 = hashlib.sha256(capture_source.encode("utf-8")).hexdigest()
    if capture_sha256 != CAPTURE_EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "unexpected v0.22.1rc1 routed-experts capture source SHA-256: "
            f"expected {CAPTURE_EXPECTED_SOURCE_SHA256}, got {capture_sha256}"
        )

    w8a8_path.write_text(patch_w8a8_source(w8a8_source), encoding="utf-8")
    capture_path.write_text(patch_tp_capture_source(capture_source), encoding="utf-8")
    print(f"DEEPSEEK_W8A8_ROUTE_CAPTURE_PATCH_APPLIED path={w8a8_path}")
    print(f"DEEPSEEK_TP8_CAPTURE_GATHER_PATCH_APPLIED path={capture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
