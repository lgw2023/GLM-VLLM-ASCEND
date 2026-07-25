#!/usr/bin/env python3
"""Apply the GLM-5.2 W8A8 routed-expert hook to vLLM-Ascend v0.22.1rc1."""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
from importlib.metadata import distribution, version
from pathlib import Path


EXPECTED_VLLM_VERSION = "0.22.1"
EXPECTED_VLLM_ASCEND_VERSION = "0.22.1rc1"
TARGET_RELATIVE_PATH = "vllm_ascend/quantization/methods/w8a8_dynamic.py"
PACKAGE_RELATIVE_PATH = "quantization/methods/w8a8_dynamic.py"
EXPECTED_SOURCE_SHA256 = "1dd59f6f8114e19824d559b99cc4a22fed04e54ff0ecd9e853aa3b6a574699e2"
PATCH_MARKER = "# GLM52_W8A8_ROUTE_CAPTURE_V1"
ANCHOR = (
    "        assert topk_ids is not None\n"
    "        assert topk_weights is not None\n"
)
CAPTURE_BLOCK = (
    "        # GLM52_W8A8_ROUTE_CAPTURE_V1\n"
    "        # Capture logical top-k IDs before zero-expert handling, load balancing,\n"
    "        # or logical-to-physical expert mapping can change their meaning.\n"
    "        capture_fn = getattr(getattr(layer, \"router\", None), \"capture_fn\", None)\n"
    "        if capture_fn is not None:\n"
    "            capture_fn(topk_ids)\n"
    "\n"
)

def target_path() -> Path:
    # Official images may install vllm-ascend as an editable source tree. In
    # that case distribution().locate_file() points under site-packages even
    # though Python imports vllm_ascend from /vllm-workspace/vllm-ascend.
    spec = find_spec("vllm_ascend")
    if spec is not None and spec.submodule_search_locations is not None:
        import_candidates = [
            Path(location) / PACKAGE_RELATIVE_PATH
            for location in spec.submodule_search_locations
        ]
        existing = [path for path in import_candidates if path.is_file()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise RuntimeError(
                "multiple installed W8A8 source files found: "
                + ", ".join(str(path) for path in existing)
            )

    package = distribution("vllm-ascend")
    path = Path(package.locate_file(TARGET_RELATIVE_PATH))
    if not path.is_file():
        raise RuntimeError(
            "installed W8A8 source file not found via Python import path or "
            f"distribution metadata; metadata candidate: {path}"
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


def patch_source(source: str) -> str:
    if PATCH_MARKER in source:
        raise RuntimeError("W8A8 route-capture patch is already present")
    if source.count(ANCHOR) != 1:
        raise RuntimeError("unexpected W8A8 source layout; capture anchor is not unique")
    patched = source.replace(ANCHOR, ANCHOR + CAPTURE_BLOCK, 1)
    compile(patched, TARGET_RELATIVE_PATH, "exec")
    return patched


def verify_source(source: str) -> None:
    if source.count(PATCH_MARKER) != 1:
        raise RuntimeError("W8A8 route-capture marker is missing or duplicated")
    marker_index = source.index(PATCH_MARKER)
    zero_expert_index = source.index("        if zero_expert_num > 0")
    force_balance_index = source.index("        if enable_force_load_balance:")
    if not marker_index < zero_expert_index < force_balance_index:
        raise RuntimeError("W8A8 capture hook is not before remapping/load balancing")
    compile(source, TARGET_RELATIVE_PATH, "exec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an already patched installed source file instead of modifying it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_package_versions()
    path = target_path()
    source = path.read_text(encoding="utf-8")
    if args.verify:
        verify_source(source)
        print(f"W8A8_ROUTE_CAPTURE_PATCH_OK path={path}")
        return 0

    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "unexpected v0.22.1rc1 W8A8 source SHA-256: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    patched = patch_source(source)
    path.write_text(patched, encoding="utf-8")
    print(f"W8A8_ROUTE_CAPTURE_PATCH_APPLIED path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
