#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

BASE_IMAGE_DEFAULT="quay.io/ascend/vllm-ascend:v0.22.1rc1"
OUTPUT_IMAGE_DEFAULT="deepseek-v4-expert-capture:v0.22.1rc1-w8a8-v7"
PATCH_ID_DEFAULT="deepseek-v4-w8a8-logical-topk-v7"
EXPECTED_VLLM_VERSION="0.22.1"
EXPECTED_VLLM_ASCEND_VERSION="0.22.1rc1"

BASE_IMAGE="${BASE_IMAGE_DEFAULT}"
OUTPUT_IMAGE="${OUTPUT_IMAGE_DEFAULT}"
PATCH_ID="${PATCH_ID_DEFAULT}"
PULL_BASE=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/07_build_capture_image.sh [options]

Build the DeepSeek-V4 W8A8 routed-expert capture image from the local official
vLLM-Ascend v0.22.1rc1 base image. It does not download a model, benchmark, or
use any NPU.

Options:
  --confirm-pull-base       Pull the official base only when it is absent.
  --base-image IMAGE        Override the base image.
  --output-image IMAGE      Derived image tag.
  --patch-id ID             Image capture-patch identifier.
  -h, --help                Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --confirm-pull-base)
            PULL_BASE=1
            shift
            ;;
        --base-image)
            BASE_IMAGE="${2:-}"
            shift 2
            ;;
        --output-image)
            OUTPUT_IMAGE="${2:-}"
            shift 2
            ;;
        --patch-id)
            PATCH_ID="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

require_cmd docker
require_cmd sed
require_var BASE_IMAGE
require_var OUTPUT_IMAGE
require_var PATCH_ID

if docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    printf 'Using existing base image: %s\n' "${BASE_IMAGE}"
elif ((PULL_BASE == 1)); then
    docker pull "${BASE_IMAGE}"
fi
docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || \
    die "base image is absent: ${BASE_IMAGE}; load an offline copy or rerun with --confirm-pull-base on a registry-enabled node"

DOCKERFILE="${DEEPSEEK_EXPERIMENT_ROOT}/patches/Dockerfile.route-capture"
PATCH_FILE="${DEEPSEEK_EXPERIMENT_ROOT}/patches/apply_w8a8_route_capture.py"
[[ -f "${DOCKERFILE}" ]] || die "capture Dockerfile is missing: ${DOCKERFILE}"
[[ -f "${PATCH_FILE}" ]] || die "capture patch is missing: ${PATCH_FILE}"

docker build \
    --file "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CAPTURE_PATCH_ID=${PATCH_ID}" \
    --tag "${OUTPUT_IMAGE}" \
    "${DEEPSEEK_EXPERIMENT_ROOT}/patches"

ACTUAL_PATCH_ID="$(docker image inspect "${OUTPUT_IMAGE}" \
    --format '{{if .Config.Labels}}{{index .Config.Labels "deepseek.capture_patch_id"}}{{end}}')"
[[ "${ACTUAL_PATCH_ID}" == "${PATCH_ID}" ]] || \
    die "derived image has unexpected deepseek.capture_patch_id: ${ACTUAL_PATCH_ID:-<empty>}"

PACKAGE_LINES="$(docker run --rm --entrypoint python "${OUTPUT_IMAGE}" -c \
    'from importlib.metadata import version
print("vllm=" + version("vllm"))
print("vllm-ascend=" + version("vllm-ascend"))')"
printf '%s\n' "${PACKAGE_LINES}"
ACTUAL_VLLM_VERSION="$(printf '%s\n' "${PACKAGE_LINES}" | sed -n 's/^vllm=//p')"
ACTUAL_VLLM_ASCEND_VERSION="$(printf '%s\n' "${PACKAGE_LINES}" | sed -n 's/^vllm-ascend=//p')"
[[ "${ACTUAL_VLLM_VERSION}" == "${EXPECTED_VLLM_VERSION}" || \
   "${ACTUAL_VLLM_VERSION}" == "${EXPECTED_VLLM_VERSION}"+* ]] || \
    die "derived image has unexpected vLLM version: ${ACTUAL_VLLM_VERSION}"
[[ "${ACTUAL_VLLM_ASCEND_VERSION}" == "${EXPECTED_VLLM_ASCEND_VERSION}" || \
   "${ACTUAL_VLLM_ASCEND_VERSION}" == "${EXPECTED_VLLM_ASCEND_VERSION}"+* ]] || \
    die "derived image has unexpected vLLM-Ascend version: ${ACTUAL_VLLM_ASCEND_VERSION}"

MARKER_COUNT="$(docker run --rm --entrypoint python "${OUTPUT_IMAGE}" -c \
    'from importlib.util import find_spec
from pathlib import Path

def one_source(import_name, relative):
    spec = find_spec(import_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"cannot resolve installed {import_name} package")
    paths = [Path(root) / relative for root in spec.submodule_search_locations]
    existing = [path for path in paths if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(f"expected one {relative}, got {existing}")
    return existing[0]

w8a8 = one_source("vllm_ascend", "quantization/methods/w8a8_dynamic.py")
source = w8a8.read_text(encoding="utf-8")
marker = "# DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V7"
if source.count(marker) != 1:
    raise RuntimeError("DeepSeek W8A8 capture marker is absent or duplicated")
if source.index(marker) > source.index("        if zero_expert_num > 0"):
    raise RuntimeError("capture marker is after logical-ID remapping")
if "capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)" not in source:
    raise RuntimeError("Ascend routed-experts capturer call is absent")

capture = one_source("vllm", "model_executor/layers/fused_moe/routed_experts_capturer.py")
capture_source = capture.read_text(encoding="utf-8")
capture_marker = "# DEEPSEEK_V4_VLLM_TP8_CAPTURE_GATHER_V7"
if capture_source.count(capture_marker) != 1:
    raise RuntimeError("DeepSeek vLLM TP8 capture-gather marker is absent or duplicated")
if "torch.tensor_split(" not in capture_source:
    raise RuntimeError("active vLLM TP8 routed-experts tensor_split gather is absent")
if "hinted_tokens > n" not in capture_source:
    raise RuntimeError("active vLLM TP8 gather missing local-hint rejection guard")
print("CAPTURE_PATCH_SOURCES_OK")')"
[[ "${MARKER_COUNT}" == CAPTURE_PATCH_SOURCES_OK ]] || \
    die "derived image does not contain the required DeepSeek capture hooks"

printf '\nCAPTURE_IMAGE_OK image_ref=%s patch_id=%s\n' "${OUTPUT_IMAGE}" "${PATCH_ID}"
printf 'Set these exact values in configs/node1_w8a8.env:\n'
printf 'IMAGE_REF=%s\n' "${OUTPUT_IMAGE}"
printf 'CAPTURE_PATCH_ID=%s\n' "${PATCH_ID}"
