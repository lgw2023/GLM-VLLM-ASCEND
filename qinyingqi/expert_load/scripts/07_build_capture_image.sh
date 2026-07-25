#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

BASE_IMAGE_DEFAULT="quay.io/ascend/vllm-ascend:v0.22.1rc1"
OUTPUT_IMAGE_DEFAULT="glm52-expert-capture:v0.22.1rc1-w8a8-v1"
PATCH_ID_DEFAULT="glm52-w8a8-logical-topk-v1"
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

Builds the W8A8 routed-expert capture image from the official vLLM-Ascend
v0.22.1rc1 release image. Run this on node1 when node1 is the registry-enabled
node, then docker save/load the resulting image onto node0.

Options:
  --confirm-pull-base       Pull the official base image only when it is absent.
                            Omit this when the exact base image already exists.
  --base-image IMAGE        Override the official release base image.
  --output-image IMAGE      Derived image tag (default is fixed for this project).
  --patch-id ID             Docker label value (default is fixed for this project).
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
elif (( PULL_BASE == 1 )); then
    docker pull "${BASE_IMAGE}"
fi
docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || \
    die "base image is absent: ${BASE_IMAGE}; pull it on a registry-enabled node or docker load an offline copy, then rerun without --confirm-pull-base"

DOCKERFILE="${EXPERT_LOAD_ROOT}/patches/Dockerfile.route-capture"
PATCH_FILE="${EXPERT_LOAD_ROOT}/patches/apply_w8a8_route_capture.py"
[[ -f "${DOCKERFILE}" ]] || die "capture Dockerfile is missing: ${DOCKERFILE}"
[[ -f "${PATCH_FILE}" ]] || die "capture patch is missing: ${PATCH_FILE}"

docker build \
    --file "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CAPTURE_PATCH_ID=${PATCH_ID}" \
    --tag "${OUTPUT_IMAGE}" \
    "${EXPERT_LOAD_ROOT}/patches"

ACTUAL_PATCH_ID="$(docker image inspect "${OUTPUT_IMAGE}" \
    --format '{{if .Config.Labels}}{{index .Config.Labels "glm52.capture_patch_id"}}{{end}}')"
[[ "${ACTUAL_PATCH_ID}" == "${PATCH_ID}" ]] || \
    die "derived image has unexpected glm52.capture_patch_id: ${ACTUAL_PATCH_ID}"

PACKAGE_LINES="$(docker run --rm --entrypoint python "${OUTPUT_IMAGE}" -c \
    'from importlib.metadata import version
print(f"vllm={version(\"vllm\")}")
print(f"vllm-ascend={version(\"vllm-ascend\")}")')"
printf '%s\n' "${PACKAGE_LINES}"
ACTUAL_VLLM_VERSION="$(printf '%s\n' "${PACKAGE_LINES}" | sed -n 's/^vllm=//p')"
ACTUAL_VLLM_ASCEND_VERSION="$(printf '%s\n' "${PACKAGE_LINES}" | sed -n 's/^vllm-ascend=//p')"
package_version_matches_release \
    "${ACTUAL_VLLM_VERSION}" "${EXPECTED_VLLM_VERSION}" || \
    die "derived image has unexpected vLLM version"
package_version_matches_release \
    "${ACTUAL_VLLM_ASCEND_VERSION}" "${EXPECTED_VLLM_ASCEND_VERSION}" || \
    die "derived image has unexpected vLLM-Ascend version"

MARKER_COUNT="$(docker run --rm --entrypoint python "${OUTPUT_IMAGE}" -c \
    'from importlib.metadata import distribution
path = distribution("vllm-ascend").locate_file("vllm_ascend/quantization/methods/w8a8_dynamic.py")
print(path.read_text(encoding="utf-8").count("# GLM52_W8A8_ROUTE_CAPTURE_V1"))')"
[[ "${MARKER_COUNT}" == 1 ]] || \
    die "derived image does not contain exactly one W8A8 route-capture hook"

printf '\nCAPTURE_IMAGE_OK image_ref=%s patch_id=%s\n' "${OUTPUT_IMAGE}" "${PATCH_ID}"
printf 'Use these exact cluster.env values on both nodes after the image is present on both nodes:\n'
printf 'RUN_PROFILE=expert_capture\n'
printf 'IMAGE_REF=%s\n' "${OUTPUT_IMAGE}"
printf 'VLLM_VERSION_OVERRIDE=%s\n' "${EXPECTED_VLLM_VERSION}"
printf 'ENABLE_ROUTE_CAPTURE=1\n'
printf 'CAPTURE_PATCH_ID=%s\n' "${PATCH_ID}"
printf 'EXPECTED_VLLM_PACKAGE_VERSION=%s\n' "${EXPECTED_VLLM_VERSION}"
printf 'EXPECTED_VLLM_ASCEND_PACKAGE_VERSION=%s\n' "${EXPECTED_VLLM_ASCEND_VERSION}"
printf 'MAX_NUM_SEQS=1\n'
printf 'API_BIND_HOST=127.0.0.1\n'
