#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
shift || true
PULL_BASE=0
if [[ "${1:-}" == --confirm-pull-base ]]; then
    PULL_BASE=1
    shift
fi
(($# == 0)) || die "unknown arguments: $*"

load_config "${CONFIG_PATH}"
for name in BASE_IMAGE IMAGE_REF CAPTURE_PATCH_ID EXPECTED_VLLM_PACKAGE_VERSION \
    EXPECTED_VLLM_ASCEND_PACKAGE_VERSION; do
    require_var "${name}"
done
require_cmd docker

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    if ((PULL_BASE == 0)); then
        die "base image is absent: ${BASE_IMAGE}; load an offline copy or add --confirm-pull-base"
    fi
    docker pull "${BASE_IMAGE}"
fi

docker build \
    --file "${GLM51_EXPERIMENT_ROOT}/patches/Dockerfile.route-capture" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CAPTURE_PATCH_ID=${CAPTURE_PATCH_ID}" \
    --tag "${IMAGE_REF}" \
    "${GLM51_EXPERIMENT_ROOT}/patches"

ACTUAL_PATCH_ID="$(docker image inspect "${IMAGE_REF}" \
    --format '{{if .Config.Labels}}{{index .Config.Labels "glm51.capture_patch_id"}}{{end}}')"
[[ "${ACTUAL_PATCH_ID}" == "${CAPTURE_PATCH_ID}" ]] || \
    die "derived image has unexpected patch label: ${ACTUAL_PATCH_ID:-<empty>}"

docker run --rm --entrypoint python "${IMAGE_REF}" - \
    "${EXPECTED_VLLM_PACKAGE_VERSION}" \
    "${EXPECTED_VLLM_ASCEND_PACKAGE_VERSION}" <<'PY'
from importlib.metadata import version
import sys

expected_vllm, expected_ascend = sys.argv[1:]
actual_vllm = version("vllm")
actual_ascend = version("vllm-ascend")
actual_transformers = version("transformers")
print(f"vllm={actual_vllm}")
print(f"vllm-ascend={actual_ascend}")
print(f"transformers={actual_transformers}")
if actual_vllm.split("+")[0] != expected_vllm:
    raise SystemExit("unexpected vLLM version")
if actual_ascend.split("+")[0] != expected_ascend:
    raise SystemExit("unexpected vLLM-Ascend version")
parts = tuple(int(piece) for piece in actual_transformers.split("+")[0].split(".")[:2])
if parts < (5, 2):
    raise SystemExit("GLM-5.1 requires transformers>=5.2")
print("GLM51_IMAGE_PACKAGES_OK")
PY

printf 'CAPTURE_IMAGE_OK image=%s patch_id=%s\n' "${IMAGE_REF}" "${CAPTURE_PATCH_ID}"

