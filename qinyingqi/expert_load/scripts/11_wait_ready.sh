#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_CONFIG_ARG="${1:-}"
NODE_CONFIG_ARG="${2:-}"
load_configs "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}"
require_run_id
[[ "${NODE_RANK}" == 0 ]] || die "run API readiness check on node0"
require_cmd curl
require_cmd date
require_cmd docker

umask 077
CONTAINER_NAME="$(container_name)"
RUN_DIR="$(host_run_dir)"
container_exists "${CONTAINER_NAME}" || die "node0 container not found: ${CONTAINER_NAME}"
mkdir -p "${RUN_DIR}"

stop_on_readiness_failure() {
    local exit_status=$?
    trap - EXIT INT TERM
    if (( exit_status != 0 )); then
        warn "readiness failed; stopping this run's node0 model container"
        bash "${SCRIPT_DIR}/19_stop_node.sh" \
            "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}" || true
        warn "stop this run's peer model container as well"
    fi
    exit "${exit_status}"
}
trap stop_on_readiness_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

HEALTH_URL="http://${API_BIND_HOST}:${API_PORT}/health"
MODELS_URL="http://${API_BIND_HOST}:${API_PORT}/v1/models"
START_SECONDS="$(date +%s)"

while true; do
    if curl --noproxy '*' -fsS --max-time 10 "${HEALTH_URL}" >"${RUN_DIR}/health.response"; then
        break
    fi
    if ! container_is_running "${CONTAINER_NAME}"; then
        docker logs --timestamps "${CONTAINER_NAME}" >"${RUN_DIR}/container.failed.log" 2>&1 || true
        die "node0 container exited before health became ready"
    fi
    NOW_SECONDS="$(date +%s)"
    if (( NOW_SECONDS - START_SECONDS >= HEALTH_TIMEOUT_SECONDS )); then
        docker logs --tail 500 --timestamps "${CONTAINER_NAME}" >"${RUN_DIR}/container.timeout.log" 2>&1 || true
        die "health endpoint was not ready within ${HEALTH_TIMEOUT_SECONDS}s"
    fi
    sleep 10
done

curl --noproxy '*' -fsS --max-time 30 "${MODELS_URL}" | tee "${RUN_DIR}/models.response.json"
docker inspect "${CONTAINER_NAME}" >"${RUN_DIR}/container.ready.inspect.json"
docker exec "${CONTAINER_NAME}" python -c \
    'from importlib.metadata import PackageNotFoundError, version
for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers"):
    try:
        print(f"{name}={version(name)}")
    except PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")' \
    | tee "${RUN_DIR}/container.packages.txt"

if [[ "${RUN_PROFILE}" == expert_capture ]]; then
    [[ "$(package_version_from_file vllm "${RUN_DIR}/container.packages.txt")" == \
       "${EXPECTED_VLLM_PACKAGE_VERSION}" ]] || \
        die "ready container has the wrong vLLM version"
    [[ "$(package_version_from_file vllm-ascend "${RUN_DIR}/container.packages.txt")" == \
       "${EXPECTED_VLLM_ASCEND_PACKAGE_VERSION}" ]] || \
        die "ready container has the wrong vLLM-Ascend version"
fi

touch "${RUN_DIR}/SERVICE_READY"
trap - EXIT INT TERM
printf '\nSERVICE_READY base_url=http://%s:%s/v1 run_dir=%s\n' \
    "${API_BIND_HOST}" "${API_PORT}" "${RUN_DIR}"
