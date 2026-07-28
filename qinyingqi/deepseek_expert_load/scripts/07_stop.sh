#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
shift || true
REMOVE=0
if [[ "${1:-}" == --remove ]]; then
    REMOVE=1
    shift
fi
(($# == 0)) || die "unknown arguments: $*"

load_config "${CONFIG_PATH}"
require_var RUN_ROOT
require_cmd docker
RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CONTAINER_NAME="$(container_name_for_run "${RUN_ID}")"

docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 || \
    die "owned container not found: ${CONTAINER_NAME}"
LABEL_RUN_ID="$(docker inspect "${CONTAINER_NAME}" \
    --format '{{index .Config.Labels "deepseek_expert_load.run_id"}}')"
[[ "${LABEL_RUN_ID}" == "${RUN_ID}" ]] || \
    die "container ownership label mismatch; refusing to stop it"
HOST_IDS="$(docker inspect "${CONTAINER_NAME}" \
    --format '{{index .Config.Labels "deepseek_expert_load.host_npu_ids"}}')"

docker logs --timestamps "${CONTAINER_NAME}" >"${RUN_DIR}/container.final.log" 2>&1 || true
if [[ "$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}')" == true ]]; then
    docker stop -t "${STOP_TIMEOUT_SECONDS:-120}" "${CONTAINER_NAME}"
fi
docker inspect "${CONTAINER_NAME}" >"${RUN_DIR}/container.final.inspect.json"
if ((REMOVE == 1)); then
    docker rm "${CONTAINER_NAME}"
fi
printf 'STOP_OK run_id=%s container=%s removed=%s\n' \
    "${RUN_ID}" "${CONTAINER_NAME}" "${REMOVE}"
printf 'Operator action required: restore the keep-alive workload only on host NPU IDs %s.\n' \
    "${HOST_IDS}"
