#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
require_run_id
REMOVE_CONTAINER=0
if [[ "${3:-}" == --remove ]]; then
    REMOVE_CONTAINER=1
elif [[ -n "${3:-}" ]]; then
    die "unknown option: ${3}"
fi
require_cmd cp
require_cmd date
require_cmd docker
require_cmd tr

umask 077
CONTAINER_NAME="$(container_name)"
RUN_DIR="$(host_run_dir)"
LOCAL_SERVICE_DIR="$(local_run_dir)/service"
mkdir -p "${LOCAL_SERVICE_DIR}"
if mkdir -p "${RUN_DIR}" 2>/dev/null; then
    EVIDENCE_DIR="${RUN_DIR}"
else
    EVIDENCE_DIR="${LOCAL_SERVICE_DIR}/stop-evidence"
    mkdir -p "${EVIDENCE_DIR}"
    warn "shared RUN_HOST_ROOT is unavailable; writing stop evidence locally"
fi

if ! container_exists "${CONTAINER_NAME}"; then
    printf 'STOP_OK container_already_absent=%s\n' "${CONTAINER_NAME}"
    exit 0
fi

[[ -f "${LOCAL_SERVICE_DIR}/container.id" ]] || \
    die "local saved container.id is missing; refusing to stop an unverified container"
[[ -f "${LOCAL_SERVICE_DIR}/ownership.env" ]] || \
    die "local container ownership record is missing"
[[ "$(gate_value run_id "${LOCAL_SERVICE_DIR}/ownership.env")" == "${RUN_ID}" ]] || \
    die "local ownership record belongs to another run"
[[ "$(gate_value node_rank "${LOCAL_SERVICE_DIR}/ownership.env")" == "${NODE_RANK}" ]] || \
    die "local ownership record belongs to another node"
[[ "$(gate_value run_profile "${LOCAL_SERVICE_DIR}/ownership.env")" == "${RUN_PROFILE}" ]] || \
    die "local ownership record has a different profile"

SAVED_CONTAINER_ID="$(tr -d '[:space:]' <"${LOCAL_SERVICE_DIR}/container.id")"
ACTUAL_CONTAINER_ID="$(docker inspect "${CONTAINER_NAME}" --format '{{.Id}}')"
[[ -n "${SAVED_CONTAINER_ID}" && "${SAVED_CONTAINER_ID}" == "${ACTUAL_CONTAINER_ID}" ]] || \
    die "container ID does not match the container created by this run"
[[ "$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{index .Config.Labels "glm52.experiment"}}')" == expert-load ]] || \
    die "container ownership label glm52.experiment is wrong"
[[ "$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{index .Config.Labels "glm52.run_id"}}')" == "${RUN_ID}" ]] || \
    die "container run_id label is wrong"
[[ "$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{index .Config.Labels "glm52.node_rank"}}')" == "${NODE_RANK}" ]] || \
    die "container node_rank label is wrong"
[[ "$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{index .Config.Labels "glm52.profile"}}')" == "${RUN_PROFILE}" ]] || \
    die "container profile label is wrong"

docker inspect "${SAVED_CONTAINER_ID}" >"${EVIDENCE_DIR}/container.before-stop.inspect.json"
docker logs --timestamps "${SAVED_CONTAINER_ID}" \
    >"${EVIDENCE_DIR}/container.before-stop.log" 2>&1 || true

STATE="$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{.State.Status}}')"
if [[ "${STATE}" == paused ]]; then
    docker unpause "${SAVED_CONTAINER_ID}"
    STATE="$(docker inspect "${SAVED_CONTAINER_ID}" --format '{{.State.Status}}')"
fi
if [[ "${STATE}" == running || "${STATE}" == restarting ]]; then
    docker stop -t "${STOP_TIMEOUT_SECONDS}" "${SAVED_CONTAINER_ID}"
elif [[ "${STATE}" != exited && "${STATE}" != dead && "${STATE}" != created ]]; then
    die "unsupported container state ${STATE}; inspect it before taking action"
fi
container_is_running "${SAVED_CONTAINER_ID}" && \
    die "container is still running after docker stop"

docker logs --timestamps "${SAVED_CONTAINER_ID}" \
    >"${EVIDENCE_DIR}/container.final.log" 2>&1 || true
docker inspect "${SAVED_CONTAINER_ID}" >"${EVIDENCE_DIR}/container.after-stop.inspect.json"
{
    printf 'stopped_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'container_id=%s\n' "${SAVED_CONTAINER_ID}"
    printf 'container_name=%s\n' "${CONTAINER_NAME}"
} >"${LOCAL_SERVICE_DIR}/CONTAINER_STOPPED"
cp -p "${LOCAL_SERVICE_DIR}/CONTAINER_STOPPED" \
    "${RUN_DIR}/CONTAINER_STOPPED" 2>/dev/null || true

if (( REMOVE_CONTAINER == 1 )); then
    docker rm "${SAVED_CONTAINER_ID}"
    printf 'STOP_OK removed=%s logs=%s\n' "${CONTAINER_NAME}" "${EVIDENCE_DIR}"
else
    printf 'STOP_OK retained=%s logs=%s\n' "${CONTAINER_NAME}" "${EVIDENCE_DIR}"
fi
