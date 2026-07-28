#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

load_config "${1:-}"
for name in RUN_ROOT API_HOST API_PORT WAIT_TIMEOUT_SECONDS SERVED_MODEL_NAME; do
    require_var "${name}"
done
require_cmd curl
require_cmd docker

RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CONTAINER_NAME="$(container_name_for_run "${RUN_ID}")"
[[ -d "${RUN_DIR}" ]] || die "run directory not found: ${RUN_DIR}"
HEALTH_URL="http://${API_HOST}:${API_PORT}/health"
MODELS_URL="http://${API_HOST}:${API_PORT}/v1/models"
DEADLINE=$((SECONDS + WAIT_TIMEOUT_SECONDS))

while ((SECONDS < DEADLINE)); do
    if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        die "container disappeared: ${CONTAINER_NAME}"
    fi
    if [[ "$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}')" != true ]]; then
        docker logs --timestamps "${CONTAINER_NAME}" >"${RUN_DIR}/container.exit.log" 2>&1 || true
        die "container exited; inspect ${RUN_DIR}/container.exit.log"
    fi
    if curl --noproxy '*' -fsS --max-time 10 "${HEALTH_URL}" >/dev/null 2>&1; then
        curl --noproxy '*' -fsS --max-time 30 "${MODELS_URL}" \
            | tee "${RUN_DIR}/models.ready.json"
        touch "${RUN_DIR}/service.ready"
        printf '\nSERVICE_READY url=http://%s:%s/v1 model=%s\n' \
            "${API_HOST}" "${API_PORT}" "${SERVED_MODEL_NAME}"
        exit 0
    fi
    printf 'waiting for DeepSeek-V4 model load: %ss elapsed\n' \
        "$((WAIT_TIMEOUT_SECONDS - (DEADLINE - SECONDS)))"
    sleep 10
done

docker logs --tail 300 --timestamps "${CONTAINER_NAME}" \
    >"${RUN_DIR}/container.wait-timeout.log" 2>&1 || true
die "service did not become ready in ${WAIT_TIMEOUT_SECONDS}s; inspect ${RUN_DIR}/container.wait-timeout.log"
