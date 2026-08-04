#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

load_config "${1:-}"
require_var RUN_ROOT
require_cmd docker
RUN_ID="$(current_run_id)"
CONTAINER_NAME="$(container_name_for_run "${RUN_ID}")"
printf 'run_id=%s container=%s\n' "${RUN_ID}" "${CONTAINER_NAME}"
docker inspect "${CONTAINER_NAME}" \
    --format 'status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}'
docker logs --tail 120 --timestamps "${CONTAINER_NAME}"

