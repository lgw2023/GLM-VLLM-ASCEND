#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
require_run_id
[[ -z "${3:-}" ]] || die "05_prepare_npus.sh no longer accepts a third argument"
require_cmd date
require_cmd docker
require_cmd tee
require_config_gate preflight
require_config_gate hccn_ping

NPU_SMI_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
OUT_DIR="$(host_run_dir)/npu-ready"
READY_FILE="${OUT_DIR}/NPU_READY"
mkdir -p "${OUT_DIR}"

printf '[NPU state before this run]\n' | tee "${OUT_DIR}/npu-smi.txt"
"${NPU_SMI_BIN}" info | tee -a "${OUT_DIR}/npu-smi.txt"

RUNNING_NPU_CONTAINERS="$(running_npu_container_ids)"
if [[ -n "${RUNNING_NPU_CONTAINERS}" ]]; then
    warn "running containers expose NPU devices: ${RUNNING_NPU_CONTAINERS}"
    warn "continuing because NPU occupancy is managed manually on this server"
    docker ps --no-trunc | tee "${OUT_DIR}/docker-ps.txt"
fi

{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'node_rank=%s\n' "${NODE_RANK}"
    printf 'authorized_npu_ids=%s\n' "${AUTHORIZED_NPU_IDS}"
    printf 'config_fingerprint=%s\n' "$(config_fingerprint)"
    printf 'source_id=%s\n' "$(source_id)"
    printf 'checked_at=%s\n' "$(date --iso-8601=seconds)"
} >"${READY_FILE}"

printf 'NPU_READY run_id=%s node=%s record=%s\n' \
    "${RUN_ID}" "${NODE_RANK}" "${READY_FILE}"
