#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
require_run_id
require_cmd cp
require_cmd date
require_cmd docker
require_cmd mv
require_cmd python3
require_cmd rmdir
require_cmd sha256sum
require_cmd tee
NPU_SMI_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"

umask 077
LOCAL_RUN_DIR="$(local_run_dir)"
STATE_DIR="${LOCAL_RUN_DIR}/keepalive"
STATE_FILE="$(keepalive_state_file)"
SHARED_EVIDENCE_DIR="$(host_run_dir)/keepalive"
[[ -f "${STATE_FILE}" ]] || die "keep-alive run state missing: ${STATE_FILE}"
acquire_lifecycle_lock restore
trap 'release_lifecycle_lock' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "${STATE_DIR}"
exec > >(tee -a "${STATE_DIR}/restore.log") 2>&1

[[ "$(gate_value run_id "${STATE_FILE}")" == "${RUN_ID}" ]] || \
    die "keep-alive state belongs to another run"
[[ "$(gate_value node_rank "${STATE_FILE}")" == "${NODE_RANK}" ]] || \
    die "keep-alive state belongs to another node"

LOCAL_STOP_SCRIPT="$(gate_value stop_script "${STATE_FILE}")"
LOCAL_START_SCRIPT="$(gate_value start_script "${STATE_FILE}")"
RECORDED_STOP_SHA="$(gate_value stop_script_sha256 "${STATE_FILE}")"
RECORDED_START_SHA="$(gate_value start_script_sha256 "${STATE_FILE}")"
SOURCE_STOP_SCRIPT="$(gate_value source_stop_script "${STATE_FILE}")"
SOURCE_START_SCRIPT="$(gate_value source_start_script "${STATE_FILE}")"
for local_script in "${LOCAL_STOP_SCRIPT}" "${LOCAL_START_SCRIPT}"; do
    [[ -f "${local_script}" ]] || die "verified local keep-alive script missing: ${local_script}"
done
[[ "${RECORDED_STOP_SHA}" =~ ^[0-9a-f]{64}$ ]] || die "invalid recorded stop-script SHA-256"
[[ "${RECORDED_START_SHA}" =~ ^[0-9a-f]{64}$ ]] || die "invalid recorded start-script SHA-256"
[[ "$(sha256sum "${LOCAL_STOP_SCRIPT}" | awk '{print $1}')" == "${RECORDED_STOP_SHA}" ]] || \
    die "verified local stop script changed"
[[ "$(sha256sum "${LOCAL_START_SCRIPT}" | awk '{print $1}')" == "${RECORDED_START_SHA}" ]] || \
    die "verified local start script changed"

LEASE_DIR="$(gate_value lease_dir "${STATE_FILE}")"
[[ "${LEASE_DIR}" == "$(npu_lease_dir)" ]] || die "state references an unexpected lease directory"
LEASE_OWNER="${LEASE_DIR}/owner.env"
LEASE_RELEASED_DIR="${STATE_DIR}/lease.released"

write_state() {
    local status="$1"
    local stopped_ids="$2"
    local restored_ids="$3"
    local restoration_status="$4"
    {
        printf 'status=%s\n' "${status}"
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'node_rank=%s\n' "${NODE_RANK}"
        printf 'stopped_card_ids=%s\n' "${stopped_ids}"
        printf 'restored_card_ids=%s\n' "${restored_ids}"
        printf 'restoration_status=%s\n' "${restoration_status}"
        printf 'stop_script=%s\n' "${LOCAL_STOP_SCRIPT}"
        printf 'start_script=%s\n' "${LOCAL_START_SCRIPT}"
        printf 'source_stop_script=%s\n' "${SOURCE_STOP_SCRIPT}"
        printf 'source_start_script=%s\n' "${SOURCE_START_SCRIPT}"
        printf 'stop_script_sha256=%s\n' "${RECORDED_STOP_SHA}"
        printf 'start_script_sha256=%s\n' "${RECORDED_START_SHA}"
        printf 'lease_dir=%s\n' "${LEASE_DIR}"
        printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${STATE_FILE}"
}

mirror_state() {
    mkdir -p "${SHARED_EVIDENCE_DIR}" 2>/dev/null || return 0
    cp -p "${STATE_FILE}" \
        "${SHARED_EVIDENCE_DIR}/state.node${NODE_RANK}.env" 2>/dev/null || \
        warn "could not mirror local keep-alive state to shared RUN_HOST_ROOT"
}

release_lease() {
    [[ -d "${LEASE_DIR}" ]] || return 0
    [[ -f "${LEASE_OWNER}" ]] || die "active NPU lease has no owner record"
    [[ "$(gate_value run_id "${LEASE_OWNER}")" == "${RUN_ID}" ]] || \
        die "refusing to release another run's NPU lease"
    [[ "$(gate_value node_rank "${LEASE_OWNER}")" == "${NODE_RANK}" ]] || \
        die "refusing to release another node's NPU lease"
    [[ ! -e "${LEASE_RELEASED_DIR}" ]] || \
        die "lease release history already exists while owner is still active"
    mv -T "${LEASE_DIR}" "${LEASE_RELEASED_DIR}"
}

CURRENT_STATUS="$(gate_value status "${STATE_FILE}")"
case "${CURRENT_STATUS}" in
    PREPARE_FAILED)
        release_lease
        printf 'KEEPALIVE_NOT_STOPPED state=%s\n' "${STATE_FILE}"
        exit 0
        ;;
    RESTORED|RESTORED_AFTER_PREPARE_FAILURE)
        RESTORED_SNAPSHOT="${STATE_DIR}/restored-idempotency-check.json"
        python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${RESTORED_SNAPSHOT}"
        python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected running "${RESTORED_SNAPSHOT}"
        python3 "${SCRIPT_DIR}/keepalive_state.py" compare \
            "${STATE_DIR}/before.json" "${RESTORED_SNAPSHOT}"
        release_lease
        printf 'KEEPALIVE_ALREADY_RESTORED state=%s\n' "${STATE_FILE}"
        exit 0
        ;;
    PREPARED|PREPARING|RESTORING|RESTORATION_FAILED)
        ;;
    *)
        die "keep-alive state is not restorable: ${CURRENT_STATUS}"
        ;;
esac

[[ -d "${LEASE_DIR}" && -f "${LEASE_OWNER}" ]] || \
    die "active node-level NPU lease is missing: ${LEASE_DIR}"
[[ "$(gate_value run_id "${LEASE_OWNER}")" == "${RUN_ID}" ]] || \
    die "node-level NPU lease belongs to another run"
[[ "$(gate_value node_rank "${LEASE_OWNER}")" == "${NODE_RANK}" ]] || \
    die "node-level NPU lease belongs to another node"

RUNNING_NPU_CONTAINERS="$(running_npu_container_ids)"
[[ -z "${RUNNING_NPU_CONTAINERS}" ]] || \
    die "refusing to restore keep-alive while NPU containers are running: ${RUNNING_NPU_CONTAINERS}"

BEFORE_SNAPSHOT="${STATE_DIR}/before.json"
NORMALIZED_SNAPSHOT="${STATE_DIR}/before-restore-stopped.json"
RESTORED_SNAPSHOT="${STATE_DIR}/restored.json"
[[ -f "${BEFORE_SNAPSHOT}" ]] || die "pre-run keep-alive snapshot is missing"
CARDS=(0 1 2 3 4 5 6 7)
CARD_CSV=0,1,2,3,4,5,6,7
RESTORE_COMPLETE=0

record_restore_failure() {
    local exit_status=$?
    local cleanup_signal_status=0
    trap - EXIT
    trap 'cleanup_signal_status=130' INT
    trap 'cleanup_signal_status=143' TERM
    if (( exit_status != 0 && RESTORE_COMPLETE == 0 )); then
        write_state RESTORATION_FAILED "${CARD_CSV}" "" failed || true
        mirror_state || true
        warn "restoration did not complete; node-level lease remains at ${LEASE_DIR}"
    fi
    release_lifecycle_lock || true
    (( cleanup_signal_status == 0 )) || exit_status=${cleanup_signal_status}
    exit "${exit_status}"
}
trap record_restore_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '[NPU state before restoration normalization]\n'
"${NPU_SMI_BIN}" info
bash "${LOCAL_STOP_SCRIPT}" "${CARDS[@]}"
sleep 2
python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${NORMALIZED_SNAPSHOT}"
python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected stopped "${NORMALIZED_SNAPSHOT}"

require_no_npu_containers "keep-alive restoration"
write_state RESTORING "${CARD_CSV}" "" in_progress
bash "${LOCAL_START_SCRIPT}" "${CARDS[@]}"
sleep 5
python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${RESTORED_SNAPSHOT}"
python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected running "${RESTORED_SNAPSHOT}"
python3 "${SCRIPT_DIR}/keepalive_state.py" compare "${BEFORE_SNAPSHOT}" "${RESTORED_SNAPSHOT}"

write_state RESTORED "${CARD_CSV}" "${CARD_CSV}" success
mirror_state
RESTORE_COMPLETE=1
release_lease
release_lifecycle_lock
trap - EXIT INT TERM

"${NPU_SMI_BIN}" info
printf 'KEEPALIVE_RESTORED run_id=%s node=%s restored_card_ids=%s state=%s\n' \
    "${RUN_ID}" "${NODE_RANK}" "${CARD_CSV}" "${STATE_FILE}"
