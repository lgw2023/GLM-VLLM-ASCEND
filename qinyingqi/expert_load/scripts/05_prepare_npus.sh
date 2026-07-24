#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_CONFIG_ARG="${1:-}"
NODE_CONFIG_ARG="${2:-}"
load_configs "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}"
require_run_id
[[ "${3:-}" == --confirm-stop-keepalive ]] || \
    die "rerun with --confirm-stop-keepalive after checking the current npu-smi output"
[[ "${NPU_USE_CONFIRMED}" == YES ]] || \
    die "set NPU_USE_CONFIRMED=YES after current occupancy review"
[[ "${NPU_LAUNCH_CONFIRMATION:-}" == "${RUN_ID}" ]] || \
    die "export NPU_LAUNCH_CONFIRMATION=${RUN_ID} for this exact run"
require_config_gate preflight
require_config_gate hccn_ping
acquire_lifecycle_lock prepare
trap 'release_lifecycle_lock' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for name in KEEPALIVE_STOP_SCRIPT KEEPALIVE_START_SCRIPT \
    KEEPALIVE_STOP_SHA256 KEEPALIVE_START_SHA256; do
    require_var "${name}"
done
[[ "${KEEPALIVE_STOP_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
    die "KEEPALIVE_STOP_SHA256 must be lowercase SHA-256"
[[ "${KEEPALIVE_START_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
    die "KEEPALIVE_START_SHA256 must be lowercase SHA-256"
[[ -f "${KEEPALIVE_STOP_SCRIPT}" ]] || \
    die "keep-alive stop script missing: ${KEEPALIVE_STOP_SCRIPT}"
[[ -f "${KEEPALIVE_START_SCRIPT}" ]] || \
    die "keep-alive start script missing: ${KEEPALIVE_START_SCRIPT}"
require_cmd cp
require_cmd date
require_cmd docker
require_cmd mv
require_cmd python3
require_cmd rmdir
require_cmd sha256sum
require_cmd tee
NPU_SMI_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"

ACTUAL_STOP_SHA="$(sha256sum "${KEEPALIVE_STOP_SCRIPT}" | awk '{print $1}')"
ACTUAL_START_SHA="$(sha256sum "${KEEPALIVE_START_SCRIPT}" | awk '{print $1}')"
[[ "${ACTUAL_STOP_SHA}" == "${KEEPALIVE_STOP_SHA256}" ]] || \
    die "keep-alive stop script SHA-256 changed"
[[ "${ACTUAL_START_SHA}" == "${KEEPALIVE_START_SHA256}" ]] || \
    die "keep-alive start script SHA-256 changed"

RUNNING_NPU_CONTAINERS="$(running_npu_container_ids)"
[[ -z "${RUNNING_NPU_CONTAINERS}" ]] || \
    die "NPU containers are already running on this node: ${RUNNING_NPU_CONTAINERS}"

umask 077
LOCAL_RUN_DIR="$(local_run_dir)"
STATE_DIR="${LOCAL_RUN_DIR}/keepalive"
STATE_FILE="$(keepalive_state_file)"
SCRIPT_COPY_DIR="${STATE_DIR}/verified-scripts"
SHARED_EVIDENCE_DIR="$(host_run_dir)/keepalive"
mkdir -p "${SCRIPT_COPY_DIR}"
[[ ! -e "${STATE_FILE}" ]] || \
    die "keep-alive state already exists for RUN_ID=${RUN_ID}: ${STATE_FILE}"

LOCAL_STOP_SCRIPT="${SCRIPT_COPY_DIR}/npu_stop.sh"
LOCAL_START_SCRIPT="${SCRIPT_COPY_DIR}/npu_keep_alive.sh"
cp -p "${KEEPALIVE_STOP_SCRIPT}" "${LOCAL_STOP_SCRIPT}"
cp -p "${KEEPALIVE_START_SCRIPT}" "${LOCAL_START_SCRIPT}"
[[ "$(sha256sum "${LOCAL_STOP_SCRIPT}" | awk '{print $1}')" == "${ACTUAL_STOP_SHA}" ]] || \
    die "local stop-script copy failed identity check"
[[ "$(sha256sum "${LOCAL_START_SCRIPT}" | awk '{print $1}')" == "${ACTUAL_START_SHA}" ]] || \
    die "local start-script copy failed identity check"

LEASE_DIR="$(npu_lease_dir)"
LEASE_OWNER="${LEASE_DIR}/owner.env"
LEASE_PENDING_DIR="$(dirname "${LEASE_DIR}")/.cards-0-7.pending.${RUN_ID}.$$-${RANDOM}"
LEASE_HELD=0

release_lease_before_recovery_ready() {
    local exit_status=$?
    local cleanup_signal_status=0
    trap - EXIT
    trap 'cleanup_signal_status=130' INT
    trap 'cleanup_signal_status=143' TERM
    if (( exit_status != 0 && LEASE_HELD == 1 )); then
        if [[ -f "${LEASE_OWNER}" && \
              "$(gate_value run_id "${LEASE_OWNER}")" == "${RUN_ID}" && \
              "$(gate_value node_rank "${LEASE_OWNER}")" == "${NODE_RANK}" ]]; then
            mkdir -p "${STATE_DIR}/lease-history" || true
            if mv -T "${LEASE_DIR}" \
                "${STATE_DIR}/lease-history/released-before-prepare-$$-${RANDOM}" \
                2>/dev/null; then
                LEASE_HELD=0
            fi
        fi
        if (( LEASE_HELD == 1 )); then
            warn "early preparation failed and the node lease could not be released; inspect ${LEASE_DIR}"
        fi
    fi
    release_lifecycle_lock || true
    (( cleanup_signal_status == 0 )) || exit_status=${cleanup_signal_status}
    exit "${exit_status}"
}

mkdir -p "$(dirname "${LEASE_DIR}")"
mkdir "${LEASE_PENDING_DIR}"
{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'node_rank=%s\n' "${NODE_RANK}"
    printf 'owner_pid=%s\n' "$$"
    printf 'acquired_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'cluster_config_sha256=%s\n' "$(cluster_config_sha256)"
    printf 'source_id=%s\n' "$(source_id)"
} >"${LEASE_PENDING_DIR}/owner.env"
if ! mv -T "${LEASE_PENDING_DIR}" "${LEASE_DIR}" 2>/dev/null; then
    mv -T "${LEASE_PENDING_DIR}" \
        "${STATE_DIR}/lease.acquire-failed.$$-${RANDOM}" 2>/dev/null || true
    if [[ -f "${LEASE_OWNER}" ]]; then
        warn "current node-level NPU lease owner:"
        sed -n 's/^\(run_id\|node_rank\|acquired_at\)=/  \1=/p' "${LEASE_OWNER}" >&2
    fi
    die "cards 0..7 already have a node-level lease: ${LEASE_DIR}"
fi
LEASE_HELD=1
trap release_lease_before_recovery_ready EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CARDS=(0 1 2 3 4 5 6 7)
CARD_CSV=0,1,2,3,4,5,6,7
BEFORE_SNAPSHOT="${STATE_DIR}/before.json"
AFTER_STOP_SNAPSHOT="${STATE_DIR}/after-stop.json"
RECOVERY_STOPPED_SNAPSHOT="${STATE_DIR}/recovery-stopped.json"
RECOVERY_SNAPSHOT="${STATE_DIR}/recovery-restored.json"
STOP_ATTEMPTED=0

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
        printf 'source_stop_script=%s\n' "${KEEPALIVE_STOP_SCRIPT}"
        printf 'source_start_script=%s\n' "${KEEPALIVE_START_SCRIPT}"
        printf 'stop_script_sha256=%s\n' "${ACTUAL_STOP_SHA}"
        printf 'start_script_sha256=%s\n' "${ACTUAL_START_SHA}"
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
    (( LEASE_HELD == 1 )) || return 0
    [[ -f "${LEASE_OWNER}" ]] || die "lease owner file disappeared: ${LEASE_OWNER}"
    [[ "$(gate_value run_id "${LEASE_OWNER}")" == "${RUN_ID}" ]] || \
        die "refusing to release another run's NPU lease"
    [[ "$(gate_value node_rank "${LEASE_OWNER}")" == "${NODE_RANK}" ]] || \
        die "refusing to release another node's NPU lease"
    [[ ! -e "${STATE_DIR}/lease.released" ]] || \
        die "lease release history already exists"
    mv -T "${LEASE_DIR}" "${STATE_DIR}/lease.released"
    LEASE_HELD=0
}

recover_on_failure() {
    local exit_status=$?
    local cleanup_signal_status=0
    trap - EXIT
    trap 'cleanup_signal_status=130' INT
    trap 'cleanup_signal_status=143' TERM
    if (( exit_status == 0 )); then
        release_lifecycle_lock || true
        return
    fi
    if (( STOP_ATTEMPTED == 0 )); then
        write_state PREPARE_FAILED "" "" not_required || true
        mirror_state || true
        release_lease || true
        release_lifecycle_lock || true
        (( cleanup_signal_status == 0 )) || exit_status=${cleanup_signal_status}
        exit "${exit_status}"
    fi

    warn "NPU preparation failed after stop was attempted; normalizing then restoring cards ${CARD_CSV}"
    set +e
    bash "${LOCAL_STOP_SCRIPT}" "${CARDS[@]}"
    local normalize_stop_status=$?
    sleep 2
    python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot \
        --output "${RECOVERY_STOPPED_SNAPSHOT}"
    local stopped_snapshot_status=$?
    python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected stopped \
        "${RECOVERY_STOPPED_SNAPSHOT}"
    local stopped_validate_status=$?

    local start_status=1
    local running_validate_status=1
    local compare_status=1
    if (( normalize_stop_status == 0 && stopped_snapshot_status == 0 && stopped_validate_status == 0 )); then
        bash "${LOCAL_START_SCRIPT}" "${CARDS[@]}"
        start_status=$?
        sleep 5
        python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${RECOVERY_SNAPSHOT}"
        python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected running \
            "${RECOVERY_SNAPSHOT}"
        running_validate_status=$?
        python3 "${SCRIPT_DIR}/keepalive_state.py" compare \
            "${BEFORE_SNAPSHOT}" "${RECOVERY_SNAPSHOT}"
        compare_status=$?
    fi
    set -e

    if (( start_status == 0 && running_validate_status == 0 && compare_status == 0 )); then
        write_state RESTORED_AFTER_PREPARE_FAILURE "${CARD_CSV}" "${CARD_CSV}" success
        mirror_state
        release_lease
    else
        write_state RESTORATION_FAILED "${CARD_CSV}" "" failed || true
        mirror_state || true
        warn "automatic restoration failed; node-level lease remains held at ${LEASE_DIR}"
    fi
    release_lifecycle_lock || true
    (( cleanup_signal_status == 0 )) || exit_status=${cleanup_signal_status}
    exit "${exit_status}"
}
trap recover_on_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec > >(tee "${STATE_DIR}/prepare.log") 2>&1
printf '[current NPU state before keep-alive stop]\n'
"${NPU_SMI_BIN}" info
python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${BEFORE_SNAPSHOT}"
python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected running "${BEFORE_SNAPSHOT}"
write_state PREPARING "" "" pending

require_no_npu_containers "keep-alive stop"
STOP_ATTEMPTED=1
bash "${LOCAL_STOP_SCRIPT}" "${CARDS[@]}"
sleep 2
python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot --output "${AFTER_STOP_SNAPSHOT}"
python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected stopped "${AFTER_STOP_SNAPSHOT}"
printf '[current NPU state after keep-alive stop]\n'
"${NPU_SMI_BIN}" info
write_state PREPARED "${CARD_CSV}" "" pending
mirror_state

release_lifecycle_lock
trap - EXIT INT TERM
printf 'NPU_PREPARED run_id=%s node=%s stopped_card_ids=%s state=%s lease=%s\n' \
    "${RUN_ID}" "${NODE_RANK}" "${CARD_CSV}" "${STATE_FILE}" "${LEASE_DIR}"
