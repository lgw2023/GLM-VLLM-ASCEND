#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_CONFIG_ARG="${1:-}"
NODE_CONFIG_ARG="${2:-}"
load_configs "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}"
require_run_id
require_cmd docker
require_cmd date
require_cmd python3
require_cmd ss
require_cmd tee
require_cmd timeout

umask 077
RUN_DIR="$(host_run_dir)"
OUT_DIR="${RUN_DIR}/hccl-collective"
mkdir -p "${OUT_DIR}"
CONTAINER_NAME="glm52-hccl-node${NODE_RANK}-${RUN_ID}"
acquire_lifecycle_lock hccl
trap 'release_lifecycle_lock' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
KEEPALIVE_STATE="$(keepalive_state_file)"
[[ -f "${KEEPALIVE_STATE}" ]] || die "run 05_prepare_npus.sh first"
[[ "$(gate_value status "${KEEPALIVE_STATE}")" == PREPARED ]] || \
    die "keep-alive state must be PREPARED before HCCL collective test"
[[ "$(gate_value run_id "${KEEPALIVE_STATE}")" == "${RUN_ID}" ]] || \
    die "keep-alive state belongs to another run"
[[ "$(gate_value node_rank "${KEEPALIVE_STATE}")" == "${NODE_RANK}" ]] || \
    die "keep-alive state belongs to another node"
[[ "$(gate_value stopped_card_ids "${KEEPALIVE_STATE}")" == 0,1,2,3,4,5,6,7 ]] || \
    die "keep-alive state does not cover exactly cards 0..7"
require_run_npu_lease

container_exists "${CONTAINER_NAME}" && \
    die "HCCL test container already exists: ${CONTAINER_NAME}"
[[ ! -e "${RUN_DIR}/launch.started" && \
   ! -e "$(local_run_dir)/service/launch.started" ]] || \
    die "model launch already started for this RUN_ID; HCCL gate may not be rerun"
RUNNING_NPU_CONTAINERS="$(running_npu_container_ids)"
[[ -z "${RUNNING_NPU_CONTAINERS}" ]] || \
    die "refusing HCCL gate while NPU containers are running: ${RUNNING_NPU_CONTAINERS}"

INVOCATION_ID="hccl-node${NODE_RANK}-$(date -u +%Y%m%dT%H%M%S)-$$-${RANDOM}"
HCCL_ATTEMPTED=0

container_owned_by_invocation() {
    container_exists "${CONTAINER_NAME}" || return 1
    [[ "$(docker container inspect "${CONTAINER_NAME}" \
        --format '{{index .Config.Labels "glm52.invocation_id"}}' 2>/dev/null)" == \
       "${INVOCATION_ID}" ]]
}

COLLECTIVE_SUCCEEDED=0
restore_after_collective_failure() {
    local exit_status=$?
    local cleanup_signal_status=0
    trap - EXIT
    trap 'cleanup_signal_status=130' INT
    trap 'cleanup_signal_status=143' TERM
    if (( exit_status != 0 && COLLECTIVE_SUCCEEDED == 0 )); then
        local safe_to_restore=1 running_consumers
        if (( HCCL_ATTEMPTED == 1 )) && \
           container_owned_by_invocation && container_is_running "${CONTAINER_NAME}"; then
            docker stop -t 30 "${CONTAINER_NAME}" || true
        fi
        if container_owned_by_invocation && container_is_running "${CONTAINER_NAME}"; then
            safe_to_restore=0
        fi
        if ! running_consumers="$(running_npu_container_ids)"; then
            warn "NPU consumer scan failed; keep-alive will not be restored"
            safe_to_restore=0
        elif [[ -n "${running_consumers}" ]]; then
            safe_to_restore=0
        fi
        if (( safe_to_restore == 1 )); then
            warn "HCCL gate failed locally; restoring keep-alive on node${NODE_RANK}"
            bash "${SCRIPT_DIR}/20_restore_npus.sh" \
                "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}" || true
        else
            warn "HCCL container is still running; keep-alive was NOT restored to avoid an NPU conflict"
        fi
    fi
    release_lifecycle_lock || true
    (( cleanup_signal_status == 0 )) || exit_status=${cleanup_signal_status}
    exit "${exit_status}"
}
trap restore_after_collective_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "${NPU_LAUNCH_CONFIRMATION:-}" == "${RUN_ID}" ]] || \
    die "export NPU_LAUNCH_CONFIRMATION=${RUN_ID} after the per-run occupancy check"
require_config_gate preflight
require_config_gate hccn_ping
require_cluster_image_gate
if [[ "${NODE_RANK}" == 0 ]] && ss -H -ltn "sport = :${HCCL_TEST_PORT}" | grep -q .; then
    die "HCCL_TEST_PORT=${HCCL_TEST_PORT} is already in use"
fi

python3 "${SCRIPT_DIR}/keepalive_state.py" snapshot \
    --output "${OUT_DIR}/keepalive-before-hccl.json"
python3 "${SCRIPT_DIR}/keepalive_state.py" validate --expected stopped \
    "${OUT_DIR}/keepalive-before-hccl.json"

NPU_SMI_HOST_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_HOST_BIN="$(resolve_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"
"${NPU_SMI_HOST_BIN}" info | tee "${OUT_DIR}/npu-smi-before-hccl.txt"
DOCKER_ARGS=(
    docker run
    --name "${CONTAINER_NAME}"
    --label "glm52.experiment=hccl-gate"
    --label "glm52.run_id=${RUN_ID}"
    --label "glm52.node_rank=${NODE_RANK}"
    --label "glm52.invocation_id=${INVOCATION_ID}"
    --net=host
    --shm-size=1g
)
for device_id in {0..7}; do
    DOCKER_ARGS+=(--device "/dev/davinci${device_id}")
done
DOCKER_ARGS+=(
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    --mount type=bind,source=/usr/local/dcmi,target=/usr/local/dcmi,readonly
    --mount "type=bind,source=${HCCN_TOOL_HOST_BIN},target=/usr/local/Ascend/driver/tools/hccn_tool,readonly"
    --mount "type=bind,source=${NPU_SMI_HOST_BIN},target=/usr/local/bin/npu-smi,readonly"
    --mount type=bind,source=/usr/local/Ascend/driver/lib64,target=/usr/local/Ascend/driver/lib64,readonly
    --mount type=bind,source=/usr/local/Ascend/driver/version.info,target=/usr/local/Ascend/driver/version.info,readonly
    --mount type=bind,source=/etc/ascend_install.info,target=/etc/ascend_install.info,readonly
    --mount "type=bind,source=${EXPERT_LOAD_ROOT},target=/workspace/expert_load,readonly"
    -e "HCCL_IF_IP=${LOCAL_IP}"
    -e "GLOO_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "TP_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "HCCL_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "HCCL_CONNECT_TIMEOUT=120"
    -e "HCCL_EXEC_TIMEOUT=200"
    -e "HCCL_TEST_TIMEOUT_SECONDS=${HCCL_TEST_TIMEOUT_SECONDS}"
    --entrypoint python
)
if [[ -f /etc/hccn.conf ]]; then
    DOCKER_ARGS+=(--mount type=bind,source=/etc/hccn.conf,target=/etc/hccn.conf,readonly)
fi

COLLECTIVE_ARGS=(
    -m torch.distributed.run
    --nnodes=2
    --nproc-per-node=8
    --node-rank="${NODE_RANK}"
    --master-addr="${NODE0_COORDINATOR_IP}"
    --master-port="${HCCL_TEST_PORT}"
    /workspace/expert_load/scripts/hccl_collective_smoke.py
)
write_shell_command "${OUT_DIR}/command.sh" \
    timeout --foreground --kill-after=30 "${HCCL_TEST_TIMEOUT_SECONDS}s" \
    "${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${COLLECTIVE_ARGS[@]}"

require_no_npu_containers "HCCL container launch"
set +e
HCCL_ATTEMPTED=1
timeout --foreground --kill-after=30 "${HCCL_TEST_TIMEOUT_SECONDS}s" \
    "${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${COLLECTIVE_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/collective.log"
COLLECTIVE_STATUS=${PIPESTATUS[0]}
set -e

if (( COLLECTIVE_STATUS != 0 )); then
    if container_owned_by_invocation; then
        if container_is_running "${CONTAINER_NAME}"; then
            docker stop -t 30 "${CONTAINER_NAME}" || true
        fi
        docker inspect "${CONTAINER_NAME}" >"${OUT_DIR}/container.failed.inspect.json" 2>/dev/null || true
        docker logs --timestamps "${CONTAINER_NAME}" >"${OUT_DIR}/container.failed.log" 2>&1 || true
    elif container_exists "${CONTAINER_NAME}"; then
        warn "same-name container is not owned by this HCCL invocation; leaving it untouched"
    fi
    die "HCCL collective gate failed with status ${COLLECTIVE_STATUS}; stop/recover the peer node"
fi

container_owned_by_invocation || \
    die "completed HCCL container is missing or has the wrong invocation label"
docker inspect "${CONTAINER_NAME}" >"${OUT_DIR}/container.inspect.json"
docker logs --timestamps "${CONTAINER_NAME}" >"${OUT_DIR}/container.log" 2>&1 || true
docker rm "${CONTAINER_NAME}"
{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'node_rank=%s\n' "${NODE_RANK}"
    printf 'invocation_id=%s\n' "${INVOCATION_ID}"
    printf 'image_id=%s\n' "$(current_image_id)"
    printf 'config_fingerprint=%s\n' "$(config_fingerprint)"
    printf 'cluster_config_sha256=%s\n' "$(cluster_config_sha256)"
    printf 'root_commit=%s\n' "$(root_commit)"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
} >"${OUT_DIR}/HCCL_COLLECTIVE_OK"
COLLECTIVE_SUCCEEDED=1
release_lifecycle_lock
trap - EXIT INT TERM
printf 'HCCL_COLLECTIVE_OK run_id=%s node=%s output=%s\n' \
    "${RUN_ID}" "${NODE_RANK}" "${OUT_DIR}"
