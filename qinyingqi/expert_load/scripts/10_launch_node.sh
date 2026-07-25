#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_CONFIG_ARG="${1:-}"
NODE_CONFIG_ARG="${2:-}"
load_configs "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}"
require_run_id
require_cmd cp
require_cmd date
require_cmd docker
require_cmd python3
require_cmd sha256sum
require_cmd ss
NPU_SMI_HOST_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_HOST_BIN="$(resolve_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"

umask 077
CONTAINER_NAME="$(container_name)"
RUN_DIR="$(host_run_dir)"
LOCAL_SERVICE_DIR="$(local_run_dir)/service"
mkdir -p "${RUN_DIR}" "${LOCAL_SERVICE_DIR}"

[[ ! -e "${RUN_DIR}/launch.started" && \
   ! -e "${LOCAL_SERVICE_DIR}/launch.started" ]] || \
    die "RUN_ID was already launched on this node; inspect it or choose a new RUN_ID"
if container_exists "${CONTAINER_NAME}"; then
    die "container already exists; inspect it instead of overwriting: ${CONTAINER_NAME}"
fi

INVOCATION_ID="launch-node${NODE_RANK}-$(date -u +%Y%m%dT%H%M%S)-$$-${RANDOM}"
LAUNCH_ATTEMPTED=0

container_owned_by_invocation() {
    container_exists "${CONTAINER_NAME}" || return 1
    [[ "$(docker container inspect "${CONTAINER_NAME}" \
        --format '{{index .Config.Labels "glm52.invocation_id"}}' 2>/dev/null)" == \
       "${INVOCATION_ID}" ]]
}

cleanup_after_launch_failure() {
    local exit_status=$?
    trap - EXIT INT TERM
    if (( exit_status != 0 )); then
        {
            printf 'failed_at=%s\n' "$(date --iso-8601=seconds)"
            printf 'exit_status=%s\n' "${exit_status}"
        } >"${LOCAL_SERVICE_DIR}/launch.failed"
        cp -p "${LOCAL_SERVICE_DIR}/launch.failed" \
            "${RUN_DIR}/launch.failed" 2>/dev/null || true
        if (( LAUNCH_ATTEMPTED == 1 )) && container_owned_by_invocation; then
            docker logs --timestamps "${CONTAINER_NAME}" \
                >"${RUN_DIR}/container.launch-failure.log" 2>&1 || true
            if container_is_running "${CONTAINER_NAME}"; then
                docker stop -t "${STOP_TIMEOUT_SECONDS}" "${CONTAINER_NAME}" || true
            fi
        elif container_exists "${CONTAINER_NAME}"; then
            warn "same-name container is not owned by this launch invocation; leaving it untouched"
        fi
        warn "if the peer model container is already running, stop that run there too"
        warn "use a fresh RUN_ID for the next attempt"
    fi
    exit "${exit_status}"
}
trap cleanup_after_launch_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_config_gate preflight
require_config_gate hccn_ping
require_cluster_image_gate
require_cluster_hccl_gate
require_npu_ready
require_model_ready
docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 || die "image not found locally: ${IMAGE_REF}"

python3 "${SCRIPT_DIR}/validate_model_files.py" \
    --model-path "${MODEL_HOST_PATH}" \
    --output "${RUN_DIR}/model-validation.prelaunch.json"

DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
DOCKER_FREE_GIB="$(available_gib "${DOCKER_ROOT}")"
(( DOCKER_FREE_GIB >= MIN_DOCKER_FREE_GIB )) || \
    die "DockerRootDir has only ${DOCKER_FREE_GIB} GiB free at launch"

printf '[NPU state immediately before launch]\n' | tee "${RUN_DIR}/launch.npu-smi.txt"
"${NPU_SMI_HOST_BIN}" info | tee -a "${RUN_DIR}/launch.npu-smi.txt"
probe_image_packages | tee "${RUN_DIR}/image.packages.txt"
IMAGE_PATCH_ID="$(image_capture_patch_id)"
printf '%s\n' "${IMAGE_PATCH_ID}" >"${RUN_DIR}/image.capture-patch-id.txt"

if [[ "${RUN_PROFILE}" == vendor_smoke ]]; then
    [[ "${ENABLE_ROUTE_CAPTURE}" == 0 ]] || \
        die "vendor_smoke requires ENABLE_ROUTE_CAPTURE=0"
elif [[ "${RUN_PROFILE}" == expert_capture ]]; then
    [[ "${ENABLE_ROUTE_CAPTURE}" == 1 ]] || \
        die "expert_capture requires ENABLE_ROUTE_CAPTURE=1"
    [[ "${MAX_NUM_SEQS}" == 1 ]] || \
        die "first expert_capture baseline requires MAX_NUM_SEQS=1"
    [[ "${VLLM_VERSION_OVERRIDE:-}" == 0.22.1 ]] || \
        die "expert_capture derived image must declare VLLM_VERSION_OVERRIDE=0.22.1"
    [[ "${MODEL_REVISION}" != master && "${MODEL_REVISION}" != main ]] || \
        die "expert_capture requires an immutable MODEL_REVISION"
    [[ "$(package_version_from_file vllm "${RUN_DIR}/image.packages.txt")" == \
       "${EXPECTED_VLLM_PACKAGE_VERSION}" ]] || \
        die "expert_capture image has the wrong installed vLLM version"
    [[ "$(package_version_from_file vllm-ascend "${RUN_DIR}/image.packages.txt")" == \
       "${EXPECTED_VLLM_ASCEND_PACKAGE_VERSION}" ]] || \
        die "expert_capture image has the wrong installed vLLM-Ascend version"
    [[ -n "${IMAGE_PATCH_ID}" && "${IMAGE_PATCH_ID}" == "${CAPTURE_PATCH_ID}" ]] || \
        die "expert_capture image label glm52.capture_patch_id does not match CAPTURE_PATCH_ID"
fi

if [[ "${NODE_RANK}" == 0 ]]; then
    if ss -H -ltn "sport = :${API_PORT}" | grep -q .; then
        die "API_PORT=${API_PORT} is already in use"
    fi
    if ss -H -ltn "sport = :${DP_RPC_PORT}" | grep -q .; then
        die "DP_RPC_PORT=${DP_RPC_PORT} is already in use"
    fi
fi

DOCKER_ARGS=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --label "glm52.experiment=expert-load"
    --label "glm52.run_id=${RUN_ID}"
    --label "glm52.node_rank=${NODE_RANK}"
    --label "glm52.profile=${RUN_PROFILE}"
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
    --mount "type=bind,source=${MODEL_HOST_PATH},target=${MODEL_CONTAINER_PATH},readonly"
    --mount "type=bind,source=${RUN_DIR},target=/runs"
)
if [[ -f /etc/hccn.conf ]]; then
    DOCKER_ARGS+=(--mount type=bind,source=/etc/hccn.conf,target=/etc/hccn.conf,readonly)
fi

DOCKER_ARGS+=(
    -e "HCCL_OP_EXPANSION_MODE=AIV"
    -e "HCCL_IF_IP=${LOCAL_IP}"
    -e "GLOO_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "TP_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "HCCL_SOCKET_IFNAME=${LOCAL_NIC}"
    -e "VLLM_RPC_TIMEOUT=360000"
    -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000"
    -e "HCCL_EXEC_TIMEOUT=200"
    -e "HCCL_CONNECT_TIMEOUT=120"
    -e "OMP_PROC_BIND=false"
    -e "OMP_NUM_THREADS=10"
    -e "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    -e "ACL_OP_INIT_MODE=1"
    -e "TASK_QUEUE_ENABLE=1"
    -e "CPU_AFFINITY_CONF=1"
    -e "VLLM_ENGINE_READY_TIMEOUT_S=1200"
    -e "VLLM_ASCEND_BALANCE_SCHEDULING=0"
    -e "DYNAMIC_EPLB=false"
    -e "EXPERT_MAP_RECORD=false"
    -e "VLLM_ASCEND_ENABLE_FUSED_MC2=0"
)
if [[ -n "${VLLM_VERSION_OVERRIDE:-}" ]]; then
    DOCKER_ARGS+=(-e "VLLM_VERSION=${VLLM_VERSION_OVERRIDE}")
fi

SERVICE_ARGS=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --data-parallel-size "${DP_SIZE}"
    --data-parallel-size-local "${DP_SIZE_LOCAL}"
    --data-parallel-address "${NODE0_COORDINATOR_IP}"
    --data-parallel-rpc-port "${DP_RPC_PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --enable-expert-parallel
    --quantization ascend
    --safetensors-load-strategy prefetch
    --block-size "${BLOCK_SIZE}"
    --seed "${SEED}"
)

if [[ "${NODE_RANK}" == 0 ]]; then
    SERVICE_ARGS+=(--host "${API_BIND_HOST}" --port "${API_PORT}" --api-server-count 1)
else
    SERVICE_ARGS+=(--headless --data-parallel-start-rank 1)
fi

if [[ "${RUN_PROFILE}" == expert_capture ]]; then
    SERVICE_ARGS+=(
        --generation-config vllm
        --enable-return-routed-experts
        --enforce-eager
        --no-async-scheduling
        --no-enable-prefix-caching
        --additional-config
        '{"enable_balance_scheduling":false,"enable_fused_mc2":0,"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":0}}'
    )
fi

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'container_name=%s\n' "${CONTAINER_NAME}"
    printf 'node_rank=%s\n' "${NODE_RANK}"
    printf 'run_profile=%s\n' "${RUN_PROFILE}"
    printf 'invocation_id=%s\n' "${INVOCATION_ID}"
    printf 'image_ref=%s\n' "${IMAGE_REF}"
    printf 'image_capture_patch_id=%s\n' "${IMAGE_PATCH_ID}"
    printf 'model_id=%s\n' "${MODEL_ID}"
    printf 'model_revision=%s\n' "${MODEL_REVISION}"
    image_metadata
} >"${RUN_DIR}/launch.metadata"
write_shell_command "${RUN_DIR}/launch.command.sh" \
    "${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${SERVICE_ARGS[@]}"

RUNNING_NPU_CONTAINERS="$(running_npu_container_ids)"
if [[ -n "${RUNNING_NPU_CONTAINERS}" ]]; then
    warn "running containers expose NPU devices: ${RUNNING_NPU_CONTAINERS}"
    warn "continuing because NPU occupancy is managed manually on this server"
fi
LAUNCH_ATTEMPTED=1
CONTAINER_ID="$("${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${SERVICE_ARGS[@]}")"
[[ "${CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]] || \
    die "docker run returned an invalid container ID: ${CONTAINER_ID}"
container_owned_by_invocation || \
    die "created container is missing or has the wrong invocation label"
[[ "$(docker container inspect "${CONTAINER_NAME}" --format '{{.Id}}')" == \
   "${CONTAINER_ID}" ]] || die "created container ID does not match its assigned name"
{
    printf 'container_id=%s\n' "${CONTAINER_ID}"
    printf 'container_name=%s\n' "${CONTAINER_NAME}"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'node_rank=%s\n' "${NODE_RANK}"
    printf 'run_profile=%s\n' "${RUN_PROFILE}"
    printf 'invocation_id=%s\n' "${INVOCATION_ID}"
} >"${LOCAL_SERVICE_DIR}/ownership.env"
printf '%s\n' "${CONTAINER_ID}" >"${LOCAL_SERVICE_DIR}/container.id"
printf '%s\n' "${CONTAINER_NAME}" >"${LOCAL_SERVICE_DIR}/container.name"
touch "${LOCAL_SERVICE_DIR}/launch.started"
printf '%s\n' "${CONTAINER_ID}" >"${RUN_DIR}/container.id"
printf '%s\n' "${CONTAINER_NAME}" >"${RUN_DIR}/container.name"
touch "${RUN_DIR}/launch.started"
docker inspect "${CONTAINER_NAME}" >"${RUN_DIR}/container.initial.inspect.json"

sleep 2
if ! container_is_running "${CONTAINER_NAME}"; then
    docker logs --timestamps "${CONTAINER_NAME}" \
        >"${RUN_DIR}/container.early-exit.log" 2>&1 || true
    die "container exited during launch; preserved for inspection: ${CONTAINER_NAME}"
fi

trap - EXIT INT TERM
printf 'LAUNCH_OK container=%s id=%s run_dir=%s\n' \
    "${CONTAINER_NAME}" "${CONTAINER_ID}" "${RUN_DIR}"
printf 'Next: launch the peer node with the same RUN_ID, then run 11_wait_ready.sh on node0.\n'
