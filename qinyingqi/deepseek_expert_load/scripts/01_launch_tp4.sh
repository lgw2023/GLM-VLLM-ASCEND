#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
shift || true
CONFIRMED_IDS=""
while (($#)); do
    case "$1" in
        --confirm-npu-ids)
            CONFIRMED_IDS="${2:-}"
            shift 2
            ;;
        -h|--help)
            printf 'Usage: bash scripts/01_launch_tp4.sh CONFIG --confirm-npu-ids ID0,ID1,ID2,ID3\n'
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

load_config "${CONFIG_PATH}"
for name in IMAGE_REF MODEL_HOST_PATH RUN_ROOT HOST_NPU_IDS SERVED_MODEL_NAME \
    API_HOST API_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS \
    GPU_MEMORY_UTILIZATION; do
    require_var "${name}"
done
for command in docker python3 npu-smi ss; do
    require_cmd "${command}"
done

VALIDATED_IDS="$(validate_npu_ids)"
[[ "${CONFIRMED_IDS}" == "${VALIDATED_IDS}" ]] || \
    die "allocation confirmation mismatch; rerun with --confirm-npu-ids ${VALIDATED_IDS}"
[[ "${MAX_NUM_SEQS}" == 1 ]] || \
    die "route baseline requires MAX_NUM_SEQS=1"
[[ "${API_HOST}" == 127.0.0.1 ]] || \
    die "this shared-server pilot requires API_HOST=127.0.0.1"
[[ -d "${MODEL_HOST_PATH}" ]] || die "model directory not found: ${MODEL_HOST_PATH}"
docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 || \
    die "Docker image is absent: ${IMAGE_REF}"
if ss -H -ltn "sport = :${API_PORT}" | grep -q .; then
    die "API port is already in use: ${API_PORT}"
fi

mkdir -p "${RUN_ROOT}"
RUN_ID="${RUN_ID:-dsv4-tp4-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_.-]+$ ]] || \
    die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
[[ "${API_PORT}" =~ ^[0-9]+$ && "${API_PORT}" -ge 1 && "${API_PORT}" -le 65535 ]] || \
    die "API_PORT must be an integer in 1..65535"
[[ "${RUN_ROOT}" != / && "${RUN_ROOT}" != "${HOME}" ]] || \
    die "RUN_ROOT is too broad: ${RUN_ROOT}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_ID}" >"${RUN_ROOT}/current-run-id"
CONTAINER_NAME="$(container_name_for_run "${RUN_ID}")"
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    die "container already exists: ${CONTAINER_NAME}"
fi

MODEL_AUDIT_PATH="${RUN_DIR}/model-audit.json"
python3 "${SCRIPT_DIR}/00_audit_model.py" \
    --model-path "${MODEL_HOST_PATH}" \
    --require-model-type deepseek_v4 \
    --require-w4a8 \
    --output "${MODEL_AUDIT_PATH}"

read -r QUANTIZATION_PROFILE VLLM_QUANTIZATION < <(
    python3 - "${MODEL_AUDIT_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    report = json.load(source)
quantization = report["quantization"]
profile = quantization.get("deployment_profile")
method = quantization.get("recommended_vllm_quantization")
allowed = {"ascend", "fp8", "deepseek_v4_fp8"}
if not isinstance(profile, str) or not profile:
    raise SystemExit("model audit did not produce a deployment profile")
if method not in allowed:
    raise SystemExit(
        f"model audit produced unsupported vLLM quantization {method!r}; "
        f"expected one of {sorted(allowed)}"
    )
print(profile, method)
PY
)
printf 'quantization_profile=%s vllm_quantization=%s\n' \
    "${QUANTIZATION_PROFILE}" "${VLLM_QUANTIZATION}"

docker run --rm --entrypoint python "${IMAGE_REF}" -c '
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
vllm = version("vllm")
ascend = version("vllm-ascend")
print(f"vllm={vllm}")
print(f"vllm-ascend={ascend}")
if not vllm.startswith("0.22.1"):
    raise SystemExit(f"expected vLLM 0.22.1, got {vllm}")
if not ascend.startswith("0.22.1rc1"):
    raise SystemExit(f"expected vLLM-Ascend 0.22.1rc1, got {ascend}")
spec = find_spec("vllm_ascend")
roots = [] if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations)
if not any((Path(root) / "models/deepseek_v4.py").is_file() for root in roots):
    raise SystemExit("image has no vllm_ascend.models.deepseek_v4")
' | tee "${RUN_DIR}/image-audit.txt"

NPU_SMI_HOST_BIN="$(resolve_host_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_HOST_BIN="$(resolve_host_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"
printf '[Host NPU state before launch; verify the selected cards are allocated and idle]\n' \
    | tee "${RUN_DIR}/npu-smi.before.txt"
"${NPU_SMI_HOST_BIN}" info | tee -a "${RUN_DIR}/npu-smi.before.txt"

DEVICE_SMOKE_ARGS=(
    docker run --rm
    --net=host
    --shm-size=1g
)
IFS=',' read -r -a PHYSICAL_IDS <<<"${VALIDATED_IDS}"
for logical_id in 0 1 2 3; do
    physical_id="${PHYSICAL_IDS[${logical_id}]}"
    [[ -e "/dev/davinci${physical_id}" ]] || \
        die "host device is absent: /dev/davinci${physical_id}"
    DEVICE_SMOKE_ARGS+=(--device "/dev/davinci${physical_id}:/dev/davinci${logical_id}")
done
DEVICE_SMOKE_ARGS+=(
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    --mount type=bind,source=/usr/local/Ascend/driver/lib64,target=/usr/local/Ascend/driver/lib64,readonly
    --mount type=bind,source=/usr/local/Ascend/driver/version.info,target=/usr/local/Ascend/driver/version.info,readonly
    --mount type=bind,source=/etc/ascend_install.info,target=/etc/ascend_install.info,readonly
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    --entrypoint python
)
"${DEVICE_SMOKE_ARGS[@]}" "${IMAGE_REF}" -c '
import torch
import torch_npu
if not torch.npu.is_available():
    raise SystemExit("torch.npu is not available")
count = torch.npu.device_count()
print(f"container_npu_count={count}")
if count != 4:
    raise SystemExit(f"expected exactly four container NPUs, got {count}")
for index in range(count):
    value = torch.ones(1, device=f"npu:{index}")
    if value.item() != 1:
        raise SystemExit(f"tiny allocation failed on npu:{index}")
torch.npu.synchronize()
print("DEVICE_MAPPING_OK")
' | tee "${RUN_DIR}/device-smoke.txt"

DOCKER_ARGS=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --label "deepseek_expert_load.project=deepseek-v4-tp4"
    --label "deepseek_expert_load.run_id=${RUN_ID}"
    --label "deepseek_expert_load.host_npu_ids=${VALIDATED_IDS}"
    --net=host
    --shm-size=128g
    --ulimit memlock=-1:-1
)
for logical_id in 0 1 2 3; do
    physical_id="${PHYSICAL_IDS[${logical_id}]}"
    [[ -e "/dev/davinci${physical_id}" ]] || \
        die "host device is absent: /dev/davinci${physical_id}"
    DOCKER_ARGS+=(--device "/dev/davinci${physical_id}:/dev/davinci${logical_id}")
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
    --mount "type=bind,source=${MODEL_HOST_PATH},target=/model,readonly"
    --mount "type=bind,source=${RUN_DIR},target=/run-output"
)
if [[ -f /etc/hccn.conf ]]; then
    DOCKER_ARGS+=(--mount type=bind,source=/etc/hccn.conf,target=/etc/hccn.conf,readonly)
fi
DOCKER_ARGS+=(
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    -e VLLM_VERSION=0.22.1
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    -e VLLM_ASCEND_ENABLE_FUSED_MC2=0
    -e OMP_PROC_BIND=false
    -e OMP_NUM_THREADS=10
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    -e HCCL_BUFFSIZE=1024
    -e HCCL_OP_EXPANSION_MODE=AIV
    -e TASK_QUEUE_ENABLE=1
    -e VLLM_RPC_TIMEOUT=360000
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e VLLM_ASCEND_BALANCE_SCHEDULING=0
    -e DYNAMIC_EPLB=false
    -e EXPERT_MAP_RECORD=false
)

SERVICE_ARGS=(
    vllm serve /model
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${API_HOST}"
    --port "${API_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size 4
    --enable-expert-parallel
    --quantization "${VLLM_QUANTIZATION}"
    --tokenizer-mode deepseek_v4
    --block-size 128
    --generation-config vllm
    --enable-return-routed-experts
    --enforce-eager
    --no-async-scheduling
    --no-enable-prefix-caching
    --safetensors-load-strategy prefetch
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":64}'
    --additional-config '{"enable_balance_scheduling":false,"enable_fused_mc2":0,"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":0}}'
)

write_shell_command "${RUN_DIR}/launch.command.sh" \
    "${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${SERVICE_ARGS[@]}"
{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'container_name=%s\n' "${CONTAINER_NAME}"
    printf 'image_ref=%s\n' "${IMAGE_REF}"
    printf 'model_host_path=%s\n' "${MODEL_HOST_PATH}"
    printf 'served_model_name=%s\n' "${SERVED_MODEL_NAME}"
    printf 'quantization_profile=%s\n' "${QUANTIZATION_PROFILE}"
    printf 'vllm_quantization=%s\n' "${VLLM_QUANTIZATION}"
    printf 'host_npu_ids=%s\n' "${VALIDATED_IDS}"
    printf 'container_npu_ids=0,1,2,3\n'
    printf 'api_url=http://%s:%s/v1\n' "${API_HOST}" "${API_PORT}"
} >"${RUN_DIR}/run.env"

CONTAINER_ID="$("${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${SERVICE_ARGS[@]}")"
[[ "${CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]] || \
    die "docker run returned an invalid container ID: ${CONTAINER_ID}"
printf '%s\n' "${CONTAINER_ID}" >"${RUN_DIR}/container.id"
docker inspect "${CONTAINER_NAME}" >"${RUN_DIR}/container.initial.inspect.json"
sleep 2
if [[ "$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}')" != true ]]; then
    docker logs --timestamps "${CONTAINER_NAME}" >"${RUN_DIR}/container.early-exit.log" 2>&1 || true
    die "container exited early; inspect ${RUN_DIR}/container.early-exit.log"
fi

printf 'LAUNCH_OK run_id=%s container=%s host_npu_ids=%s\n' \
    "${RUN_ID}" "${CONTAINER_NAME}" "${VALIDATED_IDS}"
printf 'Next: bash scripts/02_wait_ready.sh %s\n' "${CONFIG_PATH}"
