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
            printf 'Usage: bash scripts/02_launch_tp8_w4a8.sh CONFIG --confirm-npu-ids 0,1,2,3,4,5,6,7\n'
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
done

load_config "${CONFIG_PATH}"
for name in IMAGE_REF CAPTURE_PATCH_ID EXPECTED_VLLM_PACKAGE_VERSION \
    EXPECTED_VLLM_ASCEND_PACKAGE_VERSION MODEL_HOST_PATH RUN_ROOT HOST_NPU_IDS \
    SERVED_MODEL_NAME API_HOST API_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS \
    MAX_NUM_SEQS GPU_MEMORY_UTILIZATION REQUIRED_EXPERT_QUANTIZATION TARGET_SOC; do
    require_var "${name}"
done
for command in docker python3 ss; do
    require_cmd "${command}"
done

[[ "${TARGET_SOC}" == ASCEND910B1 ]] || die "Node3 profile requires TARGET_SOC=ASCEND910B1"
[[ "${REQUIRED_EXPERT_QUANTIZATION}" == w4a8 ]] || \
    die "single-node 8-card A2 profile requires W4A8, not W8A8"
[[ "${MAX_NUM_SEQS}" == 1 ]] || die "first route baseline requires MAX_NUM_SEQS=1"
[[ "${API_HOST}" == 127.0.0.1 ]] || die "shared-server API must bind to 127.0.0.1"
[[ -d "${MODEL_HOST_PATH}" ]] || die "model directory not found: ${MODEL_HOST_PATH}"
VALIDATED_IDS="$(validate_npu_ids)"
[[ "${CONFIRMED_IDS}" == "${VALIDATED_IDS}" ]] || \
    die "allocation confirmation mismatch; use --confirm-npu-ids ${VALIDATED_IDS}"
if ss -H -ltn "sport = :${API_PORT}" | grep -q .; then
    die "API port is already in use: ${API_PORT}"
fi

docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 || \
    die "capture image is absent: ${IMAGE_REF}; run scripts/01_build_capture_image.sh first"
IMAGE_PATCH_ID="$(docker image inspect "${IMAGE_REF}" \
    --format '{{if .Config.Labels}}{{index .Config.Labels "glm51.capture_patch_id"}}{{end}}')"
[[ "${IMAGE_PATCH_ID}" == "${CAPTURE_PATCH_ID}" ]] || \
    die "capture image label mismatch: expected ${CAPTURE_PATCH_ID}, got ${IMAGE_PATCH_ID:-<empty>}"

mkdir -p "${RUN_ROOT}"
RUN_ID="${RUN_ID:-glm51-w4a8-tp8-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_.-]+$ ]] || die "invalid RUN_ID: ${RUN_ID}"
[[ "${RUN_ROOT}" != / && "${RUN_ROOT}" != "${HOME}" ]] || die "RUN_ROOT is too broad"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_ID}" >"${RUN_ROOT}/current-run-id"
CONTAINER_NAME="$(container_name_for_run "${RUN_ID}")"
docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 && \
    die "container already exists: ${CONTAINER_NAME}"

python3 "${SCRIPT_DIR}/00_audit_model.py" \
    --model-path "${MODEL_HOST_PATH}" \
    --output "${RUN_DIR}/model-audit.json"

docker run --rm --entrypoint python "${IMAGE_REF}" - \
    "${EXPECTED_VLLM_PACKAGE_VERSION}" \
    "${EXPECTED_VLLM_ASCEND_PACKAGE_VERSION}" <<'PY' \
    | tee "${RUN_DIR}/image-audit.txt"
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import sys

expected_vllm, expected_ascend = sys.argv[1:]
actual_vllm = version("vllm")
actual_ascend = version("vllm-ascend")
actual_transformers = version("transformers")
print(f"vllm={actual_vllm}")
print(f"vllm-ascend={actual_ascend}")
print(f"transformers={actual_transformers}")
if actual_vllm.split("+")[0] != expected_vllm:
    raise SystemExit("unexpected vLLM version")
if actual_ascend.split("+")[0] != expected_ascend:
    raise SystemExit("unexpected vLLM-Ascend version")
if tuple(int(x) for x in actual_transformers.split("+")[0].split(".")[:2]) < (5, 2):
    raise SystemExit("GLM-5.1 requires transformers>=5.2")

def one(import_name, relative):
    spec = find_spec(import_name)
    roots = [] if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations)
    paths = [Path(root) / relative for root in roots]
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise SystemExit(f"expected one installed {relative}, got {paths}")
    return paths[0]

model_source = one("vllm", "model_executor/models/deepseek_v2.py").read_text(encoding="utf-8")
if "class GlmMoeDsaForCausalLM" not in model_source:
    raise SystemExit("image has no GlmMoeDsaForCausalLM implementation")
fused = one("vllm_ascend", "ops/fused_moe/fused_moe.py").read_text(encoding="utf-8")
runner = one("vllm_ascend", "worker/model_runner_v1.py").read_text(encoding="utf-8")
capture = one(
    "vllm", "model_executor/layers/fused_moe/routed_experts_capturer.py"
).read_text(encoding="utf-8")
if fused.count("# GLM51_FUSED_MOE_CAPTURE_BEFORE_PREPARE_V2") != 1:
    raise SystemExit("GLM-5.1 fused-MoE capture marker is absent or duplicated")
if runner.count("# GLM51_MODEL_RUNNER_BIND_CAPTURE_V2") != 1:
    raise SystemExit("GLM-5.1 model-runner capture marker is absent or duplicated")
if capture.count("# GLM51_VLLM_TP8_CAPTURE_GATHER_V2") != 1:
    raise SystemExit("GLM-5.1 TP8 route-gather marker is absent or duplicated")
print("GLM51_IMAGE_AUDIT_OK")
PY

NPU_SMI_HOST_BIN="$(resolve_host_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_HOST_BIN="$(resolve_host_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"
printf '[Host NPU state before launch; verify all selected cards are idle]\n' \
    | tee "${RUN_DIR}/npu-smi.before.txt"
"${NPU_SMI_HOST_BIN}" info | tee -a "${RUN_DIR}/npu-smi.before.txt"
docker ps --no-trunc >"${RUN_DIR}/docker-ps.before.txt"

IFS=',' read -r -a PHYSICAL_IDS <<<"${VALIDATED_IDS}"
DOCKER_ARGS=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --label "glm51_expert_load.project=glm51-w4a8-tp8"
    --label "glm51_expert_load.run_id=${RUN_ID}"
    --label "glm51_expert_load.host_npu_ids=${VALIDATED_IDS}"
    --net=host
    --shm-size=512g
    --ulimit memlock=-1:-1
)
for logical_id in 0 1 2 3 4 5 6 7; do
    physical_id="${PHYSICAL_IDS[${logical_id}]}"
    [[ -e "/dev/davinci${physical_id}" ]] || die "missing host device /dev/davinci${physical_id}"
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
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    -e VLLM_VERSION=0.22.1
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_ASCEND_ENABLE_FLASHCOMM1=0
    -e VLLM_ASCEND_ENABLE_FUSED_MC2=0
    -e VLLM_ASCEND_MLA_PARALLEL=1
    -e VLLM_ASCEND_BALANCE_SCHEDULING=0
    -e DYNAMIC_EPLB=false
    -e EXPERT_MAP_RECORD=false
    -e OMP_PROC_BIND=false
    -e OMP_NUM_THREADS=10
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    -e HCCL_BUFFSIZE=200
    -e HCCL_OP_EXPANSION_MODE=AIV
    -e TASK_QUEUE_ENABLE=1
    -e VLLM_RPC_TIMEOUT=360000
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200
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
    --data-parallel-size 1
    --tensor-parallel-size 8
    --enable-expert-parallel
    --quantization ascend
    --trust-remote-code
    --generation-config vllm
    --enable-return-routed-experts
    --enforce-eager
    --no-async-scheduling
    --no-enable-prefix-caching
    --no-enable-chunked-prefill
    --safetensors-load-strategy prefetch
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}'
    --additional-config '{"fuse_muls_add":true,"enable_flashcomm1":false,"enable_balance_scheduling":false,"enable_fused_mc2":0,"multistream_overlap_shared_expert":false,"eplb_config":{"dynamic_eplb":false,"expert_map_path":null,"expert_map_record_path":null,"num_redundant_experts":0}}'
)

write_shell_command "${RUN_DIR}/launch.command.sh" \
    "${DOCKER_ARGS[@]}" "${IMAGE_REF}" "${SERVICE_ARGS[@]}"
{
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'container_name=%s\n' "${CONTAINER_NAME}"
    printf 'image_ref=%s\n' "${IMAGE_REF}"
    printf 'capture_patch_id=%s\n' "${IMAGE_PATCH_ID}"
    printf 'model_host_path=%s\n' "${MODEL_HOST_PATH}"
    printf 'served_model_name=%s\n' "${SERVED_MODEL_NAME}"
    printf 'quantization_profile=modelslim_w4a8\n'
    printf 'parallelism=dp1_tp8_ep\n'
    printf 'host_npu_ids=%s\n' "${VALIDATED_IDS}"
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
printf 'Next: bash scripts/03_wait_ready.sh %s\n' "${CONFIG_PATH}"
