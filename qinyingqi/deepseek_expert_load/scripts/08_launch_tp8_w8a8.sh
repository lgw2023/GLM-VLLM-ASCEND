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
            printf 'Usage: bash scripts/08_launch_tp8_w8a8.sh CONFIG --confirm-npu-ids 0,1,2,3,4,5,6,7\n'
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

load_config "${CONFIG_PATH}"
for name in IMAGE_REF CAPTURE_PATCH_ID EXPECTED_VLLM_PACKAGE_VERSION \
    EXPECTED_VLLM_ASCEND_PACKAGE_VERSION MODEL_HOST_PATH RUN_ROOT HOST_NPU_IDS SERVED_MODEL_NAME \
    API_HOST API_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS \
    GPU_MEMORY_UTILIZATION TARGET_SOC REQUIRED_EXPERT_QUANTIZATION; do
    require_var "${name}"
done
for command in docker python3 npu-smi ss; do
    require_cmd "${command}"
done

[[ "${TARGET_SOC}" == ASCEND910B1 ]] || \
    die "Node1 A2 profile requires TARGET_SOC=ASCEND910B1"
[[ "${REQUIRED_EXPERT_QUANTIZATION}" == w8a8 ]] || \
    die "Node1 TP8 profile requires REQUIRED_EXPERT_QUANTIZATION=w8a8"
VALIDATED_IDS="$(validate_npu_ids_for_count 8)"
[[ "${CONFIRMED_IDS}" == "${VALIDATED_IDS}" ]] || \
    die "allocation confirmation mismatch; rerun with --confirm-npu-ids ${VALIDATED_IDS}"
[[ "${MAX_NUM_SEQS}" == 1 ]] || \
    die "route baseline requires MAX_NUM_SEQS=1"
[[ "${API_HOST}" == 127.0.0.1 ]] || \
    die "this shared-server experiment requires API_HOST=127.0.0.1"
[[ -d "${MODEL_HOST_PATH}" ]] || die "model directory not found: ${MODEL_HOST_PATH}"
docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 || \
    die "Docker image is absent: ${IMAGE_REF}"
IMAGE_PATCH_ID="$(docker image inspect "${IMAGE_REF}" \
    --format '{{if .Config.Labels}}{{index .Config.Labels "deepseek.capture_patch_id"}}{{end}}')"
[[ "${IMAGE_PATCH_ID}" == "${CAPTURE_PATCH_ID}" ]] || \
    die "route-capture image label mismatch: expected ${CAPTURE_PATCH_ID}, got ${IMAGE_PATCH_ID:-<empty>}"
if ss -H -ltn "sport = :${API_PORT}" | grep -q .; then
    die "API port is already in use: ${API_PORT}"
fi

mkdir -p "${RUN_ROOT}"
RUN_ID="${RUN_ID:-dsv4-w8a8-tp8-$(date -u +%Y%m%dT%H%M%SZ)}"
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
    --require-expert-quantization w8a8 \
    --target-soc "${TARGET_SOC}" \
    --output "${MODEL_AUDIT_PATH}"

python3 - "${MODEL_AUDIT_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    report = json.load(source)
quantization = report["quantization"]
hardware = report["hardware"]
if quantization.get("deployment_profile") != "modelslim_w8a8":
    raise SystemExit(
        "expected deployment_profile='modelslim_w8a8', got "
        f"{quantization.get('deployment_profile')!r}"
    )
if quantization.get("recommended_vllm_quantization") != "ascend":
    raise SystemExit("W8A8 model audit did not select --quantization ascend")
if hardware.get("soc_compatible") is not True:
    raise SystemExit("W8A8 model audit did not prove Ascend 910B1 compatibility")
print("W8A8_MODEL_AUDIT_OK")
PY

docker run --rm --entrypoint python "${IMAGE_REF}" -c '
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import sys

expected_vllm = sys.argv[1]
expected_ascend = sys.argv[2]
vllm = version("vllm")
ascend = version("vllm-ascend")
print(f"vllm={vllm}")
print(f"vllm-ascend={ascend}")
if vllm.split("+")[0] != expected_vllm:
    raise SystemExit(f"expected vLLM {expected_vllm}, got {vllm}")
if ascend.split("+")[0] != expected_ascend:
    raise SystemExit(f"expected vLLM-Ascend {expected_ascend}, got {ascend}")
spec = find_spec("vllm_ascend")
roots = [] if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations)
if not any((Path(root) / "models/deepseek_v4.py").is_file() for root in roots):
    raise SystemExit("image has no vllm_ascend.models.deepseek_v4")
targets = [Path(root) / "quantization/methods/w8a8_dynamic.py" for root in roots]
targets = [path for path in targets if path.is_file()]
if len(targets) != 1:
    raise SystemExit(f"expected one installed w8a8_dynamic.py, got {targets}")
marker_count = targets[0].read_text(encoding="utf-8").count(
    "# DEEPSEEK_V4_W8A8_ROUTE_CAPTURE_V3"
)
print(f"route_capture_marker_count={marker_count}")
if marker_count != 1:
    raise SystemExit("W8A8 route-capture hook is absent or duplicated")
capture_targets = [Path(root) / "patch/worker/patch_routed_experts_capture.py" for root in roots]
capture_targets = [path for path in capture_targets if path.is_file()]
if len(capture_targets) != 1:
    raise SystemExit(f"expected one routed-experts capture source file, got {capture_targets}")
capture_source = capture_targets[0].read_text(encoding="utf-8")
capture_marker_count = capture_source.count("# DEEPSEEK_V4_TP8_CAPTURE_GATHER_V3")
print(f"tp8_capture_gather_marker_count={capture_marker_count}")
if capture_marker_count != 1:
    raise SystemExit("TP8 capture-gather hook is absent or duplicated")
if "dist.all_gather(list(gathered_splits), topk_ids, get_tp_group().device_group)" not in capture_source:
    raise SystemExit("TP8 capture-gather hook does not call the TP all-gather")
' "${EXPECTED_VLLM_PACKAGE_VERSION}" "${EXPECTED_VLLM_ASCEND_PACKAGE_VERSION}" \
    | tee "${RUN_DIR}/image-audit.txt"

NPU_SMI_HOST_BIN="$(resolve_host_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_HOST_BIN="$(resolve_host_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"
printf '[Host NPU state before launch; all eight cards must be allocated and idle]\n' \
    | tee "${RUN_DIR}/npu-smi.before.txt"
"${NPU_SMI_HOST_BIN}" info | tee -a "${RUN_DIR}/npu-smi.before.txt"

RUNNING_CONTAINERS=()
while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] && RUNNING_CONTAINERS+=("${container_id}")
done < <(docker ps -q)
if ((${#RUNNING_CONTAINERS[@]} > 0)); then
    docker inspect "${RUNNING_CONTAINERS[@]}" >"${RUN_DIR}/running-containers.before.json"
else
    printf '[]\n' >"${RUN_DIR}/running-containers.before.json"
fi
python3 - "${RUN_DIR}/running-containers.before.json" "${VALIDATED_IDS}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    containers = json.load(source)
selected = {int(value) for value in sys.argv[2].split(",")}
conflicts = []
for container in containers:
    devices = (container.get("HostConfig") or {}).get("Devices") or []
    claimed = set()
    for device in devices:
        match = re.fullmatch(r"/dev/davinci(\d+)", device.get("PathOnHost", ""))
        if match:
            claimed.add(int(match.group(1)))
    overlap = sorted(selected & claimed)
    if overlap:
        name = container.get("Name", "<unnamed>").lstrip("/")
        conflicts.append(f"{name}: {overlap}")
if conflicts:
    print(
        "WARNING: selected NPUs are exposed to running containers; Docker "
        "device exposure is not proof of active NPU use: " + "; ".join(conflicts),
        file=sys.stderr,
    )
    print("RUNNING_CONTAINER_NPU_MAPPINGS_RECORDED")
else:
    print("RUNNING_CONTAINER_NPU_CHECK_OK")
PY

DEVICE_SMOKE_ARGS=(
    docker run --rm
    --net=host
    --shm-size=1g
)
IFS=',' read -r -a PHYSICAL_IDS <<<"${VALIDATED_IDS}"
for logical_id in 0 1 2 3 4 5 6 7; do
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
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    --entrypoint python
)
"${DEVICE_SMOKE_ARGS[@]}" "${IMAGE_REF}" -c '
import torch
import torch_npu
if not torch.npu.is_available():
    raise SystemExit("torch.npu is not available")
count = torch.npu.device_count()
print(f"container_npu_count={count}")
if count != 8:
    raise SystemExit(f"expected exactly eight container NPUs, got {count}")
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
    --label "deepseek_expert_load.project=deepseek-v4-w8a8-tp8"
    --label "deepseek_expert_load.run_id=${RUN_ID}"
    --label "deepseek_expert_load.host_npu_ids=${VALIDATED_IDS}"
    --net=host
    --shm-size=512g
    --ulimit memlock=-1:-1
)
for logical_id in 0 1 2 3 4 5 6 7; do
    physical_id="${PHYSICAL_IDS[${logical_id}]}"
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
    --data-parallel-size 1
    --tensor-parallel-size 8
    --enable-expert-parallel
    --quantization ascend
    --tokenizer-mode deepseek_v4
    --block-size 128
    --generation-config vllm
    --enable-return-routed-experts
    --enforce-eager
    --no-async-scheduling
    --no-enable-prefix-caching
    --safetensors-load-strategy prefetch
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":128}'
    --additional-config '{"enable_balance_scheduling":false,"enable_fused_mc2":0,"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":0}}'
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
    printf 'quantization_profile=modelslim_w8a8\n'
    printf 'vllm_quantization=ascend\n'
    printf 'parallelism=dp1_tp8_ep\n'
    printf 'host_npu_ids=%s\n' "${VALIDATED_IDS}"
    printf 'container_npu_ids=0,1,2,3,4,5,6,7\n'
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
