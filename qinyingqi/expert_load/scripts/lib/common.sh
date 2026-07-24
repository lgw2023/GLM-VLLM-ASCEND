#!/usr/bin/env bash

set -Eeuo pipefail

EXPERT_LOAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERT_LOAD_ROOT}/../.." && pwd)"
SOURCE_MANIFEST_PATH="${EXPERT_LOAD_ROOT}/SOURCE_MANIFEST.json"
SOURCE_MANIFEST_TOOL="${EXPERT_LOAD_ROOT}/scripts/source_manifest.py"
VLLM_SOURCE_LOCK=0decac0d96c42b49572498019f0a0e3600f50398
VLLM_ASCEND_SOURCE_LOCK=5f6faa0cb8830f667266f3b8121cd1383606f2a1

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

resolve_binary() {
    local name="$1"
    shift
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
        return
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    die "required executable not found: ${name}"
}

require_var() {
    local name="$1"
    local value="${!name:-}"
    [[ -n "${value}" ]] || die "required variable is empty: ${name}"
    case "${value}" in
        *REPLACE*|*CHANGEME*|xxx|xxxx)
            die "replace placeholder value for ${name}"
            ;;
    esac
}

require_uint() {
    local name="$1"
    local value="${!name:-}"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer: ${value}"
}

load_configs() {
    local cluster_config="${1:-}"
    local node_config="${2:-}"
    [[ -f "${cluster_config}" ]] || die "cluster config not found: ${cluster_config}"
    [[ -f "${node_config}" ]] || die "node config not found: ${node_config}"
    CLUSTER_CONFIG_PATH="$(cd "$(dirname "${cluster_config}")" && pwd)/$(basename "${cluster_config}")"
    NODE_CONFIG_PATH="$(cd "$(dirname "${node_config}")" && pwd)/$(basename "${node_config}")"

    # Configs are operator-owned shell assignment files and must not contain secrets.
    # shellcheck disable=SC1090
    source "${cluster_config}"
    # shellcheck disable=SC1090
    source "${node_config}"
    validate_config
}

validate_config() {
    local required=(
        CLUSTER_NAME NODE0_COORDINATOR_IP SOURCE_MANIFEST_SHA256
        MODEL_ID MODEL_REVISION MODEL_CONTAINER_PATH
        RUN_PROFILE IMAGE_REF ENABLE_ROUTE_CAPTURE SERVED_MODEL_NAME API_PORT DP_RPC_PORT
        EXPECTED_VLLM_PACKAGE_VERSION EXPECTED_VLLM_ASCEND_PACKAGE_VERSION CAPTURE_PATCH_ID
        HCCL_TEST_PORT HCCL_TEST_TIMEOUT_SECONDS
        NUM_NODES NPUS_PER_NODE TP_SIZE DP_SIZE DP_SIZE_LOCAL
        MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
        GPU_MEMORY_UTILIZATION BLOCK_SIZE SEED HEALTH_TIMEOUT_SECONDS
        STOP_TIMEOUT_SECONDS MIN_DOCKER_FREE_GIB
        MIN_MODEL_STORAGE_FREE_GIB NODE_RANK LOCAL_IP PEER_IP LOCAL_NIC
        AUTHORIZED_NPU_IDS MODEL_HOST_PATH RUN_HOST_ROOT LOCAL_STATE_ROOT
    )
    local name
    for name in "${required[@]}"; do
        require_var "${name}"
    done

    local integer_vars=(
        API_PORT DP_RPC_PORT HCCL_TEST_PORT HCCL_TEST_TIMEOUT_SECONDS
        NUM_NODES NPUS_PER_NODE TP_SIZE DP_SIZE
        DP_SIZE_LOCAL MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
        BLOCK_SIZE SEED HEALTH_TIMEOUT_SECONDS STOP_TIMEOUT_SECONDS
        MIN_DOCKER_FREE_GIB MIN_MODEL_STORAGE_FREE_GIB NODE_RANK
    )
    for name in "${integer_vars[@]}"; do
        require_uint "${name}"
    done

    [[ "${NUM_NODES}" == 2 ]] || die "this baseline requires NUM_NODES=2"
    [[ "${NPUS_PER_NODE}" == 8 ]] || die "this baseline requires NPUS_PER_NODE=8"
    [[ "${TP_SIZE}" == 8 ]] || die "this baseline requires TP_SIZE=8"
    [[ "${DP_SIZE}" == 2 ]] || die "this baseline requires DP_SIZE=2"
    [[ "${DP_SIZE_LOCAL}" == 1 ]] || die "this baseline requires DP_SIZE_LOCAL=1"
    [[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || die "NODE_RANK must be 0 or 1"
    [[ "${AUTHORIZED_NPU_IDS}" == "0,1,2,3,4,5,6,7" ]] || \
        die "GLM-5.2 W8A8 needs exactly AUTHORIZED_NPU_IDS=0,1,2,3,4,5,6,7"
    [[ "${RUN_PROFILE}" == vendor_smoke || "${RUN_PROFILE}" == expert_capture ]] || \
        die "RUN_PROFILE must be vendor_smoke or expert_capture"
    [[ "${ENABLE_ROUTE_CAPTURE:-0}" == 0 || "${ENABLE_ROUTE_CAPTURE:-0}" == 1 ]] || \
        die "ENABLE_ROUTE_CAPTURE must be 0 or 1"
    ensure_data_path "${MODEL_HOST_PATH}" MODEL_HOST_PATH
    ensure_data_path "${RUN_HOST_ROOT}" RUN_HOST_ROOT
    ensure_data_path "${LOCAL_STATE_ROOT}" LOCAL_STATE_ROOT

    (( API_PORT > 0 && API_PORT <= 65535 )) || die "API_PORT is outside 1..65535"
    (( DP_RPC_PORT > 0 && DP_RPC_PORT <= 65535 )) || die "DP_RPC_PORT is outside 1..65535"
    (( HCCL_TEST_PORT > 0 && HCCL_TEST_PORT <= 65535 )) || die "HCCL_TEST_PORT is outside 1..65535"
    [[ "${API_PORT}" != "${DP_RPC_PORT}" && \
       "${API_PORT}" != "${HCCL_TEST_PORT}" && \
       "${DP_RPC_PORT}" != "${HCCL_TEST_PORT}" ]] || \
        die "API_PORT, DP_RPC_PORT and HCCL_TEST_PORT must differ"
    [[ "${GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]] || \
        die "GPU_MEMORY_UTILIZATION must be in (0, 1]"
    [[ "${MODEL_CONTAINER_PATH}" == /* && "${MODEL_CONTAINER_PATH}" != / ]] || \
        die "MODEL_CONTAINER_PATH must be an absolute, non-root path"
    [[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
        die "MODEL_REVISION must be a pinned lowercase 40-character commit SHA"
    [[ "${SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
        die "SOURCE_MANIFEST_SHA256 must be a lowercase SHA-256"
    if [[ "${RUN_PROFILE}" == vendor_smoke ]]; then
        [[ "${CAPTURE_PATCH_ID}" == none ]] || \
            die "vendor_smoke requires CAPTURE_PATCH_ID=none"
    else
        [[ "${CAPTURE_PATCH_ID}" != none ]] || \
            die "expert_capture requires a non-none CAPTURE_PATCH_ID"
    fi
}

ensure_data_path() {
    local path="$1"
    local label="$2"
    [[ "${path}" == /data/* ]] || die "${label} must live below /data: ${path}"
    [[ "${path}" != /data && "${path}" != /data/ ]] || die "${label} is too broad"
}

existing_parent() {
    local path="$1"
    while [[ ! -d "${path}" && "${path}" != / ]]; do
        path="$(dirname "${path}")"
    done
    [[ -d "${path}" ]] || die "no existing parent for path: $1"
    printf '%s\n' "${path}"
}

available_gib() {
    local path
    path="$(existing_parent "$1")"
    df -Pk "${path}" | awk 'NR == 2 { print int($4 / 1024 / 1024) }'
}

require_run_id() {
    require_var RUN_ID
    [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || \
        die "RUN_ID must be 1-80 characters from [A-Za-z0-9_.-]"
}

container_name() {
    require_run_id
    printf 'glm52-%s-node%s-%s\n' "${RUN_PROFILE}" "${NODE_RANK}" "${RUN_ID}"
}

host_run_dir() {
    require_run_id
    printf '%s/%s/node%s\n' "${RUN_HOST_ROOT}" "${RUN_ID}" "${NODE_RANK}"
}

write_shell_command() {
    local destination="$1"
    shift
    {
        printf '# generated_at=%s\n' "$(date --iso-8601=seconds)"
        printf '%q ' "$@"
        printf '\n'
    } >"${destination}"
}

image_metadata() {
    docker image inspect "${IMAGE_REF}" \
        --format 'id={{.Id}} repo_digests={{json .RepoDigests}} created={{.Created}}'
}

current_image_id() {
    docker image inspect "${IMAGE_REF}" --format '{{.Id}}'
}

cluster_config_sha256() {
    require_cmd sha256sum
    sha256sum "${CLUSTER_CONFIG_PATH}" | awk '{print $1}'
}

verify_source_manifest() {
    require_cmd python3
    [[ -f "${SOURCE_MANIFEST_TOOL}" ]] || \
        die "source manifest verifier is missing: ${SOURCE_MANIFEST_TOOL}"
    [[ -f "${SOURCE_MANIFEST_PATH}" ]] || \
        die "source manifest is missing: ${SOURCE_MANIFEST_PATH}"
    python3 "${SOURCE_MANIFEST_TOOL}" verify \
        --package-root "${EXPERT_LOAD_ROOT}" \
        --manifest "${SOURCE_MANIFEST_PATH}" \
        --expected-sha256 "${SOURCE_MANIFEST_SHA256}" "$@"
}

source_id() {
    verify_source_manifest --quiet
    printf 'sha256:%s\n' "${SOURCE_MANIFEST_SHA256}"
}

source_identity_metadata() {
    printf 'source_id=%s\n' "$(source_id)"
    printf 'source_manifest_sha256=%s\n' "${SOURCE_MANIFEST_SHA256}"
    printf 'vllm_source_lock=%s\n' "${VLLM_SOURCE_LOCK}"
    printf 'vllm_ascend_source_lock=%s\n' "${VLLM_ASCEND_SOURCE_LOCK}"
}

image_capture_patch_id() {
    local value
    value="$(docker image inspect "${IMAGE_REF}" \
        --format '{{if .Config.Labels}}{{index .Config.Labels "glm52.capture_patch_id"}}{{end}}')"
    if [[ "${value}" == '<no value>' ]]; then
        value=""
    fi
    printf '%s\n' "${value}"
}

probe_image_packages() {
    docker run --rm --entrypoint python "${IMAGE_REF}" -c \
        'from importlib.metadata import PackageNotFoundError, version
for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers"):
    try:
        value = version(name)
    except PackageNotFoundError:
        value = "NOT_INSTALLED"
    print(f"{name}={value}")'
}

package_version_from_file() {
    local distribution="$1"
    local package_file="$2"
    sed -n "s/^${distribution}=//p" "${package_file}" | tail -n 1
}

require_cluster_image_gate() {
    local current_id current_source_id
    current_id="$(current_image_id)"
    current_source_id="$(source_id)"
    local rank gate_file gate_ref gate_id
    for rank in 0 1; do
        gate_file="${RUN_HOST_ROOT}/gates/node${rank}/image.gate"
        [[ -f "${gate_file}" ]] || die "missing image gate from node${rank}: ${gate_file}"
        gate_ref="$(gate_value image_ref "${gate_file}")"
        gate_id="$(gate_value image_id "${gate_file}")"
        [[ "${gate_ref}" == "${IMAGE_REF}" ]] || die "node${rank} image gate uses a different IMAGE_REF"
        [[ "${gate_id}" == "${current_id}" ]] || \
            die "node${rank} image ID differs from local image ID; repull the same immutable image"
        [[ "$(gate_value cluster_config_sha256 "${gate_file}")" == "$(cluster_config_sha256)" ]] || \
            die "node${rank} image gate used a different cluster.env"
        [[ "$(gate_value source_id "${gate_file}")" == "${current_source_id}" ]] || \
            die "node${rank} image gate used a different source manifest"
    done
}

require_cluster_hccl_gate() {
    local current_id current_source_id
    current_id="$(current_image_id)"
    current_source_id="$(source_id)"
    local rank gate_file
    for rank in 0 1; do
        gate_file="${RUN_HOST_ROOT}/${RUN_ID}/node${rank}/hccl-collective/HCCL_COLLECTIVE_OK"
        [[ -f "${gate_file}" ]] || \
            die "missing successful 16-rank HCCL gate from node${rank}: ${gate_file}"
        [[ "$(gate_value run_id "${gate_file}")" == "${RUN_ID}" ]] || \
            die "node${rank} HCCL gate belongs to another run"
        [[ "$(gate_value node_rank "${gate_file}")" == "${rank}" ]] || \
            die "node${rank} HCCL gate has the wrong rank"
        [[ "$(gate_value image_id "${gate_file}")" == "${current_id}" ]] || \
            die "node${rank} HCCL gate used a different image ID"
        [[ "$(gate_value cluster_config_sha256 "${gate_file}")" == "$(cluster_config_sha256)" ]] || \
            die "node${rank} HCCL gate used a different cluster.env"
        [[ "$(gate_value source_id "${gate_file}")" == "${current_source_id}" ]] || \
            die "node${rank} HCCL gate used a different source manifest"
        if [[ "${rank}" == "${NODE_RANK}" ]]; then
            [[ "$(gate_value config_fingerprint "${gate_file}")" == "$(config_fingerprint)" ]] || \
                die "local HCCL gate was created for different config contents"
        fi
    done
}

config_fingerprint() {
    require_cmd sha256sum
    {
        cluster_config_sha256
        sha256sum "${NODE_CONFIG_PATH}" | awk '{print $1}'
    } | sha256sum | awk '{print $1}'
}

gate_dir() {
    printf '%s/gates/node%s\n' "${RUN_HOST_ROOT}" "${NODE_RANK}"
}

write_config_gate() {
    local gate_name="$1"
    local destination
    destination="$(gate_dir)/${gate_name}.gate"
    mkdir -p "$(dirname "${destination}")"
    {
        printf 'gate=%s\n' "${gate_name}"
        printf 'node_rank=%s\n' "${NODE_RANK}"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'config_fingerprint=%s\n' "$(config_fingerprint)"
        printf 'cluster_config_sha256=%s\n' "$(cluster_config_sha256)"
        printf 'source_id=%s\n' "$(source_id)"
    } >"${destination}"
    printf '%s\n' "${destination}"
}

gate_value() {
    local key="$1"
    local file="$2"
    sed -n "s/^${key}=//p" "${file}" | tail -n 1
}

require_config_gate() {
    local gate_name="$1"
    local gate_file
    gate_file="$(gate_dir)/${gate_name}.gate"
    [[ -f "${gate_file}" ]] || die "missing ${gate_name} gate for node${NODE_RANK}: ${gate_file}"
    local recorded current
    recorded="$(gate_value config_fingerprint "${gate_file}")"
    current="$(config_fingerprint)"
    [[ -n "${recorded}" && "${recorded}" == "${current}" ]] || \
        die "${gate_name} gate was created for different config contents; rerun it"
    [[ "$(gate_value source_id "${gate_file}")" == "$(source_id)" ]] || \
        die "${gate_name} gate was created for a different source manifest; rerun it"
}

model_download_state_file() {
    printf '%s/.download-state/GLM-5.2-w8a8.complete\n' "$(dirname "${MODEL_HOST_PATH}")"
}

model_ready_file() {
    printf '%s/.model-ready/GLM-5.2-w8a8.ready\n' "$(dirname "${MODEL_HOST_PATH}")"
}

require_model_ready() {
    local ready_file
    ready_file="$(model_ready_file)"
    [[ -f "${ready_file}" ]] || die "model ready marker is missing: ${ready_file}"
    [[ "$(gate_value model_id "${ready_file}")" == "${MODEL_ID}" ]] || \
        die "model ready marker has a different model ID"
    [[ "$(gate_value model_revision "${ready_file}")" == "${MODEL_REVISION}" ]] || \
        die "model ready marker has a different revision"
    [[ "$(gate_value model_path "${ready_file}")" == "${MODEL_HOST_PATH}" ]] || \
        die "model ready marker has a different path"
}

local_run_dir() {
    require_run_id
    printf '%s/runs/%s\n' "${LOCAL_STATE_ROOT}" "${RUN_ID}"
}

npu_ready_file() {
    printf '%s/npu-ready/NPU_READY\n' "$(host_run_dir)"
}

require_npu_ready() {
    local ready_file
    ready_file="$(npu_ready_file)"
    [[ -f "${ready_file}" ]] || die "run 05_prepare_npus.sh first: ${ready_file}"
    [[ "$(gate_value run_id "${ready_file}")" == "${RUN_ID}" ]] || \
        die "NPU readiness record belongs to another run"
    [[ "$(gate_value node_rank "${ready_file}")" == "${NODE_RANK}" ]] || \
        die "NPU readiness record belongs to another node"
    [[ "$(gate_value config_fingerprint "${ready_file}")" == \
       "$(config_fingerprint)" ]] || \
        die "NPU readiness record has stale configuration; rerun 05_prepare_npus.sh"
    [[ "$(gate_value source_id "${ready_file}")" == "$(source_id)" ]] || \
        die "NPU readiness record has stale source; rerun 05_prepare_npus.sh"
}

running_npu_container_ids() {
    local container_ids container_id devices privileged rules requests
    container_ids="$(docker ps -q)" || die "docker ps failed during NPU consumer scan"
    while IFS= read -r container_id; do
        [[ -n "${container_id}" ]] || continue
        devices="$(docker inspect "${container_id}" \
            --format '{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}')" || \
            die "docker inspect failed during NPU consumer scan: ${container_id}"
        privileged="$(docker inspect "${container_id}" \
            --format '{{.HostConfig.Privileged}}')" || \
            die "could not inspect privileged state: ${container_id}"
        rules="$(docker inspect "${container_id}" \
            --format '{{json .HostConfig.DeviceCgroupRules}}')" || \
            die "could not inspect device cgroup rules: ${container_id}"
        requests="$(docker inspect "${container_id}" \
            --format '{{json .HostConfig.DeviceRequests}}')" || \
            die "could not inspect device requests: ${container_id}"
        if printf '%s\n' "${devices}" | \
            grep -Eq '^/dev/(davinci[0-7]|davinci_manager)$' || \
           [[ "${privileged}" == true ]] || \
           [[ "${rules}" != null && "${rules}" != '[]' && \
              "${rules}" != '<no value>' ]] || \
           [[ "${requests}" != null && "${requests}" != '[]' && \
              "${requests}" != '<no value>' ]]; then
            printf '%s\n' "${container_id}"
        fi
    done <<<"${container_ids}"
}

require_no_npu_containers() {
    local context="${1:-operation}" running_consumers
    running_consumers="$(running_npu_container_ids)" || \
        die "NPU consumer scan failed before ${context}"
    [[ -z "${running_consumers}" ]] || \
        die "refusing ${context} while potential NPU containers are running: ${running_consumers}"
}

container_exists() {
    docker container inspect "$1" >/dev/null 2>&1
}

container_is_running() {
    [[ "$(docker container inspect "$1" --format '{{.State.Running}}' 2>/dev/null)" == true ]]
}
