#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
require_cmd awk
require_cmd date
require_cmd df
require_cmd docker
require_cmd free
require_cmd findmnt
require_cmd ip
require_cmd lscpu
require_cmd mv
require_cmd ss
require_cmd sha256sum
require_cmd tee
mv --help 2>&1 | grep -q -- '--no-target-directory' || \
    die "GNU mv with -T/--no-target-directory is required for atomic safety locks"

NPU_SMI_BIN="$(resolve_binary npu-smi /usr/local/bin/npu-smi /usr/local/sbin/npu-smi)"
HCCN_TOOL_BIN="$(resolve_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"

umask 077
PREFLIGHT_ID="preflight-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RUN_HOST_ROOT}/preflight/${PREFLIGHT_ID}/node${NODE_RANK}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/environment.log"
exec > >(tee "${LOG_FILE}") 2>&1

printf 'preflight_id=%s\n' "${PREFLIGHT_ID}"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'hostname=%s\n' "$(hostname)"
printf 'node_rank=%s\n' "${NODE_RANK}"
printf 'local_nic=%s\n' "${LOCAL_NIC}"
printf 'model_host_path=%s\n' "${MODEL_HOST_PATH}"
printf 'run_host_root=%s\n' "${RUN_HOST_ROOT}"
printf 'local_state_root=%s\n' "${LOCAL_STATE_ROOT}"

mkdir -p "${LOCAL_STATE_ROOT}"
chmod 700 "${LOCAL_STATE_ROOT}"
LOCAL_STATE_FSTYPE="$(findmnt -n -o FSTYPE -T "${LOCAL_STATE_ROOT}")"
printf 'local_state_fstype=%s\n' "${LOCAL_STATE_FSTYPE}"
case "${LOCAL_STATE_FSTYPE}" in
    nfs|nfs4|cifs|smb3|fuse.sshfs)
        die "LOCAL_STATE_ROOT must be node-local, not ${LOCAL_STATE_FSTYPE}"
        ;;
esac

ARCH="$(uname -m)"
[[ "${ARCH}" == aarch64 ]] || die "expected aarch64 Atlas host, got ${ARCH}"
printf '\n[uname]\n'
uname -a
printf '\n[os-release]\n'
cat /etc/os-release
printf '\n[lscpu]\n'
lscpu
printf '\n[memory]\n'
free -h

for device_id in {0..7}; do
    [[ -c "/dev/davinci${device_id}" ]] || die "missing character device /dev/davinci${device_id}"
done
for device_path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [[ -e "${device_path}" ]] || die "missing Ascend device: ${device_path}"
done
for mount_path in \
    /usr/local/dcmi \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/version.info \
    /etc/ascend_install.info; do
    [[ -e "${mount_path}" ]] || die "missing required container mount source: ${mount_path}"
done

printf '\n[npu-smi]\n'
"${NPU_SMI_BIN}" info

printf '\n[network]\n'
ip -br addr
LOCAL_ADDRS="$(ip -o -4 addr show dev "${LOCAL_NIC}")"
printf '%s\n' "${LOCAL_ADDRS}"
printf '%s\n' "${LOCAL_ADDRS}" | grep -Fq "${LOCAL_IP}/" || \
    die "LOCAL_IP=${LOCAL_IP} is not assigned to LOCAL_NIC=${LOCAL_NIC}"
ROUTE_TO_PEER="$(ip -4 route get "${PEER_IP}")"
printf 'route_to_peer=%s\n' "${ROUTE_TO_PEER}"
printf '%s\n' "${ROUTE_TO_PEER}" | grep -Fq " dev ${LOCAL_NIC} " || \
    die "route to PEER_IP does not use LOCAL_NIC=${LOCAL_NIC}"

if [[ "${NODE_RANK}" == 0 ]]; then
    [[ "${LOCAL_IP}" == "${NODE0_COORDINATOR_IP}" ]] || \
        die "on node0, LOCAL_IP must equal NODE0_COORDINATOR_IP"
else
    [[ "${PEER_IP}" == "${NODE0_COORDINATOR_IP}" ]] || \
        die "on node1, PEER_IP must equal NODE0_COORDINATOR_IP"
fi

if [[ "${NODE_RANK}" == 0 ]]; then
    if ss -H -ltn "sport = :${API_PORT}" | grep -q .; then
        die "API_PORT=${API_PORT} is already listening on node0"
    fi
    if ss -H -ltn "sport = :${DP_RPC_PORT}" | grep -q .; then
        die "DP_RPC_PORT=${DP_RPC_PORT} is already listening on node0"
    fi
fi

printf '\n[hccn]\n'
for device_id in {0..7}; do
    printf '\n--- device %s link ---\n' "${device_id}"
    LINK_OUTPUT="$("${HCCN_TOOL_BIN}" -i "${device_id}" -link -g)"
    printf '%s\n' "${LINK_OUTPUT}"
    printf '%s\n' "${LINK_OUTPUT}" | grep -Eiq '(^|[^[:alnum:]_])DOWN([^[:alnum:]_]|$)' && \
        die "device ${device_id} HCCN link reports DOWN"
    printf '%s\n' "${LINK_OUTPUT}" | grep -Eiq '(^|[^A-Za-z])UP([^A-Za-z]|$)' || \
        die "device ${device_id} HCCN link is not UP"
    printf '%s\n' "--- device ${device_id} net_health ---"
    HEALTH_OUTPUT="$("${HCCN_TOOL_BIN}" -i "${device_id}" -net_health -g)"
    printf '%s\n' "${HEALTH_OUTPUT}"
    printf '%s\n' "${HEALTH_OUTPUT}" | \
        grep -Eiq 'fail|error|unhealthy|abnormal|not[[:space:]_-]+normal' && \
        die "device ${device_id} HCCN net_health reports failure"
    printf '%s\n' "${HEALTH_OUTPUT}" | \
        grep -Eiq '(^|[^[:alnum:]_])(success|healthy|normal)([^[:alnum:]_]|$)' || \
        die "device ${device_id} HCCN net_health did not report success"
    printf '%s\n' "--- device ${device_id} netdetect configuration ---"
    "${HCCN_TOOL_BIN}" -i "${device_id}" -netdetect -g
    printf '%s\n' "--- device ${device_id} ip ---"
    "${HCCN_TOOL_BIN}" -i "${device_id}" -ip -g
done

printf '\n[storage]\n'
df -hT "$(existing_parent "${MODEL_HOST_PATH}")" "$(existing_parent "${RUN_HOST_ROOT}")"
MODEL_FREE_GIB="$(available_gib "${MODEL_HOST_PATH}")"
printf 'model_storage_free_gib=%s\n' "${MODEL_FREE_GIB}"
if [[ ! -f "${MODEL_HOST_PATH}/config.json" ]] && \
   (( MODEL_FREE_GIB < MIN_MODEL_STORAGE_FREE_GIB )); then
    die "model storage has ${MODEL_FREE_GIB} GiB free; need at least ${MIN_MODEL_STORAGE_FREE_GIB} GiB before download"
fi

printf '\n[docker]\n'
docker version
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
printf 'docker_root=%s\n' "${DOCKER_ROOT}"
df -hT "${DOCKER_ROOT}"
DOCKER_FREE_GIB="$(available_gib "${DOCKER_ROOT}")"
printf 'docker_free_gib=%s\n' "${DOCKER_FREE_GIB}"
if (( DOCKER_FREE_GIB < MIN_DOCKER_FREE_GIB )); then
    die "DockerRootDir has ${DOCKER_FREE_GIB} GiB free; minimum is ${MIN_DOCKER_FREE_GIB} GiB"
fi
docker ps --no-trunc
if docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
    image_metadata
else
    warn "IMAGE_REF is not present locally yet: ${IMAGE_REF}"
fi

printf '\n[ascend-install]\n'
for info_file in \
    /etc/ascend_install.info \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info; do
    if [[ -f "${info_file}" ]]; then
        printf '%s\n' "--- ${info_file} ---"
        cat "${info_file}"
    fi
done

printf '\n[source manifest]\n'
verify_source_manifest

touch "${OUT_DIR}/SUCCESS.node${NODE_RANK}"
GATE_FILE="$(write_config_gate preflight)"
printf '\nPREFLIGHT_OK output=%s gate=%s\n' "${OUT_DIR}" "${GATE_FILE}"
