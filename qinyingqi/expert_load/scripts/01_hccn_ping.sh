#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
REMOTE_IP_FILE="${3:-}"
[[ -f "${REMOTE_IP_FILE}" ]] || die "remote NPU IP file not found: ${REMOTE_IP_FILE}"
HCCN_TOOL_BIN="$(resolve_binary hccn_tool /usr/local/Ascend/driver/tools/hccn_tool)"
require_cmd date
require_cmd tee

valid_ipv4() {
    local ip="$1"
    [[ "${ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
    local octet
    local -a octets
    IFS=. read -r -a octets <<<"${ip}"
    for octet in "${octets[@]}"; do
        (( 10#${octet} <= 255 )) || return 1
    done
}

REMOTE_IPS=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "${line}" ]] || continue
    valid_ipv4 "${line}" || die "invalid remote NPU IPv4 address: ${line}"
    REMOTE_IPS+=("${line}")
done <"${REMOTE_IP_FILE}"

[[ "${#REMOTE_IPS[@]}" == 8 ]] || \
    die "expected exactly 8 remote NPU IPs in device order, got ${#REMOTE_IPS[@]}"

umask 077
CHECK_ID="hccn-ping-$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RUN_HOST_ROOT}/connectivity/${CHECK_ID}/node${NODE_RANK}"
mkdir -p "${OUT_DIR}"
exec > >(tee "${OUT_DIR}/hccn-ping.log") 2>&1

for device_id in {0..7}; do
    printf '\n[local device %s -> remote NPU %s]\n' \
        "${device_id}" "${REMOTE_IPS[${device_id}]}"
    "${HCCN_TOOL_BIN}" -i "${device_id}" -ping -g \
        address "${REMOTE_IPS[${device_id}]}"
done

touch "${OUT_DIR}/SUCCESS.node${NODE_RANK}"
GATE_FILE="$(write_config_gate hccn_ping)"
printf '\nHCCN_PING_OK output=%s gate=%s\n' "${OUT_DIR}" "${GATE_FILE}"
