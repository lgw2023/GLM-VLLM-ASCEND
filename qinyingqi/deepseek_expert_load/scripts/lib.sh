#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSEEK_EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

require_var() {
    [[ -n "${!1:-}" ]] || die "required variable is empty: $1"
}

load_config() {
    local path="${1:-}"
    [[ -n "${path}" && -f "${path}" ]] || die "config file not found: ${path:-<empty>}"
    if LC_ALL=C grep -q $'\r' "${path}"; then
        die "config has CRLF line endings; run: sed -i 's/\\r$//' ${path}"
    fi
    set -a
    # shellcheck disable=SC1090
    source "${path}"
    set +a
}

validate_npu_ids() {
    validate_npu_ids_for_count 4
}

validate_npu_ids_for_count() {
    local expected_count="$1"
    python3 - "${HOST_NPU_IDS}" "${expected_count}" <<'PY'
import sys

raw = sys.argv[1]
expected_count = int(sys.argv[2])
try:
    ids = [int(value) for value in raw.split(",")]
except ValueError as exc:
    raise SystemExit(f"HOST_NPU_IDS must be comma-separated integers: {exc}")
if (
    len(ids) != expected_count
    or len(set(ids)) != expected_count
    or any(value < 0 or value > 7 for value in ids)
):
    raise SystemExit(
        f"HOST_NPU_IDS must contain {expected_count} unique IDs from 0..7"
    )
print(",".join(str(value) for value in ids))
PY
}

current_run_id() {
    local path="${RUN_ROOT}/current-run-id"
    [[ -s "${path}" ]] || die "current run ID not found: ${path}"
    tr -d '[:space:]' <"${path}"
}

sanitize_name() {
    printf '%s' "$1" | tr -c 'a-zA-Z0-9_.-' '-'
}

container_name_for_run() {
    local run_id="$1"
    sanitize_name "${CONTAINER_PREFIX:-deepseek-v4-expert}-${run_id}"
}

write_shell_command() {
    local output="$1"
    shift
    {
        printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
        printf '%q ' "$@"
        printf '\n'
    } >"${output}"
    chmod 700 "${output}"
}

resolve_host_binary() {
    local name="$1"
    shift
    local candidate
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
        return
    fi
    for candidate in "$@"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
    die "cannot resolve executable: ${name}"
}
