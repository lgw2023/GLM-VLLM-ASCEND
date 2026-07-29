#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
shift || true
ALLOW_DUPLICATE=0
while (($#)); do
    case "$1" in
        --allow-duplicate-topk)
            ALLOW_DUPLICATE=1
            shift
            ;;
        -h|--help)
            printf 'Usage: bash scripts/03_smoke_capture.sh CONFIG [--allow-duplicate-topk]\n'
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

load_config "${CONFIG_PATH}"
for name in RUN_ROOT API_HOST API_PORT SERVED_MODEL_NAME MODEL_HOST_PATH; do
    require_var "${name}"
done
require_cmd python3
RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ -f "${RUN_DIR}/service.ready" ]] || die "service-ready gate is missing"

# Fresh smoke directory each attempt; resume is for long benchmarks only.
rm -rf "${RUN_DIR}/smoke"

ARGS=(
    --prompt 'Explain why mixture-of-experts routing may depend on the task domain.'
    --benchmark smoke
    --base-url "http://${API_HOST}:${API_PORT}/v1"
    --model "${SERVED_MODEL_NAME}"
    --model-path "${MODEL_HOST_PATH}"
    --output-dir "${RUN_DIR}/smoke"
    --max-tokens 16
)
if ((ALLOW_DUPLICATE == 1)); then
    ARGS+=(--allow-duplicate-topk)
fi

python3 "${SCRIPT_DIR}/03_capture_routes.py" "${ARGS[@]}"
