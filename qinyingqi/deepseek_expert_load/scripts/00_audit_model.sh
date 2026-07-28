#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

load_config "${1:-}"
require_var MODEL_HOST_PATH
require_cmd python3

OUTPUT_ARGS=()
if [[ -n "${2:-}" ]]; then
    OUTPUT_ARGS=(--output "$2")
fi

python3 "${SCRIPT_DIR}/00_audit_model.py" \
    --model-path "${MODEL_HOST_PATH}" \
    --require-model-type deepseek_v4 \
    --require-w4a8 \
    "${OUTPUT_ARGS[@]}"
