#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

load_config "${1:-}"
for name in RUN_ROOT BENCHMARK_DATA_ROOT API_HOST API_PORT SERVED_MODEL_NAME \
    MODEL_HOST_PATH BENCHMARK_MAX_TOKENS; do
    require_var "${name}"
done
require_cmd python3
RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ -f "${RUN_DIR}/service.ready" ]] || die "service-ready gate is missing"
INPUT="${BENCHMARK_DATA_ROOT}/inputs/livecodebench.jsonl"
[[ -s "${INPUT}" ]] || die "prepared LiveCodeBench input not found: ${INPUT}"

python3 "${SCRIPT_DIR}/04_run_livecodebench_one.py" \
    --input-jsonl "${INPUT}" \
    --index 0 \
    --base-url "http://${API_HOST}:${API_PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --model-path "${MODEL_HOST_PATH}" \
    --output-dir "${RUN_DIR}/livecodebench-one" \
    --max-tokens "${BENCHMARK_MAX_TOKENS}"

