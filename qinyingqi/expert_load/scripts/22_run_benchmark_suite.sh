#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

CLUSTER_CONFIG_ARG="${1:-}"
NODE_CONFIG_ARG="${2:-}"
shift 2 || true
load_configs "${CLUSTER_CONFIG_ARG}" "${NODE_CONFIG_ARG}"
require_run_id
require_cmd curl

[[ "${NODE_RANK}" == 0 ]] || die "run the benchmark suite only on node0"
[[ "${RUN_PROFILE}" == expert_capture ]] || \
    die "benchmark capture requires RUN_PROFILE=expert_capture, not ${RUN_PROFILE}"
[[ "${ENABLE_ROUTE_CAPTURE}" == 1 ]] || \
    die "benchmark capture requires ENABLE_ROUTE_CAPTURE=1"
[[ "${CAPTURE_PATCH_ID}" != none ]] || \
    die "benchmark capture requires a derived route-capture image"

DATA_ROOT="${RUN_HOST_ROOT}/benchmark-data"
BENCHMARKS="mmlu_pro,swebench_lite,livecodebench,ruler_niah"
MAX_REQUESTS=0
MAX_TOKENS=16
MIN_OUTPUT_TOKENS=4
TIMEOUT_SECONDS=1800
RESUME=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/22_run_benchmark_suite.sh CONFIGS/cluster.env CONFIGS/node.env [options]

Options:
  --data-root PATH             Prepared benchmark data root (default: RUN_HOST_ROOT/benchmark-data)
  --benchmarks CSV             mmlu_pro,swebench_lite,livecodebench,ruler_niah
  --max-requests N             Cap requests per workload; 0 means all prepared records
  --max-tokens N               Deterministic generation length (default: 16)
  --min-output-tokens N        Require this many generated tokens (default: 4)
  --timeout-seconds N          Per request timeout (default: 1800)
  --resume                     Continue existing capture directories after digest checks
  -h, --help                   Show this message
EOF
}

while (($#)); do
    case "$1" in
        --data-root)
            DATA_ROOT="${2:-}"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="${2:-}"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="${2:-}"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="${2:-}"
            shift 2
            ;;
        --min-output-tokens)
            MIN_OUTPUT_TOKENS="${2:-}"
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECONDS="${2:-}"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

for value in "${MAX_REQUESTS}" "${MAX_TOKENS}" "${MIN_OUTPUT_TOKENS}" "${TIMEOUT_SECONDS}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "numeric options must be unsigned integers"
done
(( MAX_TOKENS >= 1 )) || die "--max-tokens must be at least 1"
(( MIN_OUTPUT_TOKENS >= 1 && MIN_OUTPUT_TOKENS <= MAX_TOKENS )) || \
    die "--min-output-tokens must be in 1..--max-tokens"
(( TIMEOUT_SECONDS >= 1 )) || die "--timeout-seconds must be positive"
[[ "${DATA_ROOT}" == /data/* ]] || die "--data-root must live below /data: ${DATA_ROOT}"

CLIENT_PYTHON="${CLIENT_PYTHON:-${EXPERT_LOAD_ROOT}/.client-venv/bin/python}"
[[ -x "${CLIENT_PYTHON}" ]] || \
    die "client Python is not executable: ${CLIENT_PYTHON}; create .client-venv first"

RUN_DIR="$(host_run_dir)"
[[ -f "${RUN_DIR}/SERVICE_READY" ]] || \
    die "current run has no SERVICE_READY marker: ${RUN_DIR}/SERVICE_READY"
API_BASE_URL="http://${API_BIND_HOST}:${API_PORT}/v1"
curl --noproxy '*' -fsS --max-time 10 \
    "http://${API_BIND_HOST}:${API_PORT}/health" >/dev/null || \
    die "local vLLM health endpoint is unavailable"

OUTPUT_ROOT="${RUN_DIR}/benchmarks"
ROUTE_GATE_DIR="${OUTPUT_ROOT}/route-gate"
mkdir -p "${OUTPUT_ROOT}/captures"

"${CLIENT_PYTHON}" "${SCRIPT_DIR}/12_smoke_request.py" \
    --base-url "${API_BASE_URL}" \
    --model "${SERVED_MODEL_NAME}" \
    --max-tokens 8 \
    --ignore-eos \
    --require-routes \
    --output-dir "${ROUTE_GATE_DIR}"

IFS=',' read -r -a BENCHMARK_LIST <<<"${BENCHMARKS}"
(( ${#BENCHMARK_LIST[@]} > 0 )) || die "--benchmarks must not be empty"
declare -A SEEN_BENCHMARKS=()
ANALYZE_ARGS=()
for benchmark in "${BENCHMARK_LIST[@]}"; do
    benchmark="${benchmark//[[:space:]]/}"
    case "${benchmark}" in
        mmlu_pro|swebench_lite|livecodebench|ruler_niah)
            ;;
        *)
            die "unsupported benchmark: ${benchmark}"
            ;;
    esac
    [[ -n "${benchmark}" ]] || die "empty benchmark name"
    [[ -z "${SEEN_BENCHMARKS[${benchmark}]:-}" ]] || \
        die "duplicate benchmark: ${benchmark}"
    SEEN_BENCHMARKS["${benchmark}"]=1

    INPUT_JSONL="${DATA_ROOT}/inputs/${benchmark}.jsonl"
    [[ -s "${INPUT_JSONL}" ]] || \
        die "missing prepared input: ${INPUT_JSONL}; run 20_prepare_benchmarks.sh on node0"
    CAPTURE_DIR="${OUTPUT_ROOT}/captures/${benchmark}"
    CAPTURE_ARGS=(
        "${CLIENT_PYTHON}" "${SCRIPT_DIR}/21_capture_expert_routes.py"
        --input-jsonl "${INPUT_JSONL}"
        --base-url "${API_BASE_URL}"
        --model "${SERVED_MODEL_NAME}"
        --output-dir "${CAPTURE_DIR}"
        --max-tokens "${MAX_TOKENS}"
        --min-output-tokens "${MIN_OUTPUT_TOKENS}"
        --timeout-seconds "${TIMEOUT_SECONDS}"
        --max-requests "${MAX_REQUESTS}"
    )
    if (( RESUME == 1 )); then
        CAPTURE_ARGS+=(--resume)
    fi
    "${CAPTURE_ARGS[@]}"
    ANALYZE_ARGS+=(--capture-dir "${benchmark}=${CAPTURE_DIR}")
done

"${CLIENT_PYTHON}" "${SCRIPT_DIR}/23_analyze_expert_load.py" \
    "${ANALYZE_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/analysis" \
    --overwrite

printf 'BENCHMARK_SUITE_OK output=%s\n' "${OUTPUT_ROOT}"
