#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CONFIG_PATH="${1:-}"
shift || true
BENCHMARKS="mmlu_pro,swebench_lite,livecodebench,ruler_niah"
RESUME=0
MAX_REQUESTS_OVERRIDE=""
MAX_TOKENS_OVERRIDE=""
UNIQUE_SCOPE="decode"
ALLOW_DUPLICATE=0
while (($#)); do
    case "$1" in
        --benchmarks)
            BENCHMARKS="${2:-}"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS_OVERRIDE="${2:-}"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS_OVERRIDE="${2:-}"
            shift 2
            ;;
        --unique-scope)
            UNIQUE_SCOPE="${2:-}"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --allow-duplicate-topk)
            ALLOW_DUPLICATE=1
            shift
            ;;
        -h|--help)
            printf 'Usage: bash scripts/04_run_benchmarks.sh CONFIG [--benchmarks a,b] [--max-requests N] [--max-tokens N] [--unique-scope decode|all|none] [--resume] [--allow-duplicate-topk]\n'
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${UNIQUE_SCOPE}" in
    all|decode|none) ;;
    *)
        die "unique-scope must be one of: all, decode, none"
        ;;
esac

load_config "${CONFIG_PATH}"
for name in RUN_ROOT BENCHMARK_DATA_ROOT API_HOST API_PORT SERVED_MODEL_NAME \
    MODEL_HOST_PATH BENCHMARK_MAX_TOKENS BENCHMARK_MAX_REQUESTS; do
    require_var "${name}"
done
require_cmd python3
RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ -f "${RUN_DIR}/service.ready" ]] || die "service-ready gate is missing"
MAX_REQUESTS="${MAX_REQUESTS_OVERRIDE:-${BENCHMARK_MAX_REQUESTS}}"
MAX_TOKENS="${MAX_TOKENS_OVERRIDE:-${BENCHMARK_MAX_TOKENS}}"
[[ "${MAX_REQUESTS}" =~ ^[0-9]+$ ]] || die "max requests must be a non-negative integer"
[[ "${MAX_TOKENS}" =~ ^[0-9]+$ && "${MAX_TOKENS}" -ge 4 ]] || \
    die "max tokens must be an integer >= 4"

IFS=',' read -r -a SELECTED <<<"${BENCHMARKS}"
for benchmark in "${SELECTED[@]}"; do
    case "${benchmark}" in
        mmlu_pro|swebench_lite|livecodebench|ruler_niah) ;;
        *) die "unsupported benchmark: ${benchmark}" ;;
    esac
    INPUT="${BENCHMARK_DATA_ROOT}/inputs/${benchmark}.jsonl"
    [[ -s "${INPUT}" ]] || die "prepared benchmark input not found: ${INPUT}"
    OUTPUT="${RUN_DIR}/benchmarks/${benchmark}"
    ARGS=(
        --input-jsonl "${INPUT}"
        --base-url "http://${API_HOST}:${API_PORT}/v1"
        --model "${SERVED_MODEL_NAME}"
        --model-path "${MODEL_HOST_PATH}"
        --output-dir "${OUTPUT}"
        --max-tokens "${MAX_TOKENS}"
        --max-requests "${MAX_REQUESTS}"
        --unique-scope "${UNIQUE_SCOPE}"
    )
    if ((RESUME == 1)); then
        ARGS+=(--resume)
    fi
    if ((ALLOW_DUPLICATE == 1)); then
        ARGS+=(--allow-duplicate-topk)
    fi
    python3 "${SCRIPT_DIR}/03_capture_routes.py" "${ARGS[@]}"
done

printf 'BENCHMARK_CAPTURE_OK run_id=%s benchmarks=%s unique_scope=%s max_tokens=%s allow_duplicate_topk=%s\n' \
    "${RUN_ID}" "${BENCHMARKS}" "${UNIQUE_SCOPE}" "${MAX_TOKENS}" "${ALLOW_DUPLICATE}"
printf 'Next: bash scripts/05_analyze.sh %s\n' "${CONFIG_PATH}"
if [[ "${UNIQUE_SCOPE}" == "decode" ]]; then
    printf 'NOTE: unique-scope=decode; trust decode-phase load only (prefill may be imperfect).\n'
fi
if ((ALLOW_DUPLICATE == 1)); then
    printf 'NOTE: imperfect routes were accepted; load analysis is diagnostic only until unique top-k is fixed.\n'
fi
