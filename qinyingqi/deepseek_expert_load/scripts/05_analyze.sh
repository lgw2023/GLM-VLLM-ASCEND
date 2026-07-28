#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

load_config "${1:-}"
require_var RUN_ROOT
require_cmd find
require_cmd python3
RUN_ID="$(current_run_id)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
BENCHMARK_DIR="${RUN_DIR}/benchmarks"
[[ -d "${BENCHMARK_DIR}" ]] || die "benchmark output directory not found: ${BENCHMARK_DIR}"

CAPTURE_DIRS=()
while IFS= read -r aggregate; do
    CAPTURE_DIRS+=("$(dirname "${aggregate}")")
done < <(find "${BENCHMARK_DIR}" -mindepth 2 -maxdepth 2 \
    -name aggregate-counts.npz -type f -print | sort)
((${#CAPTURE_DIRS[@]} > 0)) || die "no completed benchmark aggregates found"

python3 "${SCRIPT_DIR}/05_analyze_expert_load.py" \
    "${CAPTURE_DIRS[@]}" \
    --output-dir "${RUN_DIR}/analysis"
printf 'Read the main report: %s\n' "${RUN_DIR}/analysis/report.md"
