#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
FULL_SHA256=0
if [[ "${3:-}" == --full-sha256 ]]; then
    FULL_SHA256=1
elif [[ -n "${3:-}" ]]; then
    die "unknown option: ${3}"
fi

[[ -f "${MODEL_HOST_PATH}/config.json" ]] || die "model config not found: ${MODEL_HOST_PATH}/config.json"
require_cmd date
require_cmd du
require_cmd find
require_cmd sha256sum
require_cmd sort
require_cmd tee
require_cmd python3

STATE_FILE="$(model_download_state_file)"
[[ -f "${STATE_FILE}" ]] || die "revision-bound download state is missing: ${STATE_FILE}"
[[ "$(gate_value model_id "${STATE_FILE}")" == "${MODEL_ID}" ]] || die "download state model ID mismatch"
[[ "$(gate_value model_revision "${STATE_FILE}")" == "${MODEL_REVISION:-default}" ]] || \
    die "download state revision mismatch"
[[ "$(gate_value model_path "${STATE_FILE}")" == "${MODEL_HOST_PATH}" ]] || die "download state path mismatch"

umask 077
MANIFEST_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RUN_HOST_ROOT}/model-manifests/${MANIFEST_ID}"
mkdir -p "${OUT_DIR}"
exec > >(tee "${OUT_DIR}/manifest.log") 2>&1

FILE_COUNT="$(find "${MODEL_HOST_PATH}" -type f | wc -l)"
SHARD_COUNT="$(find "${MODEL_HOST_PATH}" -maxdepth 1 -type f -name '*.safetensors' | wc -l)"
TOTAL_BYTES="$(du -sb "${MODEL_HOST_PATH}" | awk '{print $1}')"
{
    printf 'generated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'model_id=%s\n' "${MODEL_ID}"
    printf 'model_revision=%s\n' "${MODEL_REVISION:-default}"
    printf 'model_path=%s\n' "${MODEL_HOST_PATH}"
    printf 'file_count=%s\n' "${FILE_COUNT}"
    printf 'safetensors_shard_count=%s\n' "${SHARD_COUNT}"
    printf 'total_bytes=%s\n' "${TOTAL_BYTES}"
} | tee "${OUT_DIR}/summary.txt"

find "${MODEL_HOST_PATH}" -type f -printf '%s\t%P\n' | \
    sort -k2,2 >"${OUT_DIR}/files.tsv"

find "${MODEL_HOST_PATH}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.py' -o -name '*.md' -o -name '*.txt' \) \
    -print0 | sort -z | xargs -0 -r sha256sum >"${OUT_DIR}/metadata.sha256"

python3 "${SCRIPT_DIR}/validate_model_files.py" \
    --model-path "${MODEL_HOST_PATH}" \
    --output "${OUT_DIR}/model-validation.json"

if (( FULL_SHA256 == 1 )); then
    warn "full SHA-256 will sequentially read every model file"
    while IFS= read -r -d '' model_file; do
        sha256sum "${model_file}"
    done < <(find "${MODEL_HOST_PATH}" -type f -print0 | sort -z) \
        >"${OUT_DIR}/all-files.sha256"
fi

READY_FILE="$(model_ready_file)"
mkdir -p "$(dirname "${READY_FILE}")"
{
    printf 'model_id=%s\n' "${MODEL_ID}"
    printf 'model_revision=%s\n' "${MODEL_REVISION:-default}"
    printf 'model_path=%s\n' "${MODEL_HOST_PATH}"
    printf 'download_state_sha256=%s\n' "$(sha256sum "${STATE_FILE}" | awk '{print $1}')"
    printf 'validation_sha256=%s\n' "$(sha256sum "${OUT_DIR}/model-validation.json" | awk '{print $1}')"
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
} >"${READY_FILE}"

printf 'MANIFEST_OK output=%s ready=%s full_sha256=%s\n' \
    "${OUT_DIR}" "${READY_FILE}" "${FULL_SHA256}"
