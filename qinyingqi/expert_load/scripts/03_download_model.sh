#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
[[ "${3:-}" == --confirm-large-download ]] || \
    die "GLM-5.2-w8a8 is approximately 774 GB; rerun with --confirm-large-download"
[[ "${NODE_RANK}" == 0 ]] || die "download the shared model once from node0"
require_var MODELSCOPE_BIN
require_uint MODEL_DOWNLOAD_WORKERS
require_cmd date
require_cmd tee

if [[ "${MODELSCOPE_BIN}" == */* ]]; then
    [[ -x "${MODELSCOPE_BIN}" ]] || die "MODELSCOPE_BIN is not executable: ${MODELSCOPE_BIN}"
    MODELSCOPE_COMMAND="${MODELSCOPE_BIN}"
else
    MODELSCOPE_COMMAND="$(command -v "${MODELSCOPE_BIN}" || true)"
    [[ -n "${MODELSCOPE_COMMAND}" ]] || die "modelscope CLI not found; install it in a data-disk venv"
fi

MODEL_FREE_GIB="$(available_gib "${MODEL_HOST_PATH}")"
(( MODEL_FREE_GIB >= MIN_MODEL_STORAGE_FREE_GIB )) || \
    die "model storage has ${MODEL_FREE_GIB} GiB free; need at least ${MIN_MODEL_STORAGE_FREE_GIB} GiB"

STATE_FILE="$(model_download_state_file)"
STATE_DIR="$(dirname "${STATE_FILE}")"
mkdir -p "${STATE_DIR}" "${MODEL_HOST_PATH}"
if [[ -f "${STATE_FILE}" && -f "${MODEL_HOST_PATH}/config.json" ]]; then
    RECORDED_ID="$(gate_value model_id "${STATE_FILE}")"
    RECORDED_REVISION="$(gate_value model_revision "${STATE_FILE}")"
    RECORDED_PATH="$(gate_value model_path "${STATE_FILE}")"
    if [[ "${RECORDED_ID}" == "${MODEL_ID}" && \
          "${RECORDED_REVISION}" == "${MODEL_REVISION:-default}" && \
          "${RECORDED_PATH}" == "${MODEL_HOST_PATH}" ]]; then
        printf 'model download already marked complete: %s\n' "${STATE_FILE}"
        exit 0
    fi
    die "existing download marker belongs to a different model ID, revision or path"
fi
umask 077
OUT_DIR="${RUN_HOST_ROOT}/model-download/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
exec > >(tee "${OUT_DIR}/download.log") 2>&1

DOWNLOAD_COMMAND=(
    "${MODELSCOPE_COMMAND}" download
    --model "${MODEL_ID}"
    --local_dir "${MODEL_HOST_PATH}"
    --max-workers "${MODEL_DOWNLOAD_WORKERS}"
)
if [[ -n "${MODEL_REVISION:-}" ]]; then
    DOWNLOAD_COMMAND+=(--revision "${MODEL_REVISION}")
fi
write_shell_command "${OUT_DIR}/command.sh" "${DOWNLOAD_COMMAND[@]}"
"${DOWNLOAD_COMMAND[@]}"

[[ -f "${MODEL_HOST_PATH}/config.json" ]] || die "download finished without config.json"
FIRST_SHARD="$(find "${MODEL_HOST_PATH}" -maxdepth 1 -type f -name '*.safetensors' -print -quit)"
[[ -n "${FIRST_SHARD}" ]] || die "download finished without safetensors shards"

{
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'model_id=%s\n' "${MODEL_ID}"
    printf 'model_revision=%s\n' "${MODEL_REVISION:-default}"
    printf 'model_path=%s\n' "${MODEL_HOST_PATH}"
} >"${STATE_FILE}"
printf 'DOWNLOAD_OK state=%s\n' "${STATE_FILE}"
