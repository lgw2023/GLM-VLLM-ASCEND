#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

load_configs "${1:-}" "${2:-}"
[[ "${3:-}" == --confirm-pull ]] || \
    die "image pull can consume substantial DockerRootDir space; rerun with --confirm-pull"
require_cmd date
require_cmd docker
require_cmd sha256sum
require_cmd tee

DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
DOCKER_FREE_GIB="$(available_gib "${DOCKER_ROOT}")"
(( DOCKER_FREE_GIB >= MIN_DOCKER_FREE_GIB )) || \
    die "DockerRootDir has only ${DOCKER_FREE_GIB} GiB free"

umask 077
OUT_DIR="${RUN_HOST_ROOT}/image-manifests/$(date -u +%Y%m%dT%H%M%SZ)/node${NODE_RANK}"
mkdir -p "${OUT_DIR}"
exec > >(tee "${OUT_DIR}/pull.log") 2>&1

printf 'image_ref=%s\n' "${IMAGE_REF}"
printf 'before_pull:\n'
docker image inspect "${IMAGE_REF}" \
    --format 'id={{.Id}} repo_digests={{json .RepoDigests}} created={{.Created}}' \
    2>/dev/null || true

docker pull "${IMAGE_REF}"

printf 'after_pull:\n'
image_metadata | tee "${OUT_DIR}/image.identity"
probe_image_packages | tee "${OUT_DIR}/packages.txt"
printf '%s\n' "$(image_capture_patch_id)" >"${OUT_DIR}/capture-patch-id.txt"
DOCKER_FREE_GIB="$(available_gib "${DOCKER_ROOT}")"
(( DOCKER_FREE_GIB >= MIN_DOCKER_FREE_GIB )) || \
    die "DockerRootDir fell below ${MIN_DOCKER_FREE_GIB} GiB after pull"
GATE_DIR="$(gate_dir)"
mkdir -p "${GATE_DIR}"
{
    printf 'image_ref=%s\n' "${IMAGE_REF}"
    printf 'image_id=%s\n' "$(current_image_id)"
    printf 'packages_sha256=%s\n' "$(sha256sum "${OUT_DIR}/packages.txt" | awk '{print $1}')"
    printf 'capture_patch_id=%s\n' "$(image_capture_patch_id)"
    printf 'cluster_config_sha256=%s\n' "$(cluster_config_sha256)"
    printf 'root_commit=%s\n' "$(root_commit)"
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
} >"${GATE_DIR}/image.gate"
printf 'PULL_OK output=%s gate=%s\n' "${OUT_DIR}" "${GATE_DIR}/image.gate"
