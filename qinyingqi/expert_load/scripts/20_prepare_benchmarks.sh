#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-${SCRIPT_DIR}/../.client-venv/bin/python}"

if [[ ! -x "${CLIENT_PYTHON}" ]]; then
    printf 'ERROR: client Python is not executable: %s\n' "${CLIENT_PYTHON}" >&2
    printf 'Create .client-venv and install requirements-client.txt, or set CLIENT_PYTHON.\n' >&2
    exit 1
fi

exec "${CLIENT_PYTHON}" "${SCRIPT_DIR}/20_prepare_benchmarks.py" "$@"
