#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TRITONBENCH_BLACKWELL_PROFILE="${TRITONBENCH_BLACKWELL_PROFILE:-workstation}"
export TRITONBENCH_PYTHON="${TRITONBENCH_PYTHON:-$PWD/.venv/bin/python}"
exec "$TRITONBENCH_PYTHON" -m tritonbench.cli eval "$@"
