#!/usr/bin/env bash
# Create a TritonBench venv with Triton 3.7.1 (requires CUDA GPU host + disk space).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
mkdir -p "$TMPDIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warning: nvidia-smi not found — install CUDA drivers before running kernel validation" >&2
fi

uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
export TRITONBENCH_PYTHON="$PWD/.venv/bin/python"
"$TRITONBENCH_PYTHON" -c "
from bench_config import TRITON_VERSION, check_runtime_versions
v = check_runtime_versions(enforce_blackwell=True)
assert v['triton'].startswith('3.7'), v
print('ok', v)
print('pinned triton', TRITON_VERSION)
print('target gpu', v['target_gpu'], 'profile', v['blackwell_profile'])
"
