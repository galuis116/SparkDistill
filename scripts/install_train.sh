#!/usr/bin/env bash
# Install Axolotl training stack for SparkDistill recipes.
#
#   scripts/install_train.sh
#
# Qwen3.5 processors require torchvision. Prefer the official FlashAttention 3
# wheel (seconds); fall back to a source FlashAttention 2 build on Blackwell.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run SparkProof scripts/install.sh first" >&2
  exit 1
fi

echo ">>> syncing SparkDistill base deps"
uv sync --extra dev

echo ">>> installing Axolotl + torchvision"
uv pip install -q axolotl torchvision numba ninja packaging einops "numpy<2.5"

install_fa2() {
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${gpu_major}0}"
  if uv run --no-sync python -c "import flash_attn" 2>/dev/null; then
    uv run --no-sync python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
    return
  fi

  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "  flash-attn: CUDA compiler not found at $CUDA_HOME/bin/nvcc (training will use SDPA)" >&2
    return
  fi

  echo ">>> building FlashAttention 2 for Blackwell (first install takes several minutes)"
  build_jobs="${MAX_JOBS:-$(nproc)}"
  echo "  CUDA arch: ${FLASH_ATTN_CUDA_ARCHS}; parallel jobs: $build_jobs ($(nproc) vCPUs available)"
  MAX_JOBS="$build_jobs" uv pip install "flash-attn==2.8.3.post1" --no-build-isolation
  uv run --no-sync python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
}

gpu_major="$(uv run --no-sync python -c "import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)")"

if [ "${SPARKDISTILL_SKIP_FLASH_ATTN:-0}" = "1" ]; then
  echo "  flash-attn: skipped (SPARKDISTILL_SKIP_FLASH_ATTN=1; training will use SDPA)"
elif [ "$gpu_major" -ge 10 ]; then
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${gpu_major}0}"
  echo ">>> Blackwell SM${FLASH_ATTN_CUDA_ARCHS}: using FlashAttention 2 (official FA3 wheel is Hopper-only)"
  install_fa2
elif uv run --no-sync python -c "from transformers.utils import is_flash_attn_3_available; import sys; sys.exit(0 if is_flash_attn_3_available() else 1)" 2>/dev/null; then
  echo "  flash-attn-3: installed"
else
  cuda_tag="$(uv run --no-sync python -c "import torch; v=torch.version.cuda or ''; print('cu' + v.replace('.', '')[:3] if v else 'cu130')")"
  fa3_index="https://download.pytorch.org/whl/${cuda_tag}"
  echo ">>> installing FlashAttention 3 wheel (${cuda_tag}, ~30s)"
  if uv pip install -q "flash-attn-3" --index-url "$fa3_index" \
    && uv run --no-sync python -c "from transformers.utils import is_flash_attn_3_available; import sys; sys.exit(0 if is_flash_attn_3_available() else 1)" 2>/dev/null; then
    echo "  flash-attn-3: installed from ${fa3_index}"
  else
    install_fa2
  fi
fi

if uv run --no-sync axolotl --help >/dev/null 2>&1; then
  echo ">>> Axolotl ready"
else
  echo "error: axolotl CLI not available after install" >&2
  exit 1
fi

echo ""
echo "Next:"
echo "  scripts/prepare_mining_sft.sh"
echo "  scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml"
