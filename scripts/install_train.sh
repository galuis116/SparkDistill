#!/usr/bin/env bash
# Install Axolotl training stack for SparkDistill recipes.
#
#   scripts/install_train.sh
#
# Qwen3.5 processors require torchvision. On Blackwell, FlashAttention is built
# for SM120 only; the first source build can take several minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run SparkProof scripts/install.sh first" >&2
  exit 1
fi

echo ">>> syncing SparkDistill base deps"
uv sync --extra dev

echo ">>> installing Axolotl + torchvision"
uv pip install -q axolotl torchvision

if [ "${SPARKDISTILL_SKIP_FLASH_ATTN:-0}" = "1" ]; then
  echo "  flash-attn: skipped (SPARKDISTILL_SKIP_FLASH_ATTN=1; training will use SDPA)"
elif uv run python -c "import flash_attn" 2>/dev/null; then
  echo "  flash-attn: installed"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "  flash-attn: CUDA compiler not found at $CUDA_HOME/bin/nvcc (training will use SDPA)" >&2
  else
    echo ">>> building FlashAttention for Blackwell SM120 (first install takes several minutes)"
    uv pip install -q ninja packaging wheel
    build_jobs="${MAX_JOBS:-$(nproc)}"
    echo "  CUDA arch: SM120; parallel jobs: $build_jobs ($(nproc) vCPUs available)"
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" \
      MAX_JOBS="$build_jobs" \
      uv pip install "flash-attn==2.8.3.post1" --no-build-isolation
    uv run python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
  fi
fi

if uv run axolotl --help >/dev/null 2>&1; then
  echo ">>> Axolotl ready"
else
  echo "error: axolotl CLI not available after install" >&2
  exit 1
fi

echo ""
echo "Next:"
echo "  scripts/prepare_mining_sft.sh"
echo "  scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml"
