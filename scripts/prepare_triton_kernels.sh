#!/usr/bin/env bash
# Build the Triton-kernel SFT mix for recipes/qwen3.5-4b-phase1/sft-triton.yaml.
#
#   scripts/prepare_triton_kernels.sh
#
# 1. Generate + GPU-verify the executable-kernel dataset (eval.gen_triton_kernels).
# 2. Export the canonical mining mix (eval.prepare_mining_sft) for reasoning breadth.
# 3. Blend: verified kernels oversampled x5 (they are the improvement signal) + mining,
#    shuffled, into data/processed/triton_sft.jsonl — the recipe's `datasets` path.
#
# Run on the Blackwell SM120 target: the generator only keeps kernels that compile,
# run, and pass torch.allclose on the GPU it runs on.
set -euo pipefail
cd "$(dirname "$0")/.."

RAW="data/processed/triton_kernels_raw.jsonl"
MINING="data/processed/sparkproof-mining_sft.jsonl"
OUT="data/processed/triton_sft.jsonl"
OVERSAMPLE="${SPARKDISTILL_TRITON_OVERSAMPLE:-5}"

echo ">>> generating + verifying Triton kernels"
uv run python -m eval.gen_triton_kernels --out "$RAW"

echo ">>> exporting canonical mining mix"
uv run python -m eval.prepare_mining_sft --out "$MINING"

echo ">>> blending (kernels x${OVERSAMPLE} + mining) -> $OUT"
uv run python - "$RAW" "$MINING" "$OUT" "$OVERSAMPLE" <<'PY'
import json, random, sys
raw, mining, out, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
random.seed(4)
kern = [json.loads(l) for l in open(raw)]
mine = [json.loads(l) for l in open(mining)]
rows = kern * k + mine
random.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"  kernels={len(kern)} x{k} + mining={len(mine)} -> {len(rows)} rows")
PY

echo ""
echo "Next: scripts/train.sh recipes/qwen3.5-4b-phase1/sft-triton.yaml"
