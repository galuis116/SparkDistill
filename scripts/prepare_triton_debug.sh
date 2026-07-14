#!/usr/bin/env bash
# Build the Triton generation+debugging SFT mix for
# recipes/qwen3.5-4b-phase1/sft-triton-debug.yaml.
#
#   scripts/prepare_triton_debug.sh
#
# Extends scripts/prepare_triton_kernels.sh with the debugging + stable-reduction rows
# from eval.gen_triton_debug, so the student learns to *fix* buggy kernels (the
# TritonBench bugfix problem) on top of writing them. All kernels are GPU-verified.
set -euo pipefail
cd "$(dirname "$0")/.."

GEN="data/processed/triton_kernels_raw.jsonl"
DBG="data/processed/triton_debug_kernels.jsonl"
MINING="data/processed/sparkproof-mining_sft.jsonl"
OUT="data/processed/triton_debug_sft.jsonl"
OVERSAMPLE="${SPARKDISTILL_TRITON_OVERSAMPLE:-5}"

echo ">>> generating + verifying executable kernels"
uv run python -m eval.gen_triton_kernels --out "$GEN"

echo ">>> generating + verifying debugging + stable-reduction kernels"
uv run python -m eval.gen_triton_debug --out "$DBG"

echo ">>> exporting canonical mining mix"
uv run python -m eval.prepare_mining_sft --out "$MINING"

echo ">>> blending ((kernels + debug) x${OVERSAMPLE} + mining) -> $OUT"
uv run python - "$GEN" "$DBG" "$MINING" "$OUT" "$OVERSAMPLE" <<'PY'
import json, random, sys
gen, dbg, mining, out, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
random.seed(4)
kern = [json.loads(l) for l in open(gen)] + [json.loads(l) for l in open(dbg)]
mine = [json.loads(l) for l in open(mining)]
rows = kern * k + mine
random.shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"  kernels+debug={len(kern)} x{k} + mining={len(mine)} -> {len(rows)} rows")
PY

echo ""
echo "Next: scripts/train.sh recipes/qwen3.5-4b-phase1/sft-triton-debug.yaml"
