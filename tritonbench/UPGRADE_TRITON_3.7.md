# TritonBench → Triton 3.7.1 + Blackwell-only (SparkDistill fork)

This directory vendors [thunlp/tritonbench](https://github.com/thunlp/tritonbench) with updates for:

- **[Triton 3.7.1](https://github.com/triton-lang/triton/releases/tag/v3.7.1)** (upstream pinned 3.1.0)
- **NVIDIA Blackwell only** — SM10x datacenter or SM12x workstation; Hopper/Ampere/Ada rejected

Upstream pinned **Triton 3.1.0** on mixed GPU generations. SparkDistill distillation targets **Blackwell + Triton 3.7** only.

## Blackwell targeting

| Profile | `TRITONBENCH_BLACKWELL_PROFILE` | CUDA | Example hardware |
|---|---|---|---|
| Workstation | `workstation` (default) | sm_120, sm_121, … | RTX 5090, RTX PRO 6000 Blackwell |
| Datacenter | `datacenter` | sm_100, sm_103, … | B200, B300, GB200 |

`scripts/validate_reference_kernels.py` and `EVAL/eval_G/2_efficiency.py` call `require_blackwell_gpu()` and fail on non-Blackwell devices.

Efficiency scoring no longer uses legacy H100 constants (2039 GB/s, 312 TFLOPS). It uses Blackwell peaks from `bench_config.py` (override with `TRITONBENCH_PEAK_GBPS` / `TRITONBENCH_PEAK_TFLOPS`).

**Important:** Datacenter (SM10x) and workstation (SM12x) Blackwell use different tensor-core ISAs (tcgen05/TMEM vs extended `mma.sync`). Kernels optimized for one may not run optimally on the other — keep profile aligned with your fleet.

## What changed in this fork

| Area | Change |
|---|---|
| **Runtime** | `pyproject.toml` pins `triton==3.7.1`, `torch>=2.6.0` |
| **Config** | `bench_config.py` — repo paths, `TRITONBENCH_PYTHON`, GPU list parsing |
| **Eval scripts** | Removed hardcoded author paths; fixed T-channel data paths; fixed `--GPUs` parsing |
| **Validation** | `scripts/validate_reference_kernels.py` — GPU smoke test for gold references |
| **Setup** | `scripts/setup_env.sh` — creates `.venv` and checks versions |

## Setup (Blackwell GPU host required)

```bash
cd tritonbench
# workstation is default; for B200/B300: export TRITONBENCH_BLACKWELL_PROFILE=datacenter
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
export TRITONBENCH_PYTHON="$PWD/.venv/bin/python"
```

## Validate reference kernels on Triton 3.7.1

```bash
cd tritonbench
export TRITONBENCH_PYTHON="$PWD/.venv/bin/python"

# TritonBench-G (184 real Triton kernels) — primary compile/correctness gate
python scripts/validate_reference_kernels.py --channel G --gpu 0 --out reports/g_ref_3.7.1.json

# TritonBench-T (166 torch reference tests — no Triton in gold, CUDA torch ops)
python scripts/validate_reference_kernels.py --channel T --gpu 0 --limit 20
```

**Expected:** Some TritonBench-G kernels may fail on 3.7.1 until individually ported. Record failures in `reports/` and patch `data/TritonBench_G_v1/*.py`.

## Known Triton 3.1 → 3.7 differences affecting this bench

From [Triton 3.7 release notes](https://github.com/triton-lang/triton/releases/tag/v3.7.0):

- `tl.make_block_ptr` — deprecated (warning); migrate to tensor descriptors over time
- FP8 (`tl.float8e4nv`) — stricter on pre-sm89 GPUs; may need fp16 paths on older cards
- `triton_kernels.matmul_*` API — BC-breaking (not used directly in most bench kernels)
- Compiler/backend changes — same source may compile with different perf; **regenerate** `performance_metrics/**/golden_results` after porting

## Eval workflow (unchanged semantics)

Run from **repo root** (`tritonbench/`):

```bash
cd EVAL/eval_G
python 0_call_acc.py --source /path/to/predictions.jsonl --target /path/to/out --GPUs 0
python 1_exe_acc.py --folder /path/to/out --GPUs 0
```

Use `--GPUs 0`, `--GPUs 0,1`, or `--GPUs [0,1,2,3]`.

## Performance golden files

`performance_metrics/perf_G/golden_results` and `perf_T/golden_results` were captured on **Triton 3.1**. After reference kernels pass validation on 3.7.1, re-benchmark and update golden JSON before trusting efficiency scores (`2_efficiency.py`).

## Upstream sync

```bash
git remote add upstream https://github.com/thunlp/tritonbench.git  # once
git fetch upstream
# merge selectively — keep our pyproject.toml, bench_config.py, eval fixes
```

## Legal

TritonBench-G includes operators derived from open-source repos. See upstream paper and dataset licenses before using outputs for model training.
