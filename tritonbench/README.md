# TritonBench (SparkDistill fork)

Vendored from [thunlp/tritonbench](https://github.com/thunlp/tritonbench) — benchmark for LLM-generated **Triton** operators.

**Targets only NVIDIA Blackwell GPUs** (Triton **3.7.1**). Hopper, Ada, and Ampere are rejected at validation/eval time.

See [UPGRADE_TRITON_3.7.md](UPGRADE_TRITON_3.7.md).

## SparkDistill Triton track

See **[PROJECT.md](PROJECT.md)** for the full architecture (Triton 3.7.1 → Qwen3.5-4B → Claude Code).

```bash
./scripts/run_eval.sh --config configs/eval_quick.yaml \
  --endpoint http://localhost:8000/v1 --model qwen-triton
```

## Blackwell profiles

| Profile | CUDA SM | Example GPUs | Env |
|---|---|---|---|
| **workstation** (default) | SM12x | RTX 5090, RTX PRO 6000 Blackwell | `TRITONBENCH_BLACKWELL_PROFILE=workstation` |
| **datacenter** | SM10x | B200, B300, GB200 | `TRITONBENCH_BLACKWELL_PROFILE=datacenter` |

SM10x (datacenter) and SM12x (workstation) are **not binary-compatible** — pick the profile that matches your hardware. Override peak efficiency limits with `TRITONBENCH_PEAK_GBPS` and `TRITONBENCH_PEAK_TFLOPS`.

## Channels

| Channel | Tasks | Gold references |
|---|---:|---|
| **TritonBench-G** | 184 real-world operators | Triton kernels in `data/TritonBench_G_v1/` |
| **TritonBench-T** | 166 PyTorch-aligned tasks | Torch refs + CUDA tests in `data/TritonBench_T_v1/` |

Paper: [TRITONBENCH (ACL 2025 Findings)](https://arxiv.org/pdf/2502.14752)

## Quick setup (Blackwell host)

```bash
cd tritonbench
# workstation is default; for B200/B300 use: export TRITONBENCH_BLACKWELL_PROFILE=datacenter
./scripts/setup_env.sh
export TRITONBENCH_PYTHON="$PWD/.venv/bin/python"
python scripts/validate_reference_kernels.py --channel G --gpu 0
```

## Python environment

- `triton==3.7.1`
- `torch>=2.6.0`
- CUDA **12.8+** (Blackwell requirement)
- Blackwell GPU required for kernel validation and eval

## Evaluation

```bash
cd EVAL/eval_G
python 0_call_acc.py --source predictions.jsonl --target ./out --GPUs 0
python 1_exe_acc.py --folder ./out --GPUs 0
python 2_efficiency.py --gen_folder /path/to/perf_results   # Blackwell peak limits
```

## SparkDistill integration

Use instructions as **prompt seeds** and this harness as **verification** for Fable 5 + GPT 5.6 Sol → Qwen3.5-4B Triton distillation. Train for **Blackwell Triton 3.7** APIs (TMA/tensor descriptors, FP8/FP4 where applicable on your SM variant).
