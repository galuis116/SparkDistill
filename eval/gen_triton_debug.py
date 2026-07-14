"""Kernel-DEBUGGING + numerically-stable-reduction SFT rows (GPU-verified).

Extends `eval.gen_triton_kernels` (executable-kernel *generation*) with the skill the
generation-only student still lacks: fixing a buggy kernel. TritonBench's `wrong_mask`
problem hands the model a broken kernel and asks for the fix, but every from-scratch
example teaches generation — so the student scores ~0.28 composite there and never
executes it. These rows teach (a) the debugging format and the masking-bug class it
targets, and (b) the numerically-stable masked reduction pattern (max with other=-inf).
Ops are disjoint from the eval problems; every "fixed" script is compiled + run +
allclose-checked on the GPU and only kept if it passes.

    python -m eval.gen_triton_debug --out data/processed/triton_debug_kernels.jsonl

Run on the Blackwell SM120 target (triton==3.7.1 + torch).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tritonbench"))
from tritonbench.features.triton_371 import system_prompt_for_triton  # noqa: E402

SYSTEM = system_prompt_for_triton()

# --- numerically-stable masked reductions (softmax-family pattern) --------------
STABLE = [
    ("logsumexp", "the row-wise log-sum-exp (numerically stable) of a 2D tensor", '''import torch
import triton
import triton.language as tl


# Numerically-stable row log-sum-exp. The masked max uses other=-inf so padded columns
# never win the max; exp is taken on the shifted values and summed over the same mask.
@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=2) for w in (4, 8, 16)], key=["n_cols"])
@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    tl.store(out_ptr + row, m + tl.log(tl.sum(e, axis=0)))


def logsumexp(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    logsumexp_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=triton.next_power_of_2(cols))
    return out


torch.manual_seed(0)
xt = torch.randn(96, 500, device="cuda", dtype=torch.float32)
assert torch.allclose(logsumexp(xt), torch.logsumexp(xt, dim=1), rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")'''),
    ("max_shift", "the row max-shift out = x - max(x) per row of a 2D tensor", '''import torch
import triton
import triton.language as tl


# Row max-shift (the softmax pre-shift): subtract each row's max, with other=-inf on the
# masked load so padded columns cannot corrupt the max.
@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=2) for w in (4, 8, 16)], key=["n_cols"])
@triton.jit
def max_shift_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    tl.store(out_ptr + row * row_stride + cols, x - tl.max(x, axis=0), mask=mask)


def max_shift(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    max_shift_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=triton.next_power_of_2(cols))
    return out


torch.manual_seed(0)
xt = torch.randn(96, 500, device="cuda", dtype=torch.float32)
assert torch.allclose(max_shift(xt), xt - xt.max(dim=1, keepdim=True).values, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")'''),
    ("row_var", "the row-wise population variance of a 2D tensor", '''import torch
import triton
import triton.language as tl


# Masked row variance: masked mean, then masked mean of squared deviations (other=0.0
# so padding contributes nothing to either sum).
@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=2) for w in (4, 8, 16)], key=["n_cols"])
@triton.jit
def row_var_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    tl.store(out_ptr + row, tl.sum(d * d, axis=0) / n_cols)


def row_var(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    row_var_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=triton.next_power_of_2(cols))
    return out


torch.manual_seed(0)
xt = torch.randn(96, 500, device="cuda", dtype=torch.float32)
assert torch.allclose(row_var(xt), xt.var(dim=1, unbiased=False), rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")'''),
]

# --- debugging: (name, buggy code shown, verified fixed code, explanation) -------
DEBUG = [
    ("scale_missing_mask",
     '''@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x * scale)


def scale(x, s):
    out = torch.empty_like(x)
    n = x.numel()
    scale_kernel[(triton.cdiv(n, 1024),)](x, out, n, s, BLOCK_SIZE=1024)
    return out''',
     '''import torch
import triton
import triton.language as tl


# Fixed: mask = offsets < n_elements guards the tail so partial blocks don't read/write
# out of bounds. Both the load and the store must use it.
@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


def scale(x: torch.Tensor, s: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    scale_kernel[(triton.cdiv(n, 1024),)](x, out, n, s, BLOCK_SIZE=1024)
    return out


torch.manual_seed(0)
xt = torch.randn(4097, device="cuda", dtype=torch.float32)
assert torch.allclose(scale(xt, 2.5), xt * 2.5, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")''',
     "The load and store had no boundary mask, so the last (partial) block indexed past the "
     "tensor. Add `mask = offsets < n_elements` on both `tl.load` and `tl.store`."),
    ("rowsum_unmasked_tail",
     '''@triton.jit
def rowsum_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * row_stride + cols)
    tl.store(out_ptr + row, tl.sum(x, axis=0))''',
     '''import torch
import triton
import triton.language as tl


# Fixed: load the tail with mask + other=0.0 so out-of-range columns contribute 0 to the
# sum (do NOT pass mask= to tl.sum; mask the data on load instead).
@triton.jit
def rowsum_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


def rowsum(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    rowsum_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=triton.next_power_of_2(cols))
    return out


torch.manual_seed(0)
xt = torch.randn(64, 500, device="cuda", dtype=torch.float32)
assert torch.allclose(rowsum(xt), xt.sum(dim=1), rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")''',
     "`BLOCK_N` is rounded up to a power of two, so the unmasked load pulled garbage past "
     "`n_cols` into the sum. Load with `mask = cols < n_cols, other=0.0` so padding adds 0."),
    ("rowmax_wrong_other",
     '''@triton.jit
def rowmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.max(x, axis=0))''',
     '''import torch
import triton
import triton.language as tl


# Fixed: for a max reduction the masked `other` must be -inf, not 0.0 — otherwise the
# padding 0.0 wins the max on all-negative rows and the result is wrong.
@triton.jit
def rowmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    tl.store(out_ptr + row, tl.max(x, axis=0))


def rowmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    rowmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=triton.next_power_of_2(cols))
    return out


torch.manual_seed(0)
xt = torch.randn(64, 500, device="cuda", dtype=torch.float32) - 5.0  # all-negative rows
assert torch.allclose(rowmax(xt), xt.max(dim=1).values, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")''',
     "The masked load used `other=0.0` for a max reduction, so padded columns injected a 0 "
     "that beat every real (negative) value. Use `other=-inf` for max reductions."),
]

STABLE_PRE = ["Here is a complete, stable implementation with a launcher and an allclose check.",
              "A full working kernel with masked reductions and a reference test follows.",
              "Below is the kernel plus launcher and a torch.allclose verification."]
STABLE_USER = ("Write a Triton 3.7.1 kernel that computes {desc}.\n\nRequirements:\n"
               "- One program per row, boundary masking (use other=-inf for max reductions)\n"
               "- @triton.autotune (vary num_warps), fp32 accumulation, numerically stable\n"
               "- A launcher and a torch.allclose correctness test vs PyTorch\n- Target Blackwell SM120")


def _verify(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        env = dict(os.environ, TRITON_PRINT_AUTOTUNING="0")
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=200, env=env)
        return ("TRITONBENCH_PASS" in p.stdout), (p.stdout + p.stderr)[-700:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/processed/triton_debug_kernels.jsonl"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for i, (name, desc, code) in enumerate(STABLE):
        ok, log = _verify(code)
        if not ok:
            failed.append(name)
            print(f"  FAIL {name}: {log.splitlines()[-1] if log.strip() else 'no output'}", file=sys.stderr)
            continue
        a = f"{STABLE_PRE[i % len(STABLE_PRE)]}\n\n```python\n{code}```"
        rows.append({"messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": STABLE_USER.format(desc=desc)},
                                  {"role": "assistant", "content": a}]})
        print(f"  PASS stable:{name}", file=sys.stderr)

    for name, buggy, fixed, why in DEBUG:
        ok, log = _verify(fixed)
        if not ok:
            failed.append(name)
            print(f"  FAIL {name}: {log.splitlines()[-1] if log.strip() else 'no output'}", file=sys.stderr)
            continue
        user = ("The following Triton 3.7.1 kernel is buggy — it gives wrong results for sizes that "
                "are not a multiple of the block. Find and fix the bug(s), and return a complete "
                f"runnable script with a torch.allclose test.\n\n```python\n{buggy}\n```")
        assistant = f"{why}\n\nHere is the corrected, runnable version:\n\n```python\n{fixed}```"
        rows.append({"messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": assistant}]})
        print(f"  PASS debug:{name}", file=sys.stderr)

    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} verified debug/stable rows to {args.out}", file=sys.stderr)
    if failed:
        print(f"skipped (did not verify on this GPU): {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
