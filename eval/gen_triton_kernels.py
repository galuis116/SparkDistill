"""Generate a GPU-verified Triton-3.7.1 kernel SFT dataset for the Phase-1 student.

Each row teaches the student to emit a COMPLETE, EXECUTABLE kernel with an autotuned
launcher, boundary masks, fp32 accumulation, and a `torch.allclose` correctness check —
the exact shape TritonBench rewards (correctness is 35% of the composite, and the frontier
scores 0 there). Only kernels that compile, run, and pass their own `torch.allclose` on the
current GPU are kept, so every training target is known-good.

The ops are deliberately DISJOINT from the TritonBench eval problems (vector_add, softmax,
wrong_mask) so decontamination passes — we teach the *skill* (masked elementwise + masked
row reductions + normalization), not the answers. Comment headers and answer preambles are
varied per row so the student does not overfit to one phrasing (uniform text induces
degenerate repetition at inference).

    python -m eval.gen_triton_kernels --out data/processed/triton_kernels_sft.jsonl

Run on a Blackwell SM120 GPU (the Phase-1 target); requires triton==3.7.1 + torch.
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

# Varied text so the student sees the pattern, not one fixed phrasing.
PREAMBLES = [
    "Here's a complete, runnable kernel with an autotuned launcher and a correctness check.",
    "Below is a self-contained implementation, including the launcher and a torch.allclose test.",
    "This solution uses boundary masks and autotuning; it verifies its own output against PyTorch.",
    "A full working version with grid launcher and reference comparison follows.",
    "Here is the kernel plus a launcher and a numerical check versus the PyTorch reference.",
]
HEADER_ALTS = [
    "# Autotuned over the block size; boundary masks handle non-aligned sizes.",
    "# Grid-strided launch with masks so any length is safe; fp32 throughout.",
    "# Block size is autotuned; the mask guards the tail of each row/vector.",
    "# Uses num_warps/num_stages hints and masked loads for correctness.",
    "# Masked loads/stores keep partial blocks correct on Blackwell SM120.",
]

# ----------------------------------------------------------------------------- specs
# (name, user-facing description, triton expr over `x`, torch reference over `xt`, init)
UNARY = [
    ("relu", "element-wise ReLU: out = max(x, 0)", "tl.where(x > 0, x, 0.0)", "torch.relu(xt)", "randn"),
    ("silu", "element-wise SiLU/Swish: out = x * sigmoid(x)", "x * tl.sigmoid(x)", "torch.nn.functional.silu(xt)", "randn"),
    ("sigmoid", "element-wise sigmoid", "tl.sigmoid(x)", "torch.sigmoid(xt)", "randn"),
    ("tanh_op", "element-wise tanh using 2*sigmoid(2x)-1", "2.0 * tl.sigmoid(2.0 * x) - 1.0", "torch.tanh(xt)", "randn"),
    ("gelu_tanh", "element-wise GELU (tanh approximation)",
     "0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608 * (x + 0.044715 * x * x * x))) - 1.0))",
     "torch.nn.functional.gelu(xt, approximate='tanh')", "randn"),
    ("leaky_relu", "element-wise LeakyReLU with slope 0.01", "tl.where(x > 0, x, 0.01 * x)",
     "torch.nn.functional.leaky_relu(xt, 0.01)", "randn"),
    ("elu", "element-wise ELU: x if x>0 else exp(x)-1", "tl.where(x > 0, x, tl.exp(x) - 1.0)",
     "torch.nn.functional.elu(xt)", "randn"),
    ("abs_op", "element-wise absolute value", "tl.abs(x)", "torch.abs(xt)", "randn"),
    ("square", "element-wise square: out = x*x", "x * x", "xt * xt", "randn"),
    ("neg", "element-wise negation", "-x", "-xt", "randn"),
    ("softplus", "element-wise softplus: log(1+exp(x))", "tl.log(1.0 + tl.exp(x))",
     "torch.nn.functional.softplus(xt)", "small"),
    ("exp_op", "element-wise exp", "tl.exp(x)", "torch.exp(xt)", "small"),
    ("reciprocal", "element-wise reciprocal 1/x", "1.0 / x", "torch.reciprocal(xt)", "pos"),
    ("sqrt_op", "element-wise sqrt (positive inputs)", "tl.sqrt(x)", "torch.sqrt(xt)", "pos"),
    ("clamp01", "element-wise clamp to [0, 1]", "tl.minimum(tl.maximum(x, 0.0), 1.0)", "torch.clamp(xt, 0.0, 1.0)", "randn"),
    ("hardswish", "element-wise HardSwish: x * relu6(x+3) / 6",
     "x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0", "torch.nn.functional.hardswish(xt)", "randn"),
    ("relu6", "element-wise ReLU6: min(max(x,0),6)", "tl.minimum(tl.maximum(x, 0.0), 6.0)",
     "torch.nn.functional.relu6(xt)", "randn"),
    ("mish", "element-wise Mish: x * tanh(softplus(x))",
     "x * (2.0 * tl.sigmoid(2.0 * tl.log(1.0 + tl.exp(x))) - 1.0)", "torch.nn.functional.mish(xt)", "randn"),
]
BINARY = [
    ("mul", "element-wise multiply: C = A * B", "a * b", "at * bt"),
    ("sub", "element-wise subtract: C = A - B", "a - b", "at - bt"),
    ("maximum_op", "element-wise maximum: C = max(A, B)", "tl.maximum(a, b)", "torch.maximum(at, bt)"),
    ("minimum_op", "element-wise minimum: C = min(A, B)", "tl.minimum(a, b)", "torch.minimum(at, bt)"),
    ("div", "element-wise divide C = A / B", "a / b", "at / bt"),
]
# (name, description, other-value, tl reduce expr, torch ref)
REDUCE = [
    ("row_sum", "the row-wise sum of a 2D tensor", "0.0", "tl.sum(x, axis=0)", "xt.sum(dim=1)"),
    ("row_max", "the row-wise maximum of a 2D tensor", "-float('inf')", "tl.max(x, axis=0)", "xt.max(dim=1).values"),
    ("row_min", "the row-wise minimum of a 2D tensor", "float('inf')", "tl.min(x, axis=0)", "xt.min(dim=1).values"),
    ("row_mean", "the row-wise mean of a 2D tensor", "0.0", "tl.sum(x, axis=0) / n_cols", "xt.mean(dim=1)"),
]
# (name, description, other, reduce-expr, elementwise-out, init, ref) — masked 2-pass normalize
NORM = [
    ("row_sum_normalize", "row-wise sum-normalization out = x / sum(x) for positive rows",
     "0.0", "tl.sum(x, axis=0)", "x / denom",
     'torch.rand(96, 500, device="cuda", dtype=torch.float32) + 0.1', "xt / xt.sum(dim=1, keepdim=True)"),
    ("row_l2_normalize", "row-wise L2 normalization out = x / sqrt(sum(x*x))",
     "0.0", "tl.sqrt(tl.sum(x * x, axis=0))", "x / denom",
     'torch.randn(96, 500, device="cuda", dtype=torch.float32)',
     "xt / torch.sqrt((xt * xt).sum(dim=1, keepdim=True))"),
    ("row_mean_center", "row-wise mean-centering out = x - mean(x)",
     "0.0", "tl.sum(x, axis=0) / n_cols", "x - denom",
     'torch.randn(96, 500, device="cuda", dtype=torch.float32)', "xt - xt.mean(dim=1, keepdim=True)"),
]

INIT = {
    "randn": 'torch.randn(4099, device="cuda", dtype=torch.float32)',
    "small": 'torch.randn(4099, device="cuda", dtype=torch.float32) * 0.5',
    "pos": 'torch.rand(4099, device="cuda", dtype=torch.float32) + 0.5',
}

UNARY_TMPL = '''import torch
import triton
import triton.language as tl


{header}
@triton.autotune(
    configs=[
        triton.Config({{"BLOCK_SIZE": 256}}, num_warps=4, num_stages=2),
        triton.Config({{"BLOCK_SIZE": 1024}}, num_warps=4, num_stages=3),
        triton.Config({{"BLOCK_SIZE": 2048}}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def {op}_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = {expr}
    tl.store(out_ptr + offsets, out, mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    """Elementwise {op} with an autotuned block size."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    {op}_kernel[grid](x, out, n_elements)
    return out


torch.manual_seed(0)
xt = {init}
out = {op}(xt)
ref = {ref}
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")
'''

BINARY_TMPL = '''import torch
import triton
import triton.language as tl


{header}
@triton.autotune(
    configs=[
        triton.Config({{"BLOCK_SIZE": 256}}, num_warps=4, num_stages=2),
        triton.Config({{"BLOCK_SIZE": 1024}}, num_warps=4, num_stages=3),
        triton.Config({{"BLOCK_SIZE": 4096}}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def {op}_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = {expr}
    tl.store(out_ptr + offsets, out, mask=mask)


def {op}(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch the {op} kernel elementwise over two same-shape tensors."""
    out = torch.empty_like(a)
    n_elements = a.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    {op}_kernel[grid](a, b, out, n_elements)
    return out


torch.manual_seed(0)
at = torch.randn(4099, device="cuda", dtype=torch.float32)
bt = torch.randn(4099, device="cuda", dtype=torch.float32){bfix}
out = {op}(at, bt)
ref = {ref}
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")
'''

REDUCE_TMPL = '''import torch
import triton
import triton.language as tl


{header}
@triton.autotune(
    configs=[triton.Config({{}}, num_warps=w, num_stages=2) for w in (4, 8, 16)],
    key=["n_cols"],
)
@triton.jit
def {op}_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other={other})
    acc = {reduce}
    tl.store(out_ptr + row, acc)


def {op}(x: torch.Tensor) -> torch.Tensor:
    """Reduce each row of a 2-D float32 tensor with a single masked block."""
    rows, cols = x.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    block_n = triton.next_power_of_2(cols)
    {op}_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=block_n)
    return out


torch.manual_seed(0)
xt = torch.randn(128, 500, device="cuda", dtype=torch.float32)
out = {op}(xt)
ref = {ref}
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")
'''

NORM_TMPL = '''import torch
import triton
import triton.language as tl


{header}
@triton.autotune(
    configs=[triton.Config({{}}, num_warps=w, num_stages=2) for w in (4, 8, 16)],
    key=["n_cols"],
)
@triton.jit
def {op}_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other={other})
    denom = {reduce}
    out = {elem}
    tl.store(out_ptr + row * row_stride + cols, out, mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    """Row-wise {op} of a 2-D float32 tensor with boundary masking."""
    rows, cols = x.shape
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(cols)
    {op}_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_N=block_n)
    return out


torch.manual_seed(0)
xt = {init}
out = {op}(xt)
ref = {ref}
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")
'''

RMSNORM = '''import torch
import triton
import triton.language as tl


{header}
@triton.autotune(
    configs=[triton.Config({{}}, num_warps=w, num_stages=2) for w in (4, 8, 16)],
    key=["n_cols"],
)
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, row_stride, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    ms = tl.sum(x * x, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    tl.store(out_ptr + row * row_stride + offs, x * inv * w, mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise RMS normalization of a 2-D float32 tensor."""
    rows, cols = x.shape
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(cols)
    rmsnorm_kernel[(rows,)](x, w, out, cols, x.stride(0), eps, BLOCK_N=block_n)
    return out


torch.manual_seed(0)
xt = torch.randn(64, 512, device="cuda", dtype=torch.float32)
wt = torch.randn(512, device="cuda", dtype=torch.float32)
out = rmsnorm(xt, wt)
ref = xt / torch.sqrt((xt * xt).mean(dim=1, keepdim=True) + 1e-6) * wt
assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "kernel output mismatch"
print("TRITONBENCH_PASS")
'''

U_ELEM = ("Write a Triton 3.7.1 kernel that computes {desc}.\n\nRequirements:\n"
          "- Arbitrary sizes with boundary masking\n"
          "- @triton.autotune with >=3 configs varying BLOCK_SIZE/num_warps/num_stages\n"
          "- A launcher with a grid and a torch.allclose test\n- Target Blackwell SM120, float32")
U_RED = ("Write a Triton 3.7.1 kernel that computes {desc}.\n\nRequirements:\n"
         "- One program per row, boundary masking for arbitrary column counts\n"
         "- @triton.autotune (vary num_warps), fp32 accumulation\n"
         "- A launcher and a torch.allclose correctness test vs PyTorch\n- Target Blackwell SM120")
U_RMS = ("Write a Triton 3.7.1 RMSNorm kernel: out = x / sqrt(mean(x^2)+eps) * weight, row-wise "
         "over a 2D float32 tensor. Use @triton.autotune, boundary masking, fp32 accumulation, a "
         "launcher, and a torch.allclose test vs a PyTorch reference. Target Blackwell SM120.")


def _build_specs():
    specs = []
    for op, desc, expr, ref, init in UNARY:
        specs.append((op, U_ELEM.format(desc=desc), UNARY_TMPL, dict(op=op, expr=expr, ref=ref, init=INIT[init])))
    for op, desc, expr, ref in BINARY:
        bfix = '\nbt = torch.where(bt.abs() < 0.5, bt + 1.0, bt)' if op == "div" else ""
        specs.append((op, U_ELEM.format(desc=desc), BINARY_TMPL, dict(op=op, expr=expr, ref=ref, bfix=bfix)))
    for op, desc, other, reduce, ref in REDUCE:
        specs.append((op, U_RED.format(desc=desc), REDUCE_TMPL, dict(op=op, other=other, reduce=reduce, ref=ref)))
    for op, desc, other, reduce, elem, init, ref in NORM:
        specs.append((op, U_RED.format(desc=desc), NORM_TMPL,
                      dict(op=op, other=other, reduce=reduce, elem=elem, init=init, ref=ref)))
    specs.append(("rmsnorm", U_RMS, RMSNORM, {}))
    return specs


def _verify(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        env = dict(os.environ, TRITON_PRINT_AUTOTUNING="0")
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=240, env=env)
        return ("TRITONBENCH_PASS" in p.stdout), (p.stdout + p.stderr)[-800:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/processed/triton_kernels_sft.jsonl"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    specs = _build_specs()
    kept, failed = [], []
    for i, (op, user, tmpl, fields) in enumerate(specs):
        header = HEADER_ALTS[i % len(HEADER_ALTS)]
        code = tmpl.format(header=header, **fields)
        ok, log = _verify(code)
        if not ok:
            failed.append(op)
            print(f"  FAIL {op}: {log.splitlines()[-1] if log.strip() else 'no output'}", file=sys.stderr)
            continue
        preamble = PREAMBLES[i % len(PREAMBLES)]
        assistant = f"{preamble}\n\n```python\n{code}```"
        kept.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
        print(f"  PASS {op}", file=sys.stderr)

    with args.out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(kept)}/{len(specs)} verified kernels to {args.out}", file=sys.stderr)
    if failed:
        print(f"skipped (did not verify on this GPU): {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
