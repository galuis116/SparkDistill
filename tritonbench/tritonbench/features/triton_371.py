"""Triton 3.7.1 feature registry for benchmark design and training prompts."""

from __future__ import annotations

TRITON_VERSION = "3.7.1"

# SparkDistill targets workstation Blackwell (SM12x) by default — not datacenter SM10x TMEM/tcgen05.
DEFAULT_GPU_TARGET = "Blackwell-SM120"

TRITON_371_FEATURES: dict = {
    "core_api": {
        "tensor_descriptors": {
            "ops": [
                "tl.make_tensor_descriptor",
                "desc.load(offsets)",
                "desc.store(offsets, value)",
            ],
            "replaces": "tl.make_block_ptr (deprecated in 3.7)",
            "hardware": "Preferred load/store path on Triton 3.7+",
            "priority": "critical",
        },
        "dot_operations": {
            "ops": [
                "tl.dot(a, b, acc, out_dtype=tl.float32)",
                "tl.dot(a, b, acc, input_precision='tf32')",
                "tl.dot(a, b, acc, input_precision='ieee')",
            ],
            "hardware": "Workstation Blackwell: extended mma.sync; datacenter SM10x: tcgen05/TMEM",
            "priority": "critical",
        },
        "data_types": {
            "fp8": ["tl.float8e4nv", "tl.float8e5m2", "tl.float8e4b8", "tl.float8e5b16"],
            "fp4": ["tl.float4e2m1"],
            "standard": ["tl.float16", "tl.bfloat16", "tl.float32", "tl.int8", "tl.int32"],
            "priority": "critical",
        },
        "scan_reduce": {
            "ops": ["tl.associative_scan", "tl.reduce", "tl.cumsum"],
            "priority": "high",
        },
    },
    "decorators": {
        "jit": "@triton.jit",
        "autotune": "@triton.autotune(configs, key)",
        "heuristics": "@triton.heuristics(values)",
    },
    "compiler_features": {
        "num_stages": "Pipeline depth",
        "num_warps": "Warp count",
        "num_ctas": "CTA clusters (when supported)",
    },
    "targets": {
        "sparkdistill_default": ["sm_120", "sm_121"],
        "sparkdistill_datacenter": ["sm_100", "sm_103"],
    },
}


def system_prompt_for_triton(*, triton_version: str = TRITON_VERSION, gpu_target: str = DEFAULT_GPU_TARGET) -> str:
    return f"""You are a Triton {triton_version} GPU kernel expert.
Target GPU: {gpu_target} (workstation Blackwell, CUDA SM12x).

Write complete, runnable code with:
1. @triton.jit kernel(s)
2. Python launcher with grid
3. @triton.autotune when tile sizes matter
4. Correctness test vs PyTorch (torch.allclose)
5. Optional benchmark via triton.testing.do_bench

Prefer tl.make_tensor_descriptor over tl.make_block_ptr (deprecated in 3.7).
Use fp32 accumulators for tl.dot. Include boundary masks for non-aligned sizes."""
