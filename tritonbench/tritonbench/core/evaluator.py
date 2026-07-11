"""Score generated Triton kernels."""

from __future__ import annotations

import re
from typing import Any


class TritonEvaluator:
    def __init__(self, gpu_target: str = "Blackwell-SM120"):
        self.gpu = gpu_target

    def score(
        self,
        problem: dict,
        generated_code: str,
        exec_pass: bool,
        exec_output: str,
    ) -> dict[str, float]:
        scores = {
            "correctness": self._score_correctness(exec_pass, generated_code),
            "api_modernity": self._score_api_modernity(generated_code),
            "perf_awareness": self._score_performance(generated_code),
            "completeness": self._score_completeness(problem, generated_code),
            "code_quality": self._score_quality(generated_code),
        }
        weights = {
            "correctness": 0.35,
            "api_modernity": 0.20,
            "perf_awareness": 0.20,
            "completeness": 0.15,
            "code_quality": 0.10,
        }
        scores["composite_score"] = sum(scores[k] * weights[k] for k in weights)
        return scores

    def _score_correctness(self, exec_pass: bool, code: str) -> float:
        """Score numerical correctness of an executed kernel.

        exec_pass only proves the script ran without raising. Full credit requires
        the generated code to contain a reference comparison — an executed script
        whose torch.allclose/assert_close check did not raise means the kernel's
        output actually matched the reference. Merely running, with nothing
        checked, is not evidence of a correct kernel.
        """
        if not exec_pass:
            return 0.0
        if "torch.allclose" in code or "torch.testing.assert_close" in code:
            return 1.0
        if re.search(r"^\s*assert\b", code, re.MULTILINE):
            return 0.8
        return 0.5

    def _score_api_modernity(self, code: str) -> float:
        score = 0.5
        for pattern, delta in {
            "tl.make_tensor_descriptor": 0.15,
            "tl.float8e4nv": 0.10,
            "tl.associative_scan": 0.10,
            "num_stages": 0.05,
            "input_precision": 0.05,
        }.items():
            if pattern in code:
                score += delta
        for pattern, delta in {"tl.make_block_ptr": -0.15, "tl.advance": -0.10}.items():
            if pattern in code:
                score += delta
        return max(0.0, min(1.0, score))

    def _score_performance(self, code: str) -> float:
        checks = [
            "@triton.autotune" in code,
            bool(re.search(r"BLOCK_[MNK]\s*[=:]\s*\d+", code)),
            "num_stages" in code,
            "num_warps" in code,
            "tl.dot(" in code,
            any(h in code.lower() for h in ("blackwell", "sm_120", "sm_121", "rtx 50", "sm120")),
        ]
        return sum(1.0 for c in checks if c) / len(checks)

    def _score_completeness(self, problem: dict, code: str) -> float:
        score = 0.0
        for check, weight in [
            ("@triton.jit" in code, 0.25),
            ("def " in code and "grid" in code, 0.25),
            ("torch.allclose" in code or "assert" in code, 0.25),
            ("@triton.autotune" in code, 0.25),
        ]:
            if check:
                score += weight
        # Problems can pin required patterns (masking, tl.load/tl.store, ...);
        # a solution missing them is incomplete no matter how well it runs.
        required = problem.get("required_patterns") or []
        if required:
            present = sum(1 for pattern in required if pattern in code)
            score = 0.5 * score + 0.5 * (present / len(required))
        return score

    def _score_quality(self, code: str) -> float:
        score = 0.5
        if '"""' in code or "'''" in code or code.count("#") >= 3:
            score += 0.2
        if ": int" in code or "tl.constexpr" in code:
            score += 0.15
        lines = len(code.strip().split("\n"))
        if 20 <= lines <= 200:
            score += 0.15
        return min(1.0, score)
