"""Unit tests for validator and evaluator (no GPU)."""

from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.validator import TritonValidator


def test_extract_code_from_markdown():
    v = TritonValidator()
    text = "Here:\n```python\nx = 1\n```"
    assert "x = 1" in v.extract_code(text)


def test_syntax_check():
    v = TritonValidator()
    assert v.check_syntax("def f():\n    return 1\n")
    assert not v.check_syntax("def f(\n")


def test_api_check_requires_jit():
    v = TritonValidator()
    code = "x = 1\n"
    r = v.check_triton_api(code)
    assert r["modern"] is False
    assert any("jit" in i.lower() for i in r["issues"])


def test_evaluator_composite_in_range():
    e = TritonEvaluator()
    scores = e.score(
        problem={},
        generated_code="@triton.jit\ndef k(): pass\ndef launch(): grid=1\n",
        exec_pass=False,
        exec_output="",
    )
    assert 0.0 <= scores["composite_score"] <= 1.0
