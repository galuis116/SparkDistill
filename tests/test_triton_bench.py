import json

from eval.benchmarks import BENCHMARKS, run_benchmark
from eval.score import score
from eval.triton_bench import latest_report, summary_scores


def _report(composite=0.71, exec_pass=0.65, correctness=0.7, syntax=0.9):
    return {
        "summary": {
            "avg_composite": composite,
            "exec_pass_rate": exec_pass,
            "avg_correctness": correctness,
            "syntax_pass_rate": syntax,
            "avg_api_modernity": 0.8,
            "avg_perf_awareness": 0.5,
            "avg_gen_time_s": 12.0,
        },
        "num_problems": 20,
    }


def test_summary_scores_flattens_headline_and_submetrics():
    scores = summary_scores(_report())
    assert scores["triton"] == 0.71
    assert scores["triton_exec_pass_rate"] == 0.65
    assert scores["triton_correctness"] == 0.7
    assert scores["triton_syntax_pass_rate"] == 0.9


def test_summary_scores_empty_report_is_zero():
    assert summary_scores({})["triton"] == 0.0


def test_latest_report_picks_newest(tmp_path):
    (tmp_path / "tritonbench_m_20260101_000000.json").write_text(json.dumps(_report(composite=0.1)))
    (tmp_path / "tritonbench_m_20260201_000000.json").write_text(json.dumps(_report(composite=0.9)))
    assert latest_report(tmp_path)["summary"]["avg_composite"] == 0.9


def test_triton_registered_in_basket():
    assert "triton" in BENCHMARKS
    assert BENCHMARKS["triton"].metric == "avg_composite"


def test_run_benchmark_dispatches_triton_to_adapter(tmp_path, monkeypatch):
    calls = {}

    def fake_run(model_path, output_dir, limit=None, endpoint=None):
        calls["args"] = (model_path, output_dir, limit)
        return 0.42

    import eval.triton_bench as tb

    monkeypatch.setattr(tb, "run_triton_benchmark", fake_run)
    result = run_benchmark(BENCHMARKS["triton"], "outputs/student", tmp_path, limit=5)
    assert result == 0.42
    assert calls["args"] == ("outputs/student", tmp_path, 5)


def test_score_tiers_triton_improvement():
    candidate = {"triton": 0.71, "gsm8k": 0.88}
    frontier = {"triton": 0.60, "gsm8k": 0.88}
    report = score(candidate, frontier)
    assert report["label"] == "eval:XL"  # 18.3% relative improvement on triton
    assert report["best_benchmark"] == "triton"


def test_score_flags_triton_regression():
    candidate = {"triton": 0.50, "gsm8k": 0.90}
    frontier = {"triton": 0.60, "gsm8k": 0.88}
    report = score(candidate, frontier)
    assert "regression-triton" in report["regressions"]
    assert report["label"] == "eval:REJECT"
