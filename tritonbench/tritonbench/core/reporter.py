"""Generate evaluation reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class Reporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: list[dict], config: Any) -> dict:
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": config.model_name,
            "triton_version": config.triton_version,
            "gpu_target": config.gpu_target,
            "num_problems": len(results),
            "summary": self._summarize(results),
            "by_level": self._by_level(results),
            "by_metric": self._by_metric(results),
            "details": results,
        }
        out_path = self.output_dir / f"tritonbench_{config.model_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.write_text(json.dumps(report, indent=2))
        self.print_summary(report)
        return report

    def print_summary(self, report: dict) -> None:
        s = report["summary"]
        print("\n" + "=" * 60)
        print(f"  TRITONBENCH — {report['model']}")
        print(f"  Triton {report['triton_version']} | GPU: {report['gpu_target']}")
        print("=" * 60)
        print(f"  Problems:        {report['num_problems']}")
        print(f"  Exec pass rate:  {s['exec_pass_rate']:.1%}")
        print(f"  Syntax pass:     {s['syntax_pass_rate']:.1%}")
        print(f"  Composite:       {s['avg_composite']:.3f}")
        print(f"  Correctness:     {s['avg_correctness']:.3f}")
        print(f"  API modernity:   {s['avg_api_modernity']:.3f}")
        print("-" * 60)
        for lvl, data in report["by_level"].items():
            print(
                f"    L{lvl}: pass={data['exec_pass_rate']:.1%}  "
                f"composite={data['avg_composite']:.3f}  (n={data['count']})"
            )
        print("=" * 60)

    def _summarize(self, results: list[dict]) -> dict:
        n = len(results) or 1
        return {
            "exec_pass_rate": sum(1 for r in results if r["exec_pass"]) / n,
            "syntax_pass_rate": sum(1 for r in results if r["syntax_ok"]) / n,
            "avg_composite": sum(r["composite_score"] for r in results) / n,
            "avg_correctness": sum(r["correctness"] for r in results) / n,
            "avg_api_modernity": sum(r["api_modernity"] for r in results) / n,
            "avg_perf_awareness": sum(r["perf_awareness"] for r in results) / n,
            "avg_gen_time_s": sum(r["gen_time_s"] for r in results) / n,
        }

    def _by_level(self, results: list[dict]) -> dict:
        levels: dict[str, list] = {}
        for r in results:
            levels.setdefault(str(r["level"]), []).append(r)
        return {
            lvl: {
                "count": len(rs),
                "exec_pass_rate": sum(1 for r in rs if r["exec_pass"]) / len(rs),
                "avg_composite": sum(r["composite_score"] for r in rs) / len(rs),
            }
            for lvl, rs in levels.items()
        }

    def _by_metric(self, results: list[dict]) -> dict:
        metrics = ["correctness", "api_modernity", "perf_awareness", "completeness", "code_quality"]
        n = len(results) or 1
        return {m: sum(r[m] for r in results) / n for m in metrics}
