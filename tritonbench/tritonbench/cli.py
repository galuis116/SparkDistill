"""CLI for TritonBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tritonbench.core.reporter import Reporter
from tritonbench.core.runner import BenchConfig, TritonBench
from tritonbench.features.triton_371 import DEFAULT_GPU_TARGET, TRITON_VERSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TritonBench — Triton 3.7.1 kernel programming benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run evaluation against an OpenAI-compatible model")
    eval_p.add_argument("--config", default="configs/eval_quick.yaml")
    eval_p.add_argument("--model", help="Override model name")
    eval_p.add_argument("--endpoint", help="Override API endpoint")
    eval_p.add_argument("--levels", nargs="+", type=int, default=None)
    eval_p.add_argument("--output", default=None, help="Override output directory")

    rep_p = sub.add_parser("report", help="Print summary from a results JSON file")
    rep_p.add_argument("--results", required=True)

    args = parser.parse_args(argv)

    if args.command == "eval":
        cfg_path = Path(args.config)
        cfg_dict = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        if args.model:
            cfg_dict["model_name"] = args.model
        if args.endpoint:
            cfg_dict["model_endpoint"] = args.endpoint
        if args.levels is not None:
            cfg_dict["levels"] = args.levels
        if args.output:
            cfg_dict["output_dir"] = args.output
        config = BenchConfig(**cfg_dict)
        bench = TritonBench(config)
        bench.run()
        return 0

    if args.command == "report":
        data = json.loads(Path(args.results).read_text())
        Reporter(".").print_summary(data)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
