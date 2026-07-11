"""Triton domain benchmark: score a student checkpoint on TritonBench.

The general basket (BFCL, GSM8K, ...) can't measure Triton expertise — it only guards
against catastrophic forgetting. This adapter is the improvement signal for the Triton
domain: it serves the student checkpoint behind an OpenAI-compatible endpoint (vLLM),
runs the vendored TritonBench harness against it (each generated kernel is compiled and
executed on the GPU, outputs checked against a reference), and reports the summary's
`avg_composite` as the headline metric for `eval.score` tiering.

Eval hygiene: TritonBench problems are quarantined from training data — SparkProof's
release gate blocks any generated row whose AST structure, prompt, or task fingerprint
matches them (see SparkProof's `decontaminate.py`). This benchmark must only ever run
TritonBench's own problems, never prompts derived from miner datasets.

Determinism caveats: pin the same GPU/driver as the frontier run (speed-derived scores
are hardware-sensitive), and keep the config's temperature/max_tokens unchanged between
candidate and frontier runs.

    # existing endpoint (e.g. vLLM already serving the checkpoint):
    python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 \\
        --endpoint http://localhost:8000/v1 --out eval/results/triton.json

    # or let it serve the checkpoint itself (requires `vllm` on PATH):
    python -m eval.triton_bench --checkpoint outputs/qwen3.5-4b-phase1 --serve \\
        --out eval/results/triton.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

HEADLINE_METRIC = "avg_composite"
_QUICK_LEVELS = [1]  # cheap re-verification, mirrors configs/eval_quick.yaml
_FULL_LEVELS = [1, 2, 3, 4]


def tritonbench_root() -> Path:
    configured = os.environ.get("SPARKDISTILL_TRITONBENCH_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "tritonbench"


def summary_scores(report: dict) -> dict[str, float]:
    """Flatten a TritonBench report's summary into the score fields we track.

    `triton` (the headline, used by eval.score) is the average composite score;
    the rest are kept alongside it in the detail json for human review.
    """
    summary = report.get("summary") or {}
    return {
        "triton": float(summary.get(HEADLINE_METRIC, 0.0)),
        "triton_exec_pass_rate": float(summary.get("exec_pass_rate", 0.0)),
        "triton_correctness": float(summary.get("avg_correctness", 0.0)),
        "triton_syntax_pass_rate": float(summary.get("syntax_pass_rate", 0.0)),
    }


def latest_report(results_dir: Path) -> dict:
    reports = sorted(results_dir.glob("tritonbench_*.json"))
    if not reports:
        raise FileNotFoundError(f"TritonBench produced no results JSON under {results_dir}")
    return json.loads(reports[-1].read_text())


def _endpoint_ready(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


@contextmanager
def serve_checkpoint(model_path: str, port: int = 8000, startup_timeout_s: int = 600):
    """Serve `model_path` via `vllm serve` and yield the OpenAI-compatible endpoint.

    Requires the `vllm` CLI on PATH (installed separately, like lm-eval and Axolotl).
    """
    endpoint = f"http://127.0.0.1:{port}/v1"
    proc = subprocess.Popen(
        ["vllm", "serve", model_path, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + startup_timeout_s
        while not _endpoint_ready(endpoint):
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early (code {proc.returncode})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"vllm serve not ready after {startup_timeout_s}s")
            time.sleep(5)
        yield endpoint
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_tritonbench(
    endpoint: str,
    model_name: str,
    results_dir: Path,
    levels: list[int],
    bench_root: Path | None = None,
    timeout_s: int = 7200,
) -> dict:
    root = bench_root or tritonbench_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"TritonBench not found at {root} — set SPARKDISTILL_TRITONBENCH_ROOT or vendor it"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tritonbench.cli",
        "eval",
        "--config",
        "configs/eval_quick.yaml",
        "--endpoint",
        endpoint,
        "--model",
        model_name,
        "--levels",
        *[str(level) for level in levels],
        "--output",
        str(results_dir.resolve()),
    ]
    subprocess.run(command, cwd=root, check=True, timeout=timeout_s)
    return latest_report(results_dir)


def run_triton_benchmark(
    model_path: str,
    output_dir: Path,
    limit: int | None = None,
    endpoint: str | None = None,
) -> float:
    """Adapter used by `eval.benchmarks.run_benchmark` for the `triton` entry.

    A non-None `limit` (the harness's cheap re-verification mode) maps to the
    level-1-only quick run; a full run covers all levels. Endpoint resolution:
    explicit argument, then SPARKDISTILL_STUDENT_ENDPOINT, else serve the
    checkpoint locally with vLLM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "_tritonbench"
    levels = _QUICK_LEVELS if limit is not None else _FULL_LEVELS
    model_name = Path(model_path).name or model_path

    endpoint = endpoint or os.environ.get("SPARKDISTILL_STUDENT_ENDPOINT")
    if endpoint:
        report = run_tritonbench(endpoint, model_name, results_dir, levels)
    else:
        with serve_checkpoint(model_path) as served:
            report = run_tritonbench(served, model_name, results_dir, levels)

    scores = summary_scores(report)
    (output_dir / "triton.json").write_text(
        json.dumps({"scores": scores, "levels": levels, "report": report}, indent=2)
    )
    return scores["triton"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local path or HF hub id of the student checkpoint")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible endpoint already serving the checkpoint")
    parser.add_argument("--serve", action="store_true", help="serve the checkpoint locally with vLLM")
    parser.add_argument("--quick", action="store_true", help="level-1-only quick run (cheap re-verification)")
    parser.add_argument("--work-dir", type=Path, default=Path("eval/results/_work"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.endpoint and not args.serve and not os.environ.get("SPARKDISTILL_STUDENT_ENDPOINT"):
        parser.error("pass --endpoint, set SPARKDISTILL_STUDENT_ENDPOINT, or use --serve")

    headline = run_triton_benchmark(
        args.checkpoint,
        args.work_dir,
        limit=1 if args.quick else None,
        endpoint=args.endpoint,
    )
    detail = json.loads((args.work_dir / "triton.json").read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoint": args.checkpoint, "scores": detail["scores"]}, indent=2))
    print(f"triton={headline:.3f} — wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
