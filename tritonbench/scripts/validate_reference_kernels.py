#!/usr/bin/env python3
"""Smoke-test TritonBench reference kernels under the pinned Triton runtime.

Runs each gold file in TritonBench-G (and optionally T) as a subprocess on one GPU.
Use after upgrading Triton to catch compile/runtime breakages.

  cd tritonbench
  uv run python scripts/validate_reference_kernels.py --channel G --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_config import DATA_G_DIR, DATA_T_DIR, PY_INTERPRETER, REPO_ROOT, check_runtime_versions, require_blackwell_gpu


def _run_file(py_path: Path, gpu: int, timeout: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        proc = subprocess.run(
            [PY_INTERPRETER, str(py_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
            env=env,
        )
        return {
            "file": py_path.name,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"file": py_path.name, "ok": False, "returncode": -1, "stderr_tail": "timeout"}


def validate_channel(channel: str, gpu: int, limit: int | None, timeout: int) -> dict:
    folder = DATA_G_DIR if channel == "G" else DATA_T_DIR
    files = sorted(folder.glob("*.py"))
    if limit is not None:
        files = files[:limit]

    results = [_run_file(p, gpu, timeout) for p in files]
    ok = sum(1 for r in results if r["ok"])
    return {
        "channel": channel,
        "total": len(results),
        "passed": ok,
        "failed": len(results) - ok,
        "pass_rate": round(100.0 * ok / len(results), 2) if results else 0.0,
        "failures": [r for r in results if not r["ok"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=["G", "T", "both"], default="G")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=None, help="write JSON report")
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="debug only — do not enforce Blackwell target",
    )
    args = parser.parse_args()

    versions = check_runtime_versions()
    print(json.dumps({"runtime": versions}, indent=2))

    gpu_info = None
    if not args.skip_gpu_check:
        try:
            gpu_info = require_blackwell_gpu(args.gpu)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps({"gpu": gpu_info}, indent=2))

    reports = []
    channels = ["G", "T"] if args.channel == "both" else [args.channel]
    for ch in channels:
        report = validate_channel(ch, args.gpu, args.limit, args.timeout)
        reports.append(report)
        print(
            f"TritonBench-{ch}: {report['passed']}/{report['total']} passed "
            f"({report['pass_rate']}%)",
            flush=True,
        )
        for fail in report["failures"][:10]:
            print(f"  FAIL {fail['file']}: {fail['stderr_tail'][:200]}", flush=True)

    if args.out:
        payload = {"runtime": versions, "reports": reports}
        if gpu_info:
            payload["gpu"] = gpu_info
        args.out.write_text(json.dumps(payload, indent=2))

    return 0 if all(r["failed"] == 0 for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
