"""Automated gate for datasets/registry.jsonl pull requests.

Miners append one JSON line per HF dataset submission. This module parses the
added lines, rejects malformed or duplicate registry entries, and runs full
SparkProof production verification against each HF repo's `proof/` artifacts
before merge.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.dataset_verify import verify_dataset_submission

REGISTRY_PATH = Path("datasets/registry.jsonl")
REQUIRED_FIELDS = ("miner", "hf_url", "trajectories_sha256", "rows_total", "dataset_version")
_HF_REPO_RE = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/#?]+)")


def _load_registry_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_added_registry_lines(base_text: str, head_text: str) -> list[dict[str, Any]]:
    """Return JSON objects newly appended in `head_text` relative to `base_text`."""
    base_lines = {line.strip() for line in base_text.splitlines() if line.strip()}
    added: list[dict[str, Any]] = []
    for line_no, line in enumerate(head_text.splitlines(), start=1):
        line = line.strip()
        if not line or line in base_lines:
            continue
        try:
            added.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"added registry line {line_no}: invalid JSON: {exc}") from exc
    return added


def hf_repo_from_url(url: str) -> str:
    match = _HF_REPO_RE.match(url.strip())
    if not match:
        raise ValueError(f"hf_url must be https://huggingface.co/datasets/<org>/<repo>, got {url!r}")
    return match.group(1)


def validate_registry_entry(entry: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in REQUIRED_FIELDS:
        if not entry.get(field):
            issues.append(f"missing required field: {field}")
    if entry.get("trajectories_sha256") and len(str(entry["trajectories_sha256"])) != 64:
        issues.append("trajectories_sha256 must be a 64-char sha256 hex digest")
    try:
        rows_total = int(entry.get("rows_total", 0))
        if rows_total <= 0:
            issues.append("rows_total must be positive")
    except (TypeError, ValueError):
        issues.append("rows_total must be an integer")
    try:
        hf_repo_from_url(str(entry.get("hf_url", "")))
    except ValueError as exc:
        issues.append(str(exc))
    return issues


def check_registry_duplicates(
    existing: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    seen_hf = {hf_repo_from_url(row["hf_url"]) for row in existing if row.get("hf_url")}
    seen_sha = {row["trajectories_sha256"] for row in existing if row.get("trajectories_sha256")}

    for entry in new_entries:
        repo = hf_repo_from_url(entry["hf_url"])
        sha = entry["trajectories_sha256"]
        if repo in seen_hf:
            issues.append(f"duplicate hf_url repo already in registry: {repo}")
        if sha in seen_sha:
            issues.append(f"duplicate trajectories_sha256 already in registry: {sha}")
        seen_hf.add(repo)
        seen_sha.add(sha)
    return issues


def gate_registry_submission(
    entry: dict[str, Any],
    *,
    sparkproof_root: Path,
    existing_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one registry entry end-to-end."""
    issues = validate_registry_entry(entry)
    if existing_registry is not None:
        issues.extend(check_registry_duplicates(existing_registry, [entry]))

    hf_repo = hf_repo_from_url(entry["hf_url"]) if entry.get("hf_url") else ""
    report: dict[str, Any] = {
        "miner": entry.get("miner"),
        "hf_repo": hf_repo,
        "trajectories_sha256": entry.get("trajectories_sha256"),
        "issues": issues,
        "verified": False,
        "label": "dataset:REJECT",
    }
    if issues:
        return report

    verification = verify_dataset_submission(
        claimed_sha256=entry["trajectories_sha256"],
        sparkproof_root=sparkproof_root,
        hf_repo=hf_repo,
        production=True,
    )
    report.update(verification)
    if int(entry.get("rows_total", 0)) != verification.get("rows_total", -1):
        report["issues"] = list(report.get("issues") or []) + [
            f"rows_total mismatch: PR claims {entry['rows_total']} but verified bundle has {verification.get('rows_total')}"
        ]
        report["verified"] = False
        report["label"] = "dataset:REJECT"
    return report


def gate_registry_pr(
    *,
    base_registry_text: str,
    head_registry_text: str,
    sparkproof_root: Path,
) -> dict[str, Any]:
    """Gate every newly appended registry line in a PR."""
    existing = []
    for line in base_registry_text.splitlines():
        line = line.strip()
        if line:
            existing.append(json.loads(line))

    added = parse_added_registry_lines(base_registry_text, head_registry_text)
    if not added:
        return {
            "verified": False,
            "label": "dataset:REJECT",
            "issues": ["PR must append exactly one new line to datasets/registry.jsonl"],
            "submissions": [],
        }
    if len(added) != 1:
        return {
            "verified": False,
            "label": "dataset:REJECT",
            "issues": [f"PR must append exactly one registry line per submission (got {len(added)})"],
            "submissions": [],
        }

    report = gate_registry_submission(added[0], sparkproof_root=sparkproof_root, existing_registry=existing)
    return {
        "verified": report.get("verified", False),
        "label": report.get("label"),
        "issues": report.get("issues", []),
        "submissions": [report],
    }


def _git_show(ref: str, path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", default="origin/main", help="git ref for the merge base registry")
    parser.add_argument("--head-ref", default="HEAD", help="git ref for the PR head registry")
    parser.add_argument("--sparkproof-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--merge-on-pass",
        action="store_true",
        help="merge the current GitHub PR when verification passes (CI only)",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    args = parser.parse_args(argv)

    base_text = _git_show(args.base_ref, REGISTRY_PATH)
    head_text = _git_show(args.head_ref, REGISTRY_PATH)
    if not head_text.strip():
        head_path = REGISTRY_PATH
        if head_path.exists():
            head_text = head_path.read_text(encoding="utf-8")

    report = gate_registry_pr(
        base_registry_text=base_text,
        head_registry_text=head_text,
        sparkproof_root=args.sparkproof_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{report['label']} verified={report['verified']} issues={len(report.get('issues') or [])}",
        file=sys.stderr,
    )
    if report.get("issues"):
        for issue in report["issues"]:
            print(f"  - {issue}", file=sys.stderr)

    if args.merge_on_pass and report.get("verified") and args.pr_number is not None:
        merge = subprocess.run(
            ["gh", "pr", "merge", str(args.pr_number), "--merge"],
            capture_output=True,
            text=True,
            check=False,
        )
        if merge.returncode != 0:
            print(merge.stderr or merge.stdout, file=sys.stderr)
            return 1
        print(f"merged PR #{args.pr_number}", file=sys.stderr)

    return 0 if report.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
