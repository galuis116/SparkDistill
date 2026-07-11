import json

from eval.registry_gate import (
    check_registry_duplicates,
    gate_registry_pr,
    parse_added_registry_lines,
    validate_registry_entry,
)


def _entry(**overrides):
    base = {
        "miner": "alice",
        "hf_url": "https://huggingface.co/datasets/org/sparkproof-triton-v0",
        "trajectories_sha256": "a" * 64,
        "rows_total": 128,
        "dataset_version": "triton-distill-v0.2",
    }
    base.update(overrides)
    return base


def test_parse_added_registry_lines():
    base = ""
    head = json.dumps(_entry()) + "\n"
    added = parse_added_registry_lines(base, head)
    assert len(added) == 1
    assert added[0]["miner"] == "alice"


def test_validate_registry_entry_requires_fields():
    issues = validate_registry_entry({"miner": "alice"})
    assert any("hf_url" in issue for issue in issues)


def test_check_registry_duplicates_rejects_repeat_sha():
    existing = [_entry()]
    issues = check_registry_duplicates(existing, [_entry(miner="bob")])
    assert any("duplicate trajectories_sha256" in issue for issue in issues)


def test_gate_registry_pr_rejects_multi_line_append():
    entry = json.dumps(_entry())
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=entry + "\n" + entry + "\n",
        sparkproof_root=__import__("pathlib").Path("."),
    )
    assert report["verified"] is False
    assert any("exactly one" in issue for issue in report["issues"])


def test_gate_registry_pr_rejects_schema_before_hf(monkeypatch):
    entry = json.dumps(_entry(rows_total=0))
    report = gate_registry_pr(
        base_registry_text="",
        head_registry_text=entry + "\n",
        sparkproof_root=__import__("pathlib").Path("."),
    )
    assert report["verified"] is False
    assert report["submissions"][0]["issues"]
