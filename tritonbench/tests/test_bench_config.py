from pathlib import Path

import pytest

from bench_config import (
    BLACKWELL_SM_MAJOR,
    REPO_ROOT,
    STATS_G_PATH,
    STATS_T_PATH,
    TARGET_GPU_FAMILY,
    blackwell_profile,
    is_blackwell_capability,
    load_json_records,
    parse_gpus,
)


def test_parse_gpus_formats():
    assert parse_gpus("0") == [0]
    assert parse_gpus("0,1,2") == [0, 1, 2]
    assert parse_gpus("[0, 1]") == [0, 1]


def test_default_profile_is_workstation(monkeypatch):
    monkeypatch.delenv("TRITONBENCH_BLACKWELL_PROFILE", raising=False)
    assert blackwell_profile() == "workstation"


def test_blackwell_capability():
    assert TARGET_GPU_FAMILY == "blackwell"
    assert is_blackwell_capability(10, 0)  # B200
    assert is_blackwell_capability(12, 0)  # RTX 5090
    assert not is_blackwell_capability(9, 0)  # Hopper
    assert not is_blackwell_capability(8, 0)  # Ampere
    assert BLACKWELL_SM_MAJOR == frozenset({10, 12})


def test_stats_files_exist():
    assert STATS_G_PATH.exists()
    assert STATS_T_PATH.exists()


def test_load_g_stats():
    records = load_json_records(STATS_G_PATH)
    assert len(records) == 184
    assert "file" in records[0]


def test_load_t_stats():
    records = load_json_records(STATS_T_PATH)
    assert len(records) == 166
    assert "file" in records[0]


def test_g_reference_dir_count():
    g_dir = REPO_ROOT / "data" / "TritonBench_G_v1"
    assert len(list(g_dir.glob("*.py"))) == 184


def test_t_reference_dir_count():
    t_dir = REPO_ROOT / "data" / "TritonBench_T_v1"
    assert len(list(t_dir.glob("*.py"))) == 166
