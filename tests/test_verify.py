from eval.verify import _no_student_endpoint_env, check_claim, check_training_claims


def test_check_claim_within_tolerance_has_no_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.885, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_check_claim_triton_compares_against_quick_subset():
    # A full-run composite (levels 1-4) legitimately differs from a level-1-only
    # re-run; the claim's triton_quick (same subset as the re-run) is the fair bar.
    claimed = {"triton": 0.55, "triton_quick": 0.82}
    rerun = {"triton": 0.815}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []
    # And a fabricated quick-subset claim still gets caught.
    assert check_claim({"triton": 0.55, "triton_quick": 0.95}, rerun, tolerance_pct=2.0) == ["triton"]


def test_check_claim_triton_falls_back_to_headline_without_quick():
    claimed = {"triton": 0.815}
    rerun = {"triton": 0.82}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_no_student_endpoint_env_hides_and_restores(monkeypatch):
    import os

    monkeypatch.setenv("SPARKDISTILL_STUDENT_ENDPOINT", "http://stale:8000/v1")
    with _no_student_endpoint_env():
        assert "SPARKDISTILL_STUDENT_ENDPOINT" not in os.environ
    assert os.environ["SPARKDISTILL_STUDENT_ENDPOINT"] == "http://stale:8000/v1"


def test_check_claim_beyond_tolerance_flags_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.70, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == ["gsm8k"]


def test_check_claim_ignores_benchmarks_not_claimed():
    claimed = {"gsm8k": 0.88}
    rerun = {"gsm8k": 0.88, "humaneval": 0.10}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_training_claims_within_budget_pass():
    manifest = {"train_hours": 4.5, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert check_training_claims(manifest, None) == []


def test_training_claims_over_budget_fail():
    manifest = {"train_hours": 6.0, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell"}
    issues = check_training_claims(manifest, None)
    assert any("budget" in issue for issue in issues)


def test_training_claims_wrong_gpu_fail():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA H100"}
    issues = check_training_claims(manifest, None)
    assert any("RTX PRO 6000" in issue for issue in issues)


def test_training_claims_absent_fields_do_not_fail():
    # Legacy bundles without training claims fall back to full retrain-verification.
    assert check_training_claims({}, None) == []


def test_training_claims_attestation_must_corroborate_gpu():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell"}
    attestation = {"passed": True, "claims": {"hwmodel": "GH100 A01 GSP BROM"}}
    issues = check_training_claims(manifest, attestation)
    assert any("corroborate" in issue for issue in issues)

    corroborating = {"passed": True, "claims": {"hwmodel": "GB202 RTX PRO 6000"}}
    assert check_training_claims(manifest, corroborating) == []
