from __future__ import annotations

from pathlib import Path

import pytest

from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult


def test_config_hash_deterministic():
    cfg1 = {"model": "deterministic_hcir", "version": "1.0.0", "seed": 42}
    cfg2 = {"version": "1.0.0", "seed": 42, "model": "deterministic_hcir"}
    h1 = EvaluationProtocol._compute_config_hash(cfg1)
    h2 = EvaluationProtocol._compute_config_hash(cfg2)
    assert h1 == h2
    assert len(h1) == 16


def test_task_evaluation_and_metrics(tmp_path: Path):
    protocol = EvaluationProtocol(reports_dir=tmp_path, git_sha="test_sha_123")

    def mock_run_fn():
        # returns: success, steps, latencies, errors
        return True, 25, [1.2, 2.5, 3.1, 4.0, 10.5], []

    task_res = protocol.evaluate_task(
        task_id="grid_nav_test",
        family="grid",
        split="dev",
        run_fn=mock_run_fn,
        max_budget=100,
    )

    assert task_res.success_at_budget is True
    assert task_res.steps == 25
    assert task_res.steps_to_first_success == 25
    assert task_res.p50_latency_ms == 3.1
    assert task_res.p95_latency_ms == 10.5


def test_compute_report_and_generality_score(tmp_path: Path):
    protocol = EvaluationProtocol(reports_dir=tmp_path, git_sha="test_sha_123")

    tasks = [
        TaskResult(
            task_id="grid_1",
            family="grid",
            split="dev",
            steps=20,
            max_budget=100,
            success_at_budget=True,
            p50_latency_ms=2.0,
            p95_latency_ms=4.0,
        ),
        TaskResult(
            task_id="continuous_1",
            family="continuous",
            split="dev",
            steps=50,
            max_budget=100,
            success_at_budget=True,
            p50_latency_ms=5.0,
            p95_latency_ms=8.0,
        ),
        TaskResult(
            task_id="symbolic_1",
            family="symbolic",
            split="dev",
            steps=100,
            max_budget=100,
            success_at_budget=False,
            p50_latency_ms=1.0,
            p95_latency_ms=2.0,
        ),
    ]

    report = protocol.compute_report(split="dev", tasks=tasks)
    assert report.split == "dev"
    assert "grid" in report.family_scores
    assert "continuous" in report.family_scores
    assert "symbolic" in report.family_scores
    assert report.family_scores["symbolic"] == 0.0
    assert report.generality_score > 0.0

    dev_files = list((tmp_path / "dev").glob("dev_*.json"))
    assert len(dev_files) == 1


def test_heldout_anti_tamper_gate_refuses_rerun(tmp_path: Path):
    git_sha = "abcd1234ef5678"
    protocol = EvaluationProtocol(reports_dir=tmp_path, git_sha=git_sha)

    # First run on heldout
    tasks = [
        TaskResult(
            task_id="heldout_1",
            family="grid",
            split="heldout",
            steps=10,
            max_budget=50,
            success_at_budget=True,
        )
    ]
    report = protocol.compute_report(split="heldout", tasks=tasks)
    assert report.git_sha == git_sha

    # Verify report written
    heldout_files = list((tmp_path / "heldout").glob(f"{git_sha}_*.json"))
    assert len(heldout_files) == 1

    # Attempting to verify or run heldout again on same git_sha MUST raise RuntimeError
    protocol_rerun = EvaluationProtocol(reports_dir=tmp_path, git_sha=git_sha, eval_tag="eval-v2")
    with pytest.raises(RuntimeError, match="Held-out evaluation already exists for git commit"):
        protocol_rerun.verify_heldout_gate()


def test_heldout_gate_requires_eval_tag_and_writes_ledger(tmp_path: Path):
    """Held-out gate rejects untagged commits and appends to LEDGER.md on tagged runs."""
    untagged = EvaluationProtocol(reports_dir=tmp_path, git_sha="deadbeef00001111", eval_tag="")
    with pytest.raises(RuntimeError, match="not tagged with 'eval-\\*'"):
        untagged.verify_heldout_gate(require_eval_tag=True)

    tagged = EvaluationProtocol(
        reports_dir=tmp_path, git_sha="cafebabe22223333", eval_tag="eval-phase2-m1"
    )
    task = TaskResult(
        task_id="ho_v2_1",
        family="partial_observability",
        split="heldout",
        steps=24,
        max_budget=140,
        seed_count=20,
        success_at_budget=True,
        zero_shot={
            "n": 20,
            "successes": [True] * 16 + [False] * 4,
            "steps_per_seed": [24] * 16 + [140] * 4,
        },
    )
    report = tagged.compute_report(split="heldout", tasks=[task])
    assert report.zero_shot["n_trials"] == 20
    assert report.zero_shot["solve_rate"] == 0.8
    ci = report.zero_shot["wilson_95_ci"]
    assert len(ci) == 2 and 0.55 < ci[0] < 0.80 < ci[1] < 0.95
    ledger = (tmp_path / "heldout" / "LEDGER.md").read_text(encoding="utf-8")
    assert "eval-phase2-m1" in ledger
    assert "cafebabe" in ledger


def test_family_scores_match_zero_shot_headline(tmp_path: Path):
    """EvaluationReport.family_scores matches zero-shot headline while lifelong stays separate."""
    protocol = EvaluationProtocol(reports_dir=tmp_path, git_sha="test_zs_lf_parity")
    tasks = [
        TaskResult(
            task_id="grid_1",
            family="grid",
            split="dev",
            steps=10,
            max_budget=100,
            success_at_budget=True,
            zero_shot={"success": False, "steps": 100},
            lifelong={"success": True, "steps": 10},
        ),
        TaskResult(
            task_id="cont_1",
            family="continuous",
            split="dev",
            steps=20,
            max_budget=100,
            success_at_budget=True,
            zero_shot={"success": True, "steps": 20},
            lifelong={"success": True, "steps": 20},
        ),
    ]
    report = protocol.compute_report(split="dev", tasks=tasks)
    assert report.family_scores == report.zero_shot["family_scores"]
    assert report.family_scores["grid"] == 0.0
    assert report.lifelong["family_scores"]["grid"] > 0.0
    assert report.generality_score == pytest.approx(
        sum(report.family_scores.values()) / len(report.family_scores), abs=1e-4
    )
    assert report.lifelong_generality_score > report.generality_score
