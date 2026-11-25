"""Tests for suite-named dev reports, dev LEDGER.md, and widened R1 ablation registration (H-5, L-6)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.ablation.faculty_ablation import WIDENED_BENCH_ENVS, _load_env_class
from hbllm.benchmarks.protocol import EvaluationProtocol, TaskResult


def test_suite_named_report_and_dev_ledger(tmp_path: Path) -> None:
    proto = EvaluationProtocol(reports_dir=tmp_path, git_sha="deadbeef1234", eval_tag="eval-test")
    tasks = [
        TaskResult(
            task_id="stochastic_wind_dev_system",
            family="stochastic_dynamics",
            split="dev",
            steps=18,
            max_budget=120,
            success_at_budget=True,
            zero_shot={"successes": [True, False], "steps_per_seed": [18, 120]},
        )
    ]

    report = proto.compute_report(split="dev", tasks=tasks, suite_name="widened_dev")
    assert report.suite_name == "widened_dev"

    dev_dir = tmp_path / "dev"
    widened_files = list(dev_dir.glob("dev_widened_dev_handcrafted_*.json"))
    default_dev_files = list(dev_dir.glob("dev_handcrafted_*.json"))
    assert len(widened_files) == 1
    assert len(default_dev_files) == 0

    ledger = dev_dir / "LEDGER.md"
    assert ledger.exists()
    text = ledger.read_text(encoding="utf-8")
    assert "widened_dev" in text
    assert widened_files[0].name in text


def test_widened_r1_ablation_env_roster_imports_cleanly() -> None:
    assert len(WIDENED_BENCH_ENVS) == 6
    for _, cls_path, budget in WIDENED_BENCH_ENVS:
        cls = _load_env_class(cls_path)
        assert callable(cls)
        assert budget > 0
