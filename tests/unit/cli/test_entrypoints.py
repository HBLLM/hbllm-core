from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hbllm.cli.main import app

runner = CliRunner()


def test_cli_help_displays_root_tree():
    """Verify that hbllm --help displays the canonical command tree."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "kernel" in result.stdout
    assert "mind" in result.stdout
    assert "bench" in result.stdout
    assert "diagnostics" in result.stdout
    assert "info" in result.stdout


def test_cli_kernel_status_reports_verified_pipeline():
    """Verify that hbllm kernel status displays verified status and 4 tiers."""
    result = runner.invoke(app, ["kernel", "status"])
    assert result.exit_code == 0
    assert "ONLINE" in result.stdout
    assert "verified=True" in result.stdout
    assert "ConstitutionalVerifier" in result.stdout
    assert "working, brain, persistent, meta" in result.stdout


def test_cli_kernel_boot_creates_verified_system():
    """Verify that hbllm kernel boot initializes verified kernel services."""
    result = runner.invoke(app, ["kernel", "boot", "--world-id", "test_cli_world"])
    assert result.exit_code == 0
    assert "Kernel booted successfully" in result.stdout
    assert "Verified TM: True" in result.stdout


def test_cli_mind_step_evaluates_observation_fixture(tmp_path: Path):
    """Verify that hbllm mind step consumes an EmbodiedObservation and prints an action."""
    obs_file = tmp_path / "obs.json"
    obs_file.write_text(
        json.dumps(
            {
                "lattice": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                ],
                "available_actions": [1, 2, 3, 4],
                "context_id": "cli_test_step",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["mind", "step", "--obs", str(obs_file), "--world-id", "cli_test"])
    assert result.exit_code == 0
    assert "Action: " in result.stdout


def test_cli_diagnostics_runs_and_displays_cognitive_status():
    """Verify that hbllm diagnostics runs and reports hardware, Rust, and cognitive subsystems."""
    result = runner.invoke(app, ["diagnostics"])
    assert result.exit_code == 0
    assert "HBLLM Runtime Diagnostics" in result.stdout
    assert "Cognitive OS & Subsystems:" in result.stdout
    assert "Constitutional Verifier" in result.stdout
    assert "Memory Tiers" in result.stdout


def test_cli_info_displays_canonical_invariants():
    """Verify that hbllm info displays core invariants and cortical loop."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Invariants:" in result.stdout
    assert "Cortical Epistemic Loop:" in result.stdout
    assert "Zero LLM dependency in Cognitive Core" in result.stdout


def test_cli_bench_help_lists_subcommands():
    """Verify that hbllm bench --help lists dev, heldout, local-suite, sge, ablation."""
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "dev" in result.stdout
    assert "heldout" in result.stdout
    assert "local-suite" in result.stdout
    assert "sge" in result.stdout
    assert "ablation" in result.stdout
