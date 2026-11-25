"""Phase P5 Graduation Gate test suite for Pure-HCIR autonomy, latency & calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval_pure_hcir_autonomy import run_pure_hcir_graduation_eval


def test_pure_hcir_graduation_gate_autonomy_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies >= 98% Pure-HCIR parse/plan/realize autonomy and ECE < 0.05 hermetically."""
    # Guarantee hermetic execution even when git-ignored checkpoints/t1_ontology is absent
    missing_db = tmp_path / "nonexistent_lexical_ontology.db"
    monkeypatch.setattr("hbllm.hcir.stores.ontology_store.DEFAULT_ONTOLOGY_DB", missing_db)
    monkeypatch.setattr("hbllm.hcir.language.ontology_store.DEFAULT_ONTOLOGY_DB", missing_db)

    report = run_pure_hcir_graduation_eval()

    assert report["total_cases"] == 80
    assert report["autonomy"]["hcir_parse_autonomy_rate"] >= 0.98
    assert report["autonomy"]["hcir_plan_autonomy_rate"] >= 0.98
    assert report["autonomy"]["hcir_realize_autonomy_rate"] >= 0.98
    assert report["autonomy"]["can_retire_llm"] is True
    assert report["overall_pass_rate"] >= 0.95
    assert report["gates"]["p95_latency_lt_25ms"] is True
    assert report["calibration"]["ece"] < 0.05
    assert report["calibration"]["brier_score"] < 0.10
