"""Tests for held-out language suite and T4 extraction recall metric (M-6)."""

from __future__ import annotations

from hbllm.benchmarks.language_heldout_eval import run_language_heldout_eval
from hbllm.hcir.language.reading_curriculum import ReadingCurriculumIngestor


def test_reading_curriculum_evaluate_extraction_recall() -> None:
    ingestor = ReadingCurriculumIngestor()
    samples = [
        {
            "id": "tax_1",
            "text": "Gravity is a fundamental force.",
            "expected": [("gravity", "is_a", "force")],
        },
        {
            "id": "dep_1",
            "text": "Photosynthesis requires chlorophyll.",
            "expected": [("photosynthesis", "requires", "chlorophyll")],
        },
        {
            "id": "miss_1",
            "text": "Silent noise.",
            "expected": [("absent", "causes", "nothing")],
        },
    ]

    metrics = ingestor.evaluate_extraction_recall(samples)
    assert metrics["expected_count"] == 3
    assert metrics["matched"] >= 2
    assert metrics["recall"] >= 0.66
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert len(metrics["per_sample"]) == 3


def test_language_heldout_eval_hermetic() -> None:
    report = run_language_heldout_eval(enable_brain_llm=False)
    assert len(report["manifest_sha256"]) == 64
    assert report["total"] == 24
    assert report["passed"] >= 20, f"Held-out language failures: {report['failures']}"
    assert report["accuracy"] >= 0.83
