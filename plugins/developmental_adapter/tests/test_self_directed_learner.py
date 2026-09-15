"""Unit tests for Autonomous Epistemic Self-Directed Learner."""

from __future__ import annotations

import pytest

from plugins.developmental_adapter.school import CognitiveSchool
from plugins.developmental_adapter.self_directed_learner import (
    EpistemicKnowledgeGap,
    KnowledgeGapDetector,
    KnowledgeGapType,
    SelfDirectedReadingEngine,
    binary_entropy,
)
from plugins.developmental_adapter.types import BabyActionType, CausalHypothesis


def test_binary_entropy_calculation() -> None:
    """Verify Shannon entropy reaches maximum of 1.0 bit at p=0.5 and decays at extremes."""
    assert binary_entropy(0.5) == pytest.approx(1.0, abs=1e-4)
    # Symmetry
    assert binary_entropy(0.1) == pytest.approx(binary_entropy(0.9), abs=1e-4)
    assert binary_entropy(0.01) < 0.1
    assert binary_entropy(0.99) < 0.1


def test_knowledge_gap_detection() -> None:
    """Verify detection of causal gaps (p=0.5) and lexical deficits (unfamiliar tokens)."""
    student = CognitiveSchool().student

    # Inject candidate hypothesis at maximum uncertainty (p=0.5)
    hyp = CausalHypothesis(
        hypothesis_id="hyp_candle_combustion",
        action=BabyActionType.PUSH,
        variable="combustion",
        operator="==",
        value="active",
        consequence="HEAT",
        confidence=0.5,
    )
    student.causal_engine.hypotheses.append(hyp)

    detector = KnowledgeGapDetector()
    gaps = detector.detect_gaps(
        student,
        recent_texts=["We observed an optical prism refract light into diverse colors."],
    )

    assert len(gaps) >= 2
    types = {g.gap_type for g in gaps}
    assert KnowledgeGapType.CAUSAL in types
    assert KnowledgeGapType.LEXICAL in types

    # Causal gap should have high entropy (1.0)
    causal_gap = next(g for g in gaps if g.gap_type == KnowledgeGapType.CAUSAL)
    assert causal_gap.prior_entropy == pytest.approx(1.0, abs=1e-3)


def test_self_directed_reading_plan_formulation() -> None:
    """Verify search ranking matches domain-specific educational texts."""
    engine = SelfDirectedReadingEngine()

    # Gap about prism and light -> should rank Newton's Opticks
    optics_gap = EpistemicKnowledgeGap(
        gap_id="gap_optics",
        gap_type=KnowledgeGapType.LEXICAL,
        topic="prism",
        prior_entropy=1.0,
        context="Optical dispersion through glass",
        suggested_query="prism light refraction optics",
    )
    plan_optics = engine.plan_reading([optics_gap])
    assert "optick" in plan_optics.selected_book.title.lower() or plan_optics.domain == "optics"

    # Gap about candle and combustion -> should rank Faraday's Chemical History of a Candle
    chem_gap = EpistemicKnowledgeGap(
        gap_id="gap_chem",
        gap_type=KnowledgeGapType.CAUSAL,
        topic="combustion",
        prior_entropy=1.0,
        context="Exothermic candle combustion",
        suggested_query="combustion candle chemical faraday",
    )
    plan_chem = engine.plan_reading([chem_gap])
    assert "candle" in plan_chem.selected_book.title.lower() or plan_chem.domain == "chemistry"


def test_execute_self_directed_reading_information_gain() -> None:
    """Verify autonomous reading reduces entropy (ΔH > 0), expands vocabulary, and preserves BWT=0."""
    student = CognitiveSchool().student
    hyp = CausalHypothesis(
        hypothesis_id="hyp_flame_combustion",
        action=BabyActionType.PUSH,
        variable="combustion",
        operator="==",
        value="true",
        consequence="HEAT",
        confidence=0.5,
    )
    student.causal_engine.hypotheses.append(hyp)

    engine = SelfDirectedReadingEngine(student=student)

    text = """
    A Discourse on the Chemical History of a Candle.
    Combustion is the rapid oxidation of fuel vapor with oxygen producing radiant heat.
    The wick acts as a capillary pump to deliver liquid wax to the zone of combustion.
    """
    gaps = engine.gap_detector.detect_gaps(student, recent_texts=[text])
    plan = engine.plan_reading(gaps)

    report = engine.execute_self_directed_reading(plan, sample_book_text=text)

    # Information Gain Dynamics
    assert report.reading_success is True
    assert report.prior_entropy > report.posterior_entropy
    assert report.delta_entropy > 0.0
    assert report.new_vocabulary_count > 0
    assert report.backward_transfer == 0.0000

    # Markdown rendering
    md = report.format_markdown()
    assert "Autonomous Epistemic Self-Directed Learning Report" in md
    assert "Information Gain" in md
    assert "Dual-Store Backward Transfer" in md
