"""Architecture invariant test: exactly one canonical implementation per cognitive faculty.

Enforces acceptance criteria for duplicate stack retirement:
- Entity tracking: ThreeTierEntityTracker (hbllm.hcir.grounding.tracking)
- Concept learning: ConceptLearner / WorkingConceptStore (hbllm.hcir.grounding.concepts)
- Belief revision: BeliefRevisionEngine (hbllm.hcir.grounding.epistemics.revision)
- Causal theory: HCIRSymbolicWorldTheory (hbllm.hcir.world.cortex_causal)
- Surprise: SurpriseEngine (hbllm.hcir.world.surprise_engine)
"""

from __future__ import annotations

import pytest


def test_single_canonical_entity_tracker() -> None:
    """Exactly one importable implementation for entity tracking."""
    from hbllm.hcir.grounding.tracking import ThreeTierEntityTracker

    assert ThreeTierEntityTracker is not None

    with pytest.raises(ImportError):
        import hbllm.brain.world.entity_registry  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.world.object_permanence  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.world.scene_graph  # noqa: F401


def test_single_canonical_concept_learning() -> None:
    """Exactly one importable implementation for concept learning."""
    from hbllm.hcir.grounding.concepts import ConceptLearner, WorkingConceptStore

    assert ConceptLearner is not None
    assert WorkingConceptStore is not None

    with pytest.raises(ImportError):
        import hbllm.brain.concepts.concept_formation_loop  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.concepts.concept_consolidator  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.concepts.feature_accumulator  # noqa: F401


def test_single_canonical_belief_revision() -> None:
    """Exactly one importable implementation for belief revision."""
    from hbllm.hcir.grounding.epistemics.revision import BeliefRevisionEngine

    assert BeliefRevisionEngine is not None

    with pytest.raises(ImportError):
        import hbllm.brain.epistemics.belief_manager  # noqa: F401


def test_single_canonical_causal_theory() -> None:
    """Exactly one importable implementation for causal world theory."""
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

    assert HCIRSymbolicWorldTheory is not None

    with pytest.raises(ImportError):
        import hbllm.brain.causality.causal_graph  # noqa: F401


def test_single_canonical_surprise_engine() -> None:
    """Exactly one importable implementation for surprise."""
    from hbllm.hcir.world.surprise_engine import SurpriseEngine

    assert SurpriseEngine is not None


def test_no_dead_epistemics_or_control_stacks() -> None:
    """Duplicate and unused epistemics and control loop modules are deleted."""
    with pytest.raises(ImportError):
        import hbllm.brain.epistemics.contradiction_engine  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.epistemics.evidence_evaluator  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.epistemics.epistemic_loop  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.control.executive_controller  # noqa: F401
    with pytest.raises(ImportError):
        import hbllm.brain.reasoning.concept_formation  # noqa: F401
