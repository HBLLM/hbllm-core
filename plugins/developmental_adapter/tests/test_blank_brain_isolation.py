"""Unit tests for Phase 2 Blank-Brain Cognitive Isolation."""

from __future__ import annotations

import pytest

from plugins.developmental_adapter.blank_brain import (
    create_blank_brain_substrate,
)


def test_blank_brain_initialization_isolation():
    substrate = create_blank_brain_substrate()

    # Layer 1: Innate machinery is present and active
    assert substrate.workspace is not None
    assert substrate.transaction_mgr is not None
    assert substrate.memory is not None
    assert substrate.mental_sandbox is not None
    assert substrate.runtime is not None
    assert substrate.active_inference is not None

    # Layer 2: All learned knowledge stores are strictly empty
    assert len(substrate.semantic_concepts) == 0
    assert len(substrate.object_categories) == 0
    assert len(substrate.causal_rules) == 0
    assert len(substrate.affordances) == 0
    assert len(substrate.spatial_schemas) == 0
    assert len(substrate.procedural_skills) == 0
    assert len(substrate.lexical_mapping) == 0

    assert substrate.verify_isolation() is True


def test_violation_when_knowledge_leaked_into_blank_brain():
    substrate = create_blank_brain_substrate()

    # Artificially inject pre-compiled concept
    substrate.semantic_concepts["ball"] = {"shape": "sphere"}

    with pytest.raises(AssertionError) as exc_info:
        substrate.verify_isolation()
    assert "Learned concepts must be empty initially" in str(exc_info.value)
