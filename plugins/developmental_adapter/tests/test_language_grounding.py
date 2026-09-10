"""Tests for Grounded Lexical Acquisition Engine (Stage D10)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine
from plugins.developmental_adapter.types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
    LexicalCategory,
)


def test_grounded_lexical_acquisition_fast_mapping():
    """Verify vocabulary acquisition through cross-situational associative grounding."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="containment_world")
    engine = LanguageGroundingEngine(substrate, env)

    # Initial state must be completely empty
    assert len(substrate.lexical_mapping) == 0

    # Paired demonstrations
    demos = [
        ("look at the red ball", {"entity_type": BabyObjectType.BALL, "color": "red"}),
        ("push the ball", {"action": BabyActionType.PUSH, "entity_type": BabyObjectType.BALL}),
        (
            "the ball is inside the box",
            {"relation": BabyRelationType.INSIDE, "entity_type": BabyObjectType.BOX},
        ),
        (
            "put the toy in the container",
            {"action": BabyActionType.PLACE, "relation": BabyRelationType.INSIDE},
        ),
    ]

    for utt, ctx in demos:
        engine.observe_paired_demonstration(utt, ctx)

    # Verify learned vocabulary in substrate
    assert "ball" in substrate.lexical_mapping
    assert substrate.lexical_mapping["ball"] == "ball"
    assert "push" in substrate.lexical_mapping
    assert substrate.lexical_mapping["push"] == "PUSH"
    assert "inside" in substrate.lexical_mapping
    assert substrate.lexical_mapping["inside"] == "INSIDE"

    # Verify lexical categories
    assert engine.lexicon["ball"].category == LexicalCategory.NOUN
    assert engine.lexicon["push"].category == LexicalCategory.VERB
    assert engine.lexicon["inside"].category == LexicalCategory.PREPOSITION
