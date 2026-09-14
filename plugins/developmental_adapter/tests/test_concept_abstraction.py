"""Tests for Concept Abstraction Engine (Stage D9)."""

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.concept_abstraction import ConceptAbstractionEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.types import (
    BabyObjectState,
    BabyObjectType,
    Vector2D,
)


def test_concept_induction_and_clustering():
    """Verify unsupervised concept clustering from sensory property bundles."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="affordance_discovery_world")
    engine = ConceptAbstractionEngine(substrate, env)

    concepts = engine.induce_concepts_from_experience()

    assert "SPHERICAL_BALL" in concepts
    assert "MANIPULABLE_BLOCK" in concepts
    assert "HEAVY_OBSTACLE" in concepts
    assert len(substrate.semantic_concepts) >= 3


def test_novel_entity_out_of_distribution_categorization():
    """Verify categorization of novel unseen objects with different dimensions and colors."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(scenario="affordance_discovery_world")
    engine = ConceptAbstractionEngine(substrate, env)
    engine.induce_concepts_from_experience()

    # Novel sphere with neon pink color
    novel_sphere = BabyObjectState(
        id="novel_neon_sphere",
        object_type=BabyObjectType.BALL,
        color="neon_pink",
        mass=0.8,
        size=Vector2D(0.15, 0.15),
        position=Vector2D(0.5, 0.5),
        rollable=True,
    )
    cat = engine.categorize_novel_entity(novel_sphere)
    assert cat == "SPHERICAL_BALL"

    # Novel heavy obstacle
    novel_pillar = BabyObjectState(
        id="novel_titanium_pillar",
        object_type=BabyObjectType.BLOCK,
        color="silver",
        mass=20.0,
        size=Vector2D(0.5, 0.8),
        position=Vector2D(0.1, 0.1),
        rollable=False,
    )
    cat = engine.categorize_novel_entity(novel_pillar)
    assert cat == "HEAVY_OBSTACLE"
