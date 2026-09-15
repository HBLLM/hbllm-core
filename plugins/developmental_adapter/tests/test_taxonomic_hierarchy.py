"""Unit tests for Taxonomic Concept Hierarchy & Zero-Shot Property Inheritance (Milestone A24)."""

from __future__ import annotations

from plugins.developmental_adapter.dictionary_store import LanguageDictionary
from plugins.developmental_adapter.taxonomy import TaxonNode, TaxonomyHierarchyEngine
from plugins.developmental_adapter.textbook_curriculum import (
    TextbookChapter,
    TextbookCurriculumCurator,
    TextbookSection,
    TextbookSectionType,
)


def test_taxonomy_preseeded_hierarchy() -> None:
    """Verify built-in taxonomic hierarchy connects root entities down to concrete objects."""
    engine = TaxonomyHierarchyEngine()

    # Verify root
    root = engine.nodes.get("physical_entity")
    assert root is not None
    assert root.level == 0

    # Verify intermediate categories
    container = engine.nodes.get("container")
    assert container is not None
    assert container.parent_concept == "physical_entity"
    assert "CONTAINER" in container.direct_affordances

    receptacle = engine.nodes.get("receptacle")
    assert receptacle is not None
    assert receptacle.parent_concept == "container"
    assert "CONTAINER" in receptacle.inherited_affordances

    # Verify leaf nodes
    box = engine.nodes.get("box")
    assert box is not None
    assert box.parent_concept == "receptacle"
    assert box.is_container is True
    assert "CONTAINER" in box.all_affordances


def test_affordance_and_property_inheritance() -> None:
    """Verify transitive inheritance of functional affordances from ancestor nodes."""
    engine = TaxonomyHierarchyEngine()

    # Pre-seeded container leaf
    crucible = engine.nodes.get("crucible")
    assert crucible is not None
    assert crucible.parent_concept == "receptacle"
    assert crucible.is_container is True
    assert "CONTAINER" in crucible.all_affordances
    assert "HOLDS_INSIDE" in crucible.all_affordances
    assert "GRASPABLE" in crucible.all_affordances

    # Pre-seeded tool leaf
    prybar = engine.nodes.get("prybar")
    assert prybar is not None
    assert prybar.parent_concept == "lever_instrument"
    assert prybar.is_tool is True
    assert "TOOL" in prybar.all_affordances
    assert "EXTENDS_REACH" in prybar.all_affordances
    assert "LEVERAGE" in prybar.all_affordances

    # Spheres
    marble = engine.nodes.get("marble")
    assert marble is not None
    assert marble.is_rollable is True
    assert "ROLLABLE" in marble.all_affordances


def test_dynamic_is_a_induction_from_natural_language() -> None:
    """Verify dynamic hypernym deduction and link registration from definition text."""
    engine = TaxonomyHierarchyEngine()

    # 1. Deduce from "type of container"
    retort_node = engine.induce_is_a_relation(
        child_term="retort",
        definition="A retort is a type of container used for distillation of chemical substances.",
    )
    assert isinstance(retort_node, TaxonNode)
    assert retort_node.name == "retort"
    assert retort_node.parent_concept == "container"
    assert retort_node.is_container is True
    assert "CONTAINER" in retort_node.all_affordances
    assert "HOLDS_INSIDE" in retort_node.all_affordances

    # 2. Deduce from "rigid tool used to..."
    grapple_node = engine.induce_is_a_relation(
        child_term="grapple",
        definition="A grapple is a rigid tool used to hook, grasp, and pull distant cargo.",
    )
    assert grapple_node.parent_concept == "tool"
    assert grapple_node.is_tool is True
    assert "TOOL" in grapple_node.all_affordances
    assert "EXTENDS_REACH" in grapple_node.all_affordances


def test_ancestor_chain_resolution() -> None:
    """Verify linear lineage traversal from leaf to ontology root."""
    engine = TaxonomyHierarchyEngine()

    chain = engine.get_ancestor_chain("box")
    assert chain == ["box", "receptacle", "container", "physical_entity"]

    tool_chain = engine.get_ancestor_chain("stick")
    assert tool_chain == ["stick", "lever_instrument", "tool", "physical_entity"]


def test_language_dictionary_taxonomy_integration() -> None:
    """Verify LanguageDictionary delegates to and enriches entries with taxonomic inheritance."""
    dictionary = LanguageDictionary()

    # Register a novel tool with definitional hypernym
    entry = dictionary.register_entry(
        word="prying_lever",
        category="noun",
        definition="A prying_lever is a tool used to multiply force and pull heavy objects.",
    )

    assert entry.is_tool is True
    assert "TOOL" in entry.inherited_affordances
    assert "EXTENDS_REACH" in entry.inherited_affordances
    assert entry.parent_concept == "tool"


def test_zero_shot_curriculum_grounding_with_taxonomy() -> None:
    """Verify TextbookCurriculumCurator correctly maps novel taxonomic entities into simulation."""
    from plugins.developmental_adapter.school import CognitiveSchool

    school = CognitiveSchool(seed=42)
    student = school.student
    curator = TextbookCurriculumCurator(dictionary=school.teacher.dictionary)

    # Chapter introducing novel vessel and novel lever
    chapter = TextbookChapter(
        chapter_id="alchemy_ch1",
        title="Apparatus and Implements",
        grade_level=3,
        sections=[
            TextbookSection(
                section_id="def_alchemy",
                title="Laboratory Definitions",
                section_type=TextbookSectionType.DEFINITIONS,
                raw_text="Definitions of laboratory vessels.",
                structured_payload={
                    "glossary": {
                        "alembic": "An alembic is a type of container used to capture and condense vapours.",
                        "tongs": "Tongs are a tool designed to grip, hold, and pull heated specimens.",
                    }
                },
            )
        ],
    )

    res = curator.teach_chapter(student, chapter)
    assert res["sections_processed"] >= 1
    assert "alembic" in res["grounded_concepts"]
    assert "tongs" in res["grounded_concepts"]

    # Verify student grounding
    assert "alembic" in student.grounding_engine.lexicon
    assert "tongs" in student.grounding_engine.lexicon
