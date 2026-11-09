"""Unit tests for Compositional Concepts and Analogical Graph Retrieval.

Verifies:
1. Part-whole and role-filler composition in GroundedConcept.
2. Higher-order relational predicates and formal transaction export.
3. WorkingConceptStore composite concept registration and lifecycle evaluation.
4. RelationalGraphMatcher integration: analogical concept retrieval across scene shifts.
5. Role mapping to new entities in target scenes.
6. AutonomousEpistemicEngine relational analogy query interface.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.experimental.relational_graph_matcher import (
    RelationalGraphMatcher,
)


def test_composite_concept_part_whole_and_role_bindings() -> None:
    """Validate that GroundedConcept supports part-whole composition, roles, and higher-order relations."""
    head_ref = GroundedConceptRef(id="C-HEAD01")
    handle_ref = GroundedConceptRef(id="C-HNDL01")
    tool_ref = GroundedConceptRef(id="C-TOOL01")

    composite = GroundedConcept(
        concept_ref=tool_ref,
        lifecycle_state=ConceptLifecycle.PROVISIONAL,
    )
    assert not composite.is_composite

    # Bind constituent parts (part-whole)
    composite.add_part("head", head_ref.id)
    composite.add_part("handle", handle_ref.id)
    assert composite.is_composite

    # Bind functional roles
    composite.bind_role("user", "avatar")
    composite.bind_role("target", "barrier")

    # Add higher-order causal relation
    composite.add_higher_order_relation(
        relation="enables",
        source=tool_ref.id,
        target="ACTION_SMASH",
        confidence=0.95,
    )

    # 1. Causal Predicates Export
    preds = composite.to_causal_predicates()
    assert len(preds) == 5  # 2 parts + 2 roles + 1 higher-order
    part_preds = [p for p in preds if p.relation == "part_of"]
    assert len(part_preds) == 2
    assert any(p.value == "head" and p.target_entity == "C-HEAD01" for p in part_preds)

    role_preds = [p for p in preds if p.relation == "role_filler"]
    assert len(role_preds) == 2
    assert any(p.value == "avatar" for p in role_preds)

    ho_preds = [p for p in preds if p.relation == "enables"]
    assert len(ho_preds) == 1
    assert ho_preds[0].target_entity == "ACTION_SMASH"

    # 2. Transaction Operations Export
    ops = composite.to_transaction_operations()
    assert len(ops) >= 6  # 1 upsert node + 5 add edge
    upsert_ops = [op for op in ops if op.op.value == "upsert_node"]
    assert len(upsert_ops) == 1
    assert upsert_ops[0].node_id == f"concept_{tool_ref.id}"

    edge_ops = [op for op in ops if op.op.value == "add_edge"]
    assert len(edge_ops) == 5


def test_working_concept_store_register_composite_concept() -> None:
    """Validate registering composite concepts in WorkingConceptStore."""
    store = WorkingConceptStore()

    concept = store.register_composite_concept(
        name_or_alias="KeyAndLockPair",
        parts={"key": "C-KEY01", "lock": "C-LOCK01"},
        role_bindings={"manipulator": "avatar"},
        higher_order_relations=[
            {"relation": "unlocks", "source": "C-KEY01", "target": "C-LOCK01", "confidence": 1.0}
        ],
    )

    assert concept.is_composite
    assert concept.parts["key"] == "C-KEY01"
    assert concept.role_bindings["manipulator"] == "avatar"
    assert len(concept.higher_order_relations) == 1

    # Verify retrieval by alias
    retrieved = store.find_concept_by_alias("keyandlockpair")
    assert retrieved is not None
    assert retrieved.concept_ref.id == concept.concept_ref.id


def test_analogical_concept_retrieval_across_scenes() -> None:
    """Validate analogical retrieval of concepts based on relational scene graph isomorphism."""
    store = WorkingConceptStore()

    # Scene A: Reference scene with 2 objects placed side by side horizontally
    grid_a = np.zeros((10, 10), dtype=np.int32)
    grid_a[2, 2:4] = 1  # Object 0: horizontal bar
    grid_a[2, 6:8] = 2  # Object 1: horizontal bar to the right
    graph_a = RelationalGraphMatcher.build_scene_graph(grid_a)
    assert len(graph_a.nodes) == 2

    # Register Scene A as a relational prototype for concept "ParallelBars"
    concept = store.register_composite_concept(
        name_or_alias="ParallelBars",
        relational_graph=graph_a,
        role_bindings={"left_bar": "0", "right_bar": "1"},
    )

    # Scene B: Target query scene with 2 objects shifted in position, recolored, and scaled
    grid_b = np.zeros((12, 12), dtype=np.int32)
    grid_b[7, 1:4] = 4  # Object 0: horizontal bar at (7, 1)
    grid_b[7, 8:11] = 5  # Object 1: horizontal bar at (7, 8)
    graph_b = RelationalGraphMatcher.build_scene_graph(grid_b)
    assert len(graph_b.nodes) == 2

    # Query WorkingConceptStore for analogical concepts
    matches = store.retrieve_analogous_concepts(query_graph=graph_b, min_match_score=0.60)
    assert len(matches) > 0
    matched_concept, match_score, correspondences = matches[0]
    assert matched_concept.concept_ref.id == concept.concept_ref.id
    assert match_score >= 0.70
    assert len(correspondences) == 2


def test_analogical_role_mapping_to_target_scene() -> None:
    """Validate mapping semantic roles from a composite concept onto target scene objects."""
    store = WorkingConceptStore()

    grid_a = np.zeros((8, 8), dtype=np.int32)
    grid_a[1, 1] = 1  # Object 0 (upper)
    grid_a[6, 1] = 2  # Object 1 (lower)
    graph_a = RelationalGraphMatcher.build_scene_graph(grid_a)

    concept = store.register_composite_concept(
        name_or_alias="VerticalAlignment",
        relational_graph=graph_a,
        role_bindings={"north_anchor": "1", "south_anchor": "2"},
    )

    # Query scene with new object IDs shifted to columns (2, 5) and (7, 5)
    grid_b = np.zeros((10, 10), dtype=np.int32)
    grid_b[2, 5] = 3  # North object
    grid_b[7, 5] = 4  # South object
    graph_b = RelationalGraphMatcher.build_scene_graph(grid_b)

    role_map = store.map_analogous_roles(graph_b, concept.concept_ref.id)
    assert "north_anchor" in role_map
    assert "south_anchor" in role_map

    north_id = role_map["north_anchor"]
    south_id = role_map["south_anchor"]
    assert graph_b.nodes[north_id].centroid[0] < graph_b.nodes[south_id].centroid[0]


def test_autonomous_epistemic_engine_relational_analogy_interface() -> None:
    """Validate AutonomousEpistemicEngine exposes relational analogy querying."""
    engine = AutonomousEpistemicEngine()

    grid_ref = np.zeros((8, 8), dtype=np.int32)
    grid_ref[2, 2] = 2
    grid_ref[2, 5] = 3
    graph_ref = RelationalGraphMatcher.build_scene_graph(grid_ref)

    engine.working_concept_store.register_composite_concept(
        name_or_alias="HorizontalPair",
        relational_graph=graph_ref,
    )

    query_grid = np.zeros((10, 10), dtype=np.int32)
    query_grid[5, 2] = 4
    query_grid[5, 6] = 5

    results = engine.retrieve_relational_analogies(query_grid, min_match_score=0.6)
    assert len(results) > 0
    matched_concept, score, _ = results[0]
    assert matched_concept.linguistic_aliases.get("horizontalpair") is not None
