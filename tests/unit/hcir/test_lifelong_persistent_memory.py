"""Lifelong persistent memory across runs.

Validates that symbolic world theory, causal predicates, ontological categories,
and learned spatial concepts survive full runtime restarts via the persistent
workspace tier backed by SQLiteEventStore.
"""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import SpelkePhysicalAxiom
from hbllm.runtime.bootstrap import (
    CognitiveSystemConfig,
    build_cognitive_system,
)


def test_lifelong_persistent_memory_roundtrip(tmp_path: Path) -> None:
    """Test that world models, predicates, and concepts persist and restore identically across restarts."""
    mem_dir = tmp_path / "cognitive_memory"
    config = CognitiveSystemConfig(
        world_id="test_world",
        memory_dir=mem_dir,
        enable_grounding=True,
    )

    # 1. Boot session 1
    system_1 = build_cognitive_system(config)
    assert isinstance(system_1.tiered_workspace.persistent.event_store, SQLiteEventStore)
    engine_1 = system_1.engine

    # Seed and learn world theory & concepts
    engine_1.symbolic_theory.goal_features.add(7)
    engine_1.symbolic_theory.candidate_goal_features.add(7)
    engine_1.symbolic_theory.barrier_features.add(4)
    engine_1.symbolic_theory.walkable_features.add(0)
    engine_1.symbolic_theory.cargo_features.add(3)
    engine_1.symbolic_theory.actuator_features.add(8)
    engine_1.symbolic_theory.telic_goal_type = "deplete"
    engine_1.symbolic_theory.telic_target_feature = 7

    # Spelke physical axiom
    engine_1.symbolic_theory.spelke_axioms[SpelkePhysicalAxiom.SOLIDITY_CONTINUITY] = (
        CausalPredicate(
            variable="barrier_collision",
            operator="==",
            value=True,
        )
    )

    # Causal discovery predicate
    custom_pred = CausalPredicate(
        variable="distal_switch",
        operator="==",
        value="opened",
        relation="triggers",
        target_entity="barrier_4",
    )
    engine_1.symbolic_theory.record_causal_predicate(custom_pred)

    # Learned spatial coordinates
    engine_1.learned_goal_positions.add((2, 5))
    engine_1.learned_barriers.add((1, 1))

    # Persist state into persistent tier
    ops = engine_1.persist_to_workspace()
    assert len(ops) > 0

    # Verify persistent graph received nodes
    persistent_graph_1 = system_1.tiered_workspace.persistent.graph
    assert persistent_graph_1.node_count > 0

    # Ensure database file exists on disk
    db_file = mem_dir / "persistent.db"
    assert db_file.exists()

    # 2. Terminate session 1 completely (simulate process shutdown)
    del engine_1
    del system_1

    # 3. Boot session 2 pointing to the same memory_dir
    system_2 = build_cognitive_system(config)
    engine_2 = system_2.engine

    # 4. Assert full knowledge restoration
    assert 7 in engine_2.symbolic_theory.goal_features
    assert 4 in engine_2.symbolic_theory.barrier_features
    assert 0 in engine_2.symbolic_theory.walkable_features
    assert 3 in engine_2.symbolic_theory.cargo_features
    assert 8 in engine_2.symbolic_theory.actuator_features
    assert engine_2.symbolic_theory.telic_goal_type == "deplete"
    assert engine_2.symbolic_theory.telic_target_feature == 7

    assert SpelkePhysicalAxiom.SOLIDITY_CONTINUITY in engine_2.symbolic_theory.spelke_axioms
    solidity_pred = engine_2.symbolic_theory.spelke_axioms[SpelkePhysicalAxiom.SOLIDITY_CONTINUITY]
    assert solidity_pred.variable == "barrier_collision"

    # Domain causal predicate restored
    assert len(engine_2.symbolic_theory.domain_predicates) == 1
    restored_pred = engine_2.symbolic_theory.domain_predicates[0]
    assert restored_pred.variable == "distal_switch"
    assert restored_pred.target_entity == "barrier_4"
    assert restored_pred.relation == "triggers"

    # Learned spatial positions restored
    assert (2, 5) in engine_2.learned_goal_positions
    assert (1, 1) in engine_2.learned_barriers


def test_lifelong_grounded_concept_and_skill_persistence(tmp_path: Path) -> None:
    """Validate that GroundedConceptNodes and SkillNodes survive process restart in persistent tier."""
    from hbllm.hcir.nodes import GroundedConceptNode, SkillNode
    from hbllm.hcir.transactions import TransactionOp, TransactionOperation

    mem_dir = tmp_path / "concept_memory"
    config = CognitiveSystemConfig(world_id="concept_world", memory_dir=mem_dir)

    # 1. Boot session 1
    system_1 = build_cognitive_system(config)
    persistent_ws_1 = system_1.tiered_workspace.persistent
    grant = system_1.tiered_workspace._system_grant

    # Create grounded concept and skill nodes
    concept_node = GroundedConceptNode(
        id="grounded_concept_key",
        concept_name="gold_key",
        feature_prototype={"color": 1, "size": 3, "solidity": 1.0},
        domain="spatial_affordance",
        properties={"role": "key", "barrier_affinity": [4]},
    )
    skill_node = SkillNode(
        id="skill_key_door_sequence",
        name="solve_key_door",
        description="Pick up key and unlock door",
        properties={"steps": ["pickup_key", "unlock_door"], "domain": "navigation"},
    )

    ops = [
        TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=concept_node.id,
            node_data=concept_node.model_dump(),
        ),
        TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=skill_node.id,
            node_data=skill_node.model_dump(),
        ),
    ]

    tx = persistent_ws_1.commit_operations(ops, author="concept_learner", grant=grant)
    assert tx.status.value == "committed"
    assert persistent_ws_1.graph.has_node("grounded_concept_key")
    assert persistent_ws_1.graph.has_node("skill_key_door_sequence")

    # 2. Terminate session 1
    del persistent_ws_1
    del system_1

    # 3. Boot session 2 pointing to the same persistent store
    system_2 = build_cognitive_system(config)
    persistent_ws_2 = system_2.tiered_workspace.persistent

    # 4. Verify nodes replayed into graph
    assert persistent_ws_2.graph.has_node("grounded_concept_key")
    assert persistent_ws_2.graph.has_node("skill_key_door_sequence")

    restored_concept = persistent_ws_2.graph.get_node("grounded_concept_key")
    assert isinstance(restored_concept, GroundedConceptNode)
    assert restored_concept.concept_name == "gold_key"
    assert restored_concept.domain == "spatial_affordance"
    assert restored_concept.properties["role"] == "key"

    restored_skill = persistent_ws_2.graph.get_node("skill_key_door_sequence")
    assert isinstance(restored_skill, SkillNode)
    assert restored_skill.name == "solve_key_door"
