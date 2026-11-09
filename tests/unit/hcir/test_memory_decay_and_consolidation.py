"""Tests for Principled Forgetting, Memory Hygiene, and CLS Consolidation.

Verifies:
1. Exponential decay of activation energies following Ebbinghaus retention curves.
2. Selective eviction of sub-threshold memories while respecting lock/permanence flags.
3. Interference-based pruning of redundant or conflicting representations.
4. Comprehensive multi-tier memory hygiene execution.
5. Complementary Learning Systems (CLS) bi-directional consolidation between
   hippocampal replay and medial prefrontal schema abstraction.
6. Execution of memory hygiene during ExecutiveRuntime offline idle consolidation.
"""

from __future__ import annotations

import time

import pytest

from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.nodes import ConceptNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace, WorkspaceTier
from hbllm.hcir.world.cortex_episodic import EpisodicTransition
from hbllm.hcir.world.medial_prefrontal_schema import MedialPrefrontalSchemaEngine


def test_exponential_decay_reduces_activation_energy() -> None:
    """Validate that elapsed time exponentially diminishes node activation energy."""
    tiered = TieredWorkspace()
    now = time.time()

    # Create node with timestamp 3600s in the past (1 half-life ago)
    node = ConceptNode(
        id="decaying_concept_1",
        label="test_decay_concept",
        properties={
            "activation_energy": 1.0,
            "last_accessed": now - 3600.0,
        },
    )
    tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=node.id, node_data=node.model_dump()
            )
        ],
        author="test_init",
    )

    # Half-life = 3600s, so energy should halve to ~0.50
    pruned = tiered.apply_exponential_decay(
        tier=WorkspaceTier.BRAIN,
        half_life_seconds=3600.0,
        min_energy=0.10,
    )
    assert "decaying_concept_1" not in pruned

    updated_node = tiered.brain.get_node("decaying_concept_1")
    assert updated_node is not None
    current_energy = float(updated_node.properties.get("activation_energy", 0.0))
    assert 0.45 <= current_energy <= 0.55


def test_exponential_decay_prunes_subthreshold_nodes() -> None:
    """Validate that nodes below minimum energy threshold are pruned unless protected."""
    tiered = TieredWorkspace()
    now = time.time()

    # Node A: 10 half-lives in the past, sub-threshold, unprotected -> should be pruned
    node_a = ConceptNode(
        id="expired_concept_a",
        label="expired_concept",
        properties={
            "activation_energy": 1.0,
            "last_accessed": now - 36000.0,
        },
    )
    # Node B: 10 half-lives in the past, but locked/permanent -> should be preserved
    node_b = ConceptNode(
        id="permanent_axiom_b",
        label="permanent_axiom",
        properties={
            "activation_energy": 1.0,
            "last_accessed": now - 36000.0,
            "permanent": True,
        },
    )

    tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=node_a.id, node_data=node_a.model_dump()
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=node_b.id, node_data=node_b.model_dump()
            ),
        ],
        author="test_init",
    )

    pruned = tiered.apply_exponential_decay(
        tier=WorkspaceTier.BRAIN,
        half_life_seconds=3600.0,
        min_energy=0.10,
    )

    assert "expired_concept_a" in pruned
    assert tiered.brain.get_node("expired_concept_a") is None

    assert "permanent_axiom_b" not in pruned
    assert tiered.brain.get_node("permanent_axiom_b") is not None


def test_interference_pruning_removes_duplicate_and_lower_confidence_nodes() -> None:
    """Validate that interfering duplicate representations are resolved in favor of higher confidence."""
    tiered = TieredWorkspace()

    # Two overlapping concept nodes
    high_conf_node = ConceptNode(
        id="concept_door_key_gold",
        label="door_key_gold",
        confidence=0.95,
        properties={"activation_energy": 1.0},
    )
    low_conf_node = ConceptNode(
        id="concept_door_key_gold_copy",
        label="door_key_gold",
        confidence=0.40,
        properties={"activation_energy": 1.0},
    )

    tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=high_conf_node.id,
                node_data=high_conf_node.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=low_conf_node.id,
                node_data=low_conf_node.model_dump(),
            ),
        ],
        author="test_init",
    )

    pruned = tiered.prune_interfering_memories(
        tier=WorkspaceTier.BRAIN,
        similarity_threshold=0.80,
    )

    assert "concept_door_key_gold_copy" in pruned
    assert tiered.brain.get_node("concept_door_key_gold_copy") is None
    assert tiered.brain.get_node("concept_door_key_gold") is not None


def test_full_memory_hygiene_cycle() -> None:
    """Validate full multi-tier memory hygiene reporting."""
    tiered = TieredWorkspace()
    now = time.time()

    # Expired node in brain
    node_exp = ConceptNode(
        id="node_to_decay",
        label="node_to_decay",
        properties={"activation_energy": 0.2, "last_accessed": now - 72000.0},
    )
    tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=node_exp.id, node_data=node_exp.model_dump()
            )
        ],
        author="test_init",
    )

    report = tiered.run_memory_hygiene_cycle(half_life_seconds=3600.0)
    assert report["decayed_brain_nodes"] >= 1
    assert report["total_pruned"] >= 1


def test_cls_bidirectional_consolidation_hippocampus_mpfc() -> None:
    """Validate Complementary Learning Systems (CLS) interaction: replay -> schema induction -> primed retrieval."""
    schema_engine = MedialPrefrontalSchemaEngine()

    # 1. Existing schema retrieval & induction from replay
    initial_schema = schema_engine.get_schema("schema_clearance_sequential")
    assert initial_schema is not None

    # Replay trajectory that matches sequential clearance sequence hints
    traj = [
        EpisodicTransition(step=0, action_id=1, avatar_pos=(1, 1)),
        EpisodicTransition(step=1, action_id=2, avatar_pos=(1, 2)),
    ]
    updated = schema_engine.consolidate_replayed_trajectory(traj, terminal_reward=1.0)
    assert updated is not None
    assert updated.activation_count >= 1

    # 2. Top-down retrieval priming
    primed = schema_engine.prime_hippocampal_retrieval(["barrier", "actor"])
    assert len(primed) > 0
    assert any(s.schema_id == "schema_clearance_sequential" for s in primed)


@pytest.mark.asyncio
async def test_executive_runtime_offline_consolidation_triggers_memory_hygiene() -> None:
    """Validate that ExecutiveRuntime idle consolidation cleans decayed memory."""
    ws = HCIRWorkspaceState()
    tiered = TieredWorkspace()
    services = KernelServices.create_default(ws, tiered_workspace=tiered)
    runtime = ExecutiveRuntime(services)

    now = time.time()
    stale_node = ConceptNode(
        id="stale_brain_concept",
        label="stale_concept",
        properties={"activation_energy": 0.1, "last_accessed": now - 100000.0},
    )
    tiered.brain.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=stale_node.id,
                node_data=stale_node.model_dump(),
            )
        ],
        author="test_seed",
    )

    # Run offline consolidation
    await runtime.run_offline_consolidation()

    # Stale node should have been pruned by memory hygiene
    assert tiered.brain.get_node("stale_brain_concept") is None
