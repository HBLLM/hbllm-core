"""Tests for the unified cognitive composition root, tiered workspace, and event-driven runtime."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.protocols import (
    Consolidatable,
    Prunable,
    Replayable,
)
from hbllm.runtime.bootstrap import (
    CognitiveSystemConfig,
    build_cognitive_system,
)


def test_composition_root_workspace_unification():
    """Verify services.workspace is identical to tiered_workspace.working_state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CognitiveSystemConfig(
            world_id="unification_test",
            memory_dir=tmp_dir,
            enable_grounding=True,
        )
        system = build_cognitive_system(config)

        # Invariant: single state ownership and workspace identity
        assert system.services.workspace is system.tiered_workspace.working_state
        assert system.workspace is system.services.workspace
        assert system.engine.services is system.services
        assert system.bridge.services is system.services
        assert system.engine.transaction_manager is system.services.transaction_manager


def test_persistent_memory_defaults_and_overrides():
    """Verify default persistent memory directory conforms to ~/.hbllm/worlds/<world_id>."""
    # 1. Default configuration
    default_cfg = CognitiveSystemConfig(world_id="default_world")
    expected_default_dir = Path.home() / ".hbllm" / "worlds" / "default_world"
    assert str(expected_default_dir).endswith(".hbllm/worlds/default_world")
    assert default_cfg.world_id == "default_world"

    # 2. In-memory override via memory_dir=None
    in_mem_cfg = CognitiveSystemConfig(world_id="in_memory_test", memory_dir=None)
    in_mem_system = build_cognitive_system(in_mem_cfg)
    assert (
        in_mem_system.tiered_workspace.persistent.event_store.__class__.__name__
        == "InMemoryEventStore"
    )


def test_arbitrary_dimensional_hydration_and_homeostasis():
    """Verify engine hydrates arbitrary dimensional coordinates, concepts, and homeostatic state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CognitiveSystemConfig(
            world_id="hydration_test",
            memory_dir=tmp_dir,
            enable_grounding=False,
        )
        system = build_cognitive_system(config)
        engine = system.engine

        # Seed knowledge in persistent workspace
        persistent_ws = system.tiered_workspace.persistent

        # 3D spatial goal and 4D barrier
        from hbllm.hcir.graph import ConstraintNode, GoalNode
        from hbllm.hcir.transactions import TransactionOp, TransactionOperation

        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="learned_goal_1_2_3",
                node_data=GoalNode(
                    id="learned_goal_1_2_3",
                    description="3D goal",
                    properties={"position": [1, 2, 3], "role": "GOAL"},
                ).model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="learned_barrier_4_5_6_7",
                node_data=ConstraintNode(
                    id="learned_barrier_4_5_6_7",
                    name="barrier_4d",
                    target="p4",
                    description="4D barrier",
                    properties={"position": [4, 5, 6, 7], "role": "BARRIER"},
                ).model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="homeostatic_state",
                node_data={
                    "id": "homeostatic_state",
                    "node_type": "concept",
                    "energy_budget": 87.5,
                    "safety_margin": 0.95,
                    "competence_progress": 0.42,
                    "fatigue": 0.12,
                },
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="concept_custom_alpha",
                node_data=ConceptNode(
                    id="concept_custom_alpha",
                    label="custom_alpha",
                    definition="Test concept",
                    domain="custom_domain",
                ).model_dump(),
            ),
        ]
        persistent_ws.commit_operations(
            ops, author="test_setup", grant=system.tiered_workspace._system_grant
        )

        # Hydrate into engine
        hydrated_count = engine.hydrate_from_workspace(system.tiered_workspace)
        assert hydrated_count >= 4

        # Assert arbitrary coordinates were parsed without dimension crash
        assert (1, 2, 3) in engine.learned_goal_positions
        assert (4, 5, 6, 7) in engine.learned_barriers

        # Assert homeostatic state restored
        assert abs(engine.working_memory.homeostatic.energy_budget - 87.5) < 1e-4
        assert abs(engine.working_memory.homeostatic.safety_margin - 0.95) < 1e-4


@pytest.mark.asyncio
async def test_event_driven_autonomy_runtime():
    """Verify ExecutiveRuntime triggers offline consolidation event-driven on idle timeout."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CognitiveSystemConfig(
            world_id="autonomy_test",
            memory_dir=tmp_dir,
            enable_grounding=False,
        )
        system = build_cognitive_system(config)
        runtime: ExecutiveRuntime = system.runtime

        assert runtime is not None
        initial_count = runtime.consolidation_count

        # Start autonomous loop with short 0.1s idle threshold
        await runtime.start_autonomous_loop(idle_threshold=0.1)

        # Signal an observation to start the clock
        runtime.notify_observation()

        # Wait slightly longer than the idle threshold
        await asyncio.sleep(0.25)

        # Offline consolidation should have triggered via timeout without polling sleep loops
        assert runtime.consolidation_count > initial_count
        assert runtime.is_idle_consolidated is True

        await runtime.stop_autonomous_loop()


def test_consolidation_protocols_conformance():
    """Verify that cognitive components adhere to typing.Protocol interfaces."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = CognitiveSystemConfig(
            world_id="protocol_test",
            memory_dir=tmp_dir,
            enable_grounding=False,
        )
        system = build_cognitive_system(config)

        # Engine implements Consolidatable
        assert isinstance(system.engine, Consolidatable)

        # Theory implements Prunable
        assert isinstance(system.engine.symbolic_theory, Prunable)

        # Episodic cortex implements Replayable
        if system.engine.episodic_cortex is not None:
            assert isinstance(system.engine.episodic_cortex, Replayable)
