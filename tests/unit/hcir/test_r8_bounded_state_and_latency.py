"""R8 unit tests: bounded event-store compaction, tier retention, node budgets, and latency."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.nodes import ConceptNode
from hbllm.hcir.snapshot import SnapshotManager
from hbllm.hcir.stores import EventType, InMemoryEventStore
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import (
    DEFAULT_TIER_POLICIES,
    TieredWorkspace,
    TierRetentionPolicy,
    WorkspaceTier,
)
from hbllm.hcir.world import AutonomousEpistemicEngine
from hbllm.hcir.world.percept import EmbodiedObservation


def _commit_concepts(ws: HCIRWorkspaceState, items: list[ConceptNode]) -> None:
    ops = [
        TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=node.id,
            node_data=node.model_dump(),
        )
        for node in items
    ]
    tx = ws.commit_operations(ops, author="r8_test")
    assert tx.is_committed


def test_inmemory_event_store_compaction_preserves_replay_state() -> None:
    """InMemoryEventStore.compact folds prefix events losslessly for SnapshotManager.replay."""
    store = InMemoryEventStore()
    mgr = SnapshotManager(store)

    for i in range(1, 21):
        mgr.record_node_added(ConceptNode(id=f"c_{i}", label=f"lbl_{i}"), author="test")
    # Modify c_5 and remove c_1..c_3 so compacted state exercises updates and deletes
    mgr.record_node_modified(
        "c_5",
        {"label": "updated_5"},
        author="test",
        node=ConceptNode(id="c_5", label="updated_5"),
    )
    for j in range(1, 4):
        mgr.record_node_removed(f"c_{j}", author="test")

    pre_graph = mgr.replay(store)
    removed = store.compact(max_events=6)
    assert removed > 0
    assert store.event_count <= 6
    assert store.latest_sequence() == 24

    events = store.get_events()
    assert events[0].event_type == EventType.SNAPSHOT_CREATED

    post_graph = mgr.replay(store)
    assert {n.id for n in post_graph.all_nodes()} == {n.id for n in pre_graph.all_nodes()}
    c5 = post_graph.get_node("c_5")
    assert c5 is not None and c5.label == "updated_5"
    assert post_graph.get_node("c_1") is None


def test_inmemory_auto_compaction_respects_max_events() -> None:
    """Configured max_events triggers automatic bounded compaction on append."""
    store = InMemoryEventStore(max_events=8)
    mgr = SnapshotManager(store)
    for i in range(1, 25):
        mgr.record_node_added(ConceptNode(id=f"auto_{i}", label=f"k_{i}"))
    assert store.event_count <= 8
    replayed = mgr.replay(store)
    assert replayed.node_count == 24


def test_sqlite_event_store_compaction_preserves_hash_chain_and_replay(tmp_path: Path) -> None:
    """SQLiteEventStore.compact bounds row count, keeps hash chain valid, and replays identically."""
    db_path = tmp_path / "compact.db"
    store = SQLiteEventStore(db_path)
    mgr = SnapshotManager(store)

    with store.batch():
        for i in range(1, 31):
            mgr.record_node_added(ConceptNode(id=f"s_{i}", label=f"item_{i}"), author="test")
        for i in range(1, 11):
            mgr.record_node_removed(f"s_{i}", author="test")
        mgr.record_node_modified(
            "s_15",
            {"label": "mut_15"},
            author="test",
            node=ConceptNode(id="s_15", label="mut_15"),
        )

    assert store.verify_chain_integrity() is True
    pre_ids = {n.id for n in mgr.replay(store).all_nodes()}
    assert len(pre_ids) == 20

    removed = store.compact(max_events=10)
    assert removed > 0
    assert store.event_count <= 10
    assert store.verify_chain_integrity() is True
    assert store.latest_sequence() == 41

    # New appends after compaction continue chaining cleanly
    mgr.record_node_added(ConceptNode(id="s_99", label="post_compact"), author="test")
    assert store.verify_chain_integrity() is True
    store.close()

    # Reopen from disk and replay cleanly
    reopened = SQLiteEventStore(db_path)
    assert reopened.verify_chain_integrity() is True
    replayed = SnapshotManager(reopened).replay(reopened)
    assert {n.id for n in replayed.all_nodes()} == pre_ids | {"s_99"}
    s15 = replayed.get_node("s_15")
    assert s15 is not None and s15.label == "mut_15"
    reopened.close()


def test_workspace_per_world_node_budget_consolidates_low_salience_nodes() -> None:
    """Per-world node budget evicts lowest-salience nodes into a summary while preserving axioms."""
    ws = HCIRWorkspaceState()
    now = time.time()
    items = [
        ConceptNode(
            id="axiom_world_a",
            label="axiom_a",
            properties={"world_id": "world_a", "is_axiom": True, "activation_energy": 0.01},
        )
    ]
    for i in range(15):
        items.append(
            ConceptNode(
                id=f"mem_a_{i}",
                label=f"mem_{i}",
                properties={
                    "world_id": "world_a",
                    "activation_energy": 0.05 * (i + 1),
                    "last_accessed": now - 100 + i,
                },
            )
        )
    for j in range(4):
        items.append(
            ConceptNode(
                id=f"mem_b_{j}",
                label=f"other_{j}",
                properties={"world_id": "world_b", "activation_energy": 0.1},
            )
        )
    _commit_concepts(ws, items)

    evicted = ws.consolidate_node_budget(max_nodes=5, world_id="world_a")
    assert len(evicted) > 0

    world_a_nodes = [n for n in ws.graph.all_nodes() if ws._node_world_scope(n) == "world_a"]
    world_b_nodes = [n for n in ws.graph.all_nodes() if ws._node_world_scope(n) == "world_b"]
    assert len(world_a_nodes) <= 5
    assert len(world_b_nodes) == 4  # World B completely untouched
    assert ws.get_node("axiom_world_a") is not None
    summary = ws.get_node("consolidated_summary_world_a")
    assert summary is not None and summary.properties.get("consolidated_count", 0) == len(evicted)


def test_tiered_workspace_retention_policy_hygiene_and_auto_compaction() -> None:
    """TieredWorkspace enforces per-tier/world node budgets and compacts excess event rows."""
    tight_policies = {
        **DEFAULT_TIER_POLICIES,
        WorkspaceTier.PERSISTENT: TierRetentionPolicy(
            tier=WorkspaceTier.PERSISTENT,
            max_nodes=8,
            max_nodes_per_world=6,
            half_life_seconds=3600.0,
            max_events=12,
            retention_scope="lifelong",
        ),
    }
    tiered = TieredWorkspace(snapshot_interval=5, tier_policies=tight_policies)
    grant = getattr(tiered, "_system_grant", None)

    nodes = [
        ConceptNode(
            id=f"p_{i}",
            label=f"persistent_{i}",
            properties={"world_id": "w1", "activation_energy": 0.1 + 0.02 * i},
        )
        for i in range(18)
    ]
    ops = [
        TransactionOperation(op=TransactionOp.UPSERT_NODE, node_id=n.id, node_data=n.model_dump())
        for n in nodes
    ]
    tiered.persistent.commit_operations(ops, author="seed", grant=grant)
    report = tiered.run_memory_hygiene_cycle()

    assert report["budget_evicted_nodes"] > 0
    assert report["compacted_events"] >= 0
    assert tiered.persistent.graph.node_count <= 8
    assert tiered.persistent.event_store.event_count <= 12


def test_bounded_episodic_cortex_archives() -> None:
    """HippocampalEpisodicCortex caps multi-episode archives cleanly at max_episodes."""
    from hbllm.hcir.world.cortex_episodic import HippocampalEpisodicCortex

    cortex = HippocampalEpisodicCortex(max_episodes=4)
    for ep in range(10):
        cortex.record_transition(step=ep, avatar_pos=(1, 1), action=1, is_success=True)
        cortex.trigger_sharp_wave_ripple_replay(is_success=True)
    for ep in range(10):
        cortex.record_transition(step=ep, avatar_pos=(1, 1), action=2, is_failure=True)
        cortex.trigger_sharp_wave_ripple_replay(is_failure=True)

    assert len(cortex.past_episodes) <= 4
    assert len(cortex.successful_trajectories) <= 4
    assert len(cortex.catastrophic_replays) <= 4


def test_executive_runtime_consolidation_bounds_markers() -> None:
    """Repeated offline consolidation bounds stale consolidation_marker_* nodes."""
    tiered = TieredWorkspace()
    services = KernelServices.create_default(HCIRWorkspaceState(), tiered_workspace=tiered)
    runtime = ExecutiveRuntime(services)
    for _ in range(8):
        import asyncio

        asyncio.run(runtime.run_offline_consolidation())
    markers = [
        n.id
        for n in tiered.persistent.graph.all_nodes()
        if n.id.startswith("consolidation_marker_")
    ]
    assert len(markers) <= 5


def test_cortical_pipeline_records_per_stage_timings() -> None:
    """CorticalPipeline.run records per-stage latency telemetry cleanly without overhead regressions."""
    grid = np.zeros((32, 32), dtype=np.int32)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[2, 2] = 4
    grid[28, 28] = 5

    engine = AutonomousEpistemicEngine()
    obs = EmbodiedObservation(
        lattice=grid,
        available_actions=[1, 2, 3, 4],
        context_id="r8_timing",
        episode_id="0",
    )
    engine.step(obs)
    timings = engine.cortical_pipeline.last_stage_timings_ms
    assert "perception_stage" in timings
    assert "assimilation_stage" in timings
    assert "simulation_stage" in timings
    assert "deliberation_stage" in timings
    assert "motor_stage" in timings
    assert all(v >= 0.0 for v in timings.values())
    assert engine.cortical_pipeline.last_duration_ms > 0.0
