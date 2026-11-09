"""Unit tests for Temporal Option Discovery and Graph Bottleneck Cuts.

Validates articulation cut detection, betweenness centrality choke points,
option induction <I, pi, beta>, reuse rate metrics, and library consolidation.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.library_learning import LibraryLearner
from hbllm.hcir.world.option_discovery import OptionDiscoveryEngine


def _create_two_room_grid() -> np.ndarray:
    """Create a 5x9 grid with two 5x4 rooms connected by a single bottleneck doorway at (2, 4)."""
    grid = np.ones((5, 9), dtype=int)
    # Room 1: (1..3, 1..3)
    grid[1:4, 1:4] = 0
    # Room 2: (1..3, 5..7)
    grid[1:4, 5:8] = 0
    # Doorway connection: (2, 4)
    grid[2, 4] = 0
    return grid


def test_articulation_point_detection_on_doorway() -> None:
    """Articulation point detection correctly identifies the doorway as a cut vertex."""
    engine = OptionDiscoveryEngine()
    grid = _create_two_room_grid()
    adjacency = engine.grid_to_adjacency(grid, walkable_val=0)

    cuts = engine.find_articulation_points(adjacency)
    assert (2, 4) in cuts


def test_betweenness_centrality_choke_point() -> None:
    """Centrality of doorway is significantly higher than interior corner cells."""
    engine = OptionDiscoveryEngine()
    grid = _create_two_room_grid()
    adjacency = engine.grid_to_adjacency(grid, walkable_val=0)

    centralities = engine.compute_betweenness_centrality(adjacency)
    door_centrality = centralities.get((2, 4), 0.0)
    corner_centrality = centralities.get((1, 1), 0.0)

    assert door_centrality > 0.4
    assert door_centrality > corner_centrality * 5


def test_discover_bottlenecks_from_grid() -> None:
    """Engine discovers bottleneck states from grid topology."""
    engine = OptionDiscoveryEngine(betweenness_threshold=0.2)
    grid = _create_two_room_grid()

    bottlenecks = engine.discover_bottlenecks_from_grid(grid, walkable_val=0)
    assert (2, 4) in bottlenecks


def test_option_induction_and_declarative_skill() -> None:
    """Discovered option converts to declarative neuro-symbolic skill and transactions."""
    engine = OptionDiscoveryEngine()
    opt = engine.induce_option(
        bottleneck=(2, 4),
        subgoal_sequence=["NAVIGATE_TO", "INTERACT"],
    )

    assert opt.bottleneck_state == (2, 4)
    assert opt.name == "ReachBottleneck_2_4"

    # Export to skill
    skill = opt.to_declarative_skill()
    assert skill.name == "ReachBottleneck_2_4"
    assert len(skill.program.subgoals) == 2

    # Export to transactions
    tx_ops = opt.to_transaction_operations()
    assert len(tx_ops) == 1
    assert tx_ops[0].op == TransactionOp.UPSERT_NODE
    assert tx_ops[0].node_id == "discovered_option_opt_bottleneck_2_4"


def test_option_reuse_tracking_and_library_consolidation() -> None:
    """Option executions update reuse metrics and consolidate into LibraryLearner."""
    engine = OptionDiscoveryEngine(min_reuse_for_consolidation=2)
    opt = engine.induce_option(bottleneck=(2, 4))

    # Primitive step
    engine.record_primitive_execution(steps=5)
    assert engine.get_option_reuse_rate() == 0.0

    # Option execution 1 (success)
    engine.record_option_execution(opt.option_id, steps=10, success=True)
    assert engine.get_option_reuse_rate() == 10 / 15
    assert opt.reuse_count == 1
    assert opt.success_rate == 1.0

    # Option execution 2 (success)
    engine.record_option_execution(opt.option_id, steps=5, success=True)
    assert opt.reuse_count == 2

    # Consolidate into LibraryLearner
    library = LibraryLearner()
    consolidated = engine.consolidate_options_to_library(library)
    assert opt.name in consolidated
    assert opt.name in library.macro_skills


def test_epistemic_engine_option_discovery_integration() -> None:
    """AutonomousEpistemicEngine discovers options and tracks reuse rate."""
    epistemic_engine = AutonomousEpistemicEngine()
    grid = _create_two_room_grid()

    options = epistemic_engine.discover_and_induce_options(grid)
    assert any(opt.bottleneck_state == (2, 4) for opt in options)

    opt_id = options[0].option_id
    epistemic_engine.record_option_usage(opt_id, steps=8, success=True)
    assert epistemic_engine.get_option_reuse_rate() > 0.0
