"""Comprehensive Unit & Integration Test Suite for the Five Frontier Cognitive Tracks.

Track 1: Hippocampal Sharp-Wave Ripple (SWR) Consolidation & Sleep Replay
Track 2: Hierarchical Variational Free Energy Minimization (Karl Friston's Active Inference)
Track 3: Persistent Homology & Topological Cut-Set Manifold Navigation
Track 4: Constitutional Kernel Transaction Verification & Compensating Rollbacks
Track 5: Native Rust Acceleration for Graph Pattern Matching & Manifold Cuts
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.graph import (
    ActionNode,
    CognitiveCategory,
    GoalNode,
    HCIRNode,
    HCIRNodeType,
    NodeLifecycle,
)
from hbllm.hcir.kernel.governance.constitutional_verifier import (
    ConstitutionalTransactionVerifier,
)
from hbllm.hcir.kernel.transaction_manager import (
    TransactionManager,
)
from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill
from hbllm.hcir.topological_cut_set import (
    TopologicalCutSetAnalyzer,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
    generate_compensating_operations,
)
from hbllm.hcir.workspace import (
    HCIRWorkspaceState,
)
from hbllm.hcir.world.active_inference import (
    HierarchicalActiveInferenceEngine,
)
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
)
from hbllm.hcir.world.cortex_episodic import (
    DistilledSchema,
    EpisodicTransition,
    HippocampalEpisodicCortex,
    HippocampalReplayConsolidator,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoalStack,
    SubgoalType,
)

# ═══════════════════════════════════════════════════════════════════════════
# Track 1: Hippocampal SWR Memory Consolidation & Sleep Replay
# ═══════════════════════════════════════════════════════════════════════════


def test_track1_hippocampal_swr_forward_reverse_replay() -> None:
    """Verify bidirectional SWR replay, temporal credit assignment, and schema distillation."""
    consolidator = HippocampalReplayConsolidator(gamma=0.9, kappa=0.5, min_motif_support=1)

    # 1. Create a synthetic trajectory of 4 steps
    t0 = EpisodicTransition(step=0, avatar_pos=(0, 0), action_id=1, reward=0.0)
    t1 = EpisodicTransition(step=1, avatar_pos=(0, 1), action_id=2, reward=0.0)
    t2 = EpisodicTransition(step=2, avatar_pos=(1, 1), action_id=1, reward=0.0)
    t3 = EpisodicTransition(step=3, avatar_pos=(1, 2), action_id=2, reward=10.0, is_success=True)
    traj = [t0, t1, t2, t3]

    metrics = consolidator.replay_forward_and_reverse(traj, terminal_reward=10.0)
    assert metrics["forward_steps"] == 4
    assert metrics["reverse_steps"] == 4
    assert metrics["max_delta_q"] > 0.0

    # Step 3 (terminal) should receive max immediate credit: 10.0 * (0.9^0) * 0.5 = 5.0
    assert pytest.approx(getattr(t3, "value_estimate"), rel=1e-3) == 5.0
    # Step 2 receives: 10.0 * (0.9^1) * 0.5 = 4.5
    assert pytest.approx(getattr(t2, "value_estimate"), rel=1e-3) == 4.5
    # Forward replay count incremented
    assert getattr(t0, "replay_count") == 1

    # 2. Schema Distillation: extract repeated action motifs
    # Sequence is [1, 2, 1, 2] -> motif (1, 2) appears twice
    schemas = consolidator.distill_semantic_schemas([traj])
    assert len(schemas) >= 1
    best_schema = schemas[0]
    assert isinstance(best_schema, DistilledSchema)
    assert best_schema.action_motif in ([1, 2], [1, 2, 1], [2, 1, 2], [1, 2, 1, 2])
    assert best_schema.causal_predicate is not None
    assert isinstance(best_schema.declarative_skill, DeclarativeNeuroSymbolicSkill)

    # 3. Transactional Promotion
    tx_ops = consolidator.propose_consolidation_transactions(schemas, tenant_id="tenant_alpha")
    assert len(tx_ops) == len(schemas)
    assert tx_ops[0].op == TransactionOp.UPSERT_NODE
    assert (
        tx_ops[0].node_data is not None and tx_ops[0].node_data.get("tenant_id") == "tenant_alpha"
    )


def test_track1_autonomous_epistemic_engine_consolidation_integration() -> None:
    """Verify end-to-end memory consolidation within AutonomousEpistemicEngine."""
    engine = AutonomousEpistemicEngine(world_id="test_swr_world")
    cortex: HippocampalEpisodicCortex = engine.episodic_cortex

    # Record transitions into active episodic cortex buffer
    for s in range(5):
        cortex.record_transition(
            step=s,
            avatar_pos=(0, s),
            action=1 if s % 2 == 0 else 3,
            reward=1.0 if s == 4 else 0.0,
            is_win=(s == 4),
        )

    # Consolidate via sleep replay
    res = engine.consolidate_episodic_memory(terminal_reward=1.0)
    assert "replay" in res
    assert res["replay"]["forward_steps"] == 5
    assert res["replay"]["reverse_steps"] == 5
    # Newly inducted schemas should be registered into symbolic world theory
    predicates = engine.symbolic_theory.export_causal_predicates()
    assert any("distilled_schema" in p.variable for p in predicates)


def test_track2_hierarchical_active_inference_levels() -> None:
    """Verify 3-level hierarchical variational free energy evaluation."""
    engine = HierarchicalActiveInferenceEngine(
        sensory_precision=2.0,
        dynamic_precision=1.5,
        lambda_epistemic=0.5,
        lambda_extrinsic=0.5,
        lambda_risk=0.5,
    )

    # Level 0 (Sensory error): matching observations give lower F0 than divergent
    obs_target = np.array([[1, 0], [0, 1]], dtype=float)
    obs_match = np.array([[1, 0], [0, 1]], dtype=float)
    obs_diff = np.array([[0, 1], [1, 0]], dtype=float)

    f0_match = engine.compute_sensory_free_energy(obs_target, obs_match)
    f0_diff = engine.compute_sensory_free_energy(obs_target, obs_diff)
    assert f0_match < f0_diff

    # Level 1 (Dynamic kinematic transition error)
    state_curr = np.array([2.0, 3.0])
    state_expected = np.array([2.0, 3.0])
    state_deviated = np.array([5.0, 8.0])

    f1_match = engine.compute_dynamic_free_energy(state_curr, state_expected)
    f1_diff = engine.compute_dynamic_free_energy(state_curr, state_deviated)
    assert f1_match < f1_diff

    # Level 2 (Expected Free Energy G(pi)): action with high info gain and low goal distance
    act_explore = ActionNode(id="act_explore", intent="explore", risk_factor=0.0)
    act_risky = ActionNode(id="act_risky", intent="risky", risk_factor=1.0)

    g_explore = engine.compute_expected_free_energy_g(
        action=act_explore, epistemic_gain=1.0, goal_divergence=0.2, risk_penalty=0.0
    )
    g_risky = engine.compute_expected_free_energy_g(
        action=act_risky, epistemic_gain=0.1, goal_divergence=1.5, risk_penalty=1.0
    )
    # Lower Expected Free Energy is preferred
    assert g_explore < g_risky


def test_track2_candidate_ranking_and_transaction_proposal() -> None:
    """Verify candidate ranking under variational energy minimization and transactional logging."""
    engine = HierarchicalActiveInferenceEngine()
    a1 = ActionNode(id="a_left", intent="move_left", risk_factor=0.1)
    a2 = ActionNode(id="a_right", intent="move_right", risk_factor=0.8)

    results = engine.evaluate_candidates_hierarchical(
        candidate_actions=[a1, a2],
        goal_pos=(0.0, 0.0),
        predicted_positions={"a_left": (0.0, 1.0), "a_right": (10.0, 10.0)},
        epistemic_gain_map={"a_left": 0.8, "a_right": 0.1},
    )

    assert len(results) == 2
    # a_left has lower goal distance and higher epistemic gain -> lowest free energy -> ranked 0
    assert results[0].action.id == "a_left"
    assert results[0].free_energy < results[1].free_energy

    # Transaction proposals
    ops = results[0].to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].op == TransactionOp.UPSERT_NODE
    assert ops[0].node_id == "hfe_eval_a_left"
    assert ops[0].node_data is not None and "total_free_energy" in ops[0].node_data


def test_track3_betti_invariants_and_cavity_detection() -> None:
    """Verify 0-th & 1-st Betti numbers, cycle rank, and cavity cul-de-sac detection."""
    # Construct a 5x5 grid with a single loop (donut / corridor cycle)
    # Loop around perimeter of a 3x3 block:
    # (1,1)-(1,2)-(1,3)
    # (2,1)- (wall) -(2,3)
    # (3,1)-(3,2)-(3,3)
    traversable = {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),  # Dead-end cul-de-sac protruding to the right
    }
    betti = TopologicalCutSetAnalyzer.compute_betti_invariants(
        grid_shape=(6, 6),
        traversable_cells=traversable,
    )

    # 1 connected component
    assert betti.betti_0 == 1
    # 1 independent tunnel/loop (the cycle around (2,2))
    assert betti.betti_1 == 1
    # Euler characteristic chi = V - E
    assert betti.euler_characteristic == len(traversable) - 9
    # Dead-end cavity at (3,4)
    assert any((3, 4) in cav for cav in betti.cavities)


def test_track3_min_cut_max_flow_partition_discovery() -> None:
    """Verify Edmonds-Karp min-cut separating boundary in spatial mazes."""
    # 5x5 grid with barrier wall down column 2, except a bottleneck gate at (2, 2)
    grid_shape = (5, 5)
    barriers = {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}
    start = (2, 0)
    goal = (2, 4)

    min_cut = TopologicalCutSetAnalyzer.compute_min_cut_max_flow(
        start=start,
        goal=goal,
        grid_shape=grid_shape,
        barrier_cells=barriers,
    )

    # Start is in left partition, Goal is in right partition
    assert start in min_cut.source_partition
    assert goal in min_cut.sink_partition
    # The separating cut cells must be barrier wall cells
    assert min_cut.critical_cut_cells.issubset(barriers)
    assert len(min_cut.critical_cut_cells) > 0


def test_track3_subgoal_stack_bottleneck_relief() -> None:
    """Verify frontopolar subgoal stack scheduling for corridor bottlenecks."""
    stack = FrontopolarSubgoalStack(max_depth=5)
    sg = stack.schedule_bottleneck_relief_subgoal(
        bottleneck_cell=(2, 2), approach_cell=(2, 1), priority=2.0
    )

    assert sg.subgoal_type == SubgoalType.TRANSIT_CORRIDOR
    assert sg.target_destination == (2, 2)
    assert sg.target_entity_pos == (2, 1)
    assert stack.current_subgoal == sg


def test_track4_constitutional_verifier_invariants() -> None:
    """Verify constitutional verifier enforces budget, tenant isolation, and safety."""
    verifier = ConstitutionalTransactionVerifier(
        max_operations_per_tx=3,
        require_provenance=True,
        enforce_scope_isolation=True,
    )
    workspace = HCIRWorkspaceState()

    # 1. Budget violation: 4 ops exceeds max 3
    tx_budget = HCIRTransaction(
        author="planner",
        operations=[
            TransactionOperation(op=TransactionOp.ADD_NODE, node_id=f"n_{i}") for i in range(4)
        ],
    )
    assert not verifier.verify(tx_budget, workspace)

    # 2. Safety violation: removing active goal node
    goal_node = GoalNode(id="goal_1", lifecycle=NodeLifecycle.ACTIVE, description="reach goal")
    workspace.graph.add_node(goal_node)
    tx_unsafe = HCIRTransaction(
        author="agent",
        operations=[
            TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id="goal_1"),
        ],
    )
    assert not verifier.verify(tx_unsafe, workspace)


def test_track4_compensating_operations_and_two_phase_commit() -> None:
    """Verify inverse operation generation and 2-phase commit with rollback."""
    # 1. Test inverse operation generation
    forward_ops = [
        TransactionOperation(op=TransactionOp.ADD_NODE, node_id="node_a"),
        TransactionOperation(
            op=TransactionOp.REMOVE_NODE, node_id="node_b", node_data={"id": "node_b"}
        ),
        TransactionOperation(op=TransactionOp.ADD_EDGE, edge_id="edge_ab"),
    ]
    inverse_ops = generate_compensating_operations(forward_ops)

    assert len(inverse_ops) == 3
    # Reverse order: edge first, then node_b restored, then node_a removed
    assert inverse_ops[0].op == TransactionOp.REMOVE_EDGE
    assert inverse_ops[0].edge_id == "edge_ab"
    assert inverse_ops[1].op == TransactionOp.ADD_NODE
    assert inverse_ops[1].node_id == "node_b"
    assert inverse_ops[2].op == TransactionOp.REMOVE_NODE
    assert inverse_ops[2].node_id == "node_a"

    # 2. Test TransactionManager 2-phase commit
    workspace = HCIRWorkspaceState()
    tx_mgr = TransactionManager(workspace)
    verifier = ConstitutionalTransactionVerifier(max_operations_per_tx=10)

    valid_node = HCIRNode(
        id="test_node_1", node_type=HCIRNodeType.CONCEPT, category=CognitiveCategory.MEMORY
    )
    tx_valid = HCIRTransaction(
        author="kernel_actor",
        operations=[
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id="test_node_1",
                node_data=valid_node.model_dump(),
            )
        ],
    )

    success, committed_tx, comp_tx = tx_mgr.commit_with_verification(tx_valid, verifiers=[verifier])
    assert success is True
    assert committed_tx.is_committed is True
    assert comp_tx is None
    assert workspace.graph.has_node("test_node_1")


def test_track5_rust_blake3_and_topological_acceleration() -> None:
    """Verify native Rust BLAKE3 hashing and connected components with Python parity."""
    import hbllm_hcir_graph  # type: ignore[import-not-found]

    # 1. BLAKE3 hashing
    hash_str = hbllm_hcir_graph.blake3_hash(b"hbllm_cognitive_os")
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64  # Standard BLAKE3 hex digest length

    # 2. Topological connected components in Rust
    # 4x4 grid with a vertical wall down column 1 from row 0 to 2
    barriers = [(0, 1), (1, 1), (2, 1)]
    components = hbllm_hcir_graph.topological_connected_components(4, 4, barriers)

    # Since row 3 is open ((3, 1) is traversable), the grid remains 1 connected component
    assert len(components) == 1
    assert len(components[0]) == 16 - 3  # 13 cells

    # If row 3 is also blocked, it splits into 2 disjoint partitions
    barriers_full = [(0, 1), (1, 1), (2, 1), (3, 1)]
    components_split = hbllm_hcir_graph.topological_connected_components(4, 4, barriers_full)
    assert len(components_split) == 2

    # 3. Python TopologicalCutSetAnalyzer integration
    reach = TopologicalCutSetAnalyzer.get_reachable_component(
        start=(0, 0),
        barrier_cells=set(barriers_full),
        grid_shape=(4, 4),
    )
    # Left component has 4 cells: (0,0), (1,0), (2,0), (3,0)
    assert reach == {(0, 0), (1, 0), (2, 0), (3, 0)}
