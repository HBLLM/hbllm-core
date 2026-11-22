"""Unit tests for unified TransitionKnowledgeBase and ontology-free shortest-path search."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from hbllm.hcir.graph import BeliefNode, LearnedRuleNode
from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_causal import (
    CausalEffectRule,
    HCIRSymbolicWorldTheory,
    TransitionBelief,
    TransitionEdgeEvaluation,
    TransitionKnowledgeBase,
    TransitionOutcome,
)
from hbllm.hcir.world.learning.causal_planner import CausalTheoryPlanner
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from hbllm.hcir.world.objects.planner import ObjectStateGraphPlanner
from hbllm.hcir.world.percept import TerminalSignal
from hbllm.hcir.world.topological_graph_induction import GridTopologyGraph


def test_empirical_outcomes_emerge_without_hardcoded_labels() -> None:
    """Verify traversability, risk, goal value, and mutation emerge from transition counts."""
    kb = TransitionKnowledgeBase()

    # Unknown feature defaults to optimistic/unknown with maximal info gain.
    unknown_eval = kb.evaluate_transition(99)
    assert isinstance(unknown_eval, TransitionEdgeEvaluation)
    assert unknown_eval.traversable is True
    assert unknown_eval.info_gain == 1.0
    assert unknown_eval.confidence == 0.0

    # Traversals increase confidence and keep cost near 1.0.
    kb.record_outcome(3, TransitionOutcome.TRAVERSED, evidence="step_ok")
    kb.record_outcome(3, TransitionOutcome.TRAVERSED, evidence="step_ok")
    trav_eval = kb.evaluate_transition(3)
    assert trav_eval.traversable is True
    assert trav_eval.step_cost < 1.3
    assert trav_eval.risk_cost == 0.0
    assert trav_eval.info_gain < 0.4

    # Repeated blocking renders feature non-traversable.
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="collision")
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="collision")
    blk_eval = kb.evaluate_transition(5)
    assert blk_eval.traversable is False
    assert math.isinf(blk_eval.step_cost)

    # Fatal outcomes induce high risk and block traversal.
    kb.record_outcome(7, TransitionOutcome.TERMINAL_FAIL, evidence="fatal")
    fatal_eval = kb.evaluate_transition(7)
    assert fatal_eval.traversable is False
    assert fatal_eval.risk_cost > 20.0

    # Terminal wins yield goal value and traversability.
    kb.record_outcome(8, TransitionOutcome.TERMINAL_WIN, evidence="win")
    win_eval = kb.evaluate_transition(8)
    assert win_eval.traversable is True
    assert win_eval.goal_value > 0.5

    # Displacement / state mutation outcomes flag mutates_state cleanly.
    kb.record_outcome(6, TransitionOutcome.MUTATED_STATE, effect_signature="delta:door")
    mut_eval = kb.evaluate_transition(6)
    assert mut_eval.mutates_state is True


def test_locus_belief_overrides_feature_belief() -> None:
    """Verify locus-scoped beliefs allow the same feature ID to act differently by coordinate."""
    kb = TransitionKnowledgeBase()
    # Feature 5 is blocked on the outer frame...
    kb.record_outcome(5, TransitionOutcome.BLOCKED, scope="feature", evidence="frame_collision")
    assert kb.evaluate_transition(5, locus=(0, 0)).traversable is False

    # ...yet an interior exit of feature 5 at (12, 36) is empirically rewarding and traversable.
    kb.record_outcome(
        (12, 36), TransitionOutcome.TERMINAL_WIN, scope="locus", evidence="interior_exit"
    )
    interior_eval = kb.evaluate_transition(5, locus=(12, 36))
    assert interior_eval.traversable is True
    assert interior_eval.goal_value > 0.5
    assert kb.evaluate_transition(5, locus=(2, 2)).traversable is False


def test_stochastic_robustness_under_rare_block() -> None:
    """Verify 1 occasional block among many traversals does not flip traversability to False."""
    kb = TransitionKnowledgeBase()
    for _ in range(5):
        kb.record_outcome(4, TransitionOutcome.TRAVERSED, evidence="ok")
    kb.record_outcome(4, TransitionOutcome.BLOCKED, evidence="wind_slip")

    ev = kb.evaluate_transition(4)
    assert ev.traversable is True
    assert ev.step_cost > 1.15


def test_causal_unlock_rule_activation() -> None:
    """Verify active_state_tokens derived from CausalEffectRules unlock blocked transitions."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(2, TransitionOutcome.BLOCKED, evidence="closed_gate")
    rule = kb.record_causal_rule(
        9, "unlocks:2", precondition_tokens=("item:9",), evidence="key_switch"
    )
    assert isinstance(rule, CausalEffectRule)
    assert kb.query_causal_rules(9)[0].effect_signature == "unlocks:2"

    assert kb.evaluate_transition(2).traversable is False
    unlocked_eval = kb.evaluate_transition(2, active_state_tokens=frozenset({"unlocks:2"}))
    assert unlocked_eval.traversable is True
    assert unlocked_eval.step_cost < 1.2


def test_hcir_graph_node_serialization_round_trip() -> None:
    """Verify TransitionBelief and CausalEffectRule round-trip via BeliefNode/LearnedRuleNode."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="wall")
    kb.record_outcome(
        (10, 15),
        TransitionOutcome.MUTATED_STATE,
        scope="locus",
        effect_signature="cycle:1",
        evidence="toggle",
    )
    kb.record_causal_rule(11, "unlocks:5", precondition_tokens=("held:11",), evidence="door")

    ops = kb.to_transaction_operations()
    assert len(ops) == 3
    nodes: list[BeliefNode | LearnedRuleNode] = []
    for op in ops:
        assert op.op == TransactionOp.UPSERT_NODE
        assert op.node_data is not None
        if op.node_id and op.node_id.startswith("kb_transition_"):
            nodes.append(BeliefNode(**op.node_data))
        else:
            nodes.append(LearnedRuleNode(**op.node_data))

    restored = TransitionKnowledgeBase()
    count = restored.hydrate_from_nodes(nodes)
    assert count == 3
    assert restored.evaluate_transition(5).traversable is False
    assert restored.evaluate_transition(0, locus=(10, 15)).mutates_state is True
    assert len(restored.query_causal_rules(11)) == 1
    assert isinstance(restored.get_belief(5), TransitionBelief)


def test_symbolic_theory_syncs_with_transition_kb() -> None:
    """Verify HCIRSymbolicWorldTheory induces, retracts, epochs, and clears TransitionKnowledgeBase."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_walkable(3, evidence="traversal")
    theory.induce_barrier(5, evidence="collision")
    theory.induce_goal(8, evidence=TerminalSignal.SUCCESS)

    assert theory.evaluate_transition(3).traversable is True
    assert theory.evaluate_transition(5).traversable is False
    assert theory.evaluate_transition(8).goal_value > 0.5

    theory.retract_barrier(5)
    assert theory.evaluate_transition(5).confidence == 0.0

    theory.begin_context_epoch()
    theory.transition_kb.record_outcome(
        (1, 2), TransitionOutcome.BLOCKED, scope="locus", evidence="temp"
    )
    assert theory.evaluate_transition(3, locus=(1, 2)).traversable is False
    theory.clear_context_feature_ontology(preserve_current_context=False)
    assert theory.evaluate_transition(3, locus=(1, 2)).confidence == 0.0

    theory.clear()
    assert len(theory.transition_kb.to_transaction_operations()) == 0


def test_shortest_path_from_kb_stride1_and_stride5() -> None:
    """Verify GridTopologyGraph.shortest_path_from_kb solves stride-1 and stride-5 grids using only KB."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(0, TransitionOutcome.TRAVERSED, evidence="floor")
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="wall")
    kb.record_outcome(7, TransitionOutcome.TERMINAL_FAIL, evidence="hazard")
    kb.record_outcome(8, TransitionOutcome.TERMINAL_WIN, evidence="goal")

    # 1. Stride-1 grid routing around a wall of 5s and a fatal 7.
    grid1 = np.array(
        [
            [0, 5, 0, 0],
            [0, 7, 0, 0],
            [0, 0, 0, 8],
        ],
        dtype=int,
    )
    actions1 = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    path1 = GridTopologyGraph.shortest_path_from_kb(
        grid1,
        (0, 0),
        {(2, 3)},
        actions1,
        lambda pos, feat: kb.evaluate_transition(feat, locus=pos),
    )
    assert path1 is not None
    assert [step[1] for step in path1][-1] == (2, 3)
    assert all(grid1[pos[0], pos[1]] not in (5, 7) for _, pos in path1)

    # 2. Stride-5 grid routing around an intermediate barrier intersecting the ray.
    grid5 = np.zeros((16, 16), dtype=int)
    grid5[0:6, 5] = 5  # Blocks eastward stride-5 move from (2, 2) -> (2, 7)
    grid5[12, 12] = 8
    actions5 = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    path5 = GridTopologyGraph.shortest_path_from_kb(
        grid5,
        (2, 2),
        {(12, 12)},
        actions5,
        lambda pos, feat: kb.evaluate_transition(feat, locus=pos),
    )
    assert path5 is not None
    assert path5[0] == (2, (7, 2))  # Must go South first to clear the (0..5, 5) wall
    assert path5[-1][1] == (12, 12)


def test_shortest_path_with_causal_ops_multistep_chain() -> None:
    """Verify two-stage precondition-gated causal op chaining unlocks blocked walls."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(0, TransitionOutcome.TRAVERSED, evidence="floor")
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="locked_gate")
    kb.record_outcome(8, TransitionOutcome.TERMINAL_WIN, evidence="exit")
    kb.record_causal_rule(9, "token:primed", precondition_tokens=(), evidence="stage1_switch")
    kb.record_causal_rule(
        4, "unlocks:5", precondition_tokens=("token:primed",), evidence="stage2_switch"
    )

    grid = np.array(
        [
            [0, 9, 0, 5, 0],
            [0, 0, 0, 5, 0],
            [0, 0, 0, 5, 8],
            [0, 0, 0, 5, 0],
            [0, 4, 0, 5, 0],
        ],
        dtype=int,
    )
    actions = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    # Without causal op chaining, column 3 (feature 5) blocks every direct path.
    assert (
        GridTopologyGraph.shortest_path_from_kb(
            grid,
            (2, 0),
            {(2, 4)},
            actions,
            lambda pos, feat: kb.evaluate_transition(feat, locus=pos),
        )
        is None
    )

    plan = GridTopologyGraph.shortest_path_with_causal_ops(
        grid,
        (2, 0),
        {(2, 4)},
        actions,
        kb,
    )
    assert plan is not None
    visited = [coord for _, coord in plan]
    assert visited[-1] == (2, 4)
    # Must trigger stage 1 (0, 1) strictly before stage 2 (4, 1), then pass col 3.
    idx_stage1 = visited.index((0, 1))
    idx_stage2 = visited.index((4, 1))
    idx_cross = next(i for i, (_, c) in enumerate(visited) if c == 3)
    assert idx_stage1 < idx_stage2 < idx_cross


def test_causal_theory_planner_uses_transition_kb_and_chained_ops() -> None:
    """Verify CausalTheoryPlanner delegates to TransitionKB for locus overrides and causal ops."""
    actions = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    grid = np.array(
        [
            [0, 9, 5, 0, 0],
            [0, 0, 5, 0, 0],
            [0, 0, 5, 0, 8],
        ],
        dtype=int,
    )

    # 1. Locus-scoped TRAVERSED override opens a direct path in CausalTheoryPlanner.plan_path.
    engine = AutonomousEpistemicEngine()
    for act_id, (dr, dc) in actions.items():
        engine.action_dynamics[act_id] = ActionDynamicsModel(
            act_id, delta_r=dr, delta_c=dc, confidence=1.0
        )
    engine.symbolic_theory.induce_walkable(0, evidence="floor")
    engine.symbolic_theory.induce_barrier(5, evidence="wall")
    engine.symbolic_theory.induce_goal(8, evidence=TerminalSignal.SUCCESS)
    engine.symbolic_theory.transition_kb.record_outcome(
        (1, 2), TransitionOutcome.TRAVERSED, scope="locus", evidence="secret_gap"
    )

    ctx = CausalTheoryPlanner.extract_planning_context(engine, grid)
    direct = CausalTheoryPlanner.plan_path((2, 0), grid, ctx)
    assert direct is not None
    pos = (2, 0)
    visited = []
    for act in direct:
        dr, dc = actions[act]
        pos = (pos[0] + dr, pos[1] + dc)
        visited.append(pos)
    assert (1, 2) in visited
    assert visited[-1] == (2, 4)

    # 2. Locus causal rule in transition_kb enables plan_chained_path without tool_barrier_affinities.
    engine2 = AutonomousEpistemicEngine()
    for act_id, (dr, dc) in actions.items():
        engine2.action_dynamics[act_id] = ActionDynamicsModel(
            act_id, delta_r=dr, delta_c=dc, confidence=1.0
        )
    engine2.symbolic_theory.induce_walkable(0, evidence="floor")
    engine2.symbolic_theory.induce_barrier(5, evidence="wall")
    engine2.symbolic_theory.induce_goal(8, evidence=TerminalSignal.SUCCESS)
    engine2.symbolic_theory.transition_kb.record_causal_rule(
        (0, 1), "unlocks:(2, 2)", evidence="lever"
    )

    ctx2 = CausalTheoryPlanner.extract_planning_context(engine2, grid)
    chained = CausalTheoryPlanner.plan_chained_path((2, 0), grid, ctx2, tool_barrier_affinities={})
    assert chained is not None
    pos2 = (2, 0)
    visited2 = []
    for act in chained:
        dr, dc = actions[act]
        pos2 = (pos2[0] + dr, pos2[1] + dc)
        visited2.append(pos2)
    assert visited2[-1] == (2, 4)
    assert visited2.index((0, 1)) < visited2.index((2, 2))


def test_object_planner_find_shortest_path_consults_transition_kb() -> None:
    """Verify ObjectPlanner.find_shortest_path delegates to shortest_path_from_kb and honors KB."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_walkable(0, evidence="floor")
    theory.induce_walkable(3, evidence="corridor")
    theory.transition_kb.record_outcome(
        (1, 1), TransitionOutcome.BLOCKED, scope="locus", evidence="hidden_lock"
    )
    stub_engine = SimpleNamespace(
        bg_feature=0,
        compute_budget=None,
        symbolic_theory=theory,
        is_feature_unverified=lambda val, bg: False,
        UNVERIFIED_FEATURE_COST=2.0,
    )
    planner = ObjectStateGraphPlanner()
    grid = np.array(
        [
            [0, 5, 0, 0],
            [0, 3, 0, 0],
            [0, 5, 0, 0],
        ],
        dtype=int,
    )
    movable = [(1, -1, 0), (2, 1, 0), (3, 0, -1), (4, 0, 1)]

    blocked_path = planner.find_shortest_path(
        start_pos=(1, 0),
        target_stances={(1, 3)},
        grid_shape=(3, 4),
        movable_actions=movable,
        engine=stub_engine,  # type: ignore[arg-type]
        failed_transitions=set(),
        static_barriers={(0, 1), (2, 1)},
        percept=None,
        curr_grid=grid,
        lethal_features=set(),
    )
    assert blocked_path is None

    theory.transition_kb.record_outcome(
        (1, 1), TransitionOutcome.TRAVERSED, scope="locus", evidence="unlocked"
    )
    open_path = planner.find_shortest_path(
        start_pos=(1, 0),
        target_stances={(1, 3)},
        grid_shape=(3, 4),
        movable_actions=movable,
        engine=stub_engine,  # type: ignore[arg-type]
        failed_transitions=set(),
        static_barriers={(0, 1), (2, 1)},
        percept=None,
        curr_grid=grid,
        lethal_features=set(),
    )
    assert open_path is not None
    assert [step.predicted_avatar_pos for step in open_path] == [(1, 1), (1, 2), (1, 3)]
