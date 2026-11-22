"""Unit tests for unified TransitionKnowledgeBase and ontology-free shortest-path search."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from hbllm.hcir.graph import BeliefNode, LearnedRuleNode
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_causal import (
    CausalEffectRule,
    HCIRSymbolicWorldTheory,
    TransitionBelief,
    TransitionEdgeEvaluation,
    TransitionKnowledgeBase,
    TransitionOutcome,
)
from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta
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


def test_theory_derives_ontology_views_solely_from_transition_kb() -> None:
    """Verify HCIRSymbolicWorldTheory derives all ontology properties solely from TransitionKB BeliefNodes."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(0, TransitionOutcome.TRAVERSED, evidence="traversal")
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="collision")
    kb.record_outcome(7, TransitionOutcome.TERMINAL_FAIL, evidence="lethal_hazard")
    kb.record_outcome(8, TransitionOutcome.TERMINAL_WIN, evidence=TerminalSignal.SUCCESS)
    kb.record_outcome(4, TransitionOutcome.DISPLACED_OTHER, evidence="push")
    kb.record_outcome(6, TransitionOutcome.RECEIVED_ENTITY, evidence="dock")
    kb.record_outcome(9, TransitionOutcome.MUTATED_STATE, evidence="toggle")

    nodes: list[BeliefNode] = []
    for op in kb.to_transaction_operations():
        assert op.op == TransactionOp.UPSERT_NODE
        assert op.node_data is not None
        assert op.node_id is not None and op.node_id.startswith("kb_transition_")
        nodes.append(BeliefNode(**op.node_data))

    theory = HCIRSymbolicWorldTheory()
    theory.hydrate_from_nodes(nodes)

    assert 0 in theory.walkable_features
    assert 5 in theory.barrier_features
    assert 7 in theory.barrier_features
    assert 8 in theory.goal_features
    assert 4 in theory.cargo_features
    assert 6 in theory.receptacle_features
    assert 9 in theory.actuator_features

    assert theory.is_walkable(0) is True
    assert theory.is_barrier(5) is True
    assert theory.is_barrier(7) is True
    assert theory.is_goal(8) is True
    assert theory.is_cargo(4) is True
    assert theory.is_receptacle(6) is True
    assert theory.is_actuator(9) is True

    for private_attr in (
        "_goal_features",
        "_barrier_features",
        "_walkable_features",
        "_cargo_features",
        "_receptacle_features",
        "_actuator_features",
    ):
        assert not hasattr(theory, private_attr)


def test_transition_kb_per_outcome_epoch_revision() -> None:
    """Verify per-outcome epoch revision prunes stale epoch outcomes while retaining fresh ones."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_goal(3, evidence="terminal_reward")
    assert 3 in theory.goal_features

    theory.begin_context_epoch()
    theory.induce_walkable(3, evidence="traversal")
    theory.clear_context_feature_ontology(preserve_current_context=True)

    assert 3 not in theory.goal_features
    assert not theory.transition_kb.has_feature_outcome(3, TransitionOutcome.TERMINAL_WIN)
    assert 3 in theory.walkable_features
    assert theory.is_walkable(3) is True


def test_engine_hydrates_transition_kb_from_persistent_workspace() -> None:
    """Verify AutonomousEpistemicEngine hydrates TransitionKB beliefs and rules from TieredWorkspace."""
    kb = TransitionKnowledgeBase()
    kb.record_outcome(5, TransitionOutcome.BLOCKED, evidence="wall")
    kb.record_outcome(8, TransitionOutcome.TERMINAL_WIN, evidence="exit")
    kb.record_causal_rule(9, "unlocks:5", evidence="lever")

    tiered = TieredWorkspace()
    tm = TransactionManager(workspace=tiered.persistent)
    tiered.persistent.attach_transaction_manager(tm, replace=True)
    tiered.persistent.commit_operations(kb.to_transaction_operations(), author="test_seed")

    engine = AutonomousEpistemicEngine(enable_logging=False)
    engine.attach_tiered_workspace(tiered, goal_id="kb_hydrate_test", hydrate_persistent=True)

    assert 5 in engine.symbolic_theory.barrier_features
    assert 8 in engine.symbolic_theory.goal_features
    assert engine.symbolic_theory.evaluate_transition(5).traversable is False
    assert engine.symbolic_theory.evaluate_transition(8).goal_value > 0.5
    rules = engine.symbolic_theory.transition_kb.query_causal_rules(9)
    assert len(rules) == 1
    assert rules[0].effect_signature == "unlocks:5"


def test_working_to_persistent_promotion_promotes_kb_nodes() -> None:
    """Verify winning an episode promotes persisted kb_transition_* and kb_rule_* nodes."""
    engine = AutonomousEpistemicEngine(enable_logging=False)
    tiered = TieredWorkspace()
    frame = engine.attach_tiered_workspace(
        tiered, goal_id="kb_promote_test", hydrate_persistent=False
    )

    engine.symbolic_theory.induce_walkable(2, evidence="traversal")
    engine.symbolic_theory.induce_barrier(7, evidence="lethal_hazard")
    engine.symbolic_theory.induce_goal(8, evidence=TerminalSignal.SUCCESS)
    engine.symbolic_theory.record_tool_barrier_affinity(9, 7)
    engine.commit_working_memory_transactions()

    assert frame.workspace.get_node("kb_transition_feature_int_2") is not None
    assert frame.workspace.get_node("kb_transition_feature_int_7") is not None
    assert frame.workspace.get_node("kb_transition_feature_int_8") is not None
    assert frame.workspace.get_node("kb_rule_int_9_unlocks_7") is not None

    curr_grid = np.zeros((6, 6), dtype=int)
    curr_grid[3, 3] = 8
    engine.avatar_pos = (3, 2)
    engine.prev_grid = curr_grid.copy()
    engine.assimilate_feedback(
        curr_grid=curr_grid,
        available_actions=[1, 2, 3, 4],
        is_success=True,
        action=4,
    )

    assert tiered.persistent.get_node("kb_transition_feature_int_2") is not None
    assert tiered.persistent.get_node("kb_transition_feature_int_7") is not None
    assert tiered.persistent.get_node("kb_transition_feature_int_8") is not None
    assert tiered.persistent.get_node("kb_rule_int_9_unlocks_7") is not None


def test_hydration_does_not_double_count_when_both_belief_and_legacy_nodes_exist() -> None:
    """Verify hydrating canonical KB nodes skips redundant legacy ConceptNode re-induction."""
    source = HCIRSymbolicWorldTheory()
    source.induce_barrier(5, evidence="collision")
    source.induce_goal(8, evidence=TerminalSignal.SUCCESS)

    nodes: list[Any] = []
    from hbllm.hcir.graph import ConceptNode

    for op in source.to_transaction_operations():
        assert op.node_data is not None
        if op.node_id and op.node_id.startswith("kb_transition_"):
            nodes.append(BeliefNode(**op.node_data))
        elif op.node_id and op.node_id.startswith("ontology_"):
            nodes.append(ConceptNode(**op.node_data))

    restored = HCIRSymbolicWorldTheory()
    restored.hydrate_from_nodes(nodes)

    b5 = restored.transition_kb.get_belief(5)
    b8 = restored.transition_kb.get_belief(8)
    assert b5 is not None and b5.counts.get(TransitionOutcome.BLOCKED.value) == 1.0
    assert b8 is not None and b8.counts.get(TransitionOutcome.TERMINAL_WIN.value) == 1.0


def test_episode_resets_clear_locus_beliefs_and_preserve_cross_context_lethal() -> None:
    """Verify retain_dynamics=False clears locus beliefs while new_context keeps lethal beliefs."""
    engine = AutonomousEpistemicEngine(enable_logging=False)
    engine.symbolic_theory.induce_barrier(5, evidence="collision")
    engine.symbolic_theory.induce_barrier(7, evidence="lethal_hazard")
    engine.symbolic_theory.transition_kb.record_outcome(
        (2, 3), TransitionOutcome.BLOCKED, scope="locus", evidence="temp_door"
    )

    # New level keeps lethal TERMINAL_FAIL beliefs while clearing non-lethal BLOCKED barriers & loci
    view = ReadOnlyEngineView(engine)
    delta = StateDelta()
    engine.episodic_cortex.consolidate_schema_and_reset(
        view, delta, retain_dynamics=True, is_new_level=True
    )
    engine.apply_delta(delta, author="test")
    assert 5 not in engine.symbolic_theory.barrier_features
    assert 7 in engine.symbolic_theory.barrier_features
    assert engine.symbolic_theory.transition_kb.get_belief((2, 3), scope="locus") is None

    # Full reset (retain_dynamics=False) clears stale locus beliefs
    engine.symbolic_theory.transition_kb.record_outcome(
        (4, 4), TransitionOutcome.BLOCKED, scope="locus", evidence="stale"
    )
    view2 = ReadOnlyEngineView(engine)
    delta2 = StateDelta()
    engine.episodic_cortex.consolidate_schema_and_reset(
        view2, delta2, retain_dynamics=False, is_new_level=False
    )
    engine.apply_delta(delta2, author="test")
    assert engine.symbolic_theory.transition_kb.get_belief((4, 4), scope="locus") is None


def test_tool_affinity_and_distal_actuation_register_kb_causal_rules() -> None:
    """Verify legacy tool_barrier_affinity and distal_actuation APIs emit KB CausalEffectRules."""
    theory = HCIRSymbolicWorldTheory()
    theory.record_tool_barrier_affinity(9, 5)
    theory.record_distal_actuation(trigger_coord=(1, 2), affected_pos=(4, 5))

    rules_9 = theory.transition_kb.query_causal_rules(9)
    rules_1_2 = theory.transition_kb.query_causal_rules((1, 2))
    assert any(r.effect_signature == "unlocks:5" for r in rules_9)
    assert any(r.effect_signature == "unlocks:(4, 5)" for r in rules_1_2)


def test_is_feature_unverified_uses_kb_confidence() -> None:
    """Verify is_feature_unverified checks empirical KB confidence across all outcomes."""
    engine = AutonomousEpistemicEngine(enable_logging=False)
    assert engine.is_feature_unverified(9, bg=0) is True

    engine.symbolic_theory.induce_actuator(9, evidence="remote_trigger")
    assert engine.is_feature_unverified(9, bg=0) is False

    engine.symbolic_theory.induce_receptacle(6, evidence="docked")
    assert engine.is_feature_unverified(6, bg=0) is False


def test_has_verified_walkable_requires_traversal_rank_evidence() -> None:
    """Verify has_verified_walkable enforces OntologyEvidenceRank >= TRAVERSAL."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_walkable(4, evidence="background_shift")
    theory.induce_walkable(0, evidence="hydration:free_space_prior")
    theory.induce_walkable(3, evidence="traversal")

    assert 4 in theory.walkable_features
    assert theory.has_verified_walkable(4) is False
    assert theory.has_verified_walkable(0) is False
    assert theory.has_verified_walkable(3) is True

    theory.transition_kb.record_outcome(2, TransitionOutcome.TRAVERSED, evidence="traversal")
    theory.transition_kb.record_outcome(6, TransitionOutcome.TRAVERSED, evidence="background_shift")
    assert theory.has_verified_walkable(2) is True
    assert theory.has_verified_walkable(6) is False
