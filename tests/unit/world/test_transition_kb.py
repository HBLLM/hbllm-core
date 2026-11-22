"""Unit tests for unified TransitionKnowledgeBase and ontology-free shortest-path search."""

from __future__ import annotations

import math

import numpy as np

from hbllm.hcir.graph import BeliefNode, LearnedRuleNode
from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.cortex_causal import (
    CausalEffectRule,
    HCIRSymbolicWorldTheory,
    TransitionBelief,
    TransitionEdgeEvaluation,
    TransitionKnowledgeBase,
    TransitionOutcome,
)
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
