"""Unit tests for Grounded Causal Induction from Utterances and Decision Explanations (AGI Gap 8).

Verifies:
1. Compositional semantics grounded in the world model (e.g. 'the block left of the key').
2. Learning testable causal predicates from natural language utterances ('red doors need red keys').
3. Causal status initialization (HYPOTHESISED) and automatic tool-barrier affinity registration.
4. Explainability: AutonomousEpistemicEngine generates human-readable decision explanations.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.compiler import HCIRCompiler
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory


def test_compositional_spatial_semantics_grounding() -> None:
    """Validate evaluating compositional relational language ('block left of the key') against grid."""
    compiler = HCIRCompiler()
    theory = HCIRSymbolicWorldTheory()

    grid = np.zeros((10, 10), dtype=np.int32)
    grid[3, 2] = 1  # blue block at (3, 2)
    grid[3, 6] = 2  # red key at (3, 6)

    ast = compiler.parse_utterance("push the blue block left of the red key")
    resolved = compiler.evaluate_grounded_expression(ast, theory, grid=grid)

    assert len(resolved) == 1
    assert resolved[0] == (3, 2)


def test_compositional_spatial_semantics_above_below() -> None:
    """Validate evaluating vertical relational expressions ('green barrier above yellow switch')."""
    compiler = HCIRCompiler()
    theory = HCIRSymbolicWorldTheory()

    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2, 5] = 3  # green barrier at (2, 5)
    grid[7, 5] = 4  # yellow switch at (7, 5)

    ast = compiler.parse_utterance("the green barrier above the yellow switch")
    resolved = compiler.evaluate_grounded_expression(ast, theory, grid=grid)

    assert len(resolved) == 1
    assert resolved[0] == (2, 5)


def test_compiling_utterance_to_testable_causal_hypothesis() -> None:
    """Validate that being told a rule creates a testable CausalPredicate hypothesis in world theory."""
    compiler = HCIRCompiler()
    theory = HCIRSymbolicWorldTheory()

    utterance = "red doors need red keys"
    pred = compiler.compile_utterance_to_causal_hypothesis(utterance, theory)

    assert pred.relation == "requires"
    assert "red_door" in pred.variable
    assert pred.target_entity == "red doors"
    assert pred.value == "red keys"
    assert pred.causal_status == "HYPOTHESISED"

    # Verify registered in theory
    assert pred in theory.causal_predicates
    # Verify tool-barrier affinity was registered (red key tool clears red door barrier)
    assert theory.has_tool_barrier_affinity(tool_feat=2, barrier_feat=2)


def test_compiling_causal_rules_variations() -> None:
    """Validate compiling diverse causal utterance constructions."""
    compiler = HCIRCompiler()
    theory = HCIRSymbolicWorldTheory()

    # 1. Unlocks relation
    pred1 = compiler.compile_utterance_to_causal_hypothesis("switches unlock gates", theory)
    assert pred1.relation == "unlocks"
    assert "switch" in pred1.variable

    # 2. Clears relation
    pred2 = compiler.compile_utterance_to_causal_hypothesis("pushing blocks clears path", theory)
    assert pred2.relation == "clears"

    # 3. Causes relation
    pred3 = compiler.compile_utterance_to_causal_hypothesis("touching lava causes reset", theory)
    assert pred3.relation == "causes"

    # 4. If-then conditional
    pred4 = compiler.compile_utterance_to_causal_hypothesis(
        "if blue button pressed then laser deactivated", theory
    )
    assert pred4.relation == "implies"


def test_autonomous_epistemic_engine_decision_explanation() -> None:
    """Validate AutonomousEpistemicEngine produces human-interpretable natural language explanations."""
    engine = AutonomousEpistemicEngine()

    # 1. Idle state explanation
    idle_explanation = engine.explain_last_decision()
    assert "idle" in idle_explanation.lower()

    # 2. Active decision explanation
    engine.last_action = 2
    engine.epistemic_phase = "Hypothesis Testing"
    engine.active_subgoal = (5, 7)

    # Condition on a causal hypothesis
    hyp_pred = CausalPredicate(
        variable="door_clearance",
        operator="==",
        value="unlocked",
        relation="requires",
        target_entity="key_01",
    )
    engine.symbolic_theory.record_causal_predicate(hyp_pred)

    explanation = engine.explain_last_decision()
    assert "action 2" in explanation.lower()
    assert "hypothesis testing" in explanation.lower()
    assert "(5, 7)" in explanation
    assert "door_clearance" in explanation
