"""Unit tests for Core HCIR Active Interventional Causal Discovery Engine."""

from __future__ import annotations

import math

from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    BeliefTransitionType,
    CausalHypothesis,
)
from hbllm.hcir.world.world_causal import CausalEdgeType, WorldCausalGraph


def test_hypothesis_entropy_calculation() -> None:
    engine = BaseCausalDiscoveryEngine()

    # Empty or zero confidence hypotheses -> 0.0 entropy
    assert engine.compute_hypothesis_entropy([]) == 0.0

    # Uniform distribution across 4 hypotheses -> log2(4) = 2.0 bits
    hyps = [
        CausalHypothesis(variable="color", value="red", confidence=0.5),
        CausalHypothesis(variable="shape", value="cube", confidence=0.5),
        CausalHypothesis(variable="texture", value="rough", confidence=0.5),
        CausalHypothesis(variable="weight", value="heavy", confidence=0.5),
    ]
    entropy = engine.compute_hypothesis_entropy(hyps)
    assert math.isclose(entropy, 2.0, rel_tol=1e-5)

    # If 3 hypotheses are falsified, remaining hypothesis has certainty -> 0.0 entropy
    hyps[0].falsified = True
    hyps[1].falsified = True
    hyps[2].falsified = True
    entropy_single = engine.compute_hypothesis_entropy(hyps)
    assert math.isclose(entropy_single, 0.0, abs_tol=1e-5)


def test_hypothesis_prediction_operators() -> None:
    eq_hyp = CausalHypothesis(variable="color", operator="==", value="blue")
    neq_hyp = CausalHypothesis(variable="color", operator="!=", value="blue")
    lt_hyp = CausalHypothesis(variable="mass", operator="<", value=5.0)
    gte_hyp = CausalHypothesis(variable="mass", operator=">=", value=5.0)

    features_match = {"color": "blue", "mass": 3.2}
    features_diff = {"color": "red", "mass": 7.8}

    assert BaseCausalDiscoveryEngine.predict_hypothesis(eq_hyp, features_match) is True
    assert BaseCausalDiscoveryEngine.predict_hypothesis(eq_hyp, features_diff) is False

    assert BaseCausalDiscoveryEngine.predict_hypothesis(neq_hyp, features_match) is False
    assert BaseCausalDiscoveryEngine.predict_hypothesis(neq_hyp, features_diff) is True

    assert BaseCausalDiscoveryEngine.predict_hypothesis(lt_hyp, features_match) is True
    assert BaseCausalDiscoveryEngine.predict_hypothesis(lt_hyp, features_diff) is False

    assert BaseCausalDiscoveryEngine.predict_hypothesis(gte_hyp, features_match) is False
    assert BaseCausalDiscoveryEngine.predict_hypothesis(gte_hyp, features_diff) is True

    # Missing feature returns False
    assert BaseCausalDiscoveryEngine.predict_hypothesis(eq_hyp, {}) is False


def test_epistemic_curiosity_candidate_ranking() -> None:
    engine = BaseCausalDiscoveryEngine()

    hyps = [
        CausalHypothesis(variable="color", operator="==", value="red"),
        CausalHypothesis(variable="mass", operator="<", value=5.0),
    ]

    # Entity 1: color="red", mass=10.0 -> hyp0 predicts True, hyp1 predicts False (DISAGREEMENT = 1)
    # Entity 2: color="red", mass=2.0  -> hyp0 predicts True, hyp1 predicts True  (DISAGREEMENT = 0)
    feature_map = {
        "ent_disagree": {"color": "red", "mass": 10.0},
        "ent_agree": {"color": "red", "mass": 2.0},
    }

    ranked = engine.rank_interventional_candidates(
        candidate_ids=["ent_agree", "ent_disagree"],
        active_hypotheses=hyps,
        feature_map=feature_map,
    )

    # Ent_disagree should be ranked highest due to maximum hypothesis conflict
    assert ranked[0][0] == "ent_disagree"
    assert ranked[0][1] > ranked[1][1]

    # Exploration penalty penalizes repeatedly probed entities
    hyps[0].supporting_episodes.append("ent_disagree")
    ranked_after_probes = engine.rank_interventional_candidates(
        candidate_ids=["ent_agree", "ent_disagree"],
        active_hypotheses=hyps,
        feature_map=feature_map,
        exploration_penalty=1.5,
    )
    # Penalized entity drops below the fresh entity
    assert ranked_after_probes[0][0] == "ent_agree"


def test_bayesian_updates_strict_falsification_and_confirmation() -> None:
    engine = BaseCausalDiscoveryEngine()

    color_hyp = CausalHypothesis(variable="color", operator="==", value="red", confidence=0.5)
    mass_hyp = CausalHypothesis(variable="mass", operator="<", value=5.0, confidence=0.5)
    hyps = [color_hyp, mass_hyp]

    # Probe 1: Red object with mass 10.0 does NOT move (did_move=False)
    # color_hyp predicted True (moves) -> Refuted! Strict falsification to 0.0
    # mass_hyp predicted False (does not move) -> Consistent! Confidence increases to 0.75
    probe_1 = {"target_id": "obj1", "color": "red", "mass": 10.0, "did_move": False}
    events_1 = engine.update_hypotheses_from_evidence(hyps, probe_1, step_index=1)

    assert color_hyp.falsified is True
    assert color_hyp.confidence == 0.0
    assert "obj1" in color_hyp.counterexamples
    assert any(e.event_type == BeliefTransitionType.HYPOTHESIS_FALSIFIED for e in events_1)

    assert mass_hyp.falsified is False
    assert mass_hyp.confidence == 0.75
    assert mass_hyp.confirmed is False

    # Probe 2: Green object with mass 2.0 DOES move (did_move=True)
    # mass_hyp predicted True -> Consistent! Confidence increases to 1.0 (>= 0.95 => CONFIRMED)
    probe_2 = {"target_id": "obj2", "color": "green", "mass": 2.0, "did_move": True}
    events_2 = engine.update_hypotheses_from_evidence(hyps, probe_2, step_index=2)

    assert mass_hyp.confirmed is True
    assert mass_hyp.confidence == 1.0
    assert any(e.event_type == BeliefTransitionType.HYPOTHESIS_CONFIRMED for e in events_2)


def test_rule_induction_and_causal_graph_wiring() -> None:
    graph = WorldCausalGraph(world_id="test_world")
    engine = BaseCausalDiscoveryEngine(causal_graph=graph)

    rule_store: list[dict] = []
    confirmed_hyp = CausalHypothesis(
        action="PUSH",
        variable="mass",
        operator="<",
        value=5.0,
        consequence="MOVES",
        confidence=1.0,
        supporting_episodes=["obj1", "obj2"],
    )

    rule, event = engine.induce_causal_rule(
        confirmed_hyp=confirmed_hyp,
        rule_store=rule_store,
        causal_graph=graph,
    )

    assert rule["rule_id"] == "causal_rule_1"
    assert rule["action"] == "PUSH"
    assert rule["precondition"]["property"] == "mass"
    assert rule["consequence"] == "MOVES"
    assert event.event_type == BeliefTransitionType.RULE_GENERALIZED
    assert len(rule_store) == 1

    # Verify causal graph edge was added
    effects = graph.get_effects_of("mass_<_5.0")
    assert len(effects) == 1
    assert effects[0].target_id == "MOVES"
    assert effects[0].relationship == CausalEdgeType.CAUSES

    # Re-inducing same rule reinforces empirical support count rather than creating duplicate
    confirmed_hyp.supporting_episodes.append("obj3")
    rule2, event2 = engine.induce_causal_rule(
        confirmed_hyp=confirmed_hyp,
        rule_store=rule_store,
        causal_graph=graph,
    )
    assert rule2["rule_id"] == "causal_rule_1"
    assert len(rule_store) == 1
    assert rule2["empirical_support_count"] > 2
    assert event2.event_type == BeliefTransitionType.RULE_REVISED


def test_held_out_generalization_evaluation() -> None:
    rules = [
        {
            "rule_id": "causal_rule_1",
            "action": "PUSH",
            "precondition": {"property": "mass_sensation", "operator": "<", "value": 5.0},
            "consequence": "MOVES",
            "confidence": 1.0,
        }
    ]

    test_objects = [
        {"id": "o1", "mass": 2.0, "surface_friction": 1.0},  # light -> moves
        {"id": "o2", "mass": 8.0, "surface_friction": 1.0},  # heavy -> static
    ]

    accuracy, records = BaseCausalDiscoveryEngine.test_generalization(test_objects, rules)
    assert accuracy == 1.0
    assert len(records) == 2
    assert all(r["is_correct"] for r in records)
