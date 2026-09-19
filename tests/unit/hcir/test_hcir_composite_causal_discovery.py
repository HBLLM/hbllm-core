"""Unit tests for Core HCIR Composite Multi-Object Causal Discovery Engine."""

from __future__ import annotations

from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    BeliefTransitionType,
    CausalHypothesis,
    CausalPredicate,
    LogicOperator,
)
from hbllm.hcir.world.world_causal import CausalEdgeType, WorldCausalGraph


def test_atomic_predicate_evaluation() -> None:
    # Numeric operators
    pred_lt = CausalPredicate(variable="mass", operator="<", value=5.0)
    assert pred_lt.evaluate({"mass": 4.2}) is True
    assert pred_lt.evaluate({"mass": 5.0}) is False
    assert pred_lt.evaluate({"mass": 6.1}) is False
    assert pred_lt.evaluate({}) is False

    pred_gte = CausalPredicate(variable="friction", operator=">=", value=0.5)
    assert pred_gte.evaluate({"friction": 0.5}) is True
    assert pred_gte.evaluate({"friction": 0.8}) is True
    assert pred_gte.evaluate({"friction": 0.4}) is False

    # Equality and membership
    pred_eq = CausalPredicate(variable="color", operator="==", value="red")
    assert pred_eq.evaluate({"color": "red"}) is True
    assert pred_eq.evaluate({"color": "blue"}) is False

    pred_in = CausalPredicate(variable="color", operator="in", value=["red", "blue"])
    assert pred_in.evaluate({"color": "red"}) is True
    assert pred_in.evaluate({"color": "green"}) is False


def test_composite_predicate_and_or_not() -> None:
    pred_mass = CausalPredicate(variable="mass", operator="<", value=5.0)
    pred_friction = CausalPredicate(variable="friction", operator="<", value=0.8)

    # Conjunction (AND)
    pred_and = CausalPredicate(
        logic_op=LogicOperator.AND,
        children=[pred_mass, pred_friction],
    )
    assert pred_and.is_composite() is True
    assert pred_and.evaluate({"mass": 3.0, "friction": 0.4}) is True
    assert pred_and.evaluate({"mass": 3.0, "friction": 0.9}) is False
    assert pred_and.evaluate({"mass": 6.0, "friction": 0.4}) is False

    # Disjunction (OR)
    pred_color = CausalPredicate(variable="color", operator="==", value="gold")
    pred_or = CausalPredicate(
        logic_op=LogicOperator.OR,
        children=[pred_and, pred_color],
    )
    assert pred_or.evaluate({"color": "gold", "mass": 100.0, "friction": 5.0}) is True
    assert pred_or.evaluate({"color": "grey", "mass": 2.0, "friction": 0.2}) is True
    assert pred_or.evaluate({"color": "grey", "mass": 10.0, "friction": 0.2}) is False

    # Negation (NOT)
    pred_not_heavy = CausalPredicate(
        logic_op=LogicOperator.NOT,
        children=[CausalPredicate(variable="mass", operator=">=", value=10.0)],
    )
    assert pred_not_heavy.evaluate({"mass": 5.0}) is True
    assert pred_not_heavy.evaluate({"mass": 15.0}) is False


def test_relational_predicate_evaluation() -> None:
    # Relational predicate: target must not be contained in box
    pred_contained = CausalPredicate(
        relation="contained_in",
        operator="==",
        target_entity="box_1",
    )
    assert pred_contained.is_relational() is True

    # Feature map has direct relation
    assert pred_contained.evaluate({"contained_in": "box_1"}) is True
    assert pred_contained.evaluate({"contained_in": "floor"}) is False

    # Context dictionary has relation
    context = {"relations": {"contained_in": "box_1"}}
    assert pred_contained.evaluate({}, context=context) is True
    assert (
        pred_contained.evaluate({}, context={"relations": {"contained_in": "other_box"}}) is False
    )

    # Compound rule: (mass < 5.0) AND NOT contained_in(box_1)
    pred_compound_relational = CausalPredicate(
        logic_op=LogicOperator.AND,
        children=[
            CausalPredicate(variable="mass", operator="<", value=5.0),
            CausalPredicate(
                logic_op=LogicOperator.NOT,
                children=[pred_contained],
            ),
        ],
    )
    assert pred_compound_relational.evaluate({"mass": 3.0, "contained_in": "open_space"}) is True
    assert pred_compound_relational.evaluate({"mass": 3.0, "contained_in": "box_1"}) is False
    assert pred_compound_relational.evaluate({"mass": 8.0, "contained_in": "open_space"}) is False


def test_hypothesis_composition() -> None:
    engine = BaseCausalDiscoveryEngine()

    h1 = CausalHypothesis(
        hypothesis_id="h_mass",
        variable="mass",
        operator="<",
        value=5.0,
        confidence=0.8,
    )
    h2 = CausalHypothesis(
        hypothesis_id="h_friction",
        variable="friction",
        operator="<",
        value=0.6,
        confidence=0.7,
    )

    comp_h = engine.compose_hypotheses([h1, h2], logic_op=LogicOperator.AND)
    assert comp_h.predicate is not None
    assert comp_h.predicate.is_composite() is True
    assert comp_h.confidence == round(0.8 * 0.7, 3)

    # Prediction tests both conditions
    assert engine.predict_hypothesis(comp_h, {"mass": 3.0, "friction": 0.4}) is True
    assert engine.predict_hypothesis(comp_h, {"mass": 3.0, "friction": 0.8}) is False
    assert engine.predict_hypothesis(comp_h, {"mass": 7.0, "friction": 0.4}) is False


def test_hypothesis_specialization_under_counterexamples() -> None:
    engine = BaseCausalDiscoveryEngine()

    # Initial naive hypothesis: "all red objects move when pushed"
    h_red = CausalHypothesis(
        hypothesis_id="h_red",
        variable="color",
        operator="==",
        value="red",
        confidence=0.6,
    )

    # Probing light red object -> moves (supports hypothesis)
    events1 = engine.update_hypotheses_from_evidence(
        hypotheses=[h_red],
        probe_result={"color": "red", "mass": 2.0, "did_move": True, "target_id": "red_light"},
        step_index=1,
    )
    assert events1[0].event_type == BeliefTransitionType.CONFIDENCE_CHANGED
    assert h_red.confidence > 0.6
    assert not h_red.falsified

    # Probing heavy red object -> fails to move (falsifies naive hypothesis!)
    events2 = engine.update_hypotheses_from_evidence(
        hypotheses=[h_red],
        probe_result={"color": "red", "mass": 50.0, "did_move": False, "target_id": "red_heavy"},
        step_index=2,
    )
    assert events2[0].event_type == BeliefTransitionType.HYPOTHESIS_FALSIFIED
    assert h_red.falsified is True

    # Specialize: Refine into composite conjunction: (color == red) AND (mass < 10.0)
    h_spec = engine.specialize_hypothesis(
        parent_hypothesis=h_red,
        discriminating_feature="mass",
        operator="<",
        value=10.0,
        logic_op=LogicOperator.AND,
    )
    assert h_spec.predicate is not None
    assert h_spec.predicate.is_composite() is True
    assert h_spec.falsified is False

    # Now verify the specialized hypothesis accurately predicts both entities
    assert engine.predict_hypothesis(h_spec, {"color": "red", "mass": 2.0}) is True
    assert engine.predict_hypothesis(h_spec, {"color": "red", "mass": 50.0}) is False
    assert engine.predict_hypothesis(h_spec, {"color": "blue", "mass": 2.0}) is False


def test_composite_rule_induction_and_causal_graph() -> None:
    causal_graph = WorldCausalGraph(world_id="test_world")
    engine = BaseCausalDiscoveryEngine(causal_graph=causal_graph)

    pred = CausalPredicate(
        logic_op=LogicOperator.AND,
        children=[
            CausalPredicate(variable="mass", operator="<", value=5.0),
            CausalPredicate(variable="clearance", operator=">", value=0.2),
        ],
    )
    confirmed_hyp = CausalHypothesis(
        hypothesis_id="h_confirmed",
        action="PUSH",
        consequence="MOVES",
        confidence=0.98,
        confirmed=True,
        predicate=pred,
        supporting_episodes=["ent_1", "ent_2", "ent_3"],
    )

    rule_store: list[dict] = []
    rule, event = engine.induce_causal_rule(
        confirmed_hyp=confirmed_hyp,
        rule_store=rule_store,
        causal_graph=causal_graph,
        step_index=5,
    )

    assert rule["is_composite"] is True
    assert rule["confidence"] == 0.98
    assert len(rule_store) == 1
    assert event.event_type == BeliefTransitionType.RULE_GENERALIZED

    # Verify edge and factors in WorldCausalGraph
    causes = causal_graph.get_causes_for("MOVES")
    assert len(causes) == 1
    assert causes[0].relationship == CausalEdgeType.CAUSES
    assert causes[0].weight == 0.98
    assert len(causes[0].factors) == 2
    assert "mass < 5.0" in causes[0].factors
    assert "clearance > 0.2" in causes[0].factors


def test_composite_predicate_serialization() -> None:
    pred = CausalPredicate(
        logic_op=LogicOperator.AND,
        children=[
            CausalPredicate(variable="mass", operator="<", value=5.0),
            CausalPredicate(relation="contained_in", operator="==", target_entity="box_1"),
        ],
    )
    d = pred.to_dict()
    restored = CausalPredicate.from_dict(d)

    assert restored.is_composite() is True
    assert restored.logic_op == LogicOperator.AND
    assert len(restored.children) == 2
    assert restored.children[0].variable == "mass"
    assert restored.children[1].relation == "contained_in"
    assert restored.children[1].target_entity == "box_1"
    assert restored.describe() == pred.describe()
