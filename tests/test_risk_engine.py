"""Tests for the Simulation Risk Engine."""

from __future__ import annotations

import pytest

from hbllm.brain.simulation.models import CounterfactualScenario, FutureWorldState, PredictionOrigin
from hbllm.brain.simulation.risk import RiskCategory, RiskEngine, RiskProfile


@pytest.fixture
def risk_engine():
    profile = RiskProfile(
        risk_thresholds={
            RiskCategory.RESOURCE: 0.8,
            RiskCategory.RELIABILITY: 0.9,
        }
    )
    return RiskEngine(profile=profile)


def test_scenario_no_risk(risk_engine):
    """A scenario with no risk retains its utility."""
    scenario = CounterfactualScenario(
        goal_id="g1",
        predicted_state=FutureWorldState(
            state_id="s1",
            base_clock=1,
            mutations={},
            predicted_confidence=1.0,
            prediction_origin=PredictionOrigin.INFERRED,
        ),
        utility_score=1.0,
        risk_categories={},
    )
    score = risk_engine.evaluate_scenario(scenario)
    assert score == 1.0
    assert scenario.predicted_state.risk_score == 0.0


def test_scenario_moderate_risk(risk_engine):
    """A scenario with moderate risk gets scaled down."""
    scenario = CounterfactualScenario(
        goal_id="g1",
        predicted_state=FutureWorldState(
            state_id="s1",
            base_clock=1,
            mutations={},
            predicted_confidence=1.0,
            prediction_origin=PredictionOrigin.INFERRED,
        ),
        utility_score=1.0,
        risk_categories={
            RiskCategory.RESOURCE.value: 0.4,  # under 0.8 threshold
            RiskCategory.RELIABILITY.value: 0.6,  # under 0.9 threshold
        },
    )
    score = risk_engine.evaluate_scenario(scenario)
    # composite_risk = (0.4 + 0.6) / 2 = 0.5
    # score = 1.0 * (1.0 - 0.5) = 0.5
    assert score == 0.5
    assert scenario.predicted_state.risk_score == 0.5


def test_scenario_exceeds_threshold(risk_engine):
    """A scenario exceeding a risk threshold is heavily penalized."""
    scenario = CounterfactualScenario(
        goal_id="g1",
        predicted_state=FutureWorldState(
            state_id="s1",
            base_clock=1,
            mutations={},
            predicted_confidence=1.0,
            prediction_origin=PredictionOrigin.INFERRED,
        ),
        utility_score=1.0,
        risk_categories={
            RiskCategory.RESOURCE.value: 0.9,  # over 0.8 threshold!
        },
    )
    score = risk_engine.evaluate_scenario(scenario)
    # utility drops to 0.1 because threshold exceeded.
    # composite_risk = 0.9.
    # final_score = 0.1 * (1.0 - 0.9) = 0.01
    assert score < 0.05
