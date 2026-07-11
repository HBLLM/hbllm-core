"""Tests for the Predictive Simulation Engine."""

from __future__ import annotations

import pytest

from hbllm.brain.autonomy.task_graph import Goal, TaskNode
from hbllm.brain.causality.causal_graph import CausalGraph
from hbllm.brain.simulation.engine import PredictiveSimulationEngine
from hbllm.brain.simulation.models import PredictionOrigin
from hbllm.brain.simulation.risk import RiskEngine
from hbllm.brain.world.world_state import WorldStateEngine


@pytest.fixture
def sim_engine(tmp_path):
    world_state = WorldStateEngine()
    causal_graph = CausalGraph(data_dir=tmp_path)
    risk_engine = RiskEngine()
    return PredictiveSimulationEngine(world_state, causal_graph, risk_engine)


def test_tier0_heuristic_simulation(sim_engine):
    goal = Goal(name="Test Goal")
    tasks = [TaskNode(name="Task 1", action_topic="test.topic")]

    scenario = sim_engine.simulate_plan(goal, tasks, tier=0)

    assert scenario is not None
    assert scenario.predicted_state.prediction_origin == PredictionOrigin.INFERRED
    # Tier 0 does not mutate
    assert len(scenario.predicted_state.mutations) == 0


def test_tier1_speculative_simulation(sim_engine):
    goal = Goal(name="Test Goal")
    # This hits deterministic rule in estimator
    tasks = [TaskNode(name="Sleep", action_topic="system.sleep")]

    scenario = sim_engine.simulate_plan(goal, tasks, tier=1)

    assert scenario is not None
    assert scenario.predicted_state.prediction_origin == PredictionOrigin.INFERRED
    assert "system_state" in scenario.predicted_state.mutations
    assert scenario.predicted_state.mutations["system_state"]["properties"]["status"] == "sleeping"


def test_speculative_fallback_risk(sim_engine):
    goal = Goal(name="Test Goal")
    # Unknown topic -> hits speculative fallback
    tasks = [TaskNode(name="Unknown", action_topic="unknown.action")]

    scenario = sim_engine.simulate_plan(goal, tasks, tier=1)

    assert scenario is not None
    assert scenario.predicted_state.prediction_origin == PredictionOrigin.SPECULATIVE
    # Speculative origin should trigger RELIABILITY risk in basic heuristic
    assert scenario.risk_categories.get("reliability") == 0.8
    # Score should be reduced due to composite risk
    assert scenario.predicted_state.risk_score == 0.8
    assert scenario.utility_score == 0.8
    # 0.8 * (1.0 - 0.8) = 0.16
    assert round(sim_engine.risk_engine.evaluate_scenario(scenario), 2) == 0.16
