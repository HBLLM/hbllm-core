import pytest

from hbllm.brain.autonomy.task_graph import Goal
from hbllm.brain.cognitive_state import CognitiveBudget, CognitivePolicy, CognitiveState
from hbllm.brain.simulation.engine import (
    LayeredSimulationEngine,
    MemoryBeliefSimulator,
    SafetySimulator,
    SocialSimulator,
)


@pytest.mark.asyncio
async def test_safety_simulator_blocks():
    state = CognitiveState(goal=Goal(), policy=CognitivePolicy())
    sim = SafetySimulator()

    # Normal action
    risk = await sim.simulate(state, {"type": "action", "action": "echo Hello"})
    assert risk == 0.0

    # Dangerous action
    risk = await sim.simulate(state, {"type": "action", "action": "rm -rf /some/path"})
    assert risk == 1.0


@pytest.mark.asyncio
async def test_social_simulator():
    # Low attention budget -> high risk for notification
    low_attn_policy = CognitivePolicy(budget=CognitiveBudget(attention_budget=0.3))
    state = CognitiveState(goal=Goal(), policy=low_attn_policy)
    sim = SocialSimulator()

    risk = await sim.simulate(state, {"type": "action", "action": "alert", "is_notification": True})
    assert risk == 0.8


@pytest.mark.asyncio
async def test_memory_belief_contradiction():
    state = CognitiveState(
        goal=Goal(), policy=CognitivePolicy(), beliefs=[{"topic": "capital", "value": "paris"}]
    )
    sim = MemoryBeliefSimulator()

    # Matching topic, different value -> contradiction!
    risk = await sim.simulate(
        state, {"type": "belief_update", "belief": {"topic": "capital", "value": "london"}}
    )
    assert risk == 0.9

    # Matching topic, same value -> no risk
    risk = await sim.simulate(
        state, {"type": "belief_update", "belief": {"topic": "capital", "value": "paris"}}
    )
    assert risk == 0.0


@pytest.mark.asyncio
async def test_engine_orchestration():
    state = CognitiveState(goal=Goal(), policy=CognitivePolicy())
    engine = LayeredSimulationEngine()

    # Allowed action
    res = await engine.simulate_mutation(state, {"type": "action", "action": "ls -la"})
    assert res["allowed"] is True
    assert res["risk_score"] < 0.8

    # Blocked action
    res = await engine.simulate_mutation(state, {"type": "action", "action": "rm -rf /"})
    assert res["allowed"] is False
    assert res["risk_score"] == 1.0
