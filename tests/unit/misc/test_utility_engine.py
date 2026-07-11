import time

import pytest

from hbllm.brain.evaluation.utility_engine import CognitiveUtilityEngine, ThoughtBudget
from hbllm.brain.planning.planner_node import PlannerNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from tests.mock_llm import MockLLM


def test_thought_budget_spent_limits():
    # Test token limit
    budget = ThoughtBudget(max_tokens=100, max_time_ms=10000.0, max_branches=5)
    assert not budget.is_exhausted()
    budget.spend_tokens(50)
    assert not budget.is_exhausted()
    budget.spend_tokens(50)
    assert budget.is_exhausted()

    # Test branch limit
    budget = ThoughtBudget(max_tokens=100, max_time_ms=10000.0, max_branches=5)
    assert not budget.is_exhausted()
    budget.spend_branch(4)
    assert not budget.is_exhausted()
    budget.spend_branch(1)
    assert budget.is_exhausted()

    # Test time limit
    budget = ThoughtBudget(max_tokens=100, max_time_ms=10.0, max_branches=5)
    assert not budget.is_exhausted()
    time.sleep(0.015)
    assert budget.is_exhausted()

    # Test serialization
    data = budget.to_dict()
    assert data["max_tokens"] == 100
    assert data["max_branches"] == 5
    assert data["is_exhausted"] is True


def test_cognitive_utility_calculation():
    engine = CognitiveUtilityEngine(
        weight_progress=1.0,
        weight_token=0.001,
        weight_latency=0.002,
        weight_risk=1.5,
    )
    # Utility = 0.8 - 0.001 * 100 - 0.002 * 50 - 1.5 * 0.2
    #         = 0.8 - 0.1 - 0.1 - 0.3
    #         = 0.3
    breakdown = engine.calculate_utility(
        progress_score=0.8,
        tokens_used=100,
        latency_ms=50,
        risk_score=0.2,
    )
    assert abs(breakdown.utility - 0.3) < 1e-6
    assert breakdown.progress_score == 0.8
    assert breakdown.token_cost == 0.1
    assert breakdown.latency_cost == 0.1
    assert abs(breakdown.risk_cost - 0.3) < 1e-6


@pytest.mark.asyncio
async def test_planner_node_with_budget_and_utility():
    bus = InProcessBus()
    await bus.start()

    mock_llm = MockLLM()

    # Mock score thought response
    async def handle_score_thought(msg: Message) -> Message:
        return msg.create_response({"score": 0.85})

    await bus.subscribe("action.score_thought", handle_score_thought)

    planner = PlannerNode(
        node_id="planner_test",
        branch_factor=2,
        max_depth=1,
        llm=mock_llm,
    )
    await planner.start(bus)

    # Dispatch request with custom thought budget
    req = Message(
        type=MessageType.QUERY,
        source_node_id="test_client",
        topic="planner.decompose",
        payload={
            "text": "test query",
            "thought_budget": {"max_tokens": 1000, "max_time_ms": 5000.0, "max_branches": 10},
        },
    )

    try:
        response = await bus.request("planner.decompose", req, timeout=5.0)
        assert response.type == MessageType.RESPONSE
        assert "thought_budget_status" in response.payload
        budget_status = response.payload["thought_budget_status"]
        assert budget_status["max_tokens"] == 1000
        assert budget_status["max_branches"] == 10
        assert budget_status["tokens_spent"] > 0
        assert budget_status["branches_spent"] > 0

        assert "best_path_utility" in response.payload
        utility_path = response.payload["best_path_utility"]
        assert len(utility_path) > 0
        # Check first breakdown
        for b in utility_path:
            if b:
                assert "utility" in b
                assert "progress_score" in b
                assert "token_cost" in b
    finally:
        await planner.stop()
        await bus.stop()
