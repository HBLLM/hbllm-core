"""Tests for fragment-level streaming in ExpressionStream."""

from __future__ import annotations

import pytest

from hbllm.brain.snn.expression.expression_stream import ExpressionStream, FragmentCallback
from hbllm.brain.snn.expression.models import (
    ExpressionResult,
    ThoughtFragment,
    ThoughtGoal,
)
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.thought_controller import ThoughtController
from hbllm.brain.snn.expression.thought_planner import ThoughtPlanner

# ── Helpers ──────────────────────────────────────────────────────────────


class StubPlanner(ThoughtPlanner):
    """Returns a fixed list of goals."""

    def __init__(self, goals: list[ThoughtGoal]) -> None:
        super().__init__()
        self._goals = goals

    def plan(self, understanding):  # noqa: ANN001
        return self._goals


async def _stub_llm(prompt: str) -> str:
    return f"Generated for: {prompt[:30]}"


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_fragment_called_for_each_goal():
    """Verify on_fragment fires once per ThoughtGoal."""
    goals = [
        ThoughtGoal(id="g1", text="explain concept A"),
        ThoughtGoal(id="g2", text="explain concept B"),
        ThoughtGoal(id="g3", text="explain concept C"),
    ]
    planner = StubPlanner(goals)
    controller = ThoughtController()
    evaluator = RewardEvaluator()

    received_fragments: list[ThoughtFragment] = []

    async def capture_fragment(fragment: ThoughtFragment) -> None:
        received_fragments.append(fragment)

    stream = ExpressionStream(
        planner=planner,
        controller=controller,
        evaluator=evaluator,
        llm_generate=_stub_llm,
        enable_gating=False,
        on_fragment=capture_fragment,
    )

    # Minimal understanding stub
    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="base thought text",
        original_query="test query",
    )

    assert isinstance(result, ExpressionResult)
    assert len(received_fragments) == len(goals), (
        f"Expected {len(goals)} fragments, got {len(received_fragments)}"
    )
    # Each fragment should correspond to a goal
    received_goal_ids = {f.goal_id for f in received_fragments}
    expected_goal_ids = {g.id for g in goals}
    assert received_goal_ids == expected_goal_ids


@pytest.mark.asyncio
async def test_on_fragment_not_called_when_none():
    """Verify no error when on_fragment is None."""
    goals = [ThoughtGoal(id="g1", text="test")]
    planner = StubPlanner(goals)
    controller = ThoughtController()
    evaluator = RewardEvaluator()

    stream = ExpressionStream(
        planner=planner,
        controller=controller,
        evaluator=evaluator,
        llm_generate=_stub_llm,
        enable_gating=False,
        on_fragment=None,
    )

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="test",
        original_query="test",
    )
    assert isinstance(result, ExpressionResult)
    assert result.thought_count == 1


@pytest.mark.asyncio
async def test_on_fragment_error_is_nonfatal():
    """Verify that a failing on_fragment callback doesn't crash the pipeline."""
    goals = [
        ThoughtGoal(id="g1", text="first"),
        ThoughtGoal(id="g2", text="second"),
    ]
    planner = StubPlanner(goals)
    controller = ThoughtController()
    evaluator = RewardEvaluator()

    call_count = 0

    async def failing_callback(fragment: ThoughtFragment) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Simulated streaming failure")

    stream = ExpressionStream(
        planner=planner,
        controller=controller,
        evaluator=evaluator,
        llm_generate=_stub_llm,
        enable_gating=False,
        on_fragment=failing_callback,
    )

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="test",
        original_query="test",
    )

    # Pipeline should still complete despite callback errors
    assert isinstance(result, ExpressionResult)
    assert len(result.fragments) == 2
    # Callback was still invoked for both fragments
    assert call_count == 2


@pytest.mark.asyncio
async def test_fragment_callback_type_export():
    """Verify FragmentCallback is importable from expression_stream."""
    assert callable(FragmentCallback) or FragmentCallback is not None


@pytest.mark.asyncio
async def test_expression_result_has_metadata():
    """Verify ExpressionResult has the metadata field."""
    result = ExpressionResult(text="test", metadata={"mode": "broca"})
    assert result.metadata["mode"] == "broca"
