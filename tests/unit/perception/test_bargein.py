"""Tests for barge-in (mid-response interruption) support."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.snn.expression.expression_stream import ExpressionStream
from hbllm.brain.snn.expression.models import (
    ThoughtFragment,
    ThoughtGoal,
)
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.thought_controller import ThoughtController
from hbllm.brain.snn.expression.thought_planner import ThoughtPlanner

# ── Helpers ──────────────────────────────────────────────────────────────


class FakePlanner(ThoughtPlanner):
    """Planner that returns a configurable number of goals."""

    def __init__(self, num_goals: int = 5):
        super().__init__()
        self._num_goals = num_goals

    def plan(self, understanding):
        return [
            ThoughtGoal(id=f"goal_{i}", text=f"Goal {i}", priority=1.0 - i * 0.1)
            for i in range(self._num_goals)
        ]


class SlowExpressionStream(ExpressionStream):
    """ExpressionStream that simulates slow fragment generation.

    Each fragment takes `delay` seconds, allowing us to interrupt mid-generation.
    """

    def __init__(self, delay: float = 0.1, num_goals: int = 5):
        planner = FakePlanner(num_goals)
        controller = ThoughtController()
        evaluator = RewardEvaluator()
        super().__init__(
            planner=planner,
            controller=controller,
            evaluator=evaluator,
            llm_generate=None,
            enable_gating=False,
        )
        self._delay = delay

    async def _generate_for_goal(self, goal, base_thought, original_query, prev_fragment_text):
        """Simulate slow LLM generation."""
        await asyncio.sleep(self._delay)
        return ThoughtFragment(
            goal_id=goal.id,
            text=f"Response for {goal.id}",
            metadata={"tokens": 10, "source": "test"},
        )


# ── ExpressionStream Cancellation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_event_not_set():
    """When cancel_event is never set, all fragments should complete."""
    stream = SlowExpressionStream(delay=0.01, num_goals=3)
    cancel = asyncio.Event()

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="base",
        original_query="test query",
        cancel_event=cancel,
    )

    # All 3 goals should produce fragments
    assert len(result.fragments) == 3
    assert result.metadata.get("interrupted") is not True


@pytest.mark.asyncio
async def test_cancel_event_interrupts_generation():
    """Setting cancel_event should stop generation between fragments."""
    stream = SlowExpressionStream(delay=0.05, num_goals=5)
    cancel = asyncio.Event()
    fragments_emitted: list[str] = []

    async def track_fragment(fragment):
        fragments_emitted.append(fragment.goal_id)

    stream.on_fragment = track_fragment

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    async def interrupt_after_delay():
        await asyncio.sleep(0.12)  # After ~2 fragments
        cancel.set()

    # Start generation and interruption concurrently
    interrupt_task = asyncio.create_task(interrupt_after_delay())

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="base",
        original_query="test query",
        cancel_event=cancel,
    )

    await interrupt_task

    # Should have fewer than 5 fragments (interrupted early)
    assert len(result.fragments) < 5
    assert result.metadata.get("interrupted") is True
    assert result.metadata.get("fragments_total") == 5
    assert result.metadata.get("fragments_completed") < 5


@pytest.mark.asyncio
async def test_cancel_event_already_set():
    """If cancel_event is set before generation starts, should produce 0 fragments."""
    stream = SlowExpressionStream(delay=0.01, num_goals=3)
    cancel = asyncio.Event()
    cancel.set()  # Already cancelled

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="fallback text",
        original_query="test query",
        cancel_event=cancel,
    )

    # No fragments should be generated
    assert len(result.fragments) == 0
    assert result.metadata.get("interrupted") is True
    # Text should be the assembled base_thought (fallback)
    assert "fallback" in result.text.lower()


@pytest.mark.asyncio
async def test_cancel_event_none():
    """When cancel_event is None, should behave normally (no interruption check)."""
    stream = SlowExpressionStream(delay=0.01, num_goals=3)

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="base",
        original_query="test query",
        cancel_event=None,  # Explicitly None
    )

    assert len(result.fragments) == 3


@pytest.mark.asyncio
async def test_partial_result_has_text():
    """Interrupted generation should still have assembled text from completed fragments."""
    stream = SlowExpressionStream(delay=0.05, num_goals=5)
    cancel = asyncio.Event()

    class FakeUnderstanding:
        concepts = []
        domain_activations = {}
        all_memories = []
        salience_map = []

    async def interrupt_soon():
        await asyncio.sleep(0.08)  # After ~1 fragment
        cancel.set()

    task = asyncio.create_task(interrupt_soon())

    result = await stream.express(
        understanding=FakeUnderstanding(),
        base_thought="base",
        original_query="query",
        cancel_event=cancel,
    )

    await task

    # Should have some text even though interrupted
    assert result.text  # Not empty
    assert len(result.fragments) >= 1


# ── SSE Interrupt Marker ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sse_interrupt_marker():
    """The __INTERRUPTED__ marker should be distinguishable from normal tokens."""
    import json

    # Simulate what the SSE generator does
    chunk = "__INTERRUPTED__"
    if chunk == "__INTERRUPTED__":
        event = json.dumps({"token": "", "done": True, "interrupted": True})
    else:
        event = json.dumps({"token": chunk, "done": False})

    parsed = json.loads(event)
    assert parsed["done"] is True
    assert parsed["interrupted"] is True
