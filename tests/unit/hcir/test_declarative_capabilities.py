"""Tests for Phase 5: Declarative Capabilities & Market-Based Scheduling."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.hcir.graph import CapabilityNode
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
)
from hbllm.hcir.kernel.scheduler import (
    CognitiveBudget,
    CognitiveProcess,
    CognitiveScheduler,
    CognitiveThread,
)

# ═══════════════════════════════════════════════════════════════════════════
# Mock Executor
# ═══════════════════════════════════════════════════════════════════════════


class MockExecutor:
    """Mock capability executor for testing."""

    def __init__(self, available: bool = True, result: dict[str, Any] | None = None) -> None:
        self._available = available
        self._result = result or {"status": "ok"}
        self.call_count = 0

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return self._result

    @property
    def is_available(self) -> bool:
        return self._available


# ═══════════════════════════════════════════════════════════════════════════
# CapabilityNode Extension Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCapabilityNodeExtensions:
    """Verify CapabilityNode declarative metadata fields."""

    def test_declarative_metadata_defaults(self) -> None:
        node = CapabilityNode(capability_name="test")
        assert node.estimated_cost == 0
        assert node.latency_ms == 0
        assert node.cooldown_seconds == 0.0
        assert node.requires_approval is False
        assert node.max_concurrent == 0
        assert node.provider == ""
        assert node.version == "1.0.0"

    def test_declarative_metadata_values(self) -> None:
        node = CapabilityNode(
            capability_name="image_gen",
            estimated_cost=5000,
            latency_ms=3000,
            cooldown_seconds=60.0,
            requires_approval=True,
            max_concurrent=2,
            provider="api",
            version="2.1.0",
        )
        assert node.estimated_cost == 5000
        assert node.latency_ms == 3000
        assert node.cooldown_seconds == 60.0
        assert node.requires_approval is True
        assert node.max_concurrent == 2
        assert node.provider == "api"
        assert node.version == "2.1.0"

    def test_backward_compatible(self) -> None:
        """Existing CapabilityNode creation without new fields should work."""
        node = CapabilityNode(
            capability_name="execute_python",
            description="Run Python code",
            input_schema={"code": "string"},
        )
        assert node.capability_name == "execute_python"
        assert node.estimated_cost == 0  # Default


# ═══════════════════════════════════════════════════════════════════════════
# Market-Based Capability Selection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMarketBasedSelection:
    """Verify resolve_cheapest and cost-based selection."""

    def setup_method(self) -> None:
        self.resolver = CapabilityResolver()
        self.cheap_exec = MockExecutor()
        self.expensive_exec = MockExecutor()
        self.fast_exec = MockExecutor()

        self.resolver.register(
            CapabilityImplementation(
                capability_name="summarize",
                implementation_id="cheap_model",
                executor=self.cheap_exec,
                estimated_cost=100,
                estimated_latency_ms=500,
                priority=5,
            )
        )
        self.resolver.register(
            CapabilityImplementation(
                capability_name="summarize",
                implementation_id="expensive_model",
                executor=self.expensive_exec,
                estimated_cost=5000,
                estimated_latency_ms=200,
                priority=10,
            )
        )
        self.resolver.register(
            CapabilityImplementation(
                capability_name="summarize",
                implementation_id="fast_model",
                executor=self.fast_exec,
                estimated_cost=300,
                estimated_latency_ms=50,
                priority=3,
            )
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_highest_priority(self) -> None:
        """Standard resolve() still picks highest priority."""
        executor = await self.resolver.resolve("summarize")
        assert executor is self.expensive_exec

    @pytest.mark.asyncio
    async def test_resolve_cheapest_no_constraints(self) -> None:
        """resolve_cheapest with no constraints returns cheapest."""
        impl = await self.resolver.resolve_cheapest("summarize")
        assert impl is not None
        assert impl.implementation_id == "cheap_model"
        assert impl.estimated_cost == 100

    @pytest.mark.asyncio
    async def test_resolve_cheapest_with_max_cost(self) -> None:
        """Filter by max_cost."""
        impl = await self.resolver.resolve_cheapest("summarize", max_cost=200)
        assert impl is not None
        assert impl.implementation_id == "cheap_model"

    @pytest.mark.asyncio
    async def test_resolve_cheapest_with_latency_constraint(self) -> None:
        """Filter by max_latency_ms — only fast_model qualifies."""
        impl = await self.resolver.resolve_cheapest("summarize", max_latency_ms=100)
        assert impl is not None
        assert impl.implementation_id == "fast_model"

    @pytest.mark.asyncio
    async def test_resolve_cheapest_no_match(self) -> None:
        """No implementation meets constraints."""
        impl = await self.resolver.resolve_cheapest("summarize", max_cost=10)
        assert impl is None

    @pytest.mark.asyncio
    async def test_resolve_cheapest_unavailable_filtered(self) -> None:
        """Unavailable executors are filtered out."""
        self.cheap_exec._available = False
        impl = await self.resolver.resolve_cheapest("summarize")
        assert impl is not None
        assert impl.implementation_id == "fast_model"  # Next cheapest


# ═══════════════════════════════════════════════════════════════════════════
# Resolve-and-Execute Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveAndExecute:
    """Verify resolve_and_execute end-to-end dispatch."""

    def setup_method(self) -> None:
        self.resolver = CapabilityResolver()
        self.executor = MockExecutor(result={"answer": 42})
        self.resolver.register(
            CapabilityImplementation(
                capability_name="compute",
                implementation_id="calculator",
                executor=self.executor,
                estimated_cost=50,
            )
        )

    @pytest.mark.asyncio
    async def test_resolve_and_execute_success(self) -> None:
        result = await self.resolver.resolve_and_execute("compute", {"x": 1})
        assert result == {"answer": 42}
        assert self.executor.call_count == 1

    @pytest.mark.asyncio
    async def test_resolve_and_execute_with_budget(self) -> None:
        result = await self.resolver.resolve_and_execute("compute", {"x": 1}, budget=100)
        assert result == {"answer": 42}
        assert self.resolver.total_cost == 50

    @pytest.mark.asyncio
    async def test_resolve_and_execute_budget_exceeded(self) -> None:
        result = await self.resolver.resolve_and_execute("compute", {"x": 1}, budget=10)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_resolve_and_execute_no_capability(self) -> None:
        result = await self.resolver.resolve_and_execute("nonexistent", {})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Budget Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveBudget:
    """Verify CognitiveBudget token tracking."""

    def test_initial_state(self) -> None:
        budget = CognitiveBudget(total_tokens=1000)
        assert budget.remaining == 1000
        assert budget.utilization == 0.0

    def test_consume(self) -> None:
        budget = CognitiveBudget(total_tokens=1000)
        assert budget.consume(300) is True
        assert budget.remaining == 700
        assert budget.consumed_tokens == 300

    def test_consume_exceeds_budget(self) -> None:
        budget = CognitiveBudget(total_tokens=100)
        assert budget.consume(50) is True
        assert budget.consume(60) is False  # Would exceed
        assert budget.remaining == 50  # Unchanged

    def test_can_afford(self) -> None:
        budget = CognitiveBudget(total_tokens=100)
        assert budget.can_afford(100) is True
        assert budget.can_afford(101) is False

    def test_reset(self) -> None:
        budget = CognitiveBudget(total_tokens=100)
        budget.consume(80)
        budget.reset()
        assert budget.remaining == 100


# ═══════════════════════════════════════════════════════════════════════════
# Budget-Aware Scheduling Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetAwareScheduling:
    """Verify CognitiveScheduler with budget constraints."""

    def test_enqueue_with_cost_within_budget(self) -> None:
        budget = CognitiveBudget(total_tokens=1000)
        scheduler = CognitiveScheduler(budget=budget)

        proc = CognitiveProcess(process_id="p1")
        thread = CognitiveThread(thread_id="t1")
        proc.add_thread(thread)
        scheduler.register_process(proc)

        result = scheduler.enqueue_with_cost("p1", "t1", estimated_cost=500)
        assert result is True
        assert budget.remaining == 500
        assert scheduler.queue_size == 1

    def test_enqueue_with_cost_exceeds_budget(self) -> None:
        budget = CognitiveBudget(total_tokens=100)
        scheduler = CognitiveScheduler(budget=budget)

        proc = CognitiveProcess(process_id="p1")
        thread = CognitiveThread(thread_id="t1")
        proc.add_thread(thread)
        scheduler.register_process(proc)

        result = scheduler.enqueue_with_cost("p1", "t1", estimated_cost=200)
        assert result is False
        assert budget.remaining == 100  # Unchanged
        assert scheduler.queue_size == 0

    def test_enqueue_without_budget(self) -> None:
        scheduler = CognitiveScheduler()  # No budget

        proc = CognitiveProcess(process_id="p1")
        thread = CognitiveThread(thread_id="t1")
        proc.add_thread(thread)
        scheduler.register_process(proc)

        result = scheduler.enqueue_with_cost("p1", "t1", estimated_cost=99999)
        assert result is True  # No budget = unlimited

    def test_budget_property(self) -> None:
        budget = CognitiveBudget(total_tokens=5000)
        scheduler = CognitiveScheduler(budget=budget)
        assert scheduler.budget is budget
        assert scheduler.budget.total_tokens == 5000
