"""Tests for ExecutionBus — event-sourced execution dispatch."""

from __future__ import annotations

import asyncio

import pytest

from hbllm.execution.bus import ExecutionBus
from hbllm.execution.events import ExecutionEvent, ExecutionEventData
from hbllm.execution.plan import ExecutionPlan
from hbllm.execution.result import ExecutionResult


@pytest.fixture()
def bus() -> ExecutionBus:
    return ExecutionBus(journal_enabled=True)


class TestExecutionBus:
    @pytest.mark.asyncio()
    async def test_submit_and_wait(self, bus: ExecutionBus) -> None:
        """Plans submitted to the bus should execute and return results."""

        # Set up a mock runtime handler
        async def mock_handler(plan: ExecutionPlan) -> ExecutionResult:
            return ExecutionResult(content="Hello from runtime", plan_id=plan.plan_id)

        bus.set_runtime_handler(mock_handler)

        plan = ExecutionPlan(provider="mock")
        handle = await bus.submit(plan)

        result = await bus.wait(handle)
        assert result.content == "Hello from runtime"
        assert result.plan_id == plan.plan_id

    @pytest.mark.asyncio()
    async def test_journal_records_events(self, bus: ExecutionBus) -> None:
        """The bus should journal all domain events."""

        async def mock_handler(plan: ExecutionPlan) -> ExecutionResult:
            return ExecutionResult(content="ok", plan_id=plan.plan_id)

        bus.set_runtime_handler(mock_handler)

        plan = ExecutionPlan()
        handle = await bus.submit(plan)
        await bus.wait(handle)

        journal = bus.journal(plan.plan_id)
        event_types = [e.event for e in journal]
        assert ExecutionEvent.SUBMITTED in event_types
        assert ExecutionEvent.STARTED in event_types
        assert ExecutionEvent.COMPLETED in event_types

    @pytest.mark.asyncio()
    async def test_journal_records_failure(self, bus: ExecutionBus) -> None:
        """Failed executions should be journaled."""

        async def failing_handler(plan: ExecutionPlan) -> ExecutionResult:
            raise RuntimeError("Provider unavailable")

        bus.set_runtime_handler(failing_handler)

        plan = ExecutionPlan()
        handle = await bus.submit(plan)

        with pytest.raises(RuntimeError, match="Provider unavailable"):
            await bus.wait(handle)

        journal = bus.journal(plan.plan_id)
        event_types = [e.event for e in journal]
        assert ExecutionEvent.FAILED in event_types

    @pytest.mark.asyncio()
    async def test_cancel(self, bus: ExecutionBus) -> None:
        """Cancelled executions should emit a cancel event."""

        async def slow_handler(plan: ExecutionPlan) -> ExecutionResult:
            await asyncio.sleep(10)
            return ExecutionResult(content="should not reach")

        bus.set_runtime_handler(slow_handler)

        plan = ExecutionPlan()
        handle = await bus.submit(plan)

        # Give it a moment to start
        await asyncio.sleep(0.05)

        await bus.cancel(handle)

        journal = bus.journal(plan.plan_id)
        event_types = [e.event for e in journal]
        assert ExecutionEvent.CANCELLED in event_types

    @pytest.mark.asyncio()
    async def test_subscribe(self, bus: ExecutionBus) -> None:
        """Subscribers should receive domain events."""
        received: list[ExecutionEventData] = []

        async def on_completed(event: ExecutionEventData) -> None:
            received.append(event)

        await bus.subscribe(ExecutionEvent.COMPLETED, on_completed)

        async def mock_handler(plan: ExecutionPlan) -> ExecutionResult:
            return ExecutionResult(content="done", plan_id=plan.plan_id)

        bus.set_runtime_handler(mock_handler)

        plan = ExecutionPlan()
        handle = await bus.submit(plan)
        await bus.wait(handle)

        assert len(received) == 1
        assert received[0].event == ExecutionEvent.COMPLETED
        assert received[0].plan_id == plan.plan_id

    @pytest.mark.asyncio()
    async def test_no_runtime_handler_raises(self, bus: ExecutionBus) -> None:
        """Submitting without a runtime handler should raise."""
        plan = ExecutionPlan()
        handle = await bus.submit(plan)

        with pytest.raises(RuntimeError, match="No runtime handler"):
            await bus.wait(handle)

    @pytest.mark.asyncio()
    async def test_pending_count(self, bus: ExecutionBus) -> None:
        """Pending count should track in-flight executions."""
        assert bus.pending_count == 0

        event = asyncio.Event()

        async def blocking_handler(plan: ExecutionPlan) -> ExecutionResult:
            await event.wait()
            return ExecutionResult(content="done")

        bus.set_runtime_handler(blocking_handler)

        plan = ExecutionPlan()
        handle = await bus.submit(plan)

        # Give dispatch a moment to start
        await asyncio.sleep(0.05)
        # It may still be pending (depends on timing)

        event.set()
        await bus.wait(handle)
        assert bus.pending_count == 0
