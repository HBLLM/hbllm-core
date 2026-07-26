"""
Execution Bus — first-class OS subsystem for event-sourced execution dispatch.

Not just transport — a pillar of the Execution OS platform.

Domain events (not callbacks):
    ExecutionSubmitted → Started → Progress →
    Completed / Failed / Cancelled / Retried

Enables:
    - Audit history (every execution is journaled)
    - Replay (replay events to reproduce execution)
    - Swarm subscription (remote nodes subscribe)
    - Observability (automatic telemetry)
    - Debugging (full event timeline per execution)
    - Distributed execution (transparent remote dispatch)
    - Cancellation and rescheduling (first-class operations)

Persists events in the same journal philosophy as HCIR.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.execution.events import ExecutionEvent, ExecutionEventData
from hbllm.execution.plan import ExecutionPlan
from hbllm.execution.result import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionHandle:
    """
    Handle returned when a plan is submitted to the bus.

    Used for tracking, cancellation, and waiting.
    """

    handle_id: str
    plan_id: str
    trace_id: str | None = None
    _future: asyncio.Future[ExecutionResult] | None = field(default=None, repr=False)


class Subscription:
    """An event subscription that can be unsubscribed."""

    def __init__(self, event: ExecutionEvent, handler: Callable[..., Any], bus: ExecutionBus):
        self.event = event
        self.handler = handler
        self._bus = bus
        self._active = True

    async def unsubscribe(self) -> None:
        """Remove this subscription."""
        if self._active:
            self._bus._remove_subscription(self)
            self._active = False


class ExecutionBus:
    """
    First-class execution subsystem.

    Sits between ExecutionOrchestrator and RuntimeRegistry.
    All execution flows through the bus, making every execution
    observable, replayable, and distributable.
    """

    def __init__(self, journal_enabled: bool = True) -> None:
        self._handlers: dict[ExecutionEvent, list[Callable[..., Any]]] = defaultdict(list)
        self._pending: dict[str, ExecutionHandle] = {}
        self._journal: list[ExecutionEventData] = []
        self._journal_enabled = journal_enabled
        self._runtime_handler: Callable[[ExecutionPlan], Any] | None = None

    def set_runtime_handler(self, handler: Callable[[ExecutionPlan], Any]) -> None:
        """
        Set the handler that actually executes plans.

        This is called by the Orchestrator to wire the bus
        to the RuntimeRegistry.
        """
        self._runtime_handler = handler

    async def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        """
        Submit a plan for execution.

        Emits ExecutionSubmitted, dispatches to runtime handler,
        returns a handle for tracking.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExecutionResult] = loop.create_future()

        handle = ExecutionHandle(
            handle_id=str(uuid.uuid4()),
            plan_id=plan.plan_id,
            trace_id=plan.trace_id,
            _future=future,
        )
        self._pending[handle.handle_id] = handle

        # Journal the submission
        await self._emit(ExecutionEventData.submitted(plan.plan_id, plan.trace_id))

        # Dispatch execution asynchronously
        asyncio.create_task(self._dispatch(plan, handle))

        return handle

    async def cancel(self, handle: ExecutionHandle) -> None:
        """Cancel a pending or running execution."""
        if handle._future and not handle._future.done():
            handle._future.cancel()
        await self._emit(ExecutionEventData.cancelled(handle.plan_id, handle.trace_id))
        self._pending.pop(handle.handle_id, None)
        logger.info("Execution cancelled: plan_id=%s", handle.plan_id)

    async def retry(self, handle: ExecutionHandle, plan: ExecutionPlan) -> ExecutionHandle:
        """Retry a failed execution with a new plan version."""
        retry_plan = plan.with_retry()
        await self._emit(
            ExecutionEventData.retried(
                plan.plan_id,
                retry_count=retry_plan.version - 1,
                trace_id=plan.trace_id,
            )
        )
        return await self.submit(retry_plan)

    async def subscribe(self, event: ExecutionEvent, handler: Callable[..., Any]) -> Subscription:
        """Subscribe to execution domain events."""
        self._handlers[event].append(handler)
        sub = Subscription(event, handler, self)
        logger.debug("Subscribed to %s", event.value)
        return sub

    async def wait(self, handle: ExecutionHandle) -> ExecutionResult:
        """Wait for an execution to complete."""
        if handle._future is None:
            raise RuntimeError(f"Handle {handle.handle_id} has no future")
        return await handle._future

    def journal(self, plan_id: str | None = None) -> list[ExecutionEventData]:
        """
        Get execution event history.

        Args:
            plan_id: Filter by plan_id. None returns all events.

        Returns:
            List of execution events in chronological order.
        """
        if plan_id is None:
            return list(self._journal)
        return [e for e in self._journal if e.plan_id == plan_id]

    @property
    def journal_size(self) -> int:
        return len(self._journal)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ── Internal ──────────────────────────────────────────────

    async def _dispatch(self, plan: ExecutionPlan, handle: ExecutionHandle) -> None:
        """Dispatch a plan to the runtime handler."""
        await self._emit(ExecutionEventData.started(plan.plan_id, plan.trace_id))

        try:
            if self._runtime_handler is None:
                raise RuntimeError("No runtime handler set on ExecutionBus")

            result = await self._runtime_handler(plan)

            await self._emit(
                ExecutionEventData.completed(
                    plan.plan_id,
                    trace_id=plan.trace_id,
                    content_length=len(result.content) if result.content else 0,
                )
            )

            if handle._future and not handle._future.done():
                handle._future.set_result(result)

        except asyncio.CancelledError:
            await self._emit(ExecutionEventData.cancelled(plan.plan_id, plan.trace_id))
            raise

        except Exception as exc:
            error_msg = str(exc)
            await self._emit(ExecutionEventData.failed(plan.plan_id, error_msg, plan.trace_id))
            if handle._future and not handle._future.done():
                handle._future.set_exception(exc)

        finally:
            self._pending.pop(handle.handle_id, None)

    async def _emit(self, event_data: ExecutionEventData) -> None:
        """Emit an event to subscribers and journal."""
        if self._journal_enabled:
            self._journal.append(event_data)

        handlers = self._handlers.get(event_data.event, [])
        for handler in handlers:
            try:
                result = handler(event_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in event handler for %s", event_data.event.value)

    def _remove_subscription(self, sub: Subscription) -> None:
        """Remove a subscription handler."""
        handlers = self._handlers.get(sub.event, [])
        if sub.handler in handlers:
            handlers.remove(sub.handler)
