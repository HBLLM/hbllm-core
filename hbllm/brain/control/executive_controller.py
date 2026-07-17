"""
Executive Controller — tripartite orchestrator for the cognitive event loop.

Coordinates three cooperating control loops (ADR 002 §1):

    1. **ReactiveController**: Sub-10ms reflex arcs, interrupts, safety.
    2. **DeliberativeController**: Multi-step GoT planning and reasoning.
    3. **ReflectiveController**: Post-execution evaluation and memory events.

The ExecutiveController itself knows **only interfaces** — every subsystem
is swappable.  It contains NO cognition of its own; it simply wires:

    IEventQueue → ReactiveController → IAttentionSelector → ICompetition
                                                                ↓
                                                            IWorkspace
                                                                ↓
                                                       ReflectiveController

Pipeline per cycle::

    1. Drain events from the queue
    2. Route urgent events through ReactiveController
    3. Score remaining events for saliency (attention)
    4. Run WTA competition (select top-K)
    5. Route winners to the workspace for reasoning
    6. Update cognitive state via deltas

The controller can run as:
    - **Single cycle** (``run_cycle``): Called from existing pipeline
    - **Continuous loop** (``run_continuous``): Async background task
      with configurable cycle rate

Design principles:
    - No direct imports of concrete classes — only interfaces
    - No business logic — just orchestration
    - Deterministic given the same inputs
    - Observable via stats and event logging
    - Tripartite dispatch: reactive → deliberative → reflective

Usage::

    from hbllm.brain.control.executive_controller import ExecutiveController

    controller = ExecutiveController(
        queue=event_queue,
        attention=saliency_evaluator,
        competition=competition_engine,
        workspace=workspace_adapter,
    )
    processed = await controller.run_cycle()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.brain.control.reactive_controller import ReactiveController
from hbllm.brain.control.reflective_controller import ReflectiveController
from hbllm.brain.core.cognitive_event import CognitiveEvent
from hbllm.brain.core.cognitive_interfaces import (
    IAttentionSelector,
    ICompetition,
    IEventQueue,
    IGoalProvider,
    IWorkspace,
)
from hbllm.brain.core.cognitive_state import (
    CognitiveStateDelta,
    CognitiveStateReducer,
    CognitiveStateSnapshot,
)

logger = logging.getLogger(__name__)


class ExecutiveController:
    """Thin orchestrator for the cognitive event loop.

    Wires: IEventQueue → IAttentionSelector → ICompetition → IWorkspace
    References: IGoalProvider (optional), CognitiveState (read-only)

    Contains NO cognition.  All intelligence lives in the plugged-in
    implementations.

    Args:
        queue: Event queue to drain events from.
        attention: Saliency evaluator to score events.
        competition: WTA engine to select winners.
        workspace: Target for winning events.
        goals: Optional goal provider for context.
        state: Initial cognitive state snapshot.
        reactive: Optional ReactiveController for reflex handling.
        reflective: Optional ReflectiveController for post-eval.
        max_batch_size: Maximum events per cycle.
        cycle_interval: Seconds between continuous cycles.
    """

    def __init__(
        self,
        queue: IEventQueue,
        attention: IAttentionSelector,
        competition: ICompetition,
        workspace: IWorkspace,
        goals: IGoalProvider | None = None,
        state: CognitiveStateSnapshot | None = None,
        reactive: ReactiveController | None = None,
        reflective: ReflectiveController | None = None,
        max_batch_size: int = 10,
        cycle_interval: float = 0.1,
    ) -> None:
        self._queue = queue
        self._attention = attention
        self._competition = competition
        self._workspace = workspace
        self._goals = goals
        self._reactive = reactive
        self._reflective = reflective

        # Cognitive state management
        self._state = state or CognitiveStateSnapshot()
        self._reducer = CognitiveStateReducer()

        # Configuration
        self._max_batch = max_batch_size
        self._cycle_interval = cycle_interval

        # Telemetry
        self._cycle_count = 0
        self._total_events_processed = 0
        self._total_winners = 0
        self._total_reactive_handled = 0
        self._running = False
        self._run_task: asyncio.Task[None] | None = None

        logger.info(
            "ExecutiveController initialized (batch=%d, interval=%.2fs, "
            "reactive=%s, reflective=%s)",
            max_batch_size,
            cycle_interval,
            "enabled" if reactive else "disabled",
            "enabled" if reflective else "disabled",
        )

    # ── State access ─────────────────────────────────────────────────

    @property
    def cognitive_state(self) -> CognitiveStateSnapshot:
        """Current cognitive state (read-only snapshot)."""
        return self._state

    def apply_delta(self, delta: CognitiveStateDelta) -> CognitiveStateSnapshot:
        """Apply a cognitive state delta and return the new state.

        Also updates the saliency evaluator's state reference.

        Args:
            delta: The change to apply.

        Returns:
            The new cognitive state.
        """
        self._state = self._reducer.apply(self._state, delta)

        # Propagate to attention selector if it supports state updates
        if hasattr(self._attention, "update_cognitive_state"):
            self._attention.update_cognitive_state(self._state)

        return self._state

    # ── Single cycle ─────────────────────────────────────────────────

    async def run_cycle(self) -> list[CognitiveEvent]:
        """Execute one cycle of the cognitive event loop.

        Pipeline:
            1. Drain up to ``max_batch_size`` events from the queue
            2. Score events for saliency
            3. Run WTA competition
            4. Route winners to workspace
            5. Return winners for caller inspection

        Returns:
            List of winning events that were routed to workspace.
        """
        cycle_start = time.monotonic()

        # 1. Drain events
        raw_events = await self._queue.drain(self._max_batch)
        if not raw_events:
            return []

        # 2. Route through ReactiveController (if available)
        deliberative_events: list[CognitiveEvent] = []
        if self._reactive:
            for event in raw_events:
                result = await self._reactive.process(event)
                if not result.handled:
                    deliberative_events.append(event)
                else:
                    self._total_reactive_handled += 1
        else:
            deliberative_events = list(raw_events)

        if not deliberative_events:
            self._cycle_count += 1
            self._total_events_processed += len(raw_events)
            return []

        # 3. Score for saliency
        scored_events = await self._attention.evaluate(deliberative_events)

        # 3. Competition (WTA)
        winners = await self._competition.compete(scored_events)

        # 4. Route winners to workspace
        for event in winners:
            try:
                await self._workspace.submit_for_reasoning(event)
            except Exception as e:
                logger.warning(
                    "Failed to route event %r to workspace: %s",
                    event,
                    e,
                )

        # 5. Telemetry
        self._cycle_count += 1
        self._total_events_processed += len(raw_events)
        self._total_winners += len(winners)

        cycle_time = time.monotonic() - cycle_start

        if self._cycle_count % 100 == 0:
            logger.info(
                "ExecutiveController cycle %d: %d events → %d scored → %d winners (%.3fs)",
                self._cycle_count,
                len(raw_events),
                len(scored_events),
                len(winners),
                cycle_time,
            )

        return winners

    # ── Continuous loop ──────────────────────────────────────────────

    async def run_continuous(self) -> None:
        """Run the cognitive loop continuously as an async task.

        Loops until ``stop()`` is called.  Sleeps for ``cycle_interval``
        between cycles if the queue is empty.
        """
        self._running = True
        logger.info("ExecutiveController: starting continuous loop")

        try:
            while self._running:
                queue_size = await self._queue.size()
                if queue_size > 0:
                    await self.run_cycle()
                else:
                    await asyncio.sleep(self._cycle_interval)
        except asyncio.CancelledError:
            logger.info("ExecutiveController: continuous loop cancelled")
        finally:
            self._running = False

    def start(self) -> None:
        """Start the continuous loop as a background task."""
        if self._running:
            logger.warning("ExecutiveController already running")
            return
        self._run_task = asyncio.ensure_future(self.run_continuous())

    async def stop(self) -> None:
        """Stop the continuous loop gracefully."""
        self._running = False
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        self._run_task = None
        logger.info("ExecutiveController: stopped")

    # ── Event submission helper ──────────────────────────────────────

    async def submit_event(self, event: CognitiveEvent) -> None:
        """Convenience: submit an event directly to the queue.

        Args:
            event: The cognitive event to submit.
        """
        await self._queue.submit(event)

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Controller statistics."""
        return {
            "cycle_count": self._cycle_count,
            "total_events_processed": self._total_events_processed,
            "total_winners": self._total_winners,
            "avg_winners_per_cycle": (round(self._total_winners / max(1, self._cycle_count), 2)),
            "cognitive_state_version": self._state.version,
            "running": self._running,
        }
