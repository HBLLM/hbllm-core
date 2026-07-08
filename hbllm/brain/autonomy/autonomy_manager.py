"""Autonomy Manager & Proactive Coordinator.

Coordinates passive presence monitoring, registers opportunity sources,
manages global budgets and rate-limits, and publishes selected opportunities.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.brain.autonomy.opportunity import Opportunity, OpportunityHistory
from hbllm.brain.autonomy.opportunity_source import OpportunityScorer, OpportunitySource
from hbllm.brain.autonomy.presence_state import PresenceState
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class PresenceMonitor:
    """Passive manager that updates a PresenceState based on reported activities."""

    def __init__(self) -> None:
        self.state = PresenceState()
        self._last_decay_time = time.time()

    def report_activity(self, source: str, category: str, timestamp: float) -> None:
        """Record activity from a source and update the PresenceState."""
        now = time.time()
        elapsed = now - self._last_decay_time
        if elapsed > 10.0:  # decay every 10 seconds
            self.state.decay_engagement(elapsed)
            self._last_decay_time = now

        if category == "user_input":
            self.state.update_user_activity(timestamp)
        elif category == "ai_output":
            self.state.update_ai_activity(timestamp)
        else:
            self.state.update_sensor_activity(source, timestamp)


class ProactiveCoordinator:
    """Central coordinator for rate-limiting, budgeting, and routing opportunities."""

    def __init__(
        self,
        history: OpportunityHistory,
        min_gap_seconds: float = 600.0,
        daily_budget: int = 10,
    ) -> None:
        self.history = history
        self.min_gap_seconds = min_gap_seconds
        self.daily_budget = daily_budget

        self._last_trigger_time = 0.0
        self._triggers_today = 0
        self._last_budget_reset = time.time()

    def _reset_budget_if_needed(self) -> None:
        now = time.time()
        if now - self._last_budget_reset >= 86400.0:
            self._triggers_today = 0
            self._last_budget_reset = now

    def evaluate_and_route(
        self,
        opportunities: list[Opportunity],
        cognitive_state: Any,
    ) -> Opportunity | None:
        """Enforce rate limits, log states in history, and return the winning Opportunity.

        Args:
            opportunities: List of candidate Opportunity objects.
            cognitive_state: The current CognitiveStateSnapshot.

        Returns:
            The selected Opportunity to evaluate, or None if suppressed by limits.
        """
        self._reset_budget_if_needed()
        now = time.time()

        if not opportunities:
            return None

        # Filter out expired ones
        active_opps = [o for o in opportunities if o.expires_at is None or now < o.expires_at]
        if not active_opps:
            return None

        # Update priorities with aging and sort (highest priority first)
        for o in active_opps:
            o.update_priority(now)
        active_opps.sort(key=lambda o: o.priority, reverse=True)
        winner = active_opps[0]

        # Log created status for auditing
        for o in active_opps:
            self.history.log_opportunity(o, "created")

        # Gating: Drop low priority candidates
        if winner.priority < 0.5:
            self.history.log_opportunity(winner, "dismissed")
            return None

        # Cooldown checks
        time_since_last = now - self._last_trigger_time
        if time_since_last < self.min_gap_seconds:
            logger.debug(
                "Proactive evaluation suppressed: cooldown active (elapsed: %ds, threshold: %ds)",
                int(time_since_last),
                int(self.min_gap_seconds),
            )
            self.history.log_opportunity(winner, "dismissed")
            return None

        if self._triggers_today >= self.daily_budget:
            logger.debug("Proactive evaluation suppressed: daily budget limit reached")
            self.history.log_opportunity(winner, "dismissed")
            return None

        # Mark executed
        self._last_trigger_time = now
        self._triggers_today += 1
        self.history.log_opportunity(winner, "executed")

        # Mark remaining active candidates as expired/dismissed in history
        for o in active_opps[1:]:
            self.history.log_opportunity(o, "expired")

        return winner


class AutonomyManager(Node):
    """Subsystem coordinator for passive monitoring and proactive routing."""

    def __init__(
        self,
        node_id: str,
        db_path: str = "./data/opportunity_history.db",
        tick_interval: float = 10.0,
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.tick_interval = tick_interval
        self.monitor = PresenceMonitor()
        self.history = OpportunityHistory(db_path)
        self.coordinator = ProactiveCoordinator(self.history)
        self.scorer = OpportunityScorer()

        self._sources: list[OpportunitySource] = []
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

    def register_source(self, source: OpportunitySource) -> None:
        """Register an opportunity detection source."""
        self._sources.append(source)

    async def on_start(self) -> None:
        """Subscribe to activity pulses and launch background tick loop."""
        await self.bus.subscribe("user.message", self._handle_user_message)
        await self.bus.subscribe("system.experience", self._handle_ai_message)
        await self.bus.subscribe("sensor.activity", self._handle_sensor_message)

        self._running = True
        self._loop_task = asyncio.create_task(self._tick_loop())
        logger.info("AutonomyManager started.")

    async def on_stop(self) -> None:
        """Stop background tasks and clean up."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("AutonomyManager stopped.")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _handle_user_message(self, msg: Message) -> None:
        self.monitor.report_activity("user", "user_input", time.time())

    async def _handle_ai_message(self, msg: Message) -> None:
        self.monitor.report_activity("ai", "ai_output", time.time())

    async def _handle_sensor_message(self, msg: Message) -> None:
        source = msg.payload.get("source", "generic_sensor")
        self.monitor.report_activity(source, "sensor", time.time())
        # Store other keys as sensor activities
        for k, v in msg.payload.items():
            if k != "source" and isinstance(v, (int, float)):
                self.monitor.state.update_sensor_activity(k, float(v))

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                await self._evaluate_opportunities()
            except Exception as e:
                logger.error("Error evaluating opportunities in tick: %s", e)
            await asyncio.sleep(self.tick_interval)

    async def _evaluate_opportunities(self) -> None:
        """Scan registered sources and route the winning opportunity to the MessageBus."""
        cog_state = None
        try:
            resp = await self.bus.request(
                "attention.query_state",
                Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    topic="attention.query_state",
                    payload={},
                ),
            )
            if resp and resp.type != MessageType.ERROR:
                cog_state = resp.payload.get("snapshot")
        except Exception:
            pass

        if cog_state is None:
            # Fallback mock cognitive state snapshot
            class MockSnapshot:
                stress = 0.0
                fatigue = 0.0
                focus_target = ""

            cog_state = MockSnapshot()

        candidates: list[Opportunity] = []
        for src in self._sources:
            try:
                opps = src.detect(self.monitor.state, cog_state)
                for o in opps:
                    # Apply contextual scoring
                    self.scorer.score(o, cog_state)
                    candidates.append(o)
            except Exception as e:
                logger.warning("Source %s failed: %s", src.source_name, e)

        winner = self.coordinator.evaluate_and_route(candidates, cog_state)
        if winner:
            logger.info(
                "Opportunity selected: %s (source: %s, priority: %.2f)",
                winner.id,
                winner.source,
                winner.priority,
            )
            event_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="brain.proactive.evaluate",
                payload={
                    "opportunity_id": winner.id,
                    "source": winner.source,
                    "category": winner.category,
                    "priority": winner.priority,
                    "urgency": winner.urgency,
                    "reason": winner.reason,
                    "context": winner.context,
                    "suggested_actions": winner.suggested_actions,
                },
            )
            await self.bus.publish("brain.proactive.evaluate", event_msg)
