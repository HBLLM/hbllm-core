"""Autonomy Core — the cognitive heartbeat of HBLLM.

Implements a hybrid event-driven + periodic tick loop that shifts HBLLM
from a reactive request→response system to a continuously operating
cognitive organism.

Architecture
────────────
Fast Path (event-driven):
    Instant wake-up on critical events via MessageBus subscriptions.
    User input, sensor anomalies, device changes trigger immediate
    cognitive processing without waiting for the next tick.

Slow Path (periodic cognition):
    Ticks at adaptive intervals controlled by the CognitiveStateMachine.
    Used for reflection, routine evaluation, memory consolidation,
    and background planning.

Tiered Invocation
─────────────────
Tier 1 — Reflex Layer:
    Deterministic rules, heuristics, scoring. Zero LLM cost.
    Microsecond latency. Always-on.

Tier 2 — Fast Cognitive Router:
    Small local model (Phi, Gemma, ONNX). Intent classification,
    urgency scoring. Millisecond latency.

Tier 3 — Heavy Reasoning:
    Large LLM for planning, synthesis, ambiguous interpretation.
    Only invoked when Tier 1/2 determine it is necessary.

Safety
──────
- Thought budgets (max thoughts per minute)
- Cooldown periods after cognitive bursts
- Maximum recursion depth for self-triggered thoughts
- Event debouncing to prevent notification storms
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.autonomy.attention import AttentionEvent, AttentionSystem, ScoredEvent
from hbllm.brain.autonomy.state_machine import CognitiveState, CognitiveStateMachine
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

# A reflex rule: (event) → optional action message
ReflexRule = Callable[[AttentionEvent], Message | None]

# An async handler for proactive cognition
ProactiveHandler = Callable[[], Coroutine[Any, Any, list[Message] | None]]


# ── Internal Thought ─────────────────────────────────────────────────────────


@dataclass
class InternalThought:
    """A self-generated cognitive event (reminder, deferred goal, reflection).

    These are produced by the system itself, not external events.
    They enter the same attention pipeline as external events.
    """

    thought_id: str = field(default_factory=lambda: f"thought_{uuid.uuid4().hex[:12]}")
    content: str = ""
    category: str = "internal"  # "reminder", "reflection", "deferred_goal"
    urgency: float = 0.3
    goal_alignment: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_attention_event(self) -> AttentionEvent:
        """Convert to an AttentionEvent for scoring."""
        return AttentionEvent(
            event_id=self.thought_id,
            source=f"internal.{self.category}",
            category="internal",
            payload={"content": self.content, **self.metadata},
            urgency=self.urgency,
            goal_alignment=self.goal_alignment,
            temporal_relevance=0.5,
        )


# ── Autonomy Core ────────────────────────────────────────────────────────────


class AutonomyCore:
    """The cognitive heartbeat — a hybrid event + tick loop.

    This is the central daemon that makes HBLLM feel "alive". It:

    1. Listens for critical events on the MessageBus (fast path)
    2. Periodically ticks at adaptive intervals (slow path)
    3. Scores all events through the AttentionSystem
    4. Routes events through Tier 1 reflexes before escalating
    5. Manages internal thoughts (self-reminders, deferred goals)
    6. Enforces safety limits (budgets, cooldowns, recursion depth)

    Usage::

        state_machine = CognitiveStateMachine()
        attention = AttentionSystem()
        core = AutonomyCore(
            state_machine=state_machine,
            attention=attention,
        )

        # Register reflex rules (Tier 1)
        core.add_reflex("low_battery", battery_reflex_rule)

        # Register proactive handlers (slow path)
        core.add_proactive_handler("routine_check", check_routines)

        await core.start(bus)
        # ... runs continuously ...
        await core.stop()
    """

    def __init__(
        self,
        state_machine: CognitiveStateMachine | None = None,
        attention: AttentionSystem | None = None,
        *,
        # Safety limits
        max_recursion_depth: int = 3,
        max_pending_thoughts: int = 50,
        # Fast-path event topics to subscribe to
        fast_path_topics: list[str] | None = None,
    ) -> None:
        self.state_machine = state_machine or CognitiveStateMachine()
        self.attention = attention or AttentionSystem()

        # Safety
        self._max_recursion_depth = max_recursion_depth
        self._max_pending_thoughts = max_pending_thoughts
        self._current_recursion_depth = 0

        # Reflex rules (Tier 1 — deterministic, zero-LLM)
        self._reflexes: dict[str, ReflexRule] = {}

        # Proactive handlers (slow path — called each tick)
        self._proactive_handlers: dict[str, ProactiveHandler] = {}

        # Internal thought queue
        self._thought_queue: list[InternalThought] = []

        # MessageBus reference
        self._bus: Any = None

        # Event loop control
        self._running = False
        self._tick_task: asyncio.Task[None] | None = None
        self._fast_path_event: asyncio.Event = asyncio.Event()

        # Fast-path topics
        self._fast_path_topics = fast_path_topics or [
            "user.input",
            "user.action",
            "sensor.anomaly",
            "device.change",
            "system.critical",
            "perception.*",
        ]

        # Pending events buffer (scored events waiting for processing)
        self._pending_events: list[ScoredEvent] = []
        self._max_pending_events = 100

        # Outbound message callback
        self._on_action: Callable[[Message], Coroutine[Any, Any, None]] | None = None

        # Telemetry
        self._ticks_completed = 0
        self._fast_path_wakes = 0
        self._reflexes_fired = 0
        self._thoughts_generated = 0
        self._actions_emitted = 0
        self._recursion_blocks = 0
        self._boot_time = time.monotonic()

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self, bus: Any) -> None:
        """Start the cognitive heartbeat.

        Args:
            bus: The MessageBus to subscribe to for fast-path events
                 and to publish proactive actions on.
        """
        self._bus = bus
        self._running = True

        # Subscribe to fast-path topics
        for topic in self._fast_path_topics:
            await self._bus.subscribe(topic, self._handle_fast_path_event)

        # Subscribe to internal thought bus
        await self._bus.subscribe("autonomy.thought", self._handle_thought_message)

        # Start the tick loop
        self._tick_task = asyncio.create_task(self._tick_loop())

        # Transition to OBSERVING
        self.state_machine.transition_to(CognitiveState.OBSERVING, reason="autonomy_boot")

        logger.info(
            "AutonomyCore started. Fast-path topics: %s, tick=%.1fs",
            self._fast_path_topics,
            self.state_machine.tick_interval,
        )

    async def stop(self) -> None:
        """Stop the cognitive heartbeat gracefully."""
        self._running = False

        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass

        self.state_machine.transition_to(CognitiveState.SLEEPING, reason="shutdown")
        logger.info(
            "AutonomyCore stopped after %d ticks, %d fast-path wakes.",
            self._ticks_completed,
            self._fast_path_wakes,
        )

    def set_action_handler(self, handler: Callable[[Message], Coroutine[Any, Any, None]]) -> None:
        """Set the callback for outbound actions (proactive messages).

        This is called when the AutonomyCore decides to emit a message
        (e.g., a proactive reminder, anomaly alert, or background result).
        """
        self._on_action = handler

    # ── Registration ──────────────────────────────────────────────────

    def add_reflex(self, name: str, rule: ReflexRule) -> None:
        """Register a Tier 1 reflex rule.

        Reflex rules are deterministic, zero-LLM functions that run on
        every scored event. If a reflex returns a ``Message``, that
        message is emitted immediately without LLM involvement.

        Example::

            def battery_low_reflex(event: AttentionEvent) -> Message | None:
                if event.source == "sensor.battery" and event.payload.get("level", 100) < 15:
                    return Message(
                        type=MessageType.COMMAND,
                        source_node_id="autonomy",
                        topic="system.power.save",
                        payload={"action": "enable_low_power"},
                    )
                return None

            core.add_reflex("battery_low", battery_low_reflex)
        """
        self._reflexes[name] = rule
        logger.debug("Registered reflex: %s", name)

    def remove_reflex(self, name: str) -> None:
        """Remove a registered reflex rule."""
        self._reflexes.pop(name, None)

    def add_proactive_handler(self, name: str, handler: ProactiveHandler) -> None:
        """Register a slow-path proactive handler.

        These are called on every cognitive tick (slow path). They can
        perform background checks (routine detection, anomaly scanning)
        and return ``Message`` objects to emit.
        """
        self._proactive_handlers[name] = handler
        logger.debug("Registered proactive handler: %s", name)

    def remove_proactive_handler(self, name: str) -> None:
        """Remove a proactive handler."""
        self._proactive_handlers.pop(name, None)

    # ── Internal Thoughts ─────────────────────────────────────────────

    def add_thought(self, thought: InternalThought) -> bool:
        """Enqueue an internal thought for future processing.

        Returns False if the thought queue is full (safety limit).
        """
        if len(self._thought_queue) >= self._max_pending_thoughts:
            logger.warning(
                "Thought queue full (%d). Dropping thought: %s",
                self._max_pending_thoughts,
                thought.content[:80],
            )
            return False

        self._thought_queue.append(thought)
        self._thoughts_generated += 1
        return True

    async def _handle_thought_message(self, msg: Message) -> None:
        """Handle internal thoughts arriving via the bus."""
        thought = InternalThought(
            content=msg.payload.get("content", ""),
            category=msg.payload.get("category", "internal"),
            urgency=msg.payload.get("urgency", 0.3),
            goal_alignment=msg.payload.get("goal_alignment", 0.0),
            metadata=msg.payload.get("metadata", {}),
        )
        self.add_thought(thought)

    # ── Fast Path (Event-Driven) ──────────────────────────────────────

    async def _handle_fast_path_event(self, msg: Message) -> None:
        """Handle an incoming fast-path event from the MessageBus.

        This is the instant wake-up path. The event is scored immediately
        and, if it exceeds the interruption threshold, the cognitive state
        is interrupted.
        """
        self._fast_path_wakes += 1

        # Convert bus message to AttentionEvent
        event = AttentionEvent(
            event_id=msg.id,
            source=msg.topic,
            category=self._classify_event_category(msg),
            payload=msg.payload,
            urgency=msg.payload.get("_urgency", 0.6),
            emotional_weight=msg.payload.get("_emotional_weight", 0.0),
            goal_alignment=msg.payload.get("_goal_alignment", 0.0),
            temporal_relevance=msg.payload.get("_temporal_relevance", 0.5),
        )

        # Score through attention system
        scored = self.attention.score_event(
            event,
            cognitive_load=self.state_machine.cognitive_load,
            user_active=event.category == "user_action",
            interruption_threshold=self.state_machine.current_profile.interruption_threshold,
        )

        # Run reflexes first (Tier 1)
        reflex_handled = await self._run_reflexes(event)
        if reflex_handled:
            return

        # Check if this should interrupt
        if scored.should_interrupt:
            self.state_machine.transition_to(
                CognitiveState.INTERRUPTED,
                reason=f"fast_path:{event.source}",
                metadata={"event_id": event.event_id, "priority": scored.priority_score},
            )

        # Buffer the scored event for processing
        self._buffer_event(scored)

        # Wake the tick loop early
        self._fast_path_event.set()

    def _classify_event_category(self, msg: Message) -> str:
        """Classify a bus message into an attention category."""
        topic = msg.topic
        if topic.startswith("user."):
            return "user_action"
        if topic.startswith("sensor.") or topic.startswith("perception."):
            return "sensor"
        if topic.startswith("system."):
            return "system_alert"
        if topic.startswith("device."):
            return "device_change"
        return "background"

    # ── Reflexes (Tier 1) ─────────────────────────────────────────────

    async def _run_reflexes(self, event: AttentionEvent) -> bool:
        """Run all registered reflex rules against an event.

        Returns True if any reflex produced an action (meaning the event
        was fully handled at Tier 1 without needing LLM escalation).
        """
        for name, rule in self._reflexes.items():
            try:
                action = rule(event)
                if action is not None:
                    self._reflexes_fired += 1
                    logger.debug("Reflex '%s' fired for event %s", name, event.source)
                    await self._emit_action(action)
                    return True
            except Exception:
                logger.exception("Error in reflex '%s'", name)
        return False

    # ── Slow Path (Periodic Tick) ─────────────────────────────────────

    async def _tick_loop(self) -> None:
        """The main cognitive tick loop (slow path).

        Uses a hybrid wait: sleeps for the adaptive tick interval but
        can be woken early by fast-path events.
        """
        while self._running:
            tick_interval = self.state_machine.tick_interval

            # Hybrid wait: sleep OR wake on fast-path event
            self._fast_path_event.clear()
            try:
                await asyncio.wait_for(
                    self._fast_path_event.wait(),
                    timeout=tick_interval,
                )
                # Woken by fast-path event — process immediately
            except asyncio.TimeoutError:
                # Normal tick — run slow-path cognition
                pass
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            await self._cognitive_tick()
            self._ticks_completed += 1

    async def _cognitive_tick(self) -> None:
        """Execute one cognitive tick.

        1. Process buffered events (highest priority first)
        2. Process internal thoughts
        3. Run proactive handlers
        4. Maintenance (attention decay, context pruning)
        5. Update cognitive load
        """
        try:
            # 1. Process buffered events
            await self._process_pending_events()

            # 2. Process internal thoughts
            await self._process_thoughts()

            # 3. Run proactive handlers (only in appropriate states)
            if self.state_machine.is_active or self.state_machine.state == CognitiveState.IDLE:
                await self._run_proactive_handlers()

            # 4. Maintenance
            self.attention.tick()

            # 5. Update cognitive load based on queue depth
            load = min(1.0, len(self._pending_events) / max(self._max_pending_events, 1))
            self.state_machine.update_cognitive_load(load)

            # 6. Auto-transition to IDLE if nothing is happening
            if (
                self.state_machine.state == CognitiveState.OBSERVING
                and not self._pending_events
                and not self._thought_queue
                and self.state_machine.state_duration > 60.0
            ):
                self.state_machine.transition_to(CognitiveState.IDLE, reason="no_activity_60s")

            # 7. Resume from interruption if the event queue is drained
            if self.state_machine.state == CognitiveState.INTERRUPTED and not self._pending_events:
                self.state_machine.resume_from_interruption(reason="event_queue_drained")

        except Exception:
            logger.exception("Error in cognitive tick")
            self.state_machine.transition_to(CognitiveState.RECOVERING, reason="tick_error")

    async def _process_pending_events(self) -> None:
        """Process buffered scored events in priority order."""
        if not self._pending_events:
            return

        # Sort by priority (highest first)
        self._pending_events.sort(key=lambda se: se.priority_score, reverse=True)

        # Process up to max_concurrent_thoughts events
        max_to_process = self.state_machine.current_profile.max_concurrent_thoughts
        to_process = self._pending_events[:max_to_process]
        self._pending_events = self._pending_events[max_to_process:]

        for scored_event in to_process:
            await self._process_scored_event(scored_event)

    async def _process_scored_event(self, scored: ScoredEvent) -> None:
        """Process a single scored event.

        This is where Tier 2/3 escalation would happen. For Phase 1,
        we emit the event as a cognitive action for downstream nodes
        (planner, executor) to handle.
        """
        # Guard against recursion
        if self._current_recursion_depth >= self._max_recursion_depth:
            self._recursion_blocks += 1
            logger.warning(
                "Recursion depth %d reached. Dropping event %s",
                self._max_recursion_depth,
                scored.event.event_id,
            )
            return

        self._current_recursion_depth += 1
        try:
            # Determine processing tier based on state machine profile
            profile = self.state_machine.current_profile

            # Build the cognitive action message
            action = Message(
                type=MessageType.EVENT,
                source_node_id="autonomy_core",
                topic="cognitive.process",
                payload={
                    "event_id": scored.event.event_id,
                    "source": scored.event.source,
                    "category": scored.event.category,
                    "priority_score": scored.priority_score,
                    "original_payload": scored.event.payload,
                    "tier": self._determine_tier(scored, profile),
                    "allow_heavy_llm": profile.allow_heavy_llm,
                    "allow_fast_router": profile.allow_fast_router,
                },
            )
            await self._emit_action(action)
        finally:
            self._current_recursion_depth -= 1

    def _determine_tier(self, scored: ScoredEvent, profile: Any) -> str:
        """Determine which processing tier an event should use."""
        if scored.priority_score < 0.3:
            return "tier1_reflex"
        if not profile.allow_heavy_llm or scored.priority_score < 0.7:
            return "tier2_fast_router"
        return "tier3_heavy_reasoning"

    async def _process_thoughts(self) -> None:
        """Process queued internal thoughts."""
        if not self._thought_queue:
            return

        # Process up to 3 thoughts per tick
        batch = self._thought_queue[:3]
        self._thought_queue = self._thought_queue[3:]

        for thought in batch:
            event = thought.to_attention_event()
            scored = self.attention.score_event(
                event,
                cognitive_load=self.state_machine.cognitive_load,
                interruption_threshold=1.0,  # Thoughts don't self-interrupt
            )
            if scored.priority_score > 0.1:
                self._buffer_event(scored)

    async def _run_proactive_handlers(self) -> None:
        """Run registered proactive handlers (slow path).

        Each handler can return a list of Messages to emit.
        """
        for name, handler in self._proactive_handlers.items():
            try:
                messages = await handler()
                if messages:
                    for msg in messages:
                        await self._emit_action(msg)
            except Exception:
                logger.exception("Error in proactive handler '%s'", name)

    # ── Output ────────────────────────────────────────────────────────

    async def _emit_action(self, msg: Message) -> None:
        """Emit a proactive action message."""
        self._actions_emitted += 1

        if self._on_action:
            await self._on_action(msg)
        elif self._bus:
            await self._bus.publish(msg.topic, msg)

    # ── Helpers ───────────────────────────────────────────────────────

    def _buffer_event(self, scored: ScoredEvent) -> None:
        """Add a scored event to the pending buffer."""
        if len(self._pending_events) >= self._max_pending_events:
            # Drop lowest-priority event
            self._pending_events.sort(key=lambda se: se.priority_score)
            self._pending_events.pop(0)

        self._pending_events.append(scored)

    # ── Introspection ─────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for telemetry / debugging."""
        return {
            "running": self._running,
            "ticks_completed": self._ticks_completed,
            "fast_path_wakes": self._fast_path_wakes,
            "reflexes_fired": self._reflexes_fired,
            "thoughts_generated": self._thoughts_generated,
            "actions_emitted": self._actions_emitted,
            "recursion_blocks": self._recursion_blocks,
            "pending_events": len(self._pending_events),
            "pending_thoughts": len(self._thought_queue),
            "reflex_count": len(self._reflexes),
            "proactive_handler_count": len(self._proactive_handlers),
            "uptime_s": round(time.monotonic() - self._boot_time, 2),
            "state_machine": self.state_machine.snapshot(),
            "attention": self.attention.snapshot(),
        }
