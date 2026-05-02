"""
Cognitive Awareness — Brain self-observation and I/O pattern detection.

A platform-agnostic awareness engine that monitors all messages flowing
through the HBLLM bus, detects noteworthy patterns, and fires cognitive
triggers.

Think of it as the brain's proprioception — it knows what it's doing,
how well it's performing, and when something has changed.

External sensors (e.g., Sentra's macOS SystemSensors) can be registered
via the ``AwarenessSensor`` protocol to feed platform-specific context
into the same pattern detection pipeline.

Usage::

    awareness = CognitiveAwareness(node_id="awareness")
    await awareness.start(bus)

    # Register a platform sensor
    awareness.register_sensor(MacOSSensor())

    # Get current brain activity snapshot
    snap = awareness.snapshot()

    # Collect pending triggers
    triggers = awareness.get_pending_triggers()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class CognitiveSnapshot:
    """Point-in-time snapshot of brain activity — no OS/platform data."""

    timestamp: float = 0.0
    queries_last_minute: int = 0
    queries_total: int = 0
    responses_total: int = 0
    active_topics: list[str] = field(default_factory=list)  # top-5 active topics
    avg_confidence: float = 0.0  # rolling avg of evaluation scores
    error_rate: float = 0.0  # errors / total in window
    tool_calls_last_minute: int = 0
    active_sessions: int = 0
    cognitive_load: float = 0.0  # 0-1, from LoadManager if available
    idle_seconds: float = 0.0  # time since last query
    memory_operations: int = 0  # store/retrieve ops in window
    skills_invoked: int = 0

    # Platform sensor data (filled by registered sensors)
    sensor_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveTrigger:
    """A proactive insight from brain self-observation."""

    trigger_type: str  # "idle_return" | "context_switch" | "degradation" | etc.
    message: str
    context: dict = field(default_factory=dict)
    priority: str = "normal"  # "low" | "normal" | "high"
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Sensor Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class AwarenessSensor(Protocol):
    """Interface for platform-specific sensors (implemented by Sentra, etc.)."""

    name: str

    async def collect(self) -> dict[str, Any]:
        """Collect sensor data. Returns a dict merged into CognitiveSnapshot.sensor_data."""
        ...


# ── Activity Window ──────────────────────────────────────────────────────────


class _ActivityWindow:
    """Fixed-size sliding window for tracking activity metrics."""

    def __init__(self, window_seconds: float = 60.0, max_events: int = 500):
        self.window_seconds = window_seconds
        self.max_events = max_events
        self._events: deque[tuple[float, str]] = deque(maxlen=max_events)

    def record(self, event_type: str, timestamp: float | None = None) -> None:
        self._events.append((timestamp or time.time(), event_type))

    def count(self, event_type: str | None = None) -> int:
        """Count events in the current window."""
        cutoff = time.time() - self.window_seconds
        if event_type:
            return sum(1 for ts, et in self._events if ts >= cutoff and et == event_type)
        return sum(1 for ts, _ in self._events if ts >= cutoff)

    def top_types(self, n: int = 5) -> list[str]:
        """Return the top-N event types by frequency in the window."""
        cutoff = time.time() - self.window_seconds
        counts: dict[str, int] = defaultdict(int)
        for ts, et in self._events:
            if ts >= cutoff:
                counts[et] += 1
        return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


# ── Pattern Detector ─────────────────────────────────────────────────────────


class _PatternDetector:
    """Detects noteworthy patterns from activity snapshots."""

    def __init__(self):
        self._last_query_time: float = 0.0
        self._was_idle: bool = False
        self._idle_threshold: float = 120.0  # 2 minutes
        self._last_topic_set: set[str] = set()
        self._consecutive_errors: int = 0
        self._confidence_history: deque[float] = deque(maxlen=20)
        self._milestone_thresholds: set[int] = {10, 50, 100, 500, 1000, 5000}
        self._last_milestone: int = 0

    def evaluate(self, snapshot: CognitiveSnapshot) -> list[CognitiveTrigger]:
        """Evaluate a snapshot and return any triggers."""
        triggers: list[CognitiveTrigger] = []
        now = snapshot.timestamp or time.time()

        # 1. Idle return detection
        if snapshot.idle_seconds >= self._idle_threshold:
            if not self._was_idle:
                self._was_idle = True
        elif self._was_idle and snapshot.queries_last_minute > 0:
            self._was_idle = False
            idle_mins = int(snapshot.idle_seconds // 60) if snapshot.idle_seconds > 0 else 0
            triggers.append(
                CognitiveTrigger(
                    trigger_type="idle_return",
                    message=f"Brain resumed after {idle_mins}+ minutes idle",
                    context={"idle_seconds": snapshot.idle_seconds},
                    priority="low",
                    timestamp=now,
                )
            )

        # 2. Topic shift detection
        current_topics = set(snapshot.active_topics[:3])
        if self._last_topic_set and current_topics:
            overlap = current_topics & self._last_topic_set
            if len(overlap) == 0 and len(current_topics) > 0:
                triggers.append(
                    CognitiveTrigger(
                        trigger_type="context_switch",
                        message=f"Context shifted from {self._last_topic_set} to {current_topics}",
                        context={
                            "previous": list(self._last_topic_set),
                            "current": list(current_topics),
                        },
                        priority="low",
                        timestamp=now,
                    )
                )
        if current_topics:
            self._last_topic_set = current_topics

        # 3. Error burst detection
        if snapshot.error_rate > 0.5 and snapshot.queries_last_minute >= 3:
            self._consecutive_errors += 1
            if self._consecutive_errors >= 3:
                triggers.append(
                    CognitiveTrigger(
                        trigger_type="degradation",
                        message=f"Error rate at {snapshot.error_rate:.0%} over last minute ({snapshot.queries_last_minute} queries)",
                        context={
                            "error_rate": snapshot.error_rate,
                            "queries": snapshot.queries_last_minute,
                        },
                        priority="high",
                        timestamp=now,
                    )
                )
        else:
            self._consecutive_errors = 0

        # 4. Quality decline detection
        if snapshot.avg_confidence > 0:
            self._confidence_history.append(snapshot.avg_confidence)
            if len(self._confidence_history) >= 10:
                recent = list(self._confidence_history)[-5:]
                older = list(self._confidence_history)[-10:-5]
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                if older_avg > 0 and (older_avg - recent_avg) / older_avg > 0.2:
                    triggers.append(
                        CognitiveTrigger(
                            trigger_type="quality_alert",
                            message=f"Avg confidence dropped from {older_avg:.2f} to {recent_avg:.2f}",
                            context={"recent_avg": recent_avg, "older_avg": older_avg},
                            priority="normal",
                            timestamp=now,
                        )
                    )

        # 5. Overload detection
        if snapshot.cognitive_load > 0.8:
            triggers.append(
                CognitiveTrigger(
                    trigger_type="overload",
                    message=f"Cognitive load at {snapshot.cognitive_load:.0%}",
                    context={"load": snapshot.cognitive_load},
                    priority="high",
                    timestamp=now,
                )
            )

        # 6. Milestone detection
        for threshold in sorted(self._milestone_thresholds):
            if snapshot.queries_total >= threshold and threshold > self._last_milestone:
                self._last_milestone = threshold
                triggers.append(
                    CognitiveTrigger(
                        trigger_type="milestone",
                        message=f"Processed {threshold} total queries",
                        context={"total_queries": snapshot.queries_total},
                        priority="low",
                        timestamp=now,
                    )
                )
                break

        return triggers


# ── Cognitive Awareness Node ─────────────────────────────────────────────────


class CognitiveAwareness(Node):
    """
    Brain self-awareness engine — observes all I/O flowing through the bus.

    Platform-agnostic. Fires cognitive triggers when noteworthy patterns
    are detected. Can be extended by platform-specific sensors.

    Subscribes to:
        sensory.input — query tracking
        sensory.output — response tracking
        sensory.stream.end — stream completion tracking
        action.tool_call — tool usage tracking
        system.error — error tracking
        system.evaluation — quality tracking
        system.load.report — cognitive load tracking
        memory.* — memory operation tracking

    Publishes:
        system.awareness.trigger — when a pattern is detected
    """

    def __init__(
        self,
        node_id: str = "cognitive_awareness",
        window_seconds: float = 60.0,
        poll_interval: float = 5.0,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["awareness", "self_monitoring"],
        )
        self._activity = _ActivityWindow(window_seconds=window_seconds)
        self._detector = _PatternDetector()
        self._sensors: dict[str, AwarenessSensor] = {}
        self._pending_triggers: list[CognitiveTrigger] = []
        self._poll_interval = poll_interval

        # Counters
        self._total_queries = 0
        self._total_responses = 0
        self._total_errors = 0
        self._total_tool_calls = 0
        self._total_memory_ops = 0
        self._total_skills = 0
        self._last_query_time: float = 0.0
        self._active_sessions: set[str] = set()

        # Rolling confidence window
        self._confidence_scores: deque[float] = deque(maxlen=50)

        # Cognitive load (from LoadManager)
        self._current_load: float = 0.0

        # Awareness loop
        self._loop_task: Any = None
        self._running = False

    # ── Sensor Registration ───────────────────────────────────────────

    def register_sensor(self, sensor: AwarenessSensor) -> None:
        """Register a platform-specific sensor for data collection."""
        self._sensors[sensor.name] = sensor
        logger.info("Registered awareness sensor: %s", sensor.name)

    def unregister_sensor(self, name: str) -> None:
        """Remove a registered sensor."""
        self._sensors.pop(name, None)

    # ── Bus Subscriptions ─────────────────────────────────────────────

    async def on_start(self) -> None:
        logger.info("Starting CognitiveAwareness")
        await self.bus.subscribe("sensory.input", self._on_query)
        await self.bus.subscribe("sensory.output", self._on_response)
        await self.bus.subscribe("sensory.stream.end", self._on_response)
        await self.bus.subscribe("action.tool_call", self._on_tool_call)
        await self.bus.subscribe("system.error", self._on_error)
        await self.bus.subscribe("system.evaluation", self._on_evaluation)
        await self.bus.subscribe("system.load.report", self._on_load_report)

        # Start background awareness loop
        import asyncio

        self._running = True
        self._loop_task = asyncio.create_task(self._awareness_loop())

    async def on_stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except Exception:
                pass
        logger.info("CognitiveAwareness stopped")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ────────────────────────────────────────────────

    async def _on_query(self, message: Message) -> None:
        self._total_queries += 1
        self._last_query_time = time.time()
        self._activity.record("query")

        # Track session
        sid = message.session_id or ""
        if sid:
            self._active_sessions.add(sid)

    async def _on_response(self, message: Message) -> None:
        self._total_responses += 1
        self._activity.record("response")

    async def _on_tool_call(self, message: Message) -> None:
        self._total_tool_calls += 1
        tool_name = message.payload.get("tool_name", "unknown")
        self._activity.record(f"tool:{tool_name}")

    async def _on_error(self, message: Message) -> None:
        self._total_errors += 1
        self._activity.record("error")

    async def _on_evaluation(self, message: Message) -> None:
        score = message.payload.get("overall_score", 0.0)
        if score > 0:
            self._confidence_scores.append(score)
        self._activity.record("evaluation")

        # Track topic
        topic = message.payload.get("thought_type", "")
        if topic:
            self._activity.record(f"topic:{topic}")

    async def _on_load_report(self, message: Message) -> None:
        self._current_load = message.payload.get("load_percent", 0.0) / 100.0

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> CognitiveSnapshot:
        """Build a point-in-time snapshot of brain activity."""
        now = time.time()

        # Calculate metrics
        queries_lm = self._activity.count("query")
        total_lm = self._activity.count()
        errors_lm = self._activity.count("error")
        tools_lm = sum(
            1
            for ts, et in self._activity._events
            if ts >= now - self._activity.window_seconds and et.startswith("tool:")
        )

        # Active topics
        all_types = self._activity.top_types(n=10)
        topics = [t.split(":", 1)[1] for t in all_types if t.startswith("topic:")][:5]

        # Confidence
        avg_conf = (
            sum(self._confidence_scores) / len(self._confidence_scores)
            if self._confidence_scores
            else 0.0
        )

        # Idle time
        idle = now - self._last_query_time if self._last_query_time > 0 else 0.0

        # Error rate
        error_rate = errors_lm / max(total_lm, 1)

        snap = CognitiveSnapshot(
            timestamp=now,
            queries_last_minute=queries_lm,
            queries_total=self._total_queries,
            responses_total=self._total_responses,
            active_topics=topics,
            avg_confidence=round(avg_conf, 3),
            error_rate=round(error_rate, 3),
            tool_calls_last_minute=tools_lm,
            active_sessions=len(self._active_sessions),
            cognitive_load=round(self._current_load, 3),
            idle_seconds=round(idle, 1),
            memory_operations=self._total_memory_ops,
            skills_invoked=self._total_skills,
        )

        return snap

    # ── Awareness Loop ────────────────────────────────────────────────

    async def _awareness_loop(self) -> None:
        """Background loop: collect snapshots, run pattern detection, fire triggers."""
        import asyncio

        while self._running:
            try:
                # Build snapshot
                snap = self.snapshot()

                # Collect from registered platform sensors
                for sensor_name, sensor in self._sensors.items():
                    try:
                        data = await sensor.collect()
                        snap.sensor_data[sensor_name] = data
                    except Exception as e:
                        logger.debug("Sensor '%s' error: %s", sensor_name, e)

                # Detect patterns
                triggers = self._detector.evaluate(snap)

                # Store and publish triggers
                for trigger in triggers:
                    self._pending_triggers.append(trigger)

                    logger.info(
                        "🧠 Awareness trigger [%s]: %s",
                        trigger.trigger_type,
                        trigger.message,
                    )

                    # Publish to bus
                    try:
                        await self.bus.publish(
                            "system.awareness.trigger",
                            Message(
                                type=MessageType.EVENT,
                                source_node_id=self.node_id,
                                topic="system.awareness.trigger",
                                payload=trigger.to_dict(),
                            ),
                        )
                    except Exception:
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Awareness loop error: %s", e)

            await asyncio.sleep(self._poll_interval)

    # ── Public API ────────────────────────────────────────────────────

    def get_pending_triggers(self) -> list[CognitiveTrigger]:
        """Pop and return all pending triggers."""
        triggers = list(self._pending_triggers)
        self._pending_triggers.clear()
        return triggers

    def stats(self) -> dict[str, Any]:
        """Return awareness statistics."""
        return {
            "total_queries": self._total_queries,
            "total_responses": self._total_responses,
            "total_errors": self._total_errors,
            "total_tool_calls": self._total_tool_calls,
            "active_sessions": len(self._active_sessions),
            "cognitive_load": self._current_load,
            "avg_confidence": (
                round(sum(self._confidence_scores) / len(self._confidence_scores), 3)
                if self._confidence_scores
                else 0.0
            ),
            "registered_sensors": list(self._sensors.keys()),
            "pending_triggers": len(self._pending_triggers),
        }
