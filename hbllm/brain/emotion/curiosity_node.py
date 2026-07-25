"""
Curiosity Engine — detects knowledge gaps and generates learning goals.

Monitors low-confidence responses, negative feedback, and error fallbacks
to identify areas where the system needs improvement. When uncertainty
accumulates beyond a threshold, it generates autonomous learning goals
and publishes them to the Planner for investigation.

Integrations:
    UserModelEngine  → Weights curiosity goals toward user interests and expertise gaps
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEvent:
    """A recorded instance of system uncertainty."""

    topic: str
    query: str
    reason: str  # "low_confidence", "negative_feedback", "error_fallback"
    confidence: float
    tenant_id: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class LearningGoal:
    """An autonomous learning objective generated from knowledge gaps."""

    topic: str
    description: str
    priority: float  # Higher = more urgent (based on frequency/recency)
    source_events: int  # Number of uncertainty events that triggered this
    status: str = "pending"  # pending, dispatched, completed


class GoalQueue:
    """Priority-sorted queue of learning goals."""

    def __init__(self, max_size: int = 50):
        self.goals: list[LearningGoal] = []
        self.max_size = max_size

    def add_or_update(
        self, topic: str, description: str, priority: float, event_count: int
    ) -> LearningGoal:
        """Add a new goal or update an existing one for the same topic."""
        for goal in self.goals:
            if goal.topic == topic and goal.status == "pending":
                goal.priority = max(goal.priority, priority)
                goal.source_events += event_count
                goal.description = description
                self._sort()
                return goal

        goal = LearningGoal(
            topic=topic,
            description=description,
            priority=priority,
            source_events=event_count,
        )
        self.goals.append(goal)
        self._sort()

        if len(self.goals) > self.max_size:
            self.goals = self.goals[: self.max_size]

        return goal

    def pop_top(self) -> LearningGoal | None:
        """Remove and return the highest-priority pending goal."""
        for i, goal in enumerate(self.goals):
            if goal.status == "pending":
                goal.status = "dispatched"
                return goal
        return None

    def get_pending(self) -> list[LearningGoal]:
        return [g for g in self.goals if g.status == "pending"]

    def _sort(self) -> None:
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.goals),
            "pending": len(self.get_pending()),
            "dispatched": sum(1 for g in self.goals if g.status == "dispatched"),
            "completed": sum(1 for g in self.goals if g.status == "completed"),
        }


class CuriosityNode(Node):
    """
    Monitors the system for signs of uncertainty and generates
    learning goals to fill knowledge gaps.

    Subscribes to:
        system.feedback — negative feedback from users
        workspace.fallback — error fallback events
        module.evaluate — low-confidence domain responses

    Publishes:
        curiosity.goal — learning goals for the Planner
        curiosity.stats — curiosity engine statistics
    """

    def __init__(
        self,
        node_id: str,
        uncertainty_threshold: int = 3,
        goal_dispatch_interval: float = 60.0,
        user_model: Any | None = None,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["curiosity", "goal_generation"],
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.goal_dispatch_interval = goal_dispatch_interval
        self._user_model = user_model  # Optional UserModelEngine for interest-weighted goals
        self.events: list[UncertaintyEvent] = []
        self.topic_counts: dict[str, int] = defaultdict(int)
        self.goal_queue = GoalQueue()
        self._last_dispatch = 0.0
        self._user_interests: list[str] = []  # Cached from UserModel

        # v2: Predictive Curiosity — track topic patterns
        self._topic_sequence: list[str] = []  # Last N topics discussed
        self._topic_cooccurrence: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # topic_a → topic_b → count
        self._max_sequence_len = 100
        self._prediction_cache: list[str] = []  # Pre-researched topics

    async def on_start(self) -> None:
        logger.info("Starting CuriosityNode (threshold=%d)", self.uncertainty_threshold)
        await self.bus.subscribe("system.feedback", self._handle_feedback)
        await self.bus.subscribe("workspace.fallback", self._handle_fallback)
        await self.bus.subscribe("workspace.thought", self._handle_low_confidence_thought)
        await self.bus.subscribe("curiosity.query", self._handle_query)
        # v2: Predictive Curiosity — observe successful queries for patterns
        await self.bus.subscribe("system.experience", self._handle_experience_for_prediction)
        # v3: UserModel integration — weight curiosity by user interests
        await self.bus.subscribe("user.model.updated", self._handle_user_model_updated)
        # v5: Autonomous Learning — contradictions and weak areas trigger curiosity
        await self.bus.subscribe("learning.contradiction", self._handle_learning_contradiction)
        await self.bus.subscribe("learning.weak_area", self._handle_learning_weak_area)

        # Start predictive exploration loop
        self._predict_loop_task = asyncio.create_task(self._predictive_exploration_loop())

    async def _predictive_exploration_loop(self) -> None:
        """Background loop: periodically run predictive research."""
        while self._running:
            await asyncio.sleep(self.goal_dispatch_interval * 2)
            await self._predictive_exploration()

    async def _predictive_exploration(self) -> None:
        """Identify trending topics and autonomously trigger web research ahead of time."""
        # Get predictions
        predictions = self.predict_next_topics(top_k=2)
        for pred in predictions:
            topic = pred["topic"]
            prob = pred["probability"]

            if prob < 0.3 or topic in self._prediction_cache:
                continue

            self._prediction_cache.append(topic)
            logger.info(
                "[CuriosityNode] Proactively researching predicted topic: '%s' (p=%.2f)",
                topic,
                prob,
            )

            # Dispatch to WebResearchNode
            await self.publish(
                "system.research.request",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="system.research.request",
                    payload={
                        "topic": topic,
                        "query": f"Learn background context about {topic}",
                        "urgency": "low",
                        "context": f"Predictive exploration based on recent patterns (prob={prob})",
                    },
                ),
            )

    async def on_stop(self) -> None:
        logger.info(
            "Stopping CuriosityNode (%d events, %d goals)",
            len(self.events),
            len(self.goal_queue.goals),
        )
        if hasattr(self, "_predict_loop_task") and self._predict_loop_task:
            self._predict_loop_task.cancel()
            try:
                await self._predict_loop_task
            except asyncio.CancelledError:
                pass

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _handle_feedback(self, message: Message) -> Message | None:
        """Process negative user feedback as uncertainty signal."""
        payload = message.payload
        rating = payload.get("rating", 0)

        if rating >= 0:
            return None  # Only negative feedback signals uncertainty

        event = UncertaintyEvent(
            topic=str(payload.get("topic", "general")),
            query=str(payload.get("prompt", payload.get("query", ""))),
            reason="negative_feedback",
            confidence=0.0,
            tenant_id=message.tenant_id,
        )
        await self._record_event(event)
        return None

    async def _handle_fallback(self, message: Message) -> Message | None:
        """Process workspace error fallbacks as critical uncertainty."""
        event = UncertaintyEvent(
            topic=str(message.payload.get("topic", "unknown")),
            query=str(message.payload.get("query", "")),
            reason="error_fallback",
            confidence=0.0,
            tenant_id=message.tenant_id,
        )
        await self._record_event(event)
        return None

    async def _handle_low_confidence_thought(self, message: Message) -> Message | None:
        """Detect low-confidence workspace thoughts as uncertainty signals."""
        payload = message.payload
        confidence = payload.get("confidence", 1.0)
        thought_type = payload.get("type", "")

        # Ignore meta-thoughts (critiques, simulation results)
        if thought_type in ("critique", "simulation_result", "curiosity_signal"):
            return None

        # Only record if confidence is notably low
        if confidence >= 0.4:
            return None

        event = UncertaintyEvent(
            topic=str(payload.get("domain", thought_type or "general")),
            query=str(payload.get("content", ""))[:200],
            reason="low_confidence",
            confidence=confidence,
            tenant_id=message.tenant_id,
        )
        await self._record_event(event)
        return None

    async def _handle_query(self, message: Message) -> Message | None:
        """Return curiosity engine stats."""
        return message.create_response(
            {
                "event_count": len(self.events),
                "goal_queue": self.goal_queue.summary(),
                "top_gaps": self._get_top_gaps(5),
            }
        )

    async def _record_event(self, event: UncertaintyEvent) -> None:
        """Record an uncertainty event and check if a goal should be generated."""
        self.events.append(event)
        # Cap events list to prevent unbounded growth
        if len(self.events) > 500:
            self.events = self.events[-500:]
        self.topic_counts[event.topic] += 1

        count = self.topic_counts[event.topic]
        if count >= self.uncertainty_threshold:
            # Apply interest boost — topics aligned with user interests get higher priority
            base_priority = min(1.0, count / 10.0)
            interest_boost = self._compute_interest_boost(event.topic)
            goal = self.goal_queue.add_or_update(
                topic=event.topic,
                description=f"Improve handling of '{event.topic}' — "
                f"{count} uncertainty events recorded",
                priority=min(1.0, base_priority * interest_boost),
                event_count=count,
            )
            logger.info(
                "Generated learning goal for topic '%s' (priority=%.2f)", event.topic, goal.priority
            )

            # Attempt to dispatch
            await self._maybe_dispatch()

    def _compute_interest_boost(self, topic: str) -> float:
        """Boost priority for topics aligned with user interests.

        Returns a multiplier (1.0 = no boost, up to 1.5 for high-interest topics).
        """
        if not self._user_interests:
            return 1.0

        topic_lower = topic.lower()
        for interest in self._user_interests:
            if interest.lower() in topic_lower or topic_lower in interest.lower():
                return 1.4  # 40% boost for interest-aligned topics

        return 1.0

    async def _handle_user_model_updated(self, message: Message) -> None:
        """Cache user interests from UserModel for curiosity weighting."""
        if not self._user_model:
            return

        tenant_id = message.payload.get("tenant_id", "default")
        try:
            model = self._user_model.get_model(tenant_id)
            self._user_interests = list(model.active_interests)[:20]
            logger.debug(
                "CuriosityNode synced %d user interests from UserModel",
                len(self._user_interests),
            )
        except Exception as e:
            logger.debug("Failed to sync user interests: %s", e)

    async def _maybe_dispatch(self) -> None:
        """Dispatch the top pending goal to SpawnerNode and SleepNode."""
        now = time.monotonic()
        if now - self._last_dispatch < self.goal_dispatch_interval:
            return

        goal = self.goal_queue.pop_top()
        if goal is None:
            return

        self._last_dispatch = now
        goal_payload = {
            "goal_topic": goal.topic,
            "description": goal.description,
            "priority": goal.priority,
            "source_events": goal.source_events,
        }

        # 1. Publish to curiosity.goal (general signal)
        await self.publish(
            "curiosity.goal",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="curiosity.goal",
                payload=goal_payload,
            ),
        )

        # 2. Dispatch to SpawnerNode for data synthesis + training
        await self.publish(
            "system.spawn",
            Message(
                type=MessageType.SPAWN_REQUEST,
                source_node_id=self.node_id,
                topic="system.spawn",
                payload={
                    "topic": goal.topic,
                    "trigger_query": goal.description,
                    "confidence_score": 0.0,
                    "from_curiosity": True,
                },
            ),
        )

        # 3. Queue for SleepNode consolidation during idle time
        await self.publish(
            "system.sleep.goal",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.sleep.goal",
                payload=goal_payload,
            ),
        )

        logger.info("Dispatched learning goal to spawn+sleep: %s", goal.description)

    def _get_top_gaps(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the topics with the most uncertainty events."""
        sorted_topics = sorted(self.topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"topic": topic, "event_count": count} for topic, count in sorted_topics[:top_k]]

    # ── v2: Predictive Curiosity ──────────────────────────────────────

    async def _handle_experience_for_prediction(self, message: Message) -> None:
        """Track topic patterns from ALL queries (not just failures) for prediction."""
        payload = message.payload
        topic = payload.get("intent", payload.get("domain", ""))
        if not topic:
            return

        # Record the topic in the sequence
        if self._topic_sequence:
            prev_topic = self._topic_sequence[-1]
            if prev_topic != topic:
                self._topic_cooccurrence[prev_topic][topic] += 1

        self._topic_sequence.append(topic)
        if len(self._topic_sequence) > self._max_sequence_len:
            self._topic_sequence = self._topic_sequence[-self._max_sequence_len :]

    def predict_next_topics(self, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Predict what the user is likely to ask about next.

        Uses topic co-occurrence patterns: if the user just discussed topic A,
        and historically A is often followed by B, predict B.

        Returns a list of {topic, probability} dicts.
        """
        if not self._topic_sequence:
            return []

        current_topic = self._topic_sequence[-1]
        followers = self._topic_cooccurrence.get(current_topic, {})

        if not followers:
            return []

        total = sum(followers.values())
        predictions = [
            {"topic": topic, "probability": round(count / total, 3)}
            for topic, count in sorted(followers.items(), key=lambda x: x[1], reverse=True)
        ]
        return predictions[:top_k]

    async def generate_predictive_goals(self) -> int:
        """
        Pre-research predicted topics — queue research goals for topics
        the user is likely to ask about next.

        Called by SleepCycleNode during the curiosity replay phase.
        Returns the number of predictive goals generated.
        """
        predictions = self.predict_next_topics(top_k=3)
        generated = 0

        for pred in predictions:
            topic = pred["topic"]
            prob = pred["probability"]

            # Only pre-research if the probability is meaningful
            if prob < 0.2:
                continue

            # Don't re-research topics we've already explored
            if topic in self._prediction_cache:
                continue

            self.goal_queue.add_or_update(
                topic=topic,
                description=f"Predictive research: user likely to ask about '{topic}' "
                f"(probability={prob:.0%})",
                priority=prob,
                event_count=0,
            )
            self._prediction_cache.append(topic)
            generated += 1
            logger.info(
                "[CuriosityNode] Predictive goal: '%s' (p=%.0f%%)",
                topic,
                prob * 100,
            )

        # Cap prediction cache to prevent unbounded growth
        if len(self._prediction_cache) > 50:
            self._prediction_cache = self._prediction_cache[-50:]

        return generated

    # ── Autonomous Learning Integration ──────────────────────────────────

    async def _handle_learning_contradiction(self, message: Message) -> None:
        """Convert learning contradictions into curiosity goals.

        When the autonomous learner detects contradictory knowledge,
        curiosity triggers a deeper investigation.
        """
        concept = message.payload.get("concept", "")
        if not concept:
            return

        topic = f"Resolve contradiction about {concept}"
        self.note_uncertainty(
            topic=topic,
            confidence=0.3,
            metadata={"source": "learning.contradiction", **message.payload},
        )
        logger.info("[CuriosityNode] Contradiction-driven curiosity: '%s'", concept)

    async def _handle_learning_weak_area(self, message: Message) -> None:
        """Convert weak learning areas into high-priority curiosity goals.

        When the autonomous learner finds weak understanding,
        curiosity schedules re-learning with higher urgency.
        """
        concept = message.payload.get("concept", "")
        score = message.payload.get("score", 0.0)
        if not concept:
            return

        topic = f"Deepen understanding of {concept}"
        self.note_uncertainty(
            topic=topic,
            confidence=score,
            metadata={"source": "learning.weak_area", **message.payload},
        )
        logger.info(
            "[CuriosityNode] Weak-area curiosity: '%s' (score=%.2f)",
            concept,
            score,
        )
