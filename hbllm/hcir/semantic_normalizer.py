"""
Semantic Event Normalizer — canonical cognitive event vocabulary.

Maps raw domain events from different nodes into a unified vocabulary.
Different subsystems may describe the same cognitive fact differently:

    Planner: GoalCreated   ─┐
    Decision: NewGoal       ├→  CognitiveEventKind.GOAL_CREATED
    Router: IntentGoal      │
    Learning: GeneratedGoal ─┘

Without normalization, the projection layer becomes a giant collection
of special cases.  The normalizer keeps it to a single canonical mapping.

Usage::

    normalizer = SemanticNormalizer()
    kind = normalizer.normalize("memory.store", message)
    # → CognitiveEventKind.MEMORY_STORED
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Canonical Cognitive Event Vocabulary
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveEventKind(StrEnum):
    """Canonical vocabulary for all cognitive state transitions.

    Every raw bus event normalizes to exactly one of these kinds.
    The projection layer maps kinds to graph operations.
    """

    # ── Directives ───────────────────────────────────────────────────
    GOAL_CREATED = "goal.created"
    GOAL_COMPLETED = "goal.completed"
    GOAL_ABANDONED = "goal.abandoned"
    GOAL_BLOCKED = "goal.blocked"

    # ── Epistemology ─────────────────────────────────────────────────
    OBSERVATION_RECEIVED = "observation.received"
    BELIEF_UPDATED = "belief.updated"
    BELIEF_REVISED = "belief.revised"
    PREDICTION_MADE = "prediction.made"
    PREDICTION_VERIFIED = "prediction.verified"
    PREDICTION_ERROR = "prediction.error"

    # ── Execution ────────────────────────────────────────────────────
    DECISION_MADE = "decision.made"
    ACTION_PLANNED = "action.planned"
    ACTION_EXECUTED = "action.executed"
    ACTION_RESULT = "action.result"
    CAPABILITY_INVOKED = "capability.invoked"

    # ── Memory ───────────────────────────────────────────────────────
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_CONSOLIDATED = "memory.consolidated"
    SKILL_LEARNED = "skill.learned"

    # ── Governance ───────────────────────────────────────────────────
    GOVERNANCE_EVALUATED = "governance.evaluated"
    GOVERNANCE_BLOCKED = "governance.blocked"

    # ── Cognitive State ──────────────────────────────────────────────
    COGNITIVE_STATE_CHANGED = "cognitive_state.changed"
    ATTENTION_SHIFTED = "attention.shifted"

    # ── Perception ───────────────────────────────────────────────────
    PERCEPTION_RECEIVED = "perception.received"

    # ── Routing ──────────────────────────────────────────────────────
    ROUTING_DECIDED = "routing.decided"

    # ── Learning ─────────────────────────────────────────────────────
    LEARNING_EVENT = "learning.event"

    # ── Emotion ──────────────────────────────────────────────────────
    EMOTION_CHANGED = "emotion.changed"

    # ── World Model ──────────────────────────────────────────────────
    WORLD_STATE_UPDATED = "world.state_updated"


# ═══════════════════════════════════════════════════════════════════════════
# Default Topic → Kind Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Bus topics that map directly to canonical kinds.
# Wildcards are resolved by prefix matching: "memory.*" matches
# "memory.store", "memory.search", etc.
_DEFAULT_TOPIC_MAP: dict[str, CognitiveEventKind] = {
    # Memory
    "memory.store": CognitiveEventKind.MEMORY_STORED,
    "memory.browse": CognitiveEventKind.MEMORY_STORED,
    "memory.search": CognitiveEventKind.MEMORY_RECALLED,
    "memory.recall": CognitiveEventKind.MEMORY_RECALLED,
    "memory.reflection": CognitiveEventKind.MEMORY_CONSOLIDATED,
    "memory.feedback": CognitiveEventKind.LEARNING_EVENT,
    # Perception
    "perception.vision": CognitiveEventKind.PERCEPTION_RECEIVED,
    "perception.audio": CognitiveEventKind.PERCEPTION_RECEIVED,
    "perception.video": CognitiveEventKind.PERCEPTION_RECEIVED,
    "perception.multimodal": CognitiveEventKind.PERCEPTION_RECEIVED,
    # Decision
    "decision.made": CognitiveEventKind.DECISION_MADE,
    "decision.result": CognitiveEventKind.ACTION_RESULT,
    # Planning
    "planning.goal_created": CognitiveEventKind.GOAL_CREATED,
    "planning.goal_completed": CognitiveEventKind.GOAL_COMPLETED,
    "planning.plan_created": CognitiveEventKind.ACTION_PLANNED,
    # Actions
    "action.executed": CognitiveEventKind.ACTION_EXECUTED,
    "action.result": CognitiveEventKind.ACTION_RESULT,
    # Governance
    "governance.evaluation": CognitiveEventKind.GOVERNANCE_EVALUATED,
    "governance.violation": CognitiveEventKind.GOVERNANCE_BLOCKED,
    # Router
    "router.query": CognitiveEventKind.ROUTING_DECIDED,
    "router.decision": CognitiveEventKind.ROUTING_DECIDED,
    # Learning
    "learning.skill": CognitiveEventKind.SKILL_LEARNED,
    "learning.update": CognitiveEventKind.LEARNING_EVENT,
    # Cognitive state
    "cognitive_state.updated": CognitiveEventKind.COGNITIVE_STATE_CHANGED,
    "cognitive_state.snapshot": CognitiveEventKind.COGNITIVE_STATE_CHANGED,
    # Emotion
    "emotion.state": CognitiveEventKind.EMOTION_CHANGED,
    "emotion.update": CognitiveEventKind.EMOTION_CHANGED,
    # World model
    "world.state": CognitiveEventKind.WORLD_STATE_UPDATED,
    "world.prediction": CognitiveEventKind.PREDICTION_MADE,
}

# Aliases: raw event names from various nodes that should normalize
# to the canonical kind.  This prevents the projection layer from
# growing into a special-case collection.
_DEFAULT_ALIAS_MAP: dict[str, CognitiveEventKind] = {
    # Goal aliases across subsystems
    "GoalCreated": CognitiveEventKind.GOAL_CREATED,
    "NewGoal": CognitiveEventKind.GOAL_CREATED,
    "IntentGoal": CognitiveEventKind.GOAL_CREATED,
    "GeneratedGoal": CognitiveEventKind.GOAL_CREATED,
    "goal_created": CognitiveEventKind.GOAL_CREATED,
    "GoalCompleted": CognitiveEventKind.GOAL_COMPLETED,
    "goal_completed": CognitiveEventKind.GOAL_COMPLETED,
    "GoalAbandoned": CognitiveEventKind.GOAL_ABANDONED,
    "goal_abandoned": CognitiveEventKind.GOAL_ABANDONED,
    # Decision aliases
    "DecisionMade": CognitiveEventKind.DECISION_MADE,
    "decision_made": CognitiveEventKind.DECISION_MADE,
    "RouteDecision": CognitiveEventKind.ROUTING_DECIDED,
    # Memory aliases
    "MemoryStored": CognitiveEventKind.MEMORY_STORED,
    "memory_stored": CognitiveEventKind.MEMORY_STORED,
    "MemoryRecalled": CognitiveEventKind.MEMORY_RECALLED,
    "memory_recalled": CognitiveEventKind.MEMORY_RECALLED,
    # Prediction aliases
    "PredictionMade": CognitiveEventKind.PREDICTION_MADE,
    "prediction_made": CognitiveEventKind.PREDICTION_MADE,
    "PredictionVerified": CognitiveEventKind.PREDICTION_VERIFIED,
    "PredictionError": CognitiveEventKind.PREDICTION_ERROR,
    # Skill / Learning aliases
    "SkillLearned": CognitiveEventKind.SKILL_LEARNED,
    "skill_learned": CognitiveEventKind.SKILL_LEARNED,
    # Belief aliases
    "BeliefUpdated": CognitiveEventKind.BELIEF_UPDATED,
    "belief_updated": CognitiveEventKind.BELIEF_UPDATED,
    "BeliefRevised": CognitiveEventKind.BELIEF_REVISED,
}


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Normalizer
# ═══════════════════════════════════════════════════════════════════════════


class SemanticNormalizer:
    """Maps raw bus topics + message metadata to canonical CognitiveEventKind.

    Resolution order:
        1. Exact topic match in ``_topic_map``
        2. Prefix match (e.g., ``"memory.*"`` matches ``"memory.store"``)
        3. Alias match against ``message.type`` or ``message.data.get("event_name")``
        4. ``None`` — event is not a recognized cognitive event

    Usage::

        normalizer = SemanticNormalizer()
        kind = normalizer.normalize("memory.store", message)
        # → CognitiveEventKind.MEMORY_STORED

        normalizer.register_alias("MyCustomGoal", CognitiveEventKind.GOAL_CREATED)
    """

    def __init__(
        self,
        topic_map: dict[str, CognitiveEventKind] | None = None,
        alias_map: dict[str, CognitiveEventKind] | None = None,
    ) -> None:
        self._topic_map = dict(topic_map or _DEFAULT_TOPIC_MAP)
        self._alias_map = dict(alias_map or _DEFAULT_ALIAS_MAP)

    # ── Public API ───────────────────────────────────────────────────

    def normalize(
        self,
        topic: str,
        message: Any = None,
    ) -> CognitiveEventKind | None:
        """Resolve a raw bus event to a canonical CognitiveEventKind.

        Args:
            topic: The bus topic the message was published on.
            message: The ``Message`` object (optional).  Used for alias
                     resolution against ``message.type`` and
                     ``message.data``.

        Returns:
            The canonical event kind, or ``None`` if unrecognized.
        """
        # 1. Exact topic match
        if topic in self._topic_map:
            return self._topic_map[topic]

        # 2. Prefix match — find the longest prefix that matches
        best_match: CognitiveEventKind | None = None
        best_prefix_len = 0
        for prefix, kind in self._topic_map.items():
            if topic.startswith(prefix) and len(prefix) > best_prefix_len:
                best_match = kind
                best_prefix_len = len(prefix)
        if best_match is not None:
            return best_match

        # 3. Alias match against message metadata
        if message is not None:
            alias_kind = self._resolve_alias(message)
            if alias_kind is not None:
                return alias_kind

        logger.debug("SemanticNormalizer: no canonical kind for topic=%s", topic)
        return None

    def register_topic(self, topic: str, kind: CognitiveEventKind) -> None:
        """Register a new topic → kind mapping."""
        self._topic_map[topic] = kind

    def register_alias(self, raw_name: str, kind: CognitiveEventKind) -> None:
        """Register a new raw event name → kind alias."""
        self._alias_map[raw_name] = kind

    # ── Internal ─────────────────────────────────────────────────────

    def _resolve_alias(self, message: Any) -> CognitiveEventKind | None:
        """Try to resolve a canonical kind from message metadata."""
        # Check message.type (MessageType enum value)
        msg_type = getattr(message, "type", None)
        if msg_type is not None:
            type_str = str(msg_type)
            if type_str in self._alias_map:
                return self._alias_map[type_str]

        # Check message.data for an explicit event_name
        msg_data = getattr(message, "data", None)
        if isinstance(msg_data, dict):
            event_name = msg_data.get("event_name") or msg_data.get("event_kind")
            if event_name and event_name in self._alias_map:
                return self._alias_map[event_name]

        return None
