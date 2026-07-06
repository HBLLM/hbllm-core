"""
Cognitive Events — typed event objects for the cognitive loop.

Events are the primary communication mechanism between cognitive subsystems.
Instead of direct function calls, subsystems emit ``CognitiveEvent`` objects
that flow through the event queue → saliency evaluator → competition engine
→ executive controller → workspace pipeline.

Each event has:
    - A typed category (``CognitiveEventType``)
    - Source node provenance
    - SNN-computed saliency score
    - Arbitrary payload for subsystem-specific data

Event Types::

    USER_SPOKE           ← Perception layer
    MEMORY_UPDATED       ← MemoryNode
    MEMORY_CONFLICT      ← BeliefGraph
    PREDICTION_FAILED    ← PredictionEngine
    ATTENTION_SPIKE      ← AttentionManager
    GOAL_ADDED           ← GoalMemory
    GOAL_COMPLETED       ← GoalMemory
    EMOTION_CHANGED      ← EmotionEngine
    REWARD_RECEIVED      ← RewardEvaluator
    TASK_COMPLETED       ← ExecutiveController
    IDLE_DETECTED        ← SleepNode
    SIMULATION_COMPLETE  ← SimulationEngine

Usage::

    from hbllm.brain.cognitive_event import CognitiveEvent, CognitiveEventType

    event = CognitiveEvent(
        type=CognitiveEventType.USER_SPOKE,
        source_node="perception_fuser",
        timestamp=time.time(),
        priority=0.9,
        payload={"text": "Hello!", "modality": "text"},
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CognitiveEventType(StrEnum):
    """Typed categories for cognitive events.

    Each variant maps to a specific subsystem or cognitive process.
    Used for routing, filtering, and domain-specific saliency scoring.
    """

    # ── Perception ──
    USER_SPOKE = "user.spoke"

    # ── Memory ──
    MEMORY_UPDATED = "memory.updated"
    MEMORY_CONFLICT = "memory.conflict"

    # ── Prediction ──
    PREDICTION_FAILED = "prediction.failed"

    # ── Attention ──
    ATTENTION_SPIKE = "attention.spike"

    # ── Goals ──
    GOAL_ADDED = "goal.added"
    GOAL_COMPLETED = "goal.completed"
    GOAL_FAILED = "goal.failed"

    # ── Emotion / Reward ──
    EMOTION_CHANGED = "emotion.changed"
    REWARD_RECEIVED = "reward.received"

    # ── Execution ──
    TASK_COMPLETED = "task.completed"

    # ── Idle / Sleep ──
    IDLE_DETECTED = "idle.detected"

    # ── Simulation ──
    SIMULATION_COMPLETE = "simulation.complete"


@dataclass
class CognitiveEvent:
    """A single event in the cognitive loop.

    Events flow through the pipeline::

        Source Node → EventQueue → SaliencyEvaluator → CompetitionEngine
                                                            ↓
                                                    ExecutiveController
                                                            ↓
                                                        Workspace

    Attributes:
        type: The event category (determines routing and scoring).
        source_node: ID of the node that produced this event.
        timestamp: Epoch time (seconds) when the event was created.
        priority: Initial priority hint from the source [0.0, 1.0].
            This is the *pre-saliency* priority — saliency evaluation
            may adjust it based on cognitive state.
        payload: Subsystem-specific data.  The schema depends on the
            event type (e.g., USER_SPOKE includes ``text``).
        snn_saliency: SNN-computed saliency score set by the
            ``SaliencyEvaluator``.  Initially 0.0 until scored.
        tenant_id: Multi-tenant isolation key.
        correlation_id: Optional ID linking related events (e.g., a
            user query and its response).
    """

    type: CognitiveEventType
    source_node: str
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5
    payload: dict[str, Any] = field(default_factory=dict)
    snn_saliency: float = 0.0
    tenant_id: str = "default"
    correlation_id: str = ""

    @property
    def effective_priority(self) -> float:
        """Combined priority: max of initial priority and SNN saliency.

        This ensures that both explicit priority hints and SNN-computed
        saliency contribute to event ordering.
        """
        return max(self.priority, self.snn_saliency)

    def with_saliency(self, saliency: float) -> CognitiveEvent:
        """Return a copy with updated snn_saliency.

        Non-mutating — preserves the original event.
        """
        return CognitiveEvent(
            type=self.type,
            source_node=self.source_node,
            timestamp=self.timestamp,
            priority=self.priority,
            payload=self.payload,
            snn_saliency=saliency,
            tenant_id=self.tenant_id,
            correlation_id=self.correlation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and persistence."""
        return {
            "type": self.type.value,
            "source_node": self.source_node,
            "timestamp": self.timestamp,
            "priority": round(self.priority, 4),
            "snn_saliency": round(self.snn_saliency, 4),
            "effective_priority": round(self.effective_priority, 4),
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }

    def __repr__(self) -> str:
        return (
            f"CognitiveEvent(type={self.type.value!r}, "
            f"src={self.source_node!r}, "
            f"pri={self.effective_priority:.2f})"
        )
