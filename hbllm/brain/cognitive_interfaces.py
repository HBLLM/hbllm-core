"""
Cognitive Interfaces — abstract contracts for the event-driven scheduler.

The ``ExecutiveController`` depends **only** on these interfaces, never on
concrete implementations.  This enables:

    - Testability: mock any subsystem
    - Swappability: replace SNN saliency with a heuristic scorer
    - Decoupling: subsystems evolve independently

Interface hierarchy::

    IEventQueue         — submit / drain events
    IAttentionSelector  — score events by saliency
    ICompetition        — WTA among scored events
    IGoalProvider       — active goals for context
    IWorkspace          — receive winning events for reasoning
    IPredictor          — observe / predict sequences
    IMemory             — store / retrieve memory cubes
    ISimulator          — mental rehearsal before action

Usage::

    from hbllm.brain.cognitive_interfaces import IEventQueue, IAttentionSelector

    class MyQueue(IEventQueue):
        async def submit(self, event): ...
        async def drain(self, max_batch=10): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# Lazy import type references — avoid circular imports
# ═══════════════════════════════════════════════════════════════════════════

# CognitiveEvent is imported lazily by concrete implementations.
# Interfaces use string annotations via ``from __future__ import annotations``.


# ═══════════════════════════════════════════════════════════════════════════
# IEventQueue — priority queue with cognitive semantics
# ═══════════════════════════════════════════════════════════════════════════


class IEventQueue(ABC):
    """Abstract event queue for the cognitive loop.

    Events are submitted by cognitive nodes and drained in batches
    by the ``ExecutiveController`` for saliency evaluation.
    """

    @abstractmethod
    async def submit(self, event: Any) -> None:
        """Submit a cognitive event to the queue.

        Args:
            event: A ``CognitiveEvent`` instance.
        """
        ...

    @abstractmethod
    async def drain(self, max_batch: int = 10) -> list[Any]:
        """Drain up to ``max_batch`` events from the queue.

        Returns events in priority order (highest first).
        Draining removes events from the queue.

        Args:
            max_batch: Maximum number of events to return.

        Returns:
            List of ``CognitiveEvent`` instances, ordered by priority.
        """
        ...

    @abstractmethod
    async def size(self) -> int:
        """Return the current number of events in the queue."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IAttentionSelector — SNN-based saliency scoring
# ═══════════════════════════════════════════════════════════════════════════


class IAttentionSelector(ABC):
    """Scores events by saliency using SNN and cognitive state.

    Takes a batch of events and returns them with ``snn_saliency``
    set based on current cognitive state, neuromodulation, and
    event type relevance.
    """

    @abstractmethod
    async def evaluate(self, events: list[Any]) -> list[Any]:
        """Score a batch of events for saliency.

        Args:
            events: List of ``CognitiveEvent`` instances to score.

        Returns:
            The same events with ``snn_saliency`` updated.
            May reorder events by saliency.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ICompetition — WTA among scored events
# ═══════════════════════════════════════════════════════════════════════════


class ICompetition(ABC):
    """Winner-Take-All competition among saliency-scored events.

    Uses lateral inhibition (``HierarchicalWTA``) to select the
    top-K events that should actually be processed.
    """

    @abstractmethod
    async def compete(self, scored_events: list[Any]) -> list[Any]:
        """Run competition on scored events.

        Args:
            scored_events: Events with ``snn_saliency`` already set.

        Returns:
            Winning events that survived competition.
            Typically a small subset (top 1–5).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IGoalProvider — goal context for saliency and planning
# ═══════════════════════════════════════════════════════════════════════════


class IGoalProvider(ABC):
    """Provides active goal context for saliency and planning.

    The executive controller reads goals to bias saliency scoring
    and to route events to goal-relevant subsystems.
    """

    @abstractmethod
    async def get_active_goals(self, tenant_id: str) -> list[Any]:
        """Return all active goals for a tenant.

        Args:
            tenant_id: Multi-tenant isolation key.

        Returns:
            List of active goal objects.
        """
        ...

    @abstractmethod
    async def get_urgent_goals(self, horizon: float = 3600.0) -> list[Any]:
        """Return goals with deadlines within the horizon.

        Args:
            horizon: Time window in seconds (default 1 hour).

        Returns:
            List of urgent goal objects.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IWorkspace — receive winning events for reasoning
# ═══════════════════════════════════════════════════════════════════════════


class IWorkspace(ABC):
    """Abstract workspace that receives winning events.

    The workspace is the "blackboard" where selected events are
    processed by reasoning subsystems.
    """

    @abstractmethod
    async def submit_for_reasoning(self, event: Any) -> None:
        """Submit a winning event for workspace reasoning.

        Args:
            event: A ``CognitiveEvent`` that won competition.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IPredictor — sequence prediction (Markov, etc.)
# ═══════════════════════════════════════════════════════════════════════════


class IPredictor(ABC):
    """Online sequence predictor for cognitive anticipation.

    Observes values over time and predicts future values.
    Used for query prediction, goal transitions, tool use, etc.
    """

    @abstractmethod
    async def observe(self, value: str, timestamp: float) -> None:
        """Record an observed value in the sequence.

        Args:
            value: The observed value (e.g., event type, query domain).
            timestamp: When the observation occurred.
        """
        ...

    @abstractmethod
    async def predict(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Predict the next values.

        Args:
            top_k: Number of top predictions to return.

        Returns:
            List of ``(predicted_value, confidence)`` tuples,
            sorted by confidence descending.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IMemory — store and retrieve memory cubes
# ═══════════════════════════════════════════════════════════════════════════


class IMemory(ABC):
    """Abstract memory interface for store/retrieve.

    Implemented by ``MemoryNode`` in Milestone 2.
    """

    @abstractmethod
    async def store(
        self, content: str, memory_type: str = "episodic", metadata: dict[str, Any] | None = None
    ) -> str:
        """Store content as a memory.

        Args:
            content: The content to memorize.
            memory_type: Category of memory.
            metadata: Optional metadata dict.

        Returns:
            The memory ID.
        """
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
        """Retrieve relevant memories.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of matching memory objects.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ISimulator — mental rehearsal (Milestone 4)
# ═══════════════════════════════════════════════════════════════════════════


class ISimulator(ABC):
    """Mental rehearsal — simulate actions before executing.

    Implemented by ``SimulationEngine`` in Milestone 4.
    """

    @abstractmethod
    async def simulate(self, action: str, context: Any) -> list[Any]:
        """Simulate an action and predict consequences.

        Args:
            action: The candidate action to simulate.
            context: Current cognitive context (state, goals, etc.).

        Returns:
            List of predicted consequence objects.
        """
        ...
