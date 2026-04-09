"""
Attention Budget Manager — importance-weighted memory retention and focus allocation.

Controls how cognitive resources are distributed across:
  - Memory types (episodic, semantic, procedural, value)
  - Domain focus (which topics get priority context window space)
  - Retention decisions (which memories to keep vs prune)

This prevents the system from drowning in low-value information
while ensuring critical knowledge is always accessible.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class MemoryBudget:
    """Budget allocation for a single memory type."""

    memory_type: str  # episodic, semantic, procedural, value
    max_items: int
    current_items: int = 0
    priority_weight: float = 1.0  # higher = more importance
    last_pruned: float = 0.0

    @property
    def utilization(self) -> float:
        return self.current_items / max(self.max_items, 1)

    @property
    def needs_pruning(self) -> bool:
        return self.utilization > 0.9


@dataclass
class FocusAllocation:
    """Focus budget for a domain/topic."""

    domain: str
    priority: float  # 0-1, higher = more focus
    context_tokens: int  # allocated context window tokens
    recent_queries: int = 0
    success_rate: float = 0.5
    last_active: float = field(default_factory=time.time)


class AttentionManager(Node):
    """
    Manages cognitive resource distribution.

    Responsibilities:
      1. Memory budget — how many items each memory type can hold
      2. Focus allocation — which domains get priority in context window
      3. Retention scoring — which memories are worth keeping
      4. Pruning orchestration — triggers cleanup during sleep cycle

    Subscribes to:
        system.evaluation — tracks domain activity
        system.sleep.prune_trigger — runs pruning during sleep
        attention.query — returns current allocations

    Publishes:
        system.attention.pruned — after memory pruning
        system.attention.rebalanced — after focus reallocation
    """

    def __init__(
        self,
        node_id: str,
        total_context_budget: int = 4096,
        default_memory_limits: dict[str, int] | None = None,
        importance_decay: float = 0.95,
        min_importance: float = 0.1,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["attention_management", "memory_budgets"],
        )
        self.total_context_budget = total_context_budget
        self.importance_decay = importance_decay
        self.min_importance = min_importance

        # Memory budgets
        limits = default_memory_limits or {
            "episodic": 500,
            "semantic": 1000,
            "procedural": 200,
            "value": 100,
        }
        self._budgets: dict[str, MemoryBudget] = {
            name: MemoryBudget(memory_type=name, max_items=limit) for name, limit in limits.items()
        }

        # Focus allocations (dynamic per domain)
        self._focus: dict[str, FocusAllocation] = {}

        # Importance scores for individual memories
        self._importance_scores: dict[str, float] = {}
        self._max_tracked = 5000

        # Stats
        self._prune_count = 0
        self._rebalance_count = 0

    async def on_start(self) -> None:
        logger.info("Starting AttentionManager (budget=%d tokens)", self.total_context_budget)
        await self.bus.subscribe("system.evaluation", self._track_activity)
        await self.bus.subscribe("system.sleep.prune_trigger", self._handle_prune)
        await self.bus.subscribe("attention.query", self._handle_query)
        await self.bus.subscribe("attention.score", self._handle_score_request)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping AttentionManager — prunes=%d rebalances=%d",
            self._prune_count,
            self._rebalance_count,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Memory Budget Management ─────────────────────────────────────

    def get_budget(self, memory_type: str) -> MemoryBudget | None:
        """Get current budget for a memory type."""
        return self._budgets.get(memory_type)

    def update_item_count(self, memory_type: str, count: int) -> None:
        """Update the current item count for a memory type."""
        if memory_type in self._budgets:
            self._budgets[memory_type].current_items = count

    def should_accept(self, memory_type: str, importance: float = 0.5) -> bool:
        """Decide whether to accept a new memory item."""
        budget = self._budgets.get(memory_type)
        if not budget:
            return True

        # Always accept if under budget
        if budget.utilization < 0.8:
            return True

        # Near capacity: only accept if importance exceeds threshold
        threshold = budget.utilization  # higher utilization = higher bar
        return importance > threshold

    # ── Importance Scoring ───────────────────────────────────────────

    def score_importance(
        self,
        memory_id: str,
        recency: float = 0.5,
        frequency: float = 0.5,
        relevance: float = 0.5,
        emotional_weight: float = 0.0,
    ) -> float:
        """
        Score a memory's importance for retention decisions.

        Factors:
          - Recency: how recently was it accessed?
          - Frequency: how often is it accessed?
          - Relevance: how relevant to current goals?
          - Emotional weight: salience from experience node
        """
        score = 0.3 * recency + 0.3 * frequency + 0.25 * relevance + 0.15 * emotional_weight
        score = max(self.min_importance, min(1.0, score))

        self._importance_scores[memory_id] = score
        if len(self._importance_scores) > self._max_tracked:
            # Prune lowest scores
            sorted_ids = sorted(
                self._importance_scores,
                key=self._importance_scores.get,  # type: ignore[arg-type]
            )
            for mid in sorted_ids[: len(sorted_ids) // 4]:
                del self._importance_scores[mid]

        return score

    def get_importance(self, memory_id: str) -> float:
        """Get cached importance score for a memory."""
        return self._importance_scores.get(memory_id, 0.5)

    def get_pruning_candidates(self, memory_type: str, count: int = 10) -> list[tuple[str, float]]:
        """Get lowest-importance memory IDs for pruning."""
        # Filter to memories of this type (by prefix convention)
        candidates = [
            (mid, score)
            for mid, score in self._importance_scores.items()
            if mid.startswith(f"{memory_type}:")
        ]
        candidates.sort(key=lambda x: x[1])
        return candidates[:count]

    def decay_all_scores(self) -> None:
        """Apply temporal decay to all importance scores."""
        for mid in self._importance_scores:
            self._importance_scores[mid] = max(
                self.min_importance,
                self._importance_scores[mid] * self.importance_decay,
            )

    # ── Focus Allocation ─────────────────────────────────────────────

    def allocate_focus(self, domain: str, priority: float = 0.5) -> FocusAllocation:
        """Allocate or update focus for a domain."""
        if domain not in self._focus:
            # New domain gets proportional share
            n_domains = max(len(self._focus) + 1, 1)
            tokens = self.total_context_budget // n_domains
            self._focus[domain] = FocusAllocation(
                domain=domain,
                priority=priority,
                context_tokens=tokens,
            )
        else:
            self._focus[domain].priority = priority
            self._focus[domain].last_active = time.time()

        return self._focus[domain]

    def get_focus(self, domain: str) -> FocusAllocation | None:
        """Get current focus allocation for a domain."""
        return self._focus.get(domain)

    def rebalance_focus(self) -> dict[str, int]:
        """Rebalance context token allocation based on priorities."""
        if not self._focus:
            return {}

        # Weight by priority × recency
        now = time.time()
        total_weight = 0.0
        weights: dict[str, float] = {}

        for domain, alloc in self._focus.items():
            recency_factor = max(0.1, 1.0 - (now - alloc.last_active) / 3600)
            w = alloc.priority * recency_factor
            weights[domain] = w
            total_weight += w

        # Allocate tokens proportionally
        result: dict[str, int] = {}
        for domain, w in weights.items():
            share = int(self.total_context_budget * (w / max(total_weight, 0.01)))
            share = max(256, share)  # minimum 256 tokens per domain
            self._focus[domain].context_tokens = share
            result[domain] = share

        self._rebalance_count += 1
        return result

    # ── Event Handlers ───────────────────────────────────────────────

    async def _track_activity(self, message: Message) -> None:
        """Track domain activity from evaluation events."""
        payload = message.payload
        # Try to determine domain from evaluation context
        domain = payload.get("thought_type", "general")
        score = payload.get("overall_score", 0.5)

        alloc = self.allocate_focus(domain)
        alloc.recent_queries += 1
        alloc.success_rate = (alloc.success_rate + score) / 2

    async def _handle_prune(self, message: Message) -> Message | None:
        """Run memory pruning during sleep cycle."""
        logger.info("[AttentionManager] Running memory pruning cycle")

        # 1. Decay all importance scores
        self.decay_all_scores()

        # 2. Identify budgets that need pruning
        pruned_count = 0
        for mem_type, budget in self._budgets.items():
            if budget.needs_pruning:
                candidates = self.get_pruning_candidates(mem_type, count=20)
                for mid, _score in candidates:
                    del self._importance_scores[mid]
                    pruned_count += 1

                logger.info(
                    "[AttentionManager] Pruned %d low-importance items from %s",
                    len(candidates),
                    mem_type,
                )

        # 3. Rebalance focus
        allocations = self.rebalance_focus()

        self._prune_count += 1

        # Publish results
        await self.bus.publish(
            "system.attention.pruned",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.attention.pruned",
                payload={
                    "pruned_count": pruned_count,
                    "focus_allocations": allocations,
                },
            ),
        )

        return None

    async def _handle_score_request(self, message: Message) -> Message | None:
        """Score a memory item's importance on request."""
        payload = message.payload
        memory_id = payload.get("memory_id", "")
        score = self.score_importance(
            memory_id=memory_id,
            recency=payload.get("recency", 0.5),
            frequency=payload.get("frequency", 0.5),
            relevance=payload.get("relevance", 0.5),
            emotional_weight=payload.get("emotional_weight", 0.0),
        )
        return message.create_response({"memory_id": memory_id, "importance": score})

    async def _handle_query(self, message: Message) -> Message | None:
        """Return attention stats."""
        return message.create_response(self.stats())

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_context_budget": self.total_context_budget,
            "memory_budgets": {
                name: {
                    "max_items": b.max_items,
                    "current_items": b.current_items,
                    "utilization": round(b.utilization, 3),
                    "needs_pruning": b.needs_pruning,
                }
                for name, b in self._budgets.items()
            },
            "focus_allocations": {
                domain: {
                    "priority": round(a.priority, 3),
                    "context_tokens": a.context_tokens,
                    "recent_queries": a.recent_queries,
                    "success_rate": round(a.success_rate, 3),
                }
                for domain, a in self._focus.items()
            },
            "tracked_memories": len(self._importance_scores),
            "prune_cycles": self._prune_count,
            "rebalance_cycles": self._rebalance_count,
        }
