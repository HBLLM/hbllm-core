"""
Attention Budget Manager — importance-weighted memory retention, focus allocation,
and cognitive task scheduling.

Controls how cognitive resources are distributed across:
  - Memory types (episodic, semantic, procedural, value)
  - Domain focus (which topics get priority context window space)
  - Retention decisions (which memories to keep vs prune)
  - Cognitive tasks (which contradictions, curiosities, reviews to pursue)

This prevents the system from drowning in low-value information
while ensuring critical knowledge is always accessible.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
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


# ── Cognitive Priority Scheduler ─────────────────────────────────────────────


class CognitiveTaskType(StrEnum):
    """Types of cognitive tasks that compete for attention."""

    CONTRADICTION = "contradiction"  # Resolve conflicting beliefs
    CURIOSITY = "curiosity"  # Investigate unknown territory
    MECHANISM_REFINEMENT = "mechanism_refinement"  # Strengthen weak mechanisms
    BELIEF_REVISION = "belief_revision"  # Update outdated beliefs
    CONCEPT_FORMATION = "concept_formation"  # Abstract recurring patterns
    LEARNING = "learning"  # Acquire new knowledge
    MAINTENANCE = "maintenance"  # Prune, consolidate, decay


@dataclass
class CognitiveTask:
    """A cognitive task competing for attention.

    Priority is computed as a weighted sum (not multiplication)
    to prevent any single zero factor from permanently burying a task.
    """

    task_id: str = field(default_factory=lambda: f"ct_{uuid.uuid4().hex[:10]}")
    task_type: CognitiveTaskType = CognitiveTaskType.LEARNING
    domain: str = "general"
    source: str = ""  # which node generated this
    description: str = ""
    created_at: float = field(default_factory=time.time)

    # Scoring factors (0.0 to 1.0)
    uncertainty: float = 0.5  # How uncertain is the current state?
    goal_relevance: float = 0.5  # How relevant to active goals?
    contradiction_severity: float = 0.0  # Severity if contradiction-type
    novelty: float = 0.5  # How novel is the domain/concept?
    expected_value: float = 0.5  # Expected cognitive value of resolution

    # Lifecycle
    priority_score: float = 0.0  # Computed, not set directly
    claimed_by: str | None = None  # Node that picked this up
    completed: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    def compute_priority(self) -> float:
        """Weighted sum priority with age boost.

        priority =
            0.30 * uncertainty
            + 0.25 * goal_relevance
            + 0.20 * contradiction_severity
            + 0.15 * novelty
            + 0.10 * expected_value

        Then:
            priority *= age_boost

        Age boost prevents starvation — old tasks gradually
        rise in priority even if their base score is low.
        """
        base = (
            0.30 * self.uncertainty
            + 0.25 * self.goal_relevance
            + 0.20 * self.contradiction_severity
            + 0.15 * self.novelty
            + 0.10 * self.expected_value
        )

        # Age boost: logarithmic growth, caps at ~2x after 1 hour
        age_seconds = max(0.0, time.time() - self.created_at)
        age_boost = 1.0 + 0.15 * math.log1p(age_seconds / 60.0)

        self.priority_score = base * age_boost
        return self.priority_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "domain": self.domain,
            "source": self.source,
            "description": self.description,
            "priority_score": round(self.priority_score, 4),
            "uncertainty": self.uncertainty,
            "goal_relevance": self.goal_relevance,
            "contradiction_severity": self.contradiction_severity,
            "novelty": self.novelty,
            "expected_value": self.expected_value,
            "claimed_by": self.claimed_by,
            "completed": self.completed,
            "age_s": round(time.time() - self.created_at, 1),
        }


class CognitivePriorityScheduler:
    """Decides which cognitive task deserves attention next.

    This is the prefrontal attention system — the global scheduler
    for all cognitive work that isn't immediate query handling.

    Consumers:
        AutonomousLearner — picks up learning tasks during idle
        SleepNode — picks up maintenance/consolidation tasks
        CuriosityNode — picks up curiosity tasks
    """

    def __init__(self, max_pending: int = 200) -> None:
        self._tasks: dict[str, CognitiveTask] = {}
        self._max_pending = max_pending
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_pruned": 0,
        }

    def submit(self, task: CognitiveTask) -> CognitiveTask:
        """Submit a cognitive task for scheduling."""
        task.compute_priority()
        self._tasks[task.task_id] = task
        self._stats["tasks_created"] += 1

        # Prune if over capacity (drop lowest priority completed/old)
        if len(self._tasks) > self._max_pending:
            self._prune()

        return task

    def next_task(
        self,
        task_type: CognitiveTaskType | None = None,
        claimer: str | None = None,
    ) -> CognitiveTask | None:
        """Get the highest-priority unclaimed task.

        Optionally filter by task_type and auto-claim for a node.
        """
        candidates = [
            t for t in self._tasks.values()
            if not t.completed and t.claimed_by is None
        ]
        if task_type is not None:
            candidates = [t for t in candidates if t.task_type == task_type]

        if not candidates:
            return None

        # Recompute priorities (age boost changes over time)
        for t in candidates:
            t.compute_priority()

        candidates.sort(key=lambda t: t.priority_score, reverse=True)
        best = candidates[0]

        if claimer:
            best.claimed_by = claimer

        return best

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self._tasks:
            self._tasks[task_id].completed = True
            self._stats["tasks_completed"] += 1

    def get_pending(self, limit: int = 20) -> list[CognitiveTask]:
        """Get top pending tasks by priority."""
        pending = [t for t in self._tasks.values() if not t.completed]
        for t in pending:
            t.compute_priority()
        pending.sort(key=lambda t: t.priority_score, reverse=True)
        return pending[:limit]

    def _prune(self) -> None:
        """Remove completed and lowest-priority tasks to stay within budget."""
        # Remove completed first
        completed = [tid for tid, t in self._tasks.items() if t.completed]
        for tid in completed:
            del self._tasks[tid]
            self._stats["tasks_pruned"] += 1

        # If still over, remove lowest priority
        if len(self._tasks) > self._max_pending:
            tasks_by_priority = sorted(
                self._tasks.items(),
                key=lambda kv: kv[1].priority_score,
            )
            to_remove = len(self._tasks) - self._max_pending
            for tid, _ in tasks_by_priority[:to_remove]:
                del self._tasks[tid]
                self._stats["tasks_pruned"] += 1

    def stats(self) -> dict[str, Any]:
        pending = sum(1 for t in self._tasks.values() if not t.completed)
        claimed = sum(1 for t in self._tasks.values() if t.claimed_by and not t.completed)
        return {
            **self._stats,
            "pending": pending,
            "claimed": claimed,
            "total_tracked": len(self._tasks),
        }


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
        learning.contradiction.discovered — creates contradiction tasks
        learning.weak_area — creates learning tasks
        learning.session.complete — creates concept formation tasks
        curiosity.investigate — creates curiosity tasks

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
            "episodic": 10000,
            "semantic": 5000,
            "procedural": 500,
            "value": 200,
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
        self._polling_task: asyncio.Task[None] | None = None

        # Cognitive Priority Scheduler
        self.scheduler = CognitivePriorityScheduler()

    async def on_start(self) -> None:
        logger.info("Starting AttentionManager (budget=%d tokens)", self.total_context_budget)
        await self.bus.subscribe("system.evaluation", self._track_activity)
        await self.bus.subscribe("system.sleep.prune_trigger", self._handle_prune)
        await self.bus.subscribe("attention.query", self._handle_query)
        await self.bus.subscribe("attention.score", self._handle_score_request)
        await self.bus.subscribe("workspace.thought", self._handle_thought_utility)
        # Coordinate with LoadManager
        await self.bus.subscribe("system.load.policy_update", self._handle_load_policy)
        # Cognitive task ingestion
        await self.bus.subscribe(
            "learning.contradiction.discovered", self._ingest_contradiction
        )
        await self.bus.subscribe("learning.weak_area", self._ingest_weak_area)
        await self.bus.subscribe(
            "learning.session.complete", self._ingest_session_complete
        )
        await self.bus.subscribe("curiosity.investigate", self._ingest_curiosity)
        await self.bus.subscribe("attention.next_task", self._handle_next_task)
        self._polling_task = asyncio.create_task(self._poll_memory_stats())

    async def on_stop(self) -> None:
        logger.info(
            "Stopping AttentionManager — prunes=%d rebalances=%d",
            self._prune_count,
            self._rebalance_count,
        )
        if self._polling_task:
            self._polling_task.cancel()

    async def _handle_load_policy(self, message: Message) -> None:
        """Adjust context budget when LoadManager changes degradation policy."""
        try:
            new_max = message.payload.get("max_context_tokens")
            if new_max is not None and isinstance(new_max, int):
                old_budget = self.total_context_budget
                self.total_context_budget = new_max
                # Rebalance focus allocations to fit the new budget
                self.rebalance_focus()
                logger.info(
                    "[AttentionManager] Context budget adjusted %d → %d (load policy: %s)",
                    old_budget,
                    new_max,
                    message.payload.get("level", "unknown"),
                )
        except Exception as e:
            logger.warning("[AttentionManager] Error handling load policy update: %s", e)

    async def _poll_memory_stats(self) -> None:
        """Periodically poll memory nodes for item counts to update budgets."""
        while self._running:
            try:
                # Wait for initial bus registration
                await asyncio.sleep(5.0)
                if not self._running:
                    break

                msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    tenant_id="default",
                    topic="memory.stats",
                    payload={"tenant_id": "default"},
                )
                reply = await self.bus.request("memory.stats", msg, timeout=4.0)
                if reply and reply.type != MessageType.ERROR:
                    stats = reply.payload
                    if "episodic" in stats:
                        self.update_item_count("episodic", stats["episodic"].get("turns", 0))
                    if "semantic" in stats:
                        self.update_item_count("semantic", stats["semantic"].get("documents", 0))
                    if "procedural" in stats:
                        self.update_item_count("procedural", stats["procedural"].get("skills", 0))
                    if "value" in stats:
                        self.update_item_count("value", stats["value"].get("rewards", 0))
            except Exception as e:
                logger.debug("Failed to poll memory stats for attention budget: %s", e)

            # Poll every 10 seconds
            await asyncio.sleep(10.0)

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
                key=lambda mid: self._importance_scores.get(mid, 0.0),
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
            "cognitive_scheduler": self.scheduler.stats(),
        }

    async def _handle_thought_utility(self, message: Message) -> None:
        """Dynamically adjust context allocations using utility scores."""
        try:
            metadata = message.payload.get("metadata", {})
            utility = metadata.get("utility_score")
            domain = metadata.get("domain", "general")
            if utility is not None and domain in self._focus:
                alloc = self._focus[domain]
                # If utility is high, allocate more context space
                if utility > 0.7:
                    alloc.context_tokens = min(alloc.context_tokens + 256, 4096)
                    alloc.priority = min(alloc.priority + 0.1, 1.0)
                    logger.info(
                        "[AttentionManager] Increased context tokens for domain '%s' to %d due to high utility: %.2f",
                        domain,
                        alloc.context_tokens,
                        utility,
                    )
                # If utility is low, contract context space
                elif utility < 0.3:
                    alloc.context_tokens = max(alloc.context_tokens - 256, 512)
                    alloc.priority = max(alloc.priority - 0.1, 0.1)
                    logger.info(
                        "[AttentionManager] Contracted context tokens for domain '%s' to %d due to low utility: %.2f",
                        domain,
                        alloc.context_tokens,
                        utility,
                    )
        except Exception as e:
            logger.debug("[AttentionManager] Error handling thought utility: %s", e)

    # ── Cognitive Task Ingestion ──────────────────────────────────────

    async def _ingest_contradiction(self, message: Message) -> None:
        """Create a cognitive task from a discovered contradiction."""
        payload = message.payload
        self.scheduler.submit(CognitiveTask(
            task_type=CognitiveTaskType.CONTRADICTION,
            domain=payload.get("concept", "general"),
            source=message.source_node_id,
            description=(
                f"Resolve: '{payload.get('claim_a', '?')[:50]}' "
                f"vs '{payload.get('claim_b', '?')[:50]}'"
            ),
            uncertainty=0.8,
            contradiction_severity=payload.get("severity", 0.5),
            novelty=0.6,
            expected_value=0.7,
            payload=payload,
        ))

    async def _ingest_weak_area(self, message: Message) -> None:
        """Create a cognitive task from a weak learning area."""
        payload = message.payload
        score = payload.get("score", 0.5)
        self.scheduler.submit(CognitiveTask(
            task_type=CognitiveTaskType.LEARNING,
            domain=payload.get("goal_topic", "general"),
            source=message.source_node_id,
            description=f"Strengthen weak area: {payload.get('concept', '?')}",
            uncertainty=1.0 - score,
            goal_relevance=0.7,
            novelty=0.4,
            expected_value=0.6,
            payload=payload,
        ))

    async def _ingest_session_complete(self, message: Message) -> None:
        """Create concept formation task after learning session."""
        payload = message.payload
        models_built = payload.get("causal_models_built", 0)
        if models_built >= 2:
            # Enough models to attempt abstraction
            self.scheduler.submit(CognitiveTask(
                task_type=CognitiveTaskType.CONCEPT_FORMATION,
                domain=payload.get("topic", "general"),
                source=message.source_node_id,
                description=(
                    f"Abstract patterns from {models_built} models "
                    f"in '{payload.get('topic', '?')}'"
                ),
                uncertainty=0.5,
                goal_relevance=0.4,
                novelty=0.7,
                expected_value=0.6,
                payload=payload,
            ))

    async def _ingest_curiosity(self, message: Message) -> None:
        """Create a cognitive task from a curiosity signal."""
        payload = message.payload
        priority_map = {"high": 0.8, "medium": 0.5, "low": 0.3}
        priority_str = payload.get("priority", "medium")
        self.scheduler.submit(CognitiveTask(
            task_type=CognitiveTaskType.CURIOSITY,
            domain=payload.get("domain", "general"),
            source=message.source_node_id,
            description=payload.get("question", "Unknown curiosity")[:200],
            uncertainty=0.7,
            goal_relevance=priority_map.get(priority_str, 0.5),
            novelty=0.8,
            expected_value=0.5,
            payload=payload,
        ))

    async def _handle_next_task(self, message: Message) -> Message | None:
        """Return the highest-priority cognitive task."""
        task_type_str = message.payload.get("task_type")
        claimer = message.payload.get("claimer")

        task_type = None
        if task_type_str:
            try:
                task_type = CognitiveTaskType(task_type_str)
            except ValueError:
                pass

        task = self.scheduler.next_task(task_type=task_type, claimer=claimer)
        if task:
            return message.create_response(task.to_dict())
        return message.create_response({"task": None})
