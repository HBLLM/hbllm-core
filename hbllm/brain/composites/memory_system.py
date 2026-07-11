"""
MemorySystem — unified memory lifecycle node.

Consolidates: MemoryNode + ExperienceNode + SleepCycleNode

These three are tightly coupled: ExperienceNode records into Memory,
SleepCycleNode consolidates Memory state. Combining them eliminates
cross-node memory coordination overhead.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.core.provider_adapter import ProviderLLM
    from hbllm.network.messages import Message
    from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class MemorySystem(Node):
    """
    Composite node that unifies the memory lifecycle.

    Wraps MemoryNode (episodic/semantic/procedural/value/KG),
    ExperienceNode (interaction recording + salience detection),
    and SleepCycleNode (offline memory consolidation).
    """

    def __init__(
        self,
        node_id: str = "memory_system",
        *,
        llm: ProviderLLM | None = None,
        registry: ServiceRegistry | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MEMORY,
            capabilities=[
                "memory",
                "episodic_memory",
                "semantic_memory",
                "procedural_memory",
                "value_memory",
                "knowledge_graph",
                "experience_recording",
                "salience_detection",
                "memory_consolidation",
                "sleep_cycle",
            ],
        )
        self.description = "Unified memory lifecycle (store → experience → consolidate)"
        self._llm = llm
        self._registry = registry
        self._db_path = db_path

        # Sub-nodes
        self._memory: Any = None
        self._experience: Any = None
        self._sleep: Any = None

    async def on_start(self) -> None:
        """Create and start all memory sub-nodes."""
        from hbllm.brain.emotion.sleep_node import SleepCycleNode
        from hbllm.brain.learning.experience_node import ExperienceNode
        from hbllm.memory.memory_node import MemoryNode

        self._memory = MemoryNode(
            node_id=f"{self.node_id}.memory",
            db_path=self._db_path or "working_memory.db",
            registry=self._registry,
        )
        self._memory.node_identity = self.node_identity
        self._experience = ExperienceNode(
            node_id=f"{self.node_id}.experience",
            llm=self._llm,
        )
        self._experience.node_identity = self.node_identity
        self._sleep = SleepCycleNode(
            node_id=f"{self.node_id}.sleep",
            llm=self._llm,
        )
        self._sleep.node_identity = self.node_identity

        bus = self.bus
        for sub in [self._memory, self._experience, self._sleep]:
            await sub.start(bus)

        logger.info("MemorySystem started with sub-nodes: memory, experience, sleep")

        # Trigger proactive memory warming (non-blocking)
        _warm_task = asyncio.create_task(self._warm_memory_cache())
        _warm_task.add_done_callback(
            lambda t: (
                logger.error("[MemorySystem] _warm_memory_cache raised: %s", t.exception())
                if not t.cancelled() and t.exception()
                else None
            )
        )

    async def _warm_memory_cache(self) -> None:
        """Proactively warm the semantic cache with recent high-salience concepts."""
        try:
            # Short delay to let the rest of the system boot
            import asyncio

            await asyncio.sleep(2.0)

            if self._memory and hasattr(self._memory, "knowledge_graph"):
                kg = self._memory.knowledge_graph
                # Get the last 5 entities added to the KG
                entities = list(kg._entities.values())[-5:]
                if not entities:
                    logger.debug("[MemorySystem] No entities in KG to warm cache.")
                    return

                for entity in entities:
                    logger.debug("[MemorySystem] Warming cache for concept: %s", entity.label)
                    await asyncio.to_thread(
                        self._memory.semantic_db.search, entity.label, top_k=5, tenant_id="default"
                    )
            logger.info("[MemorySystem] Proactive memory warming complete.")
        except Exception as e:
            logger.warning("[MemorySystem] Failed to warm memory cache: %s", e)

    async def on_stop(self) -> None:
        for sub in [self._memory, self._experience, self._sleep]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        sub_healths = []
        for sub in [self._memory, self._experience, self._sleep]:
            if sub is not None:
                sub_healths.append(await sub.health_check())

        statuses = [h.status for h in sub_healths]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return NodeHealth(
            node_id=self.node_id,
            status=overall,
            uptime_seconds=self.uptime,
            capabilities_available=self.capabilities,
            message=f"Composite: {len(sub_healths)} sub-nodes",
        )

    # ── Direct access ────────────────────────────────────────────────

    @property
    def memory(self):
        return self._memory

    @property
    def experience(self):
        return self._experience

    @property
    def sleep(self):
        return self._sleep
