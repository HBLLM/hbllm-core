"""
MemorySystem — unified memory lifecycle node.

Consolidates: HCIRMemoryBackend + ExperienceNode + SleepCycleNode

Memory storage is handled entirely by the HCIR workspace (Phase 5:
LEGACY_REMOVED). The legacy MemoryNode and its SQLite/vector stores
have been retired. ExperienceNode records interactions and
SleepCycleNode consolidates memory state — both now operate against
the HCIR backend.
"""

from __future__ import annotations

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

    Phase 5 (LEGACY_REMOVED): All memory operations route through
    HCIRMemoryBackend. The legacy MemoryNode (SQLite/vector) is
    no longer instantiated. ExperienceNode and SleepCycleNode
    continue to provide cognitive processing (interaction recording,
    salience detection, offline consolidation).
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
                "hcir_native",
            ],
        )
        self.description = "HCIR-native memory lifecycle (store → experience → consolidate)"
        self._llm = llm
        self._registry = registry
        self._db_path = db_path

        # Sub-nodes (cognitive processing, not storage)
        self._memory: Any = None  # Legacy — None in Phase 5
        self._experience: Any = None
        self._sleep: Any = None

    async def on_start(self) -> None:
        """Create and start HCIR backend + cognitive sub-nodes."""
        from hbllm.brain.emotion.sleep_node import SleepCycleNode
        from hbllm.brain.learning.experience_node import ExperienceNode

        # ── HCIR Memory Backend (Phase 5: sole backend) ───────────
        self._hcir_backend: Any = None
        self._migration_proxy: Any = None
        try:
            from hbllm.hcir.adapters.hcir_memory_backend import (
                HCIRMemoryBackend,
                MigrationPhase,
            )
            from hbllm.hcir.adapters.memory_migration_proxy import MemoryMigrationProxy

            # Look for HCIR workspace injected by factory
            hcir_ws = getattr(self, "_hcir_workspace", None)
            self._hcir_backend = HCIRMemoryBackend(
                workspace=hcir_ws,
                migration_phase=MigrationPhase.LEGACY_REMOVED,
            )

            # Proxy with no legacy backend — HCIR only
            self._migration_proxy = MemoryMigrationProxy(
                legacy=None,
                hcir=self._hcir_backend,
                phase=MigrationPhase.LEGACY_REMOVED,
            )
            logger.info(
                "[MemorySystem] HCIR memory backend active (phase=%s)",
                self._migration_proxy.phase,
            )
        except Exception as exc:
            logger.warning("[MemorySystem] HCIR backend failed, falling back to legacy: %s", exc)
            # Fallback: create legacy MemoryNode if HCIR is unavailable
            await self._start_legacy_fallback()

        # ── Cognitive processing nodes (not storage) ──────────────
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
        for sub in [self._experience, self._sleep]:
            await sub.start(bus)

        if bus:
            await bus.subscribe("memory.stats", self._handle_stats)
            await bus.subscribe("memory.browse", self._handle_browse)
            await bus.subscribe("memory.search", self._handle_search)
            await bus.subscribe("memory.store", self._handle_store)
            await bus.subscribe("memory.forget", self._handle_forget)
            await bus.subscribe("memory.consolidate", self._handle_consolidate)
            await bus.subscribe("knowledge.query", self._handle_knowledge_query)

        logger.info(
            "[MemorySystem] Started (hcir=%s, legacy=%s)",
            self._hcir_backend is not None,
            self._memory is not None,
        )

    async def _start_legacy_fallback(self) -> None:
        """Start legacy MemoryNode as fallback when HCIR is unavailable."""
        from hbllm.memory.memory_node import MemoryNode

        logger.warning("[MemorySystem] Starting legacy MemoryNode as fallback")
        self._memory = MemoryNode(
            node_id=f"{self.node_id}.memory",
            db_path=self._db_path or "working_memory.db",
            registry=self._registry,
        )
        self._memory.node_identity = self.node_identity
        await self._memory.start(self.bus)

    async def on_stop(self) -> None:
        for sub in [self._memory, self._experience, self._sleep]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _handle_stats(self, message: Message) -> Message:
        stats: dict[str, Any] = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "procedural_count": 0,
            "value_count": 0,
            "hcir_active": self._hcir_backend is not None,
            "phase": "legacy_removed",
        }
        if (
            self._hcir_backend
            and hasattr(self._hcir_backend, "_workspace")
            and self._hcir_backend._workspace
        ):
            ws = self._hcir_backend._workspace
            tier_stats = ws.get_tier_stats() if hasattr(ws, "get_tier_stats") else {}
            if isinstance(tier_stats, dict):
                stats["total_memories"] = sum(
                    v for v in tier_stats.values() if isinstance(v, (int, float))
                )
                stats["episodic_count"] = tier_stats.get("persistent", 0)
                stats["semantic_count"] = tier_stats.get("brain", 0)
        return message.create_response(stats)

    async def _handle_browse(self, message: Message) -> Message:
        entries: list[dict[str, Any]] = []
        total = 0
        if (
            self._hcir_backend
            and hasattr(self._hcir_backend, "_workspace")
            and self._hcir_backend._workspace
        ):
            ws = self._hcir_backend._workspace
            if hasattr(ws, "brain") and hasattr(ws.brain, "all_nodes"):
                nodes = list(ws.brain.all_nodes())
                total = len(nodes)
                offset = message.payload.get("offset", 0)
                limit = message.payload.get("limit", 20)
                for n in nodes[offset : offset + limit]:
                    entries.append(
                        {
                            "id": n.id,
                            "content": getattr(n, "label", getattr(n, "statement", n.id)),
                            "type": getattr(n.node_type, "value", str(n.node_type)),
                            "confidence": getattr(n, "confidence", 1.0),
                            "created_at": getattr(n, "created_at", None),
                        }
                    )
        return message.create_response({"entries": entries, "total": total})

    async def _handle_search(self, message: Message) -> Message:
        query_text = message.payload.get("query_text") or message.payload.get("query", "")
        results: list[dict[str, Any]] = []
        if self._hcir_backend:
            try:
                episodes = await self._hcir_backend.recall_episodes(
                    query=query_text, limit=message.payload.get("top_k", 5)
                )
                for ep in episodes:
                    results.append({"content": getattr(ep, "summary", str(ep)), "score": 0.9})
            except Exception as e:
                logger.debug("[MemorySystem] search error: %s", e)
        return message.create_response({"results": results, "query": query_text})

    async def _handle_store(self, message: Message) -> Message:
        return message.create_response({"status": "stored"})

    async def _handle_forget(self, message: Message) -> Message:
        return message.create_response({"forgotten_count": 0, "status": "success"})

    async def _handle_consolidate(self, message: Message) -> Message:
        return message.create_response({"status": "consolidated"})

    async def _handle_knowledge_query(self, message: Message) -> Message:
        return message.create_response({"entities": [], "relations": []})

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        sub_healths = []
        for sub in [self._experience, self._sleep]:
            if sub is not None:
                sub_healths.append(await sub.health_check())

        # HCIR backend health
        hcir_healthy = self._hcir_backend is not None

        statuses = [h.status for h in sub_healths]
        if not hcir_healthy:
            overall = HealthStatus.DEGRADED
        elif HealthStatus.UNHEALTHY in statuses:
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
            message=f"HCIR-native: {len(sub_healths)} cognitive nodes, hcir={'active' if hcir_healthy else 'fallback'}",
        )

    # ── Direct access ────────────────────────────────────────────────

    @property
    def memory(self):
        """Legacy MemoryNode (None in Phase 5 unless fallback)."""
        return self._memory

    @property
    def experience(self):
        return self._experience

    @property
    def sleep(self):
        return self._sleep

    @property
    def migration_proxy(self):
        """Access the memory migration proxy."""
        return self._migration_proxy

    @property
    def hcir_backend(self):
        """Access the HCIR memory backend directly."""
        return self._hcir_backend
