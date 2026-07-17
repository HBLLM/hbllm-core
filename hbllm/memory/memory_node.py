"""Shared Memory Node.

Acts as the single source of truth for conversation history across
all domain modules. It delegates to decomposed handlers for storage,
recall, reflection, persistence, and subscription management.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from hbllm.memory.episodic import EpisodicMemory
from hbllm.memory.handlers import (
    PersistenceHandler,
    RecallHandler,
    ReflectionHandler,
    StorageHandler,
    SubscriptionHandler,
)
from hbllm.memory.interface import MemoryType, SearchResult, UnifiedMemoryInterface
from hbllm.memory.knowledge_graph import KnowledgeGraph
from hbllm.memory.priming import WorkingMemoryPrimer
from hbllm.memory.procedural import ProceduralMemory
from hbllm.memory.semantic import SemanticMemory
from hbllm.memory.value_memory import ValueMemory
from hbllm.network.messages import Message
from hbllm.network.node import Node, NodeType
from hbllm.network.registry import ServiceRegistry
from hbllm.security.tenant_guard import require_tenant

logger = logging.getLogger(__name__)


class MemoryNode(Node, UnifiedMemoryInterface):
    """Service node that persists and recalls conversation context.

    Delegates specific responsibilities to helper handlers to remain slim
    and cohesive.
    """

    def __init__(
        self,
        node_id: str,
        db_path: str | Path = "working_memory.db",
        registry: ServiceRegistry | None = None,
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.MEMORY,
            capabilities=[
                "episodic_storage",
                "semantic_retrieval",
                "procedural_skills",
                "value_tracking",
            ],
        )
        self.registry = registry

        # Ensure the memory directory exists
        if str(db_path) != ":memory:":
            db_path = Path(db_path).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            db_path = Path(db_path)

        self.db = EpisodicMemory(db_path)
        self.procedural_db = ProceduralMemory(
            db_path.parent / "procedural_memory.db" if str(db_path) != ":memory:" else ":memory:"
        )
        self.value_db = ValueMemory(
            db_path.parent / "value_memory.db" if str(db_path) != ":memory:" else ":memory:"
        )

        # Load SemanticMemory + KnowledgeGraph from disk if previously persisted
        self._persistence_dir = db_path.parent
        _use_persistence = str(db_path) != ":memory:"
        semantic_dir = self._persistence_dir / "semantic"
        kg_path = self._persistence_dir / "knowledge_graph.json"

        self.semantic_db = (
            SemanticMemory.load_from_disk(semantic_dir)
            if (_use_persistence and semantic_dir.exists())
            else SemanticMemory()
        )
        self.knowledge_graph = (
            KnowledgeGraph.load_from_disk(kg_path)
            if (_use_persistence and kg_path.exists())
            else KnowledgeGraph()
        )
        self._knowledge_graphs: dict[str, KnowledgeGraph] = {"default": self.knowledge_graph}

        # Synaptic Priming Layer
        self.primer = WorkingMemoryPrimer()

        # PII Redactor (injected by brain factory)
        self._pii_redactor: Any | None = None

        # Track background tasks for graceful shutdown
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._improvement_tasks: set[asyncio.Task[Any]] = set()

        # Instantiate handlers for storage, recall, reflection, persistence, and subscription
        self.storage_handler = StorageHandler(self)
        self.recall_handler = RecallHandler(self)
        self.reflection_handler = ReflectionHandler(self)
        self.persistence_handler = PersistenceHandler(self)
        self.subscription_handler = SubscriptionHandler(self)

    async def on_start(self) -> None:
        """Initialize databases and register subscriptions on the bus."""
        logger.info("Starting MemoryNode with DB at %s", self.db.db_path)
        await self.db.init_db()
        await self.procedural_db.init_db()
        await self.value_db.init_db()
        await self.subscription_handler.register_subscriptions()

    async def on_stop(self) -> None:
        """Gracefully stop memory node and persist changes to disk."""
        await self.persistence_handler.shutdown()

    @staticmethod
    def _handle_background_task_result(task: asyncio.Task[Any]) -> None:
        """Callback for fire-and-forget tasks to log errors instead of swallowing them."""
        try:
            exc = task.exception()
            if exc is not None:
                logger.error("Background semantic store task failed: %s", exc)
        except asyncio.CancelledError:
            pass

    def _get_kg(self, tenant_id: str | None) -> KnowledgeGraph:
        """Get or create the KnowledgeGraph instance for a tenant."""
        tid = tenant_id or "default"
        if tid not in self._knowledge_graphs:
            db_path = self.db.db_path
            _use_persistence = str(db_path) != ":memory:" and str(self._persistence_dir) != "."
            kg_path = self._persistence_dir / f"knowledge_graph_{tid}.json"
            if _use_persistence and kg_path.exists():
                self._knowledge_graphs[tid] = KnowledgeGraph.load_from_disk(kg_path)
            else:
                self._knowledge_graphs[tid] = KnowledgeGraph()
        return self._knowledge_graphs[tid]

    # ── UnifiedMemoryInterface Implementation ──

    async def store(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        """Store information using the StorageHandler."""
        return await self.storage_handler.store(memory_type, data, **kwargs)

    async def retrieve(self, memory_type: MemoryType, query: Any, **kwargs: Any) -> list[Any]:
        """Retrieve information using the RecallHandler."""
        return await self.recall_handler.retrieve(memory_type, query, **kwargs)

    async def search(
        self, query: str, memory_types: list[MemoryType] | None = None, **kwargs: Any
    ) -> list[SearchResult]:
        """Search information using the RecallHandler."""
        return await self.recall_handler.search(query, memory_types, **kwargs)

    async def forget(self, memory_type: MemoryType, **kwargs: Any) -> int:
        """Selective forget interface."""
        return 0

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        """Return counts from all memory subsystems."""
        kg = self._get_kg(tenant_id)
        try:
            episodic_count = await self.db.get_turn_count(tenant_id)
        except Exception:
            episodic_count = 0
        return {
            "episodic": {"count": episodic_count},
            "semantic": {"count": self.semantic_db.count},
            "knowledge_graph": {"entities": kg.entity_count},
        }

    # ── General Message Handlers ──

    @require_tenant
    async def handle_stats(self, message: Message) -> Message | None:
        """Handles `memory.stats` — return counts from all memory subsystems."""
        try:
            tenant_id = message.payload.get("tenant_id") or message.tenant_id or "default"

            episodic_turns = await self.db.get_turn_count(tenant_id)
            episodic_sessions = await self.db.get_session_count(tenant_id)
            semantic_count = self.semantic_db.count
            procedural_count = 0
            value_count = 0

            try:
                async with self.procedural_db.pool.acquire() as conn:
                    async with conn.execute(
                        "SELECT COUNT(*) FROM skills WHERE tenant_id = ?", (tenant_id,)
                    ) as cursor:
                        row = await cursor.fetchone()
                        procedural_count = row[0] if row else 0
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
                pass

            try:
                value_count = await self.value_db.get_signal_count(tenant_id)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
                pass

            kg = self._get_kg(tenant_id)
            kg_entities = kg.entity_count
            kg_relations = kg.relation_count

            return message.create_response(
                {
                    "episodic": {
                        "turns": episodic_turns,
                        "sessions": episodic_sessions,
                    },
                    "semantic": {
                        "documents": semantic_count,
                    },
                    "procedural": {
                        "skills": procedural_count,
                    },
                    "value": {
                        "rewards": value_count,
                    },
                    "knowledge_graph": {
                        "entities": kg_entities,
                        "relations": kg_relations,
                    },
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory stats failed: %s", e)
            return message.create_error(str(e))

    @require_tenant
    async def handle_feedback(self, message: Message) -> Message | None:
        """Handles `memory.feedback` topics."""
        try:
            payload = message.payload
            note_id = payload.get("note_id")
            useful = payload.get("useful", True)

            if not note_id:
                return message.create_error("Missing 'note_id'")

            new_usefulness = await asyncio.to_thread(self.semantic_db.feedback, note_id, useful)

            if new_usefulness is None:
                return message.create_error(f"Memory with ID {note_id} not found", code="NOT_FOUND")

            return message.create_response(
                {"status": "updated", "note_id": note_id, "usefulness": new_usefulness}
            )
        except Exception as e:
            logger.error("Memory feedback failed: %s", e)
            return message.create_error(str(e))

    async def handle_message(self, message: Message) -> Message | None:
        """Generic handler fallback."""
        return None
