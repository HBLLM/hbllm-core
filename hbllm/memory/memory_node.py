"""
Shared Memory Node.

Acts as the single source of truth for conversation history across
all domain modules. It listens to `memory.store` and `memory.retrieve_recent`
messages on the bus, reading/writing to the local SQLite database.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

from hbllm.memory.episodic import EpisodicMemory
from hbllm.memory.interface import MemoryType, SearchResult, UnifiedMemoryInterface
from hbllm.memory.knowledge_graph import KnowledgeGraph
from hbllm.memory.procedural import ProceduralMemory
from hbllm.memory.semantic import SemanticMemory
from hbllm.memory.value_memory import ValueMemory
from hbllm.network.messages import (
    MemoryRetrievePayload,
    MemorySearchPayload,
    MemoryStorePayload,
    Message,
    MessageType,
)
from hbllm.network.node import Node, NodeType
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class MemoryNode(Node, UnifiedMemoryInterface):
    """
    Service node that persists and recalls conversation context.
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
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = EpisodicMemory(db_path)
        self.procedural_db = ProceduralMemory(db_path.parent / "procedural_memory.db")
        self.value_db = ValueMemory(db_path.parent / "value_memory.db")

        # Load SemanticMemory + KnowledgeGraph from disk if previously persisted
        # Skip for in-memory db paths (e.g. ":memory:" used in tests)
        self._persistence_dir = db_path.parent
        _use_persistence = str(db_path) != ":memory:" and str(self._persistence_dir) != "."
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
        # Track background tasks for graceful shutdown
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    async def on_start(self) -> None:
        """Subscribe to memory lifecycle verbs."""
        logger.info("Starting MemoryNode with DB at %s", self.db.db_path)
        await self.bus.subscribe("memory.store", self.handle_store)
        await self.bus.subscribe("memory.retrieve_recent", self.handle_retrieve)
        await self.bus.subscribe("memory.search", self.handle_search)
        await self.bus.subscribe("memory.skill.store", self.handle_skill_store)
        await self.bus.subscribe("memory.skill.find", self.handle_skill_find)
        await self.bus.subscribe("memory.reward.record", self.handle_reward_record)
        await self.bus.subscribe("memory.reward.query", self.handle_reward_query)
        await self.bus.subscribe("system.salience", self.handle_salience)
        await self.bus.subscribe("system.improve", self.handle_improvement)
        await self.bus.subscribe("system.reflection", self.handle_reflection)
        await self.bus.subscribe("knowledge.query", self.handle_knowledge_query)
        await self.bus.subscribe("memory.browse", self.handle_browse)
        await self.bus.subscribe("memory.forget", self.handle_forget)
        await self.bus.subscribe("memory.stats", self.handle_stats)
        # Tracking for handle_improvement which was previously untracked
        self._improvement_tasks: set[asyncio.Task[Any]] = set()

    async def on_stop(self) -> None:
        """Persist in-memory data to disk and clean up."""
        # Await any in-flight background storage tasks before persisting
        all_tasks = self._pending_tasks.union(self._improvement_tasks)
        if all_tasks:
            logger.info(
                "Awaiting %d pending background tasks before shutdown (timeout=5s)", len(all_tasks)
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for background tasks during MemoryNode stop")
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
                logger.error("Error during background task cleanup: %s", e)
            finally:
                self._pending_tasks.clear()
                self._improvement_tasks.clear()

        logger.info("Stopping MemoryNode — persisting semantic memory and knowledge graph")
        try:
            self.semantic_db.save_to_disk(self._persistence_dir / "semantic")
            self.knowledge_graph.save_to_disk(self._persistence_dir / "knowledge_graph.json")
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Failed to persist memory to disk: %s", e)

        # Close connections and release locks
        if hasattr(self.semantic_db, "close"):
            self.semantic_db.close()
        if hasattr(self.db, "close"):
            self.db.close()
        if hasattr(self.procedural_db, "close"):
            self.procedural_db.close()
        if hasattr(self.value_db, "close"):
            self.value_db.close()
        logger.info("MemoryNode stopped gracefully")

    @staticmethod
    def _handle_background_task_result(task: asyncio.Task[Any]) -> None:
        """Callback for fire-and-forget tasks to log errors instead of swallowing them."""
        try:
            exc = task.exception()
            if exc is not None:
                logger.error("Background semantic store task failed: %s", exc)
        except asyncio.CancelledError:
            pass

    async def handle_improvement(self, message: Message) -> None:
        """
        Listen for improvement/reflection signals (Node M) and extract patterns (Node N)
        into Semantic Memory (Node O).
        """
        if message.type != MessageType.SYSTEM_IMPROVE:
            return

        payload = message.payload
        domain = payload.get("domain")
        reasoning = payload.get("reasoning")

        logger.info(
            "[MemoryNode] Extracting patterns from reflection on domain '%s' (Node N)", domain
        )

        # Store a summary fact in Semantic Memory.
        pattern_content = f"Learned pattern in domain '{domain}': {reasoning}"

        # Store a summary fact in Semantic Memory.
        pattern_content = f"Learned pattern in domain '{domain}': {reasoning}"

        task = asyncio.create_task(
            asyncio.to_thread(
                self.semantic_db.store,
                pattern_content,
                {"source": "reflection_engine", "domain": domain, "tenant_id": message.tenant_id},
                is_priority=False,  # Patterns grow general semantic memory
                tenant_id=message.tenant_id,
            )
        )
        self._improvement_tasks.add(task)
        task.add_done_callback(self._improvement_tasks.discard)
        await task

    async def handle_salience(self, message: Message) -> None:
        """
        Handle salience scores. High-salience experiences are stored in
        priority semantic memory (Node K).
        """
        if message.type != MessageType.SALIENCE_SCORE:
            return

        payload = message.payload
        is_priority = payload.get("is_priority", False)
        content = payload.get("content", "")

        if is_priority and content:
            logger.info(
                "[MemoryNode] Archiving high-salience experience to Priority Memory (Node K)"
            )
            await asyncio.to_thread(
                self.semantic_db.store,
                content,
                {
                    "source": "salience_detector",
                    "message_id": payload.get("message_id"),
                    "tenant_id": message.tenant_id,
                },
                is_priority=True,
                tenant_id=message.tenant_id,
            )

    # ── UnifiedMemoryInterface Implementation ──

    async def store(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        if memory_type == MemoryType.EPISODIC:
            return str(
                self.db.store_turn(
                    session_id=kwargs.get("session_id", ""),
                    role=kwargs.get("role", "user"),
                    content=data,
                    tenant_id=kwargs.get("tenant_id", "default"),
                )
            )
        elif memory_type == MemoryType.SEMANTIC:
            await asyncio.to_thread(
                self.semantic_db.store,
                data,
                kwargs.get("metadata", {}),
                is_priority=kwargs.get("is_priority", False),
                tenant_id=kwargs.get("tenant_id", "default"),
            )
            return "stored"
        elif memory_type == MemoryType.PROCEDURAL:
            self.procedural_db.store_skill(
                tenant_id=kwargs.get("tenant_id", "default"),
                skill_name=kwargs.get("name", ""),
                trigger_pattern=data,
                steps=kwargs.get("steps", kwargs.get("code", [])),
            )
            return "stored"
        elif memory_type == MemoryType.VALUE:
            self.value_db.record_reward(
                tenant_id=kwargs.get("tenant_id", "default"),
                topic=kwargs.get("topic", "general"),
                action=kwargs.get("action_name", ""),
                reward=data,
                user_id=kwargs.get("session_id", ""),
            )
            return "stored"
        elif memory_type == MemoryType.KNOWLEDGE_GRAPH:
            return "stored"
        return ""

    async def retrieve(self, memory_type: MemoryType, query: Any, **kwargs: Any) -> list[Any]:
        if memory_type == MemoryType.EPISODIC:
            return self.db.retrieve_recent(
                session_id=kwargs.get("session_id", ""),
                limit=kwargs.get("limit", 10),
                tenant_id=kwargs.get("tenant_id", "default"),
            )
        return []

    async def search(
        self, query: str, memory_types: list[MemoryType] | None = None, **kwargs: Any
    ) -> list[SearchResult]:
        types_to_search = memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC]
        results = []

        tenant_id = kwargs.get("tenant_id", "default")
        limit = kwargs.get("limit", 5)

        if MemoryType.SEMANTIC in types_to_search:
            sem_res = await asyncio.to_thread(
                self.semantic_db.search, query, top_k=limit, tenant_id=tenant_id
            )
            for r in sem_res:
                results.append(
                    SearchResult(
                        memory_type=MemoryType.SEMANTIC,
                        id=r.get("id", ""),
                        content=r.get("content"),
                        score=r.get("score", 1.0),
                        metadata=r,
                    )
                )

        # Episodic search (mocked as retrieve for unified demo)
        if MemoryType.EPISODIC in types_to_search:
            ep_res = self.db.retrieve_recent(
                session_id=kwargs.get("session_id", ""), limit=limit, tenant_id=tenant_id
            )
            for i, r in enumerate(ep_res):
                results.append(
                    SearchResult(
                        memory_type=MemoryType.EPISODIC,
                        id=str(i),
                        content=r.get("content"),
                        score=1.0,
                        metadata=r,
                    )
                )

        return sorted(results, key=lambda x: x.score, reverse=True)[:limit]

    async def forget(self, memory_type: MemoryType, **kwargs: Any) -> int:
        return 0

    async def stats(self, tenant_id: str) -> dict[str, Any]:
        return {
            "episodic": {"count": 0},
            "semantic": {"count": len(self.semantic_db._data)},
            "knowledge_graph": {"entities": self.knowledge_graph.entity_count},
        }

    async def handle_message(self, message: Message) -> Message | None:
        """
        Generic handler fallback, but we register explicit topics in on_start.
        """
        return None

    async def handle_reflection(self, message: Message) -> Message | None:
        """
        Handle deep reflection events — ingest entities and relations
        into the KnowledgeGraph.
        """
        payload = message.payload
        content = payload.get("content", "")
        entities = payload.get("entities", [])
        rules = payload.get("rules", [])

        # 1. Ingest entities from reflection
        for entity_info in entities:
            self.knowledge_graph.add_entity(
                label=entity_info.get("label", ""),
                entity_type=entity_info.get("type", "concept"),
            )

        # 2. Extract and add relations from the content text
        if content:
            self.knowledge_graph.ingest_text(content, source=payload.get("category", "reflection"))

        # 3. Add rule-derived relations (condition → action)
        for rule in rules:
            condition = rule.get("condition", "")
            action = rule.get("action", "")
            if condition and action:
                self.knowledge_graph.add_relation(
                    source_label=condition,
                    target_label=action,
                    relation_type="leads_to",
                    weight=rule.get("confidence", 0.5),
                    metadata={
                        "rule_id": rule.get("rule_id", ""),
                        "category": rule.get("category", ""),
                    },
                )

        logger.info(
            "[MemoryNode] KnowledgeGraph updated — %d entities, %d relations",
            self.knowledge_graph.entity_count,
            self.knowledge_graph.relation_count,
        )
        return None

    async def handle_knowledge_query(self, message: Message) -> Message | None:
        """
        Handle knowledge graph queries.

        Supported actions:
          - neighbors: Get neighbors of an entity
          - path: Find shortest path between two entities
          - subgraph: Extract subgraph around an entity
          - stats: Get graph statistics
          - all_entities: Get list of active entities for GraphRAG
          - add_community: Inject a hierarchical community node
        """
        payload = message.payload
        action = payload.get("action", "neighbors")

        if action == "neighbors":
            label = payload.get("entity", "")
            direction = payload.get("direction", "both")
            rel_type = payload.get("relation_type")
            results = self.knowledge_graph.neighbors(label, direction, rel_type)
            return message.create_response({"neighbors": results, "entity": label})

        elif action == "path":
            from_label = payload.get("from", "")
            to_label = payload.get("to", "")
            max_depth = payload.get("max_depth", 5)
            path = self.knowledge_graph.shortest_path(from_label, to_label, max_depth)
            return message.create_response({"path": path, "from": from_label, "to": to_label})

        elif action == "subgraph":
            label = payload.get("entity", "")
            depth = payload.get("depth", 2)
            sg = self.knowledge_graph.subgraph(label, depth)
            return message.create_response({"subgraph": sg, "entity": label})

        elif action == "stats":
            return message.create_response(
                {
                    "entity_count": self.knowledge_graph.entity_count,
                    "relation_count": self.knowledge_graph.relation_count,
                }
            )

        elif action == "all_entities":
            limit = payload.get("limit", 100)
            entities = [
                {"id": e.id, "label": e.label, "type": e.entity_type}
                for e in list(self.knowledge_graph._entities.values())[-limit:]
            ]
            return message.create_response({"entities": entities})

        elif action == "add_community":
            community_label = payload.get("community_label")
            member_labels = payload.get("member_labels", [])
            summary = payload.get("summary", "")
            if community_label and member_labels:
                self.knowledge_graph.add_community(community_label, member_labels, summary)
            return message.create_response({"status": "success"})

        return None

    # Specific topic handlers below:

    from hbllm.security.tenant_guard import require_tenant

    @require_tenant
    async def handle_store(self, message: Message) -> Message | None:
        """
        Handles `memory.store` topics.
        """
        # CapBAC: Check permissions
        if self.registry:
            payload_data = message.payload
            scope = payload_data.get("scope", "episodic")
            if not await self.registry.has_permission(message.source_node_id, scope):
                logger.warning(
                    "[MemoryNode] Permission denied: Node '%s' cannot store in scope '%s'",
                    message.source_node_id,
                    scope,
                )
                return message.create_error(f"Access denied to scope '{scope}'", code="FORBIDDEN")

        try:
            payload = MemoryStorePayload(**message.payload)
            session_id = payload.session_id
            role = payload.role
            content = payload.content

            turn_id = self.db.store_turn(
                session_id=session_id,
                role=role,
                content=content,
                domain=payload.domain,
                metadata=payload.metadata,
                tenant_id=payload.tenant_id or (message.tenant_id or "default"),
                user_id=payload.user_id or message.user_id,
                device_id=payload.device_id or message.device_id,
                scope=payload.scope,
                vector_clock=message.vector_clock,
                authority_score=message.payload.get("authority_score", 50),
                parent_memory_id=payload.parent_memory_id,
            )

            # Sensitive Memory Guard: Skip semantic indexing for sensitive/working data
            if payload.scope not in ["sensitive", "working"]:
                # Offload semantic storage to a background thread with error handling
                task = asyncio.create_task(
                    asyncio.to_thread(
                        self.semantic_db.store,
                        content,
                        {
                            "session_id": session_id,
                            "role": role,
                            "tenant_id": payload.tenant_id or message.tenant_id,
                            "user_id": payload.user_id or message.user_id,
                            "device_id": payload.device_id or message.device_id,
                            "scope": payload.scope,
                        },
                        is_priority=False,
                        tenant_id=payload.tenant_id or message.tenant_id,
                        user_id=payload.user_id or message.user_id,
                        device_id=payload.device_id or message.device_id,
                        vector_clock=message.vector_clock,
                        authority_score=message.payload.get("authority_score", 50),
                    )
                )
                task.add_done_callback(self._handle_background_task_result)
                # Track for graceful shutdown
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            else:
                logger.debug(
                    "[MemoryNode] Skipping semantic storage for %s scoped entry", payload.scope
                )

            # Fire and forget mostly, but we reply with success
            return message.create_response({"status": "stored", "turn_id": turn_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory store failed: %s", e)
            return message.create_error(str(e))

    async def handle_retrieve(self, message: Message) -> Message | None:
        """
        Handles `memory.retrieve_recent` topics.
        """
        # CapBAC: Check permissions (retrieve defaults to episodic if not specified)
        if self.registry:
            scope = message.payload.get("scope", "episodic")
            if not await self.registry.has_permission(message.source_node_id, scope):
                return message.create_error(f"Access denied to scope '{scope}'", code="FORBIDDEN")

        try:
            payload = MemoryRetrievePayload(**message.payload)
            session_id = payload.session_id
            limit = payload.limit

            turns = self.db.retrieve_recent(
                session_id,
                limit=limit,
                tenant_id=payload.tenant_id or (message.tenant_id or "default"),
            )

            return message.create_response(
                {
                    "session_id": session_id,
                    "turns": turns,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory retrieval failed: %s", e)
            return message.create_error(str(e))

    from hbllm.security.tenant_guard import require_tenant

    @require_tenant
    async def handle_search(self, message: Message) -> Message | None:
        """
        Handles `memory.search` topics for long-term semantic RAG.
        """
        try:
            payload = MemorySearchPayload(**message.payload)
            query = payload.query_text
            limit = payload.top_k

            if not query:
                return message.create_error("Missing 'query_text'")

            results = await asyncio.to_thread(
                self.semantic_db.search,
                query,
                limit,
                tenant_id=message.tenant_id,
                user_id=message.user_id,
                device_id=message.device_id,
            )

            return message.create_response(
                {
                    "results": results,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Semantic search failed: %s", e)
            return message.create_error(str(e))

    async def handle_skill_store(self, message: Message) -> Message | None:
        """
        Handles `memory.skill.store` topics.
        Expected payload:
            skill_name: str
            trigger_pattern: str
            steps: list[dict]
        """
        try:
            payload = message.payload
            skill_name = payload.get("skill_name")
            trigger_pattern = payload.get("trigger_pattern", "")
            steps = payload.get("steps", [])
            tenant_id = message.tenant_id or "default"

            if not skill_name or not steps:
                return message.create_error("Missing 'skill_name' or 'steps'")

            skill_id = self.procedural_db.store_skill(
                tenant_id=tenant_id,
                skill_name=skill_name,
                trigger_pattern=trigger_pattern,
                steps=steps,
                source_node=message.source_node_id,
            )
            return message.create_response({"status": "stored", "skill_id": skill_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Skill store failed: %s", e)
            return message.create_error(str(e))

    async def handle_skill_find(self, message: Message) -> Message | None:
        """
        Handles `memory.skill.find` topics.
        Expected payload:
            query: str
            top_k: int (default = 3)
        """
        try:
            payload = message.payload
            query = payload.get("query", "")
            top_k = int(payload.get("top_k", 3))
            tenant_id = message.tenant_id or "default"

            if not query:
                return message.create_error("Missing 'query'")

            skills = self.procedural_db.find_skill(tenant_id, query, top_k)
            return message.create_response({"skills": skills})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Skill find failed: %s", e)
            return message.create_error(str(e))

    async def handle_reward_record(self, message: Message) -> Message | None:
        """
        Handles `memory.reward.record` topics.
        Expected payload:
            topic: str — category of the preference
            action: str — specific action being rated
            reward: float — signal (-1.0 to 1.0)
            context: Optional[dict]
        """
        try:
            payload = message.payload
            topic = payload.get("topic", "general")
            action = payload.get("action", "")
            reward = float(payload.get("reward", 0.0))
            tenant_id = message.tenant_id or "default"

            if not action:
                return message.create_error("Missing 'action'")

            reward_id = self.value_db.record_reward(
                tenant_id=tenant_id,
                topic=topic,
                action=action,
                reward=reward,
                context=payload.get("context"),
                user_id=message.user_id,
                device_id=message.device_id,
            )
            return message.create_response({"status": "recorded", "reward_id": reward_id})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Reward record failed: %s", e)
            return message.create_error(str(e))

    async def handle_reward_query(self, message: Message) -> Message | None:
        """
        Handles `memory.reward.query` topics.
        Expected payload:
            topic: Optional[str] — if set, get preferences for this topic
            top_k: int (default = 5)
        """
        try:
            payload = message.payload
            topic = payload.get("topic")
            top_k = int(payload.get("top_k", 5))
            tenant_id = message.tenant_id or "default"

            if topic:
                preferences = self.value_db.get_preference(
                    tenant_id, topic, message.user_id, message.device_id
                )
                return message.create_response({"topic": topic, "preferences": preferences})
            else:
                top_prefs = self.value_db.get_top_preferences(
                    tenant_id, top_k, message.user_id, message.device_id
                )
                return message.create_response({"top_preferences": top_prefs})

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Reward query failed: %s", e)
            return message.create_error(str(e))

    async def handle_browse(self, message: Message) -> Message | None:
        """
        Handles `memory.browse` — paginated episodic memory retrieval.

        Expected payload:
            offset: int (default 0)
            limit: int (default 20)
            session_id: Optional[str]
            tenant_id: Optional[str]
        """
        try:
            payload = message.payload
            offset = int(payload.get("offset", 0))
            limit = min(int(payload.get("limit", 20)), 100)
            session_id = payload.get("session_id")
            tenant_id = payload.get("tenant_id") or message.tenant_id or "default"

            conn = self.db._get_conn()
            conn.row_factory = sqlite3.Row

            if session_id:
                rows = conn.execute(
                    "SELECT * FROM turns WHERE tenant_id = ? AND session_id = ? "
                    "ORDER BY timestamp_iso DESC LIMIT ? OFFSET ?",
                    (tenant_id, session_id, limit, offset),
                ).fetchall()
                total_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM turns WHERE tenant_id = ? AND session_id = ?",
                    (tenant_id, session_id),
                ).fetchone()
            else:
                rows = conn.execute(
                    "SELECT * FROM turns WHERE tenant_id = ? "
                    "ORDER BY timestamp_iso DESC LIMIT ? OFFSET ?",
                    (tenant_id, limit, offset),
                ).fetchall()
                total_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM turns WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()

            conn.row_factory = None
            import json as _json

            entries = []
            for row in rows:
                entries.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "role": row["role"],
                        "content": row["content"],
                        "domain": row["domain"],
                        "timestamp": row["timestamp_iso"],
                        "metadata": _json.loads(row["metadata"]) if row["metadata"] else {},
                    }
                )

            total = total_row["cnt"] if total_row else 0
            return message.create_response(
                {
                    "entries": entries,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory browse failed: %s", e)
            return message.create_error(str(e))

    async def handle_forget(self, message: Message) -> Message | None:
        """
        Handles `memory.forget` — selective amnesia.

        Expected payload:
            query: Optional[str] — content substring to match and delete
            session_id: Optional[str] — delete all turns in this session
            before: Optional[str] — ISO timestamp, delete turns older than this
            after: Optional[str] — ISO timestamp, delete turns newer than this
            entry_ids: Optional[list[str]] — specific turn IDs to delete
            forget_semantic: bool — also purge matching semantic docs (default True)
        """
        try:
            payload = message.payload
            tenant_id = payload.get("tenant_id") or message.tenant_id or "default"
            query = payload.get("query")
            session_id = payload.get("session_id")
            before = payload.get("before")
            after = payload.get("after")
            entry_ids = payload.get("entry_ids", [])
            forget_semantic = payload.get("forget_semantic", True)

            conn = self.db._get_conn()
            deleted_episodic = 0
            deleted_semantic = 0

            # Delete specific entries by ID
            if entry_ids:
                placeholders = ",".join("?" * len(entry_ids))
                cursor = conn.execute(
                    f"DELETE FROM turns WHERE tenant_id = ? AND id IN ({placeholders})",
                    [tenant_id] + list(entry_ids),
                )
                deleted_episodic += cursor.rowcount
                conn.commit()

            # Delete by session
            if session_id:
                deleted_episodic += self.db.clear_session(session_id, tenant_id)

            # Delete by content query
            if query:
                # First find matching IDs for reporting
                conn.row_factory = sqlite3.Row
                matches = conn.execute(
                    "SELECT id, content FROM turns WHERE tenant_id = ? AND content LIKE ?",
                    (tenant_id, f"%{query}%"),
                ).fetchall()
                conn.row_factory = None

                if matches:
                    match_ids = [m["id"] for m in matches]
                    placeholders = ",".join("?" * len(match_ids))
                    cursor = conn.execute(
                        f"DELETE FROM turns WHERE id IN ({placeholders})",
                        match_ids,
                    )
                    deleted_episodic += cursor.rowcount
                    conn.commit()

                # Also delete from semantic memory
                if forget_semantic:
                    results = await asyncio.to_thread(
                        self.semantic_db.search, query, top_k=50, tenant_id=tenant_id
                    )
                    for r in results:
                        doc_id = r.get("id")
                        if doc_id and self.semantic_db.delete(doc_id):
                            deleted_semantic += 1

            # Delete by time range
            if before or after:
                conditions = ["tenant_id = ?"]
                params: list[Any] = [tenant_id]
                if before:
                    conditions.append("timestamp_iso < ?")
                    params.append(before)
                if after:
                    conditions.append("timestamp_iso > ?")
                    params.append(after)

                where = " AND ".join(conditions)
                cursor = conn.execute(f"DELETE FROM turns WHERE {where}", params)
                deleted_episodic += cursor.rowcount
                conn.commit()

            logger.info(
                "[MemoryNode] Forget completed: %d episodic, %d semantic entries removed",
                deleted_episodic,
                deleted_semantic,
            )
            return message.create_response(
                {
                    "status": "forgotten",
                    "deleted_episodic": deleted_episodic,
                    "deleted_semantic": deleted_semantic,
                }
            )

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Memory forget failed: %s", e)
            return message.create_error(str(e))

    async def handle_stats(self, message: Message) -> Message | None:
        """
        Handles `memory.stats` — return counts from all memory subsystems.
        """
        try:
            tenant_id = message.payload.get("tenant_id") or message.tenant_id or "default"

            episodic_turns = self.db.get_turn_count(tenant_id)
            episodic_sessions = self.db.get_session_count(tenant_id)
            semantic_count = self.semantic_db.count
            procedural_count = 0
            value_count = 0

            try:
                proc_conn = self.procedural_db._get_conn()
                row = proc_conn.execute(
                    "SELECT COUNT(*) FROM skills WHERE tenant_id = ?", (tenant_id,)
                ).fetchone()
                procedural_count = row[0] if row else 0
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
                pass

            try:
                val_conn = self.value_db._get_conn()
                row = val_conn.execute(
                    "SELECT COUNT(*) FROM rewards WHERE tenant_id = ?", (tenant_id,)
                ).fetchone()
                value_count = row[0] if row else 0
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError):
                pass

            kg_entities = self.knowledge_graph.entity_count
            kg_relations = self.knowledge_graph.relation_count

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
