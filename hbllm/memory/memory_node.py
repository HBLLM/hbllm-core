"""
Shared Memory Node.

Acts as the single source of truth for conversation history across 
all domain modules. It listens to `memory.store` and `memory.retrieve_recent` 
messages on the bus, reading/writing to the local SQLite database.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from hbllm.memory.episodic import EpisodicMemory
from hbllm.memory.semantic import SemanticMemory
from hbllm.memory.procedural import ProceduralMemory
from hbllm.memory.value_memory import ValueMemory
from hbllm.memory.knowledge_graph import KnowledgeGraph
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class MemoryNode(Node):
    """
    Service node that persists and recalls conversation context.
    """

    def __init__(self, node_id: str, db_path: str | Path = "working_memory.db"):
        super().__init__(node_id=node_id, node_type=NodeType.MEMORY, capabilities=["episodic_storage", "semantic_retrieval", "procedural_skills", "value_tracking"])
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
        self.semantic_db = SemanticMemory.load_from_disk(semantic_dir) if (_use_persistence and semantic_dir.exists()) else SemanticMemory()
        self.knowledge_graph = KnowledgeGraph.load_from_disk(kg_path) if (_use_persistence and kg_path.exists()) else KnowledgeGraph()
        # Track background tasks for graceful shutdown
        self._pending_tasks: set[asyncio.Task] = set()

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

    async def on_stop(self) -> None:
        """Persist in-memory data to disk and clean up."""
        # Await any in-flight background storage tasks before persisting
        if self._pending_tasks:
            logger.info("Awaiting %d pending background tasks before shutdown", len(self._pending_tasks))
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        logger.info("Stopping MemoryNode — persisting semantic memory and knowledge graph")
        try:
            self.semantic_db.save_to_disk(self._persistence_dir / "semantic")
            self.knowledge_graph.save_to_disk(self._persistence_dir / "knowledge_graph.json")
        except Exception as e:
            logger.error("Failed to persist memory to disk: %s", e)

    @staticmethod
    def _handle_background_task_result(task: asyncio.Task) -> None:
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
        
        logger.info("[MemoryNode] Extracting patterns from reflection on domain '%s' (Node N)", domain)
        
        # Store a summary fact in Semantic Memory.
        pattern_content = f"Learned pattern in domain '{domain}': {reasoning}"
        
        await asyncio.to_thread(
            self.semantic_db.store, 
            pattern_content, 
            {"source": "reflection_engine", "domain": domain},
            is_priority=False  # Patterns grow general semantic memory
        )

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
            logger.info("[MemoryNode] Archiving high-salience experience to Priority Memory (Node K)")
            await asyncio.to_thread(
                self.semantic_db.store, 
                content, 
                {"source": "salience_detector", "message_id": payload.get("message_id")},
                is_priority=True
            )

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
            self.knowledge_graph.ingest_text(
                content, source=payload.get("category", "reflection")
            )

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
                    metadata={"rule_id": rule.get("rule_id", ""), "category": rule.get("category", "")},
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
        """
        payload = message.payload
        action = payload.get("action", "neighbors")

        if action == "neighbors":
            label = payload.get("entity", "")
            direction = payload.get("direction", "both")
            rel_type = payload.get("relation_type")
            results = self.knowledge_graph.neighbors(label, direction, rel_type)
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="knowledge.response",
                payload={"neighbors": results, "entity": label},
                correlation_id=message.id,
            )

        elif action == "path":
            from_label = payload.get("from", "")
            to_label = payload.get("to", "")
            max_depth = payload.get("max_depth", 5)
            path = self.knowledge_graph.shortest_path(from_label, to_label, max_depth)
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="knowledge.response",
                payload={"path": path, "from": from_label, "to": to_label},
                correlation_id=message.id,
            )

        elif action == "subgraph":
            label = payload.get("entity", "")
            depth = payload.get("depth", 2)
            sg = self.knowledge_graph.subgraph(label, depth)
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="knowledge.response",
                payload={"subgraph": sg, "entity": label},
                correlation_id=message.id,
            )

        elif action == "stats":
            return Message(
                type=MessageType.RESPONSE,
                source_node_id=self.node_id,
                topic="knowledge.response",
                payload={
                    "entity_count": self.knowledge_graph.entity_count,
                    "relation_count": self.knowledge_graph.relation_count,
                },
                correlation_id=message.id,
            )

        return None

    # Specific topic handlers below:

    async def handle_store(self, message: Message) -> Message | None:
        """
        Handles `memory.store` topics.
        Expected payload:
            session_id: str
            role: "user" | "assistant"
            content: str
            domain: Optional[str]
            metadata: Optional[dict]
        """
        try:
            payload = message.payload
            session_id = payload.get("session_id", "default_session")
            role = payload.get("role")
            content = payload.get("content")
            
            if not role or not content:
                return message.create_error("Missing 'role' or 'content' in store payload")

            turn_id = self.db.store_turn(
                session_id=session_id,
                role=role,
                content=content,
                domain=payload.get("domain"),
                metadata=payload.get("metadata"),
                tenant_id=payload.get("tenant_id", message.tenant_id or "default"),
            )
            
            # Offload semantic storage to a background thread with error handling
            task = asyncio.create_task(
                asyncio.to_thread(self.semantic_db.store, content, {"session_id": session_id, "role": role})
            )
            task.add_done_callback(self._handle_background_task_result)
            # Track for graceful shutdown
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            
            # Fire and forget mostly, but we reply with success
            return message.create_response({"status": "stored", "turn_id": turn_id})
            
        except Exception as e:
            logger.error("Memory store failed: %s", e)
            return message.create_error(str(e))

    async def handle_retrieve(self, message: Message) -> Message | None:
        """
        Handles `memory.retrieve_recent` topics.
        Expected payload:
            session_id: str
            limit: int (default = 10)
        """
        try:
            payload = message.payload
            session_id = payload.get("session_id", "default_session")
            limit = int(payload.get("limit", 10))
            
            turns = self.db.retrieve_recent(
                session_id, 
                limit=limit,
                tenant_id=payload.get("tenant_id", message.tenant_id or "default"),
            )
            
            return message.create_response({
                "session_id": session_id,
                "turns": turns,
            })
            
        except Exception as e:
            logger.error("Memory retrieval failed: %s", e)
            return message.create_error(str(e))

    async def handle_search(self, message: Message) -> Message | None:
        """
        Handles `memory.search` topics for long-term semantic RAG.
        Expected payload:
            query_text: str
            limit: int (default = 3)
        """
        try:
            payload = message.payload
            query = payload.get("query_text")
            limit = int(payload.get("limit", 3))
            
            if not query:
                return message.create_error("Missing 'query_text'")
                
            results = await asyncio.to_thread(self.semantic_db.search, query, limit)
            
            return message.create_response({
                "results": results,
            })
            
        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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
            )
            return message.create_response({"status": "recorded", "reward_id": reward_id})

        except Exception as e:
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
                preferences = self.value_db.get_preference(tenant_id, topic)
                return message.create_response({"topic": topic, "preferences": preferences})
            else:
                top_prefs = self.value_db.get_top_preferences(tenant_id, top_k)
                return message.create_response({"top_preferences": top_prefs})

        except Exception as e:
            logger.error("Reward query failed: %s", e)
            return message.create_error(str(e))

