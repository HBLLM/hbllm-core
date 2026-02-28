"""
Shared Memory Node.

Acts as the single source of truth for conversation history across 
all domain modules. It listens to `memory.store` and `memory.retrieve_recent` 
messages on the bus, reading/writing to the local SQLite database.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hbllm.memory.episodic import EpisodicMemory
from hbllm.memory.semantic import SemanticMemory
from hbllm.memory.procedural import ProceduralMemory
from hbllm.memory.value_memory import ValueMemory
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class MemoryNode(Node):
    """
    Service node that persists and recalls conversation context.
    """

    def __init__(self, node_id: str, db_path: str | Path = "working_memory.db"):
        super().__init__(node_id=node_id, node_type=NodeType.MEMORY, capabilities=["episodic_storage", "semantic_retrieval", "procedural_skills", "value_tracking"])
        self.db = EpisodicMemory(db_path)
        self.semantic_db = SemanticMemory()
        self.procedural_db = ProceduralMemory(Path(db_path).parent / "procedural_memory.db")
        self.value_db = ValueMemory(Path(db_path).parent / "value_memory.db")

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

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping MemoryNode")

    async def handle_message(self, message: Message) -> Message | None:
        """
        Generic handler fallback, but we register explicit topics in on_start.
        """
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
            
            # Submitting to Semantic DB could be offloaded to a thread to unblock the loop
            import asyncio
            asyncio.create_task(
                asyncio.to_thread(self.semantic_db.store, content, {"session_id": session_id, "role": role})
            )
            
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
                
            import asyncio
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

