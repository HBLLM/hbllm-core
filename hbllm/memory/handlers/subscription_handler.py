"""Subscription Handler for MemoryNode.

Subscribes the coordinator to all memory lifecycle and routing verbs on the message bus.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SubscriptionHandler:
    """Registers Node message bus subscriptions and maps topics to handlers."""

    def __init__(self, node: Any) -> None:
        self.node = node

    async def register_subscriptions(self) -> None:
        """Subscribe to memory and system topics on the node's message bus."""
        # Storage topics
        await self.node.bus.subscribe("memory.store", self.node.storage_handler.handle_store)
        await self.node.bus.subscribe("memory.browse", self.node.storage_handler.handle_browse)
        await self.node.bus.subscribe("memory.forget", self.node.storage_handler.handle_forget)
        await self.node.bus.subscribe(
            "memory.skill.store", self.node.storage_handler.handle_skill_store
        )
        await self.node.bus.subscribe(
            "memory.reward.record", self.node.storage_handler.handle_reward_record
        )

        # Recall topics
        await self.node.bus.subscribe(
            "memory.retrieve_recent", self.node.recall_handler.handle_retrieve
        )
        await self.node.bus.subscribe("memory.search", self.node.recall_handler.handle_search)
        await self.node.bus.subscribe(
            "memory.skill.find", self.node.recall_handler.handle_skill_find
        )
        await self.node.bus.subscribe(
            "memory.reward.query", self.node.recall_handler.handle_reward_query
        )

        # Reflection & Consolidation topics
        await self.node.bus.subscribe(
            "system.salience", self.node.reflection_handler.handle_salience
        )
        await self.node.bus.subscribe(
            "system.improve", self.node.reflection_handler.handle_improvement
        )
        await self.node.bus.subscribe(
            "system.reflection", self.node.reflection_handler.handle_reflection
        )
        await self.node.bus.subscribe(
            "knowledge.query", self.node.reflection_handler.handle_knowledge_query
        )
        await self.node.bus.subscribe(
            "memory.consolidate", self.node.reflection_handler.handle_consolidate
        )

        # Synaptic priming & workspace updates
        await self.node.bus.subscribe("router.query", self.node.recall_handler._on_router_query)
        await self.node.bus.subscribe(
            "workspace.thought", self.node.recall_handler._on_workspace_thought
        )

        # General / coordinator queries
        await self.node.bus.subscribe("memory.stats", self.node.handle_stats)
        await self.node.bus.subscribe("memory.feedback", self.node.handle_feedback)
