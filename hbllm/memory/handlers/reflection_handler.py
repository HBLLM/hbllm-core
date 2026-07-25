"""Reflection Handler for MemoryNode.

Handles reflection events (ingesting entities/relations into the KnowledgeGraph),
improvement signals (extracting patterns), salience scoring (archiving to Priority Memory),
knowledge graph queries (neighbors, paths, subgraphs), and memory consolidation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.security.tenant_guard import require_tenant

logger = logging.getLogger(__name__)


class ReflectionHandler:
    """Handles reflection, pattern extraction, salience scoring, consolidation, and KG queries."""

    def __init__(self, node: Any) -> None:
        self.node = node

    @require_tenant
    async def handle_improvement(self, message: Message) -> None:
        """Listen for improvement/reflection signals and extract patterns into Semantic Memory."""
        if message.type != MessageType.SYSTEM_IMPROVE:
            return

        payload = message.payload
        domain = payload.get("domain")
        reasoning = payload.get("reasoning")

        logger.info(
            "[MemoryNode] Extracting patterns from reflection on domain '%s' (Node N)", domain
        )

        pattern_content = f"Learned pattern in domain '{domain}': {reasoning}"

        task = asyncio.create_task(
            asyncio.to_thread(
                self.node.semantic_db.store,
                pattern_content,
                metadata={
                    "source": "reflection_engine",
                    "domain": domain,
                    "tenant_id": message.tenant_id,
                },
                is_priority=False,  # Patterns grow general semantic memory
                tenant_id=message.tenant_id,
            )
        )
        self.node._improvement_tasks.add(task)
        task.add_done_callback(self.node._improvement_tasks.discard)
        await task

    @require_tenant
    async def handle_salience(self, message: Message) -> None:
        """Handle salience scores. High-salience experiences are stored in priority semantic memory."""
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
                self.node.semantic_db.store,
                content,
                {
                    "source": "salience_detector",
                    "message_id": payload.get("message_id"),
                    "tenant_id": message.tenant_id,
                },
                is_priority=True,
                tenant_id=message.tenant_id,
            )

    @require_tenant
    async def handle_reflection(self, message: Message) -> Message | None:
        """Handle deep reflection events — ingest entities and relations into the KnowledgeGraph."""
        payload = message.payload
        content = payload.get("content", "")
        entities = payload.get("entities", [])
        rules = payload.get("rules", [])

        # 1. Ingest entities from reflection
        kg = self.node._get_kg(message.tenant_id)
        for entity_info in entities:
            kg.add_entity(
                label=entity_info.get("label", ""),
                entity_type=entity_info.get("type", "concept"),
            )

        # 2. Extract and add relations from the content text
        if content:
            kg.ingest_text(content, source=payload.get("category", "reflection"))

        # 3. Add rule-derived relations (condition → action)
        for rule in rules:
            condition = rule.get("condition", "")
            action = rule.get("action", "")
            if condition and action:
                kg.add_relation(
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
            kg.entity_count,
            kg.relation_count,
        )
        return None

    @require_tenant
    async def handle_knowledge_query(self, message: Message) -> Message | None:
        """Handle knowledge graph queries (neighbors, path, subgraph, stats, etc.)."""
        payload = message.payload
        action = payload.get("action", "neighbors")
        tid = message.tenant_id or "default"
        kg = self.node._get_kg(tid)

        if action == "neighbors":
            label = payload.get("entity", "")
            direction = payload.get("direction", "both")
            rel_type = payload.get("relation_type")
            results = kg.neighbors(label, direction, rel_type)
            return message.create_response({"neighbors": results, "entity": label})

        elif action == "path":
            from_label = payload.get("from", "")
            to_label = payload.get("to", "")
            max_depth = payload.get("max_depth", 5)
            path = kg.shortest_path(from_label, to_label, max_depth)
            return message.create_response({"path": path, "from": from_label, "to": to_label})

        elif action == "subgraph":
            label = payload.get("entity", "")
            depth = payload.get("depth", 2)
            sg = kg.subgraph(label, depth)
            return message.create_response({"subgraph": sg, "entity": label})

        elif action == "stats":
            return message.create_response(
                {
                    "entity_count": kg.entity_count,
                    "relation_count": kg.relation_count,
                }
            )

        elif action == "all_entities":
            limit = payload.get("limit", 100)
            entities = [
                {"id": e.id, "label": e.label, "type": e.entity_type}
                for e in list(kg._entities.values())[-limit:]
            ]
            return message.create_response({"entities": entities})

        elif action == "add_community":
            community_label = payload.get("community_label")
            member_labels = payload.get("member_labels", [])
            summary = payload.get("summary", "")
            if community_label and member_labels:
                kg.add_community(community_label, member_labels, summary)
            return message.create_response({"status": "success"})

        elif action == "all_relations":
            limit = payload.get("limit", 100)
            relations = [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                    "created_at": r.created_at,
                    "metadata": r.metadata,
                }
                for r in list(kg._relations.values())[-limit:]
            ]
            return message.create_response({"relations": relations})

        elif action == "remove_relation":
            source_id = payload.get("source_id")
            target_id = payload.get("target_id")
            relation_type = payload.get("relation_type")
            if source_id and target_id and relation_type:
                kg.remove_relation(source_id, target_id, relation_type)
            return message.create_response({"status": "success"})

        return None

    @require_tenant
    async def handle_consolidate(self, message: Message) -> Message | None:
        """Handles `memory.consolidate` topics."""
        try:
            payload = message.payload
            threshold = payload.get("threshold")
            tenant_id = message.tenant_id or "default"

            res = await asyncio.to_thread(
                self.node.semantic_db.consolidate, tenant_id=tenant_id, threshold=threshold
            )

            return message.create_response(res)
        except Exception as e:
            logger.error("Memory consolidate failed: %s", e)
            return message.create_error(str(e))
