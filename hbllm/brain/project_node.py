"""
ProjectNode — bus-connected adapter for ProjectGraph.

Listens for interaction events and auto-detects/associates projects.
Publishes project context reactivation for downstream consumers.

Bus Topics:
    Subscribes:
        system.experience     → Auto-detect project from query context
        goal.created          → Auto-link new goals via has_goal
        goal.completed        → Update milestone status
        curiosity.goal        → Add as project question node

    Publishes:
        project.reactivated   → Context summary for active project
        project.milestone     → Milestone status change
        project.detected      → Auto-detection matched a project
"""

from __future__ import annotations

import logging

from hbllm.brain.project_graph import ProjectGraph
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ProjectNode(Node):
    """Bus-connected wrapper for ProjectGraph.

    On every interaction, attempts to match the query to an active project.
    If matched, publishes the project's reactivation context for LLM injection.
    """

    def __init__(
        self,
        node_id: str,
        project_graph: ProjectGraph | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["project_tracking", "context_reactivation"],
        )
        self.graph = project_graph or ProjectGraph(data_dir=data_dir)
        self._reactivations = 0

    async def on_start(self) -> None:
        logger.info("Starting ProjectNode")
        await self.bus.subscribe("system.experience", self._handle_experience)
        await self.bus.subscribe("goal.created", self._handle_goal_created)
        await self.bus.subscribe("goal.completed", self._handle_goal_completed)
        await self.bus.subscribe("curiosity.goal", self._handle_curiosity_goal)
        await self.bus.subscribe("project.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping ProjectNode — %d reactivations",
            self._reactivations,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _handle_experience(self, message: Message) -> None:
        """Auto-detect project from query context."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        query = payload.get("text", payload.get("query", ""))
        files = payload.get("files", [])

        if not query:
            return

        project = self.graph.auto_detect_project(query, files=files, tenant_id=tenant_id)
        if project:
            # Associate this conversation with the project
            topic = payload.get("topic", query[:50])
            self.graph.associate_conversation(project.entity_id, topic=topic, files=files)

            # Generate and publish reactivation context
            context = self.graph.reactivate(project.entity_id)
            if context:
                self._reactivations += 1
                await self.publish(
                    "project.reactivated",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        topic="project.reactivated",
                        tenant_id=tenant_id,
                        payload={
                            "project_id": project.entity_id,
                            "project_name": project.name,
                            "context": context,
                        },
                    ),
                )

    async def _handle_goal_created(self, message: Message) -> None:
        """Auto-link new goals to the most recently active project."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        goal_desc = payload.get("description", payload.get("name", ""))
        goal_id = payload.get("goal_id", "")

        if not goal_desc:
            return

        # Try to match goal to an active project
        project = self.graph.auto_detect_project(goal_desc, tenant_id=tenant_id)
        if project and goal_id:
            self.graph.add_relation(project.entity_id, goal_id, "has_goal")
            logger.info("Linked goal '%s' to project '%s'", goal_desc[:40], project.name)

    async def _handle_goal_completed(self, message: Message) -> None:
        """Update milestone/goal status when completed."""
        payload = message.payload
        entity_id = payload.get("goal_id", "")
        if entity_id:
            self.graph.update_status(entity_id, "completed")
            await self.publish(
                "project.milestone",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="project.milestone",
                    tenant_id=message.tenant_id,
                    payload={
                        "entity_id": entity_id,
                        "status": "completed",
                    },
                ),
            )

    async def _handle_curiosity_goal(self, message: Message) -> None:
        """Add curiosity goals as project questions."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        topic = payload.get("topic", payload.get("description", ""))
        if not topic:
            return

        project = self.graph.auto_detect_project(topic, tenant_id=tenant_id)
        if project:
            self.graph.add_entity(project.entity_id, "question", topic, tenant_id=tenant_id)

    async def _handle_query(self, message: Message) -> Message | None:
        """Return project graph stats."""
        return message.create_response(self.graph.stats())
