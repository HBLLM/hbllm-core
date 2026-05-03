"""
SocialLayer — unified multi-tenant coordination node.

Consolidates: CollectiveNode + IdentityNode

CollectiveNode handles multi-instance knowledge sharing and consensus,
IdentityNode manages per-tenant persona profiles. Together they form
the social/multi-tenant coordination surface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.skill_registry import SkillRegistry
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class SocialLayer(Node):
    """
    Composite node for multi-tenant coordination: identity management
    and collective intelligence.
    """

    def __init__(
        self,
        node_id: str = "social_layer",
        *,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=[
                "collective_intelligence",
                "knowledge_sharing",
                "consensus_voting",
                "task_delegation",
                "identity",
                "persona_management",
            ],
        )
        self.description = "Unified social layer (collective intelligence + identity)"
        self._skill_registry = skill_registry

        # Sub-nodes
        self._collective: Any = None
        self._identity: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.collective_node import CollectiveNode
        from hbllm.brain.identity_node import IdentityNode

        self._collective = CollectiveNode(
            node_id=f"{self.node_id}.collective",
            skill_registry=self._skill_registry,
        )

        self._identity = IdentityNode(
            node_id=f"{self.node_id}.identity",
        )

        bus = self.bus
        for sub in [self._collective, self._identity]:
            await sub.start(bus)

        logger.info("SocialLayer started with sub-nodes: collective, identity")

    async def on_stop(self) -> None:
        for sub in [self._collective, self._identity]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        subs = [self._collective, self._identity]
        sub_healths = []
        for sub in subs:
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

    @property
    def collective(self):
        return self._collective

    @property
    def identity(self):
        return self._identity
