"""
GovernanceGuard — unified governance and quality control node.

Consolidates: SentinelNode + PolicyEngine + ConfidenceEstimator

All three components serve the governance function: PolicyEngine evaluates
rules, SentinelNode monitors proactively, ConfidenceEstimator scores
quality. Combining them creates a single governance checkpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class GovernanceGuard(Node):
    """
    Composite node for governance: policy enforcement, proactive monitoring,
    and confidence scoring.
    """

    def __init__(
        self,
        node_id: str = "governance_guard",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=[
                "governance",
                "policy_enforcement",
                "sentinel_monitoring",
                "confidence_scoring",
            ],
        )
        self.description = "Unified governance (policy → sentinel → confidence)"

        # Components (some are Node subclasses, some are plain classes)
        self._sentinel: Any = None
        self._policy_engine: Any = None
        self._confidence_estimator: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.confidence_estimator import ConfidenceEstimator
        from hbllm.brain.policy_engine import PolicyEngine
        from hbllm.brain.sentinel_node import SentinelNode

        # PolicyEngine is a plain class (not a Node)
        self._policy_engine = PolicyEngine()

        # ConfidenceEstimator is a plain class (not a Node)
        self._confidence_estimator = ConfidenceEstimator()

        # SentinelNode IS a Node subclass — start it on the bus
        self._sentinel = SentinelNode(
            node_id=f"{self.node_id}.sentinel",
            policy_engine=self._policy_engine,
        )
        await self._sentinel.start(self.bus)

        logger.info(
            "GovernanceGuard started with: policy_engine, sentinel, confidence_estimator"
        )

    async def on_stop(self) -> None:
        if self._sentinel is not None:
            await self._sentinel.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        sentinel_health = None
        if self._sentinel is not None:
            sentinel_health = await self._sentinel.health_check()

        status = (
            sentinel_health.status if sentinel_health else HealthStatus.HEALTHY
        )

        return NodeHealth(
            node_id=self.node_id,
            status=status,
            uptime_seconds=self.uptime,
            capabilities_available=self.capabilities,
            message="Composite: sentinel + policy_engine + confidence_estimator",
        )

    # ── Direct access ────────────────────────────────────────────────

    @property
    def sentinel(self):
        return self._sentinel

    @property
    def policy_engine(self):
        return self._policy_engine

    @property
    def confidence_estimator(self):
        return self._confidence_estimator
