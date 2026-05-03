"""
LearningLoop — unified learning and simulation node.

Consolidates: LearnerNode + WorldModelNode

LearnerNode handles online learning from feedback (DPO),
WorldModelNode simulates actions for learning. Together they
form the continuous learning pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class LearningLoop(Node):
    """
    Composite node for continuous learning: online feedback learning
    and environment simulation.
    """

    def __init__(
        self,
        node_id: str = "learning_loop",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.LEARNER,
            capabilities=[
                "online_learning",
                "dpo_training",
                "world_simulation",
                "action_simulation",
            ],
        )
        self.description = "Unified learning loop (feedback → simulate → learn)"

        # Sub-nodes
        self._learner: Any = None
        self._world_model: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.learner_node import LearnerNode
        from hbllm.brain.world_model_node import WorldModelNode

        self._learner = LearnerNode(node_id=f"{self.node_id}.learner")
        self._world_model = WorldModelNode(node_id=f"{self.node_id}.world_model")

        bus = self.bus
        for sub in [self._learner, self._world_model]:
            await sub.start(bus)

        logger.info("LearningLoop started with sub-nodes: learner, world_model")

    async def on_stop(self) -> None:
        for sub in [self._learner, self._world_model]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        subs = [self._learner, self._world_model]
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
    def learner(self):
        return self._learner

    @property
    def world_model(self):
        return self._world_model
