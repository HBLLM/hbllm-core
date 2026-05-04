"""
MetaCognition — unified self-monitoring and reflection node.

Consolidates: MetaReasoningNode + EvaluationNode + ReflectionNode + CuriosityNode

All four nodes monitor cognitive performance from different angles.
Combining them creates a single self-aware monitoring surface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.cognitive_metrics import CognitiveMetrics
    from hbllm.brain.goal_manager import GoalManager
    from hbllm.brain.self_model import SelfModel
    from hbllm.brain.skill_registry import SkillRegistry
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class MetaCognition(Node):
    """
    Composite node for self-monitoring, evaluation, reflection, and curiosity.
    """

    def __init__(
        self,
        node_id: str = "meta_cognition",
        *,
        cognitive_metrics: CognitiveMetrics | None = None,
        goal_manager: GoalManager | None = None,
        self_model: SelfModel | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=[
                "meta_reasoning",
                "evaluation",
                "reflection",
                "curiosity",
                "knowledge_gap_detection",
            ],
        )
        self.description = "Unified meta-cognition (evaluate → reflect → curiosity)"
        self._cognitive_metrics = cognitive_metrics
        self._goal_manager = goal_manager
        self._self_model = self_model
        self._skill_registry = skill_registry

        # Sub-nodes
        self._meta: Any = None
        self._evaluation: Any = None
        self._reflection: Any = None
        self._curiosity: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.curiosity_node import CuriosityNode
        from hbllm.brain.evaluation_node import EvaluationNode
        from hbllm.brain.meta_node import MetaReasoningNode
        from hbllm.brain.reflection_node import ReflectionNode

        self._meta = MetaReasoningNode(node_id=f"{self.node_id}.meta")

        self._evaluation = EvaluationNode(
            node_id=f"{self.node_id}.evaluation",
            cognitive_metrics=self._cognitive_metrics,
            goal_manager=self._goal_manager,
            self_model=self._self_model,
            skill_registry=self._skill_registry,
        )

        self._reflection = ReflectionNode(
            node_id=f"{self.node_id}.reflection",
            cognitive_metrics=self._cognitive_metrics,
            goal_manager=self._goal_manager,
            self_model=self._self_model,
            skill_registry=self._skill_registry,
        )

        self._curiosity = CuriosityNode(node_id=f"{self.node_id}.curiosity")

        bus = self.bus
        for sub in [self._meta, self._evaluation, self._reflection, self._curiosity]:
            await sub.start(bus)

        logger.info("MetaCognition started with sub-nodes: meta, evaluation, reflection, curiosity")

    async def on_stop(self) -> None:
        for sub in [self._meta, self._evaluation, self._reflection, self._curiosity]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        sub_healths = []
        for sub in [self._meta, self._evaluation, self._reflection, self._curiosity]:
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
    def meta(self):
        return self._meta

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def reflection(self):
        return self._reflection

    @property
    def curiosity(self):
        return self._curiosity
