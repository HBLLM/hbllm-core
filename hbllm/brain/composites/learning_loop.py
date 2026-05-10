"""
LearningLoop — unified learning and simulation node.

Consolidates: LearnerNode + WorldModelNode + ProcessRewardNode

LearnerNode handles online learning from feedback (DPO),
WorldModelNode simulates actions for learning,
ProcessRewardNode scores reasoning steps for MCTS quality.

Together they form the continuous learning pipeline.
ProcessRewardNode was moved here from ReasoningCore because it is
a learning/evaluation concern, not inference-critical. It still
subscribes to `action.score_thought` on the bus so PlannerNode's
MCTS can reach it transparently.
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
    Composite node for continuous learning: online feedback learning,
    environment simulation, and process reward evaluation.
    """

    def __init__(
        self,
        node_id: str = "learning_loop",
        *,
        llm: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.LEARNER,
            capabilities=[
                "online_learning",
                "dpo_training",
                "world_simulation",
                "action_simulation",
                "process_reward",
                "thought_evaluation",
                "self_expansion",
            ],
        )
        self.description = "Unified learning loop (feedback → simulate → learn → reward → expand)"
        self._llm = llm

        # Sub-nodes
        self._learner: Any = None
        self._world_model: Any = None
        self._process_reward: Any = None
        self._spawner: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.learner_node import LearnerNode
        from hbllm.brain.world_model_node import WorldModelNode

        self._learner = LearnerNode(node_id=f"{self.node_id}.learner")
        self._world_model = WorldModelNode(node_id=f"{self.node_id}.world_model")

        bus = self.bus
        for sub in [self._learner, self._world_model]:
            await sub.start(bus)

        # Self-Expansion / Spawner Node
        if self._llm and hasattr(self._llm, "provider"):
            from hbllm.serving.provider import LocalProvider

            provider = self._llm.provider
            if isinstance(provider, LocalProvider):
                from hbllm.brain.spawner_node import SpawnerNode

                self._spawner = SpawnerNode(
                    node_id=f"{self.node_id}.spawner",
                    model=provider._model,
                    tokenizer=provider._tokenizer,
                )
                await self._spawner.start(bus)

        # Process Reward Model (optional — may require ML model / torch)
        try:
            from hbllm.brain.process_reward_node import ProcessRewardNode

            self._process_reward = ProcessRewardNode(
                node_id=f"{self.node_id}.prm",
            )
            await self._process_reward.start(bus)
        except Exception:
            logger.debug("ProcessRewardNode not available, skipping")
            self._process_reward = None

        logger.info(
            "LearningLoop started with sub-nodes: learner, world_model%s%s",
            ", spawner" if self._spawner else "",
            ", prm" if self._process_reward else "",
        )

    async def on_stop(self) -> None:
        for sub in [self._learner, self._world_model, self._process_reward, self._spawner]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        subs = [self._learner, self._world_model, self._process_reward, self._spawner]
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

    @property
    def process_reward(self):
        """Access the underlying ProcessRewardNode (may be None)."""
        return self._process_reward
