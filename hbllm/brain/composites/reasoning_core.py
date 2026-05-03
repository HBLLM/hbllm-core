"""
ReasoningCore — unified reasoning pipeline.

Consolidates: RouterNode + PlannerNode + CriticNode + DecisionNode
             + RevisionNode + ProcessRewardNode

These nodes always execute in sequence during every query, so combining
them eliminates 5 bus round-trips per request while preserving each
sub-node's bus subscriptions for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.policy_engine import PolicyEngine
    from hbllm.brain.provider_adapter import ProviderLLM
    from hbllm.modules.domain_registry import DomainRegistry
    from hbllm.network.bus import MessageBus
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class ReasoningCore(Node):
    """
    Composite node that unifies the core reasoning pipeline.

    Internally delegates to the original node implementations while
    presenting a single node on the bus with merged capabilities.
    """

    def __init__(
        self,
        node_id: str = "reasoning_core",
        *,
        llm: ProviderLLM | None = None,
        policy_engine: PolicyEngine | None = None,
        domain_registry: DomainRegistry | None = None,
        branch_factor: int = 3,
        max_depth: int = 2,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=[
                "routing",
                "intent_classification",
                "planning",
                "critic",
                "evaluation",
                "halting",
                "decision",
                "revision",
                "process_reward",
                "thought_evaluation",
            ],
        )
        self.description = "Unified reasoning pipeline (route → plan → critique → decide)"

        self._llm = llm
        self._policy_engine = policy_engine
        self._domain_registry = domain_registry
        self._branch_factor = branch_factor
        self._max_depth = max_depth
        self._data_dir = data_dir

        # Sub-nodes (lazily created in on_start)
        self._router: Any = None
        self._planner: Any = None
        self._critic: Any = None
        self._decision: Any = None
        self._revision: Any = None
        self._process_reward: Any = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Create and start all sub-nodes, registering their subscriptions."""
        from pathlib import Path

        from hbllm.brain.critic_node import CriticNode
        from hbllm.brain.decision_node import DecisionNode
        from hbllm.brain.planner_node import PlannerNode
        from hbllm.brain.revision_node import RevisionNode
        from hbllm.brain.router_node import RouterNode

        # Router
        self._router = RouterNode(
            node_id=f"{self.node_id}.router",
            llm=self._llm,
            domain_registry=self._domain_registry,
        )
        self._router._centroids_path = Path(self._data_dir) / "router_centroids.json"

        # Planner
        self._planner = PlannerNode(
            node_id=f"{self.node_id}.planner",
            branch_factor=self._branch_factor,
            max_depth=self._max_depth,
            policy_engine=self._policy_engine,
        )

        # Critic
        self._critic = CriticNode(
            node_id=f"{self.node_id}.critic",
            llm=self._llm,
        )

        # Decision
        self._decision = DecisionNode(
            node_id=f"{self.node_id}.decision",
            llm=self._llm,
            policy_engine=self._policy_engine,
        )

        # Revision (not a Node subclass — no bus start needed)
        self._revision = RevisionNode()

        # Process Reward (optional — may require ML model)
        try:
            from hbllm.brain.process_reward_node import ProcessRewardNode

            self._process_reward = ProcessRewardNode(
                node_id=f"{self.node_id}.prm",
            )
        except Exception:
            logger.debug("ProcessRewardNode not available, skipping")
            self._process_reward = None

        # Start all Node-subclass sub-nodes on the same bus
        bus = self.bus
        for sub in [self._router, self._planner, self._critic, self._decision]:
            await sub.start(bus)

        if self._process_reward is not None:
            await self._process_reward.start(bus)

        logger.info(
            "ReasoningCore started with sub-nodes: router, planner, critic, decision, "
            "revision%s",
            ", prm" if self._process_reward else "",
        )

    async def on_stop(self) -> None:
        """Stop all sub-nodes."""
        for sub in [
            self._router,
            self._planner,
            self._critic,
            self._decision,
            self._process_reward,
        ]:
            if sub is not None and hasattr(sub, "stop"):
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        """Route to the appropriate sub-node based on message topic."""
        # The sub-nodes register their own subscriptions in on_start,
        # so messages flow directly to them. This handler is a fallback.
        return None

    async def health_check(self):
        """Aggregate health from all sub-nodes."""
        from hbllm.network.node import HealthStatus, NodeHealth

        sub_healths = []
        for sub in [self._router, self._planner, self._critic, self._decision]:
            if sub is not None:
                h = await sub.health_check()
                sub_healths.append(h)

        # Aggregate: worst status wins
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

    # ── Direct access to sub-nodes ───────────────────────────────────

    @property
    def router(self):
        """Access the underlying RouterNode."""
        return self._router

    @property
    def planner(self):
        """Access the underlying PlannerNode."""
        return self._planner

    @property
    def critic(self):
        """Access the underlying CriticNode."""
        return self._critic

    @property
    def decision(self):
        """Access the underlying DecisionNode."""
        return self._decision

    @property
    def revision(self):
        """Access the underlying RevisionNode."""
        return self._revision

    @property
    def process_reward(self):
        """Access the underlying ProcessRewardNode (may be None)."""
        return self._process_reward
