"""
ResourceManager — unified resource control and scheduling node.

Consolidates: WorkspaceNode + AttentionManager + LoadManager + SchedulerNode

All four manage shared resources: WorkspaceNode is the cognitive blackboard,
AttentionManager handles memory budgets, LoadManager monitors system
resources, and SchedulerNode handles proactive task scheduling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class ResourceManager(Node):
    """
    Composite node for resource management: workspace, attention,
    load monitoring, and scheduling.
    """

    def __init__(
        self,
        node_id: str = "resource_manager",
        *,
        data_dir: str = "data",
        monitor_interval: float = 60.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=[
                "workspace",
                "blackboard",
                "attention",
                "memory_budgets",
                "load_monitoring",
                "resource_degradation",
                "scheduling",
                "proactive_execution",
            ],
        )
        self.description = "Unified resource management (workspace + attention + load + scheduler)"
        self._data_dir = data_dir
        self._monitor_interval = monitor_interval

        # Sub-nodes
        self._workspace: Any = None
        self._attention: Any = None
        self._load_manager: Any = None
        self._scheduler: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.attention_manager import AttentionManager
        from hbllm.brain.load_manager import LoadManager
        from hbllm.brain.scheduler_node import SchedulerNode
        from hbllm.brain.workspace_node import WorkspaceNode

        self._workspace = WorkspaceNode(node_id=f"{self.node_id}.workspace")

        self._attention = AttentionManager(node_id=f"{self.node_id}.attention")

        self._load_manager = LoadManager(
            node_id=f"{self.node_id}.load",
            monitor_interval=self._monitor_interval,
        )

        self._scheduler = SchedulerNode(
            node_id=f"{self.node_id}.scheduler",
            data_dir=self._data_dir,
        )

        bus = self.bus
        for sub in [self._workspace, self._attention, self._load_manager, self._scheduler]:
            await sub.start(bus)

        logger.info(
            "ResourceManager started with sub-nodes: workspace, attention, load, scheduler"
        )

    async def on_stop(self) -> None:
        for sub in [self._workspace, self._attention, self._load_manager, self._scheduler]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        subs = [self._workspace, self._attention, self._load_manager, self._scheduler]
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
    def workspace(self):
        return self._workspace

    @property
    def attention(self):
        return self._attention

    @property
    def load_manager(self):
        return self._load_manager

    @property
    def scheduler(self):
        return self._scheduler
