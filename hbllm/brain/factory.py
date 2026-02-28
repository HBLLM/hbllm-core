"""
Brain Factory — one-line setup for the full cognitive pipeline.

Creates all brain nodes (Router, Planner, Critic, Decision, Workspace,
Memory) with an LLM provider injected, wires them to a message bus,
and returns a running Brain instance.

Usage::

    from hbllm.brain.factory import BrainFactory

    brain = await BrainFactory.create("openai/gpt-4o-mini")
    result = await brain.process("What is quantum computing?")
    print(result.text)
    await brain.shutdown()

    # Or with Anthropic:
    brain = await BrainFactory.create("anthropic/claude-sonnet-4-20250514")

    # Or with custom provider kwargs:
    brain = await BrainFactory.create("openai", model="gpt-4o", api_key="sk-...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult
from hbllm.serving.provider import LLMProvider, get_provider

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for Brain creation."""
    inject_memory: bool = True
    inject_identity: bool = True
    inject_curiosity: bool = True
    total_timeout: float = 60.0
    planner_branch_factor: int = 3
    planner_max_depth: int = 2
    system_prompt: str = "You are a helpful AI assistant."


class Brain:
    """
    A fully wired, running HBLLM cognitive brain.
    
    Holds references to all nodes, the bus, registry, and pipeline.
    Use ``process()`` to send queries through the full cognitive loop.
    """

    def __init__(
        self,
        bus: MessageBus,
        registry: ServiceRegistry,
        pipeline: CognitivePipeline,
        llm: ProviderLLM,
        nodes: list[Any],
        provider: LLMProvider,
    ):
        self.bus = bus
        self.registry = registry
        self.pipeline = pipeline
        self.llm = llm
        self.nodes = nodes
        self.provider = provider

    async def process(
        self,
        text: str,
        tenant_id: str = "default",
        session_id: str = "default",
    ) -> PipelineResult:
        """Send a query through the full cognitive pipeline."""
        return await self.pipeline.process(
            text=text,
            tenant_id=tenant_id,
            session_id=session_id,
        )

    async def shutdown(self) -> None:
        """Stop all nodes, pipeline, and bus."""
        await self.pipeline.stop()
        for node in reversed(self.nodes):
            try:
                await node.stop()
            except Exception:
                pass
        await self.registry.stop()
        await self.bus.stop()
        logger.info("Brain shutdown complete")

    @property
    def usage(self) -> dict[str, int]:
        """Accumulated LLM usage statistics."""
        return self.llm.usage


class BrainFactory:
    """
    Factory for creating a fully wired Brain with one line.
    
    Handles provider creation, node instantiation, bus wiring, and startup.
    """

    @staticmethod
    async def create(
        provider: str | LLMProvider = "openai/gpt-4o-mini",
        config: BrainConfig | None = None,
        bus: MessageBus | None = None,
        **provider_kwargs: Any,
    ) -> Brain:
        """
        Create and start a fully wired Brain.
        
        Args:
            provider: Provider name (e.g., "openai/gpt-4o-mini", "anthropic") 
                      or an LLMProvider instance.
            config: Brain configuration. Defaults to BrainConfig().
            bus: Custom message bus. Defaults to InProcessBus.
            **provider_kwargs: Extra args passed to get_provider().
            
        Returns:
            A running Brain instance ready for queries.
        """
        cfg = config or BrainConfig()

        # 1. Create provider
        if isinstance(provider, str):
            llm_provider = get_provider(provider, **provider_kwargs)
        else:
            llm_provider = provider

        # 2. Create adapter
        llm = ProviderLLM(llm_provider, system_prompt=cfg.system_prompt)

        # 3. Create bus and registry
        message_bus = bus or InProcessBus()
        await message_bus.start()

        registry = ServiceRegistry()
        await registry.start()

        # 4. Create cognitive nodes with LLM injected
        from hbllm.brain.router_node import RouterNode
        from hbllm.brain.planner_node import PlannerNode
        from hbllm.brain.critic_node import CriticNode
        from hbllm.brain.decision_node import DecisionNode
        from hbllm.brain.workspace_node import WorkspaceNode

        nodes = [
            RouterNode(node_id="router", llm=llm),
            PlannerNode(
                node_id="planner",
                branch_factor=cfg.planner_branch_factor,
                max_depth=cfg.planner_max_depth,
            ),
            CriticNode(node_id="critic", llm=llm),
            DecisionNode(node_id="decision", llm=llm),
            WorkspaceNode(node_id="workspace"),
        ]

        # Inject LLM into planner
        nodes[1].llm = llm

        # 5. Start all nodes on the bus
        for node in nodes:
            await node.start(message_bus)
            # Register with service registry
            from hbllm.network.node import NodeInfo, NodeHealth, HealthStatus
            await registry.register(NodeInfo(
                node_id=node.node_id,
                node_type=node.node_type,
                capabilities=node.capabilities,
            ))
            await registry.update_health(NodeHealth(
                node_id=node.node_id,
                status=HealthStatus.HEALTHY,
            ))

        # 6. Create and start pipeline
        pipeline_config = PipelineConfig(
            total_timeout=cfg.total_timeout,
            inject_memory=cfg.inject_memory,
            inject_identity=cfg.inject_identity,
            inject_curiosity=cfg.inject_curiosity,
        )
        pipeline = CognitivePipeline(
            bus=message_bus,
            registry=registry,
            config=pipeline_config,
        )
        await pipeline.start()

        logger.info(
            "Brain created with %s provider, %d nodes, pipeline ready",
            llm_provider.name,
            len(nodes),
        )

        return Brain(
            bus=message_bus,
            registry=registry,
            pipeline=pipeline,
            llm=llm,
            nodes=nodes,
            provider=llm_provider,
        )
