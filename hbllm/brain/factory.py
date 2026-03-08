"""
Brain Factory — one-line setup for the full cognitive pipeline.

Creates all brain nodes (Router, Planner, Critic, Decision, Workspace,
Memory) with an LLM provider injected, wires them to a message bus,
and returns a running Brain instance.

Usage::

    from hbllm.brain.factory import BrainFactory

    # Using external provider:
    brain = await BrainFactory.create("openai/gpt-4o-mini")

    # Using LOCAL model (no API keys needed):
    brain = await BrainFactory.create_local("./checkpoints/sft/my_domain")

    # Or auto-detect local checkpoint:
    brain = await BrainFactory.create_local()

    result = await brain.process("What is quantum computing?")
    print(result.text)
    await brain.shutdown()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    inject_perception: bool = False  # Audio/Vision nodes (require ML models)
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

        return await BrainFactory._build_brain(llm_provider, cfg, bus)

    @staticmethod
    async def create_local(
        checkpoint_path: str | Path | None = None,
        model_size: str = "125m",
        config: BrainConfig | None = None,
        bus: MessageBus | None = None,
        device: str = "auto",
        lora_adapter_path: str | Path | None = None,
    ) -> Brain:
        """
        Create a Brain powered entirely by a local HBLLM model.
        
        No API keys or internet required.
        
        Args:
            checkpoint_path: Path to a model checkpoint (.pt file or directory).
                             If None, searches default locations.
            model_size: Model preset (125m, 500m, 1.5b) when no checkpoint found.
            config: Brain configuration. Defaults to BrainConfig().
            bus: Custom message bus. Defaults to InProcessBus.
            device: Device for inference ("auto", "cpu", "cuda", "mps").
            lora_adapter_path: Optional LoRA adapter .pt file to load on top.
            
        Returns:
            A running Brain instance using local model inference.
        """
        import torch
        from hbllm.model.config import get_config
        from hbllm.model.tokenizer import Tokenizer
        from hbllm.model.transformer import HBLLMForCausalLM
        from hbllm.serving.provider import LocalProvider

        cfg = config or BrainConfig()

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device

        # Load model
        model_config = get_config(model_size)
        model = HBLLMForCausalLM(model_config)
        tokenizer = Tokenizer()

        # Try to load checkpoint
        ckpt_loaded = False
        search_paths = []
        
        if checkpoint_path:
            search_paths.append(Path(checkpoint_path))
        else:
            # Search default locations
            search_paths.extend([
                Path("./checkpoints/sft"),
                Path("./checkpoints/self_improve"),
                Path("./checkpoints"),
            ])

        for ckpt_dir in search_paths:
            if ckpt_dir.is_file() and ckpt_dir.suffix == ".pt":
                logger.info("Loading checkpoint: %s", ckpt_dir)
                ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
                model.load_state_dict(
                    ckpt.get("model_state_dict", ckpt), strict=False
                )
                ckpt_loaded = True
                break
            elif ckpt_dir.is_dir():
                pts = sorted(ckpt_dir.rglob("step_*.pt"))
                if pts:
                    logger.info("Loading latest checkpoint: %s", pts[-1])
                    ckpt = torch.load(pts[-1], map_location="cpu", weights_only=False)
                    model.load_state_dict(
                        ckpt.get("model_state_dict", ckpt), strict=False
                    )
                    ckpt_loaded = True
                    break

        if not ckpt_loaded:
            logger.warning(
                "No checkpoint found — using randomly initialized %s model. "
                "Train a model first with `hbllm sft` or `hbllm train`.",
                model_config.name,
            )

        # Load LoRA adapter if specified
        if lora_adapter_path:
            adapter_path = Path(lora_adapter_path)
            if adapter_path.exists():
                from hbllm.modules.lora import LoRAManager
                LoRAManager.inject(model)
                state = torch.load(adapter_path, map_location="cpu", weights_only=True)
                LoRAManager.load_lora_state_dict(model, state)
                logger.info("Loaded LoRA adapter from %s", adapter_path)

        model = model.to(dev)
        model.eval()

        logger.info(
            "Local model ready: %s on %s (%s params)",
            model_config.name, dev, f"{model_config.num_params_estimate:,}",
        )

        # Create LocalProvider
        local_provider = LocalProvider(model=model, tokenizer=tokenizer, device=dev)

        return await BrainFactory._build_brain(local_provider, cfg, bus)

    @staticmethod
    async def _build_brain(
        llm_provider: LLMProvider,
        cfg: BrainConfig,
        bus: MessageBus | None = None,
    ) -> Brain:
        """Shared logic for wiring nodes and starting the brain."""
        # 1. Create adapter
        llm = ProviderLLM(llm_provider, system_prompt=cfg.system_prompt)

        # 2. Create bus and registry
        message_bus = bus or InProcessBus()
        await message_bus.start()

        registry = ServiceRegistry()
        await registry.start()

        # 3. Create cognitive nodes with LLM injected
        from hbllm.brain.router_node import RouterNode
        from hbllm.brain.planner_node import PlannerNode
        from hbllm.brain.critic_node import CriticNode
        from hbllm.brain.decision_node import DecisionNode
        from hbllm.brain.workspace_node import WorkspaceNode
        from hbllm.brain.experience_node import ExperienceNode
        from hbllm.brain.meta_node import MetaReasoningNode
        from hbllm.brain.identity_node import IdentityNode
        from hbllm.brain.sleep_node import SleepCycleNode
        from hbllm.brain.rule_extractor import RuleExtractorNode
        from hbllm.brain.curiosity_node import CuriosityNode
        from hbllm.brain.collective_node import CollectiveNode
        from hbllm.brain.learner_node import LearnerNode
        from hbllm.brain.world_model_node import WorldModelNode
        from hbllm.memory.memory_node import MemoryNode

        nodes = [
            # Core cognitive pipeline
            RouterNode(node_id="router", llm=llm),
            PlannerNode(
                node_id="planner",
                branch_factor=cfg.planner_branch_factor,
                max_depth=cfg.planner_max_depth,
            ),
            CriticNode(node_id="critic", llm=llm),
            DecisionNode(node_id="decision", llm=llm),
            WorkspaceNode(node_id="workspace"),

            # Memory (episodic + semantic + procedural + value + knowledge graph)
            MemoryNode(node_id="memory"),

            # Experience & meta-cognitive layer
            ExperienceNode(node_id="experience", llm=llm),
            MetaReasoningNode(node_id="meta"),
            RuleExtractorNode(node_id="rule_extractor"),

            # Curiosity-driven goal generation
            CuriosityNode(node_id="curiosity"),

            # Collective intelligence (multi-instance knowledge sharing)
            CollectiveNode(node_id="collective"),

            # Online learning from feedback (DPO)
            LearnerNode(node_id="learner"),

            # World model (code simulation & sandboxed execution)
            WorldModelNode(node_id="world_model"),

            # Memory consolidation during idle
            SleepCycleNode(node_id="sleep"),
        ]

        # Optional nodes based on config
        if cfg.inject_identity:
            nodes.append(IdentityNode(node_id="identity"))

        # Perception nodes (optional — require ML models to be downloaded)
        if cfg.inject_perception:
            from hbllm.perception.audio_in_node import AudioInputNode
            from hbllm.perception.audio_out_node import AudioOutputNode
            from hbllm.perception.vision_node import VisionNode
            nodes.extend([
                AudioInputNode(node_id="audio_in"),
                AudioOutputNode(node_id="audio_out"),
                VisionNode(node_id="vision"),
            ])

        # Inject LLM into planner
        nodes[1].llm = llm

        # 4. Start all nodes on the bus
        for node in nodes:
            await node.start(message_bus)
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

        # 5. Create and start pipeline
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

