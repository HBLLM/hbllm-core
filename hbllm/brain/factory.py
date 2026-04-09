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

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hbllm.actions.tool_memory import ToolMemory

# v2: Resource Intelligence
from hbllm.brain.attention_manager import AttentionManager
from hbllm.brain.cognitive_metrics import CognitiveMetrics
from hbllm.brain.confidence_estimator import ConfidenceEstimator

# v2: Intelligence Feedback Loop
from hbllm.brain.evaluation_node import EvaluationNode
from hbllm.brain.goal_manager import GoalManager
from hbllm.brain.load_manager import LoadManager
from hbllm.brain.owner_rules import OwnerRuleStore
from hbllm.brain.policy_engine import PolicyEngine
from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.brain.reflection_node import ReflectionNode
from hbllm.brain.revision_node import RevisionNode
from hbllm.brain.self_model import SelfModel
from hbllm.brain.skill_compiler_node import SkillCompilerNode

# New cognitive modules
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.brain.world_simulator import WorldSimulator
from hbllm.data.interaction_miner import InteractionMiner
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.cognition_router import CognitionRouter
from hbllm.network.node import Node
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult
from hbllm.serving.provider import LLMProvider, get_provider
from hbllm.serving.token_optimizer import TokenOptimizer
from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import RewardModel

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for Brain creation."""

    inject_memory: bool = True
    inject_identity: bool = True
    inject_curiosity: bool = True
    inject_perception: bool = False  # Audio/Vision nodes (require ML models)
    inject_revision: bool = True  # Self-critique loop
    inject_goals: bool = True  # Autonomous goal system
    inject_self_model: bool = True  # Capability tracking
    inject_metrics: bool = True  # Live cognitive metrics
    inject_cost_optimizer: bool = True  # Token optimization
    inject_policy_engine: bool = True  # Governance policy enforcement
    inject_owner_rules: bool = True  # Owner-defined behavioral rules
    inject_sentinel: bool = True  # Proactive governance monitoring
    inject_evaluation: bool = True  # v2: Intelligence feedback loop
    inject_reflection: bool = True  # v2: Periodic batch reflection
    inject_skill_compiler: bool = True  # v2: Auto-skill extraction
    inject_attention: bool = True  # v2: Attention budget management
    inject_load_manager: bool = True  # v2: Cognitive load management
    inject_fuzzy_logic: bool = False  # Fuzzy reasoning (requires scikit-fuzzy)
    inject_symbolic_logic: bool = False  # Z3 theorem prover (requires z3-solver)
    total_timeout: float = 60.0
    planner_branch_factor: int = 3
    planner_max_depth: int = 2
    data_dir: str = "data"
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
        nodes: list[Node],
        provider: LLMProvider,
    ) -> None:
        self.bus = bus
        self.registry = registry
        self.pipeline = pipeline
        self.llm = llm
        self.nodes = nodes
        self.provider = provider

        # Cognitive subsystems (initialized by factory)
        self.skill_registry: SkillRegistry | None = None
        self.goal_manager: GoalManager | None = None
        self.self_model: SelfModel | None = None
        self.cognitive_metrics: CognitiveMetrics | None = None
        self.world_simulator: WorldSimulator | None = None
        self.revision_node: RevisionNode | None = None
        self.confidence_estimator: ConfidenceEstimator | None = None
        self.tool_memory: ToolMemory | None = None
        self.concept_extractor: ConceptExtractor | None = None
        self.cognition_router: CognitionRouter | None = None
        self.token_optimizer: TokenOptimizer | None = None
        self.reward_model: RewardModel | None = None
        self.policy_optimizer: PolicyOptimizer | None = None
        self.interaction_miner: InteractionMiner | None = None
        self.policy_engine: PolicyEngine | None = None
        self.owner_rules: OwnerRuleStore | None = None
        self.sentinel: Any = None  # SentinelNode reference

        # v2: Intelligence Feedback Loop
        self.evaluation_node: EvaluationNode | None = None
        self.reflection_node: ReflectionNode | None = None
        self.skill_compiler_node: SkillCompilerNode | None = None

        # v2: Resource Intelligence
        self.attention_manager: AttentionManager | None = None
        self.load_manager: LoadManager | None = None

        self._hardware_loop_task: asyncio.Task[None] | None = None

    async def process(
        self,
        text: str,
        tenant_id: str = "default",
        session_id: str = "default",
    ) -> PipelineResult:
        """Send a query through the full cognitive pipeline."""
        import time as _time

        _start = _time.monotonic()

        # Start hardware monitor on first query if not running
        if not self._hardware_loop_task:
            self._hardware_loop_task = asyncio.create_task(self._hardware_monitor_loop())

        # Token optimization (pre-process)
        if self.token_optimizer:
            self.token_optimizer.optimize(text)

        result = await self.pipeline.process(
            text=text,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        _elapsed = (_time.monotonic() - _start) * 1000

        # Post-process: record cognitive metrics
        if self.cognitive_metrics:
            self.cognitive_metrics.record_latency(_elapsed, "pipeline")
            self.cognitive_metrics.record_reasoning(result.confidence)

        # Post-process: self-model tracking
        if self.self_model:
            domain = result.metadata.get("domain_hint", "general")
            self.self_model.record_outcome(
                domain,
                success=not result.error,
                confidence=result.confidence,
                latency_ms=_elapsed,
            )

        # Post-process: interaction mining
        if self.interaction_miner and not result.error:
            self.interaction_miner.record_interaction(
                query=text,
                response=result.text,
                reward=result.confidence,
                tenant_id=tenant_id,
            )

        return result

    async def _hardware_monitor_loop(self) -> None:
        """Periodic hardware health check for dynamic model offloading."""
        try:
            from hbllm.modules.hardware_hal import HardwareHAL
            from hbllm.network.messages import Message, MessageType
        except ImportError:
            return

        while True:
            await asyncio.sleep(60)  # Check every minute

            # Simulated model footprint in memory (assume dynamic tracking)
            # Threshold: > 90%
            try:
                import psutil  # type: ignore[import-untyped]

                mem_percent = psutil.virtual_memory().percent
            except ImportError:
                mem_percent = 50.0

            if mem_percent > 90.0:
                logger.warning(
                    "[HardwareMonitor] System RAM >90%% (%.1f%%). Triggering memory pressure event.",
                    mem_percent,
                )
                await self.bus.publish(
                    "system.hardware.critical",
                    Message(
                        type=MessageType.EVENT,
                        topic="system.hardware.critical",
                        source_node_id="system",
                        payload={"ram_percent": mem_percent, "action": "offload_experts_requested"},
                    ),
                )

    async def shutdown(self) -> None:
        """Stop all nodes, pipeline, and bus."""
        if self._hardware_loop_task:
            self._hardware_loop_task.cancel()
        await self.pipeline.stop()
        for node in reversed(self.nodes):
            try:
                await node.stop()
            except Exception:
                logger.debug("Error stopping node %s during shutdown", node.node_id, exc_info=True)
        await self.registry.stop()
        await self.bus.stop()
        logger.info("Brain shutdown complete")

    @property
    def usage(self) -> dict[str, int]:
        """Accumulated LLM usage statistics."""
        return self.llm.usage

    def cognitive_stats(self) -> dict[str, Any]:
        """Get stats from all cognitive subsystems."""
        stats = {}
        if self.cognitive_metrics:
            stats["metrics"] = self.cognitive_metrics.get_dashboard_metrics()
        if self.self_model:
            stats["self_model"] = self.self_model.get_metrics()
        if self.skill_registry:
            stats["skills"] = self.skill_registry.stats()
        if self.goal_manager:
            stats["goals"] = self.goal_manager.stats()
        if self.tool_memory:
            stats["tool_memory"] = self.tool_memory.stats()
        if self.token_optimizer:
            stats["token_optimizer"] = self.token_optimizer.stats()
        if self.reward_model:
            stats["rewards"] = self.reward_model.stats()
        return stats


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
        from hbllm.model.tokenizer import HBLLMTokenizer
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
        tokenizer = HBLLMTokenizer()

        # Try to load checkpoint
        ckpt_loaded = False
        search_paths = []

        if checkpoint_path:
            search_paths.append(Path(checkpoint_path))
        else:
            # Search default locations
            search_paths.extend(
                [
                    Path("./checkpoints/sft"),
                    Path("./checkpoints/self_improve"),
                    Path("./checkpoints"),
                ]
            )

        for ckpt_dir in search_paths:
            if ckpt_dir.is_file() and ckpt_dir.suffix == ".pt":
                logger.info("Loading checkpoint: %s", ckpt_dir)
                from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

                ckpt = load_checkpoint(ckpt_dir)
                model.load_state_dict(extract_model_state(ckpt), strict=False)
                ckpt_loaded = True
                break
            elif ckpt_dir.is_dir():
                pts = sorted(ckpt_dir.rglob("step_*.pt"))
                if pts:
                    logger.info("Loading latest checkpoint: %s", pts[-1])
                    from hbllm.utils.checkpoint import extract_model_state, load_checkpoint

                    ckpt = load_checkpoint(pts[-1])
                    model.load_state_dict(extract_model_state(ckpt), strict=False)
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
            model_config.name,
            dev,
            f"{model_config.num_params_estimate:,}",
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
        from hbllm.brain.collective_node import CollectiveNode
        from hbllm.brain.critic_node import CriticNode
        from hbllm.brain.curiosity_node import CuriosityNode
        from hbllm.brain.decision_node import DecisionNode
        from hbllm.brain.experience_node import ExperienceNode
        from hbllm.brain.identity_node import IdentityNode
        from hbllm.brain.learner_node import LearnerNode
        from hbllm.brain.meta_node import MetaReasoningNode
        from hbllm.brain.planner_node import PlannerNode
        from hbllm.brain.router_node import RouterNode
        from hbllm.brain.rule_extractor import RuleExtractorNode
        from hbllm.brain.sentinel_node import SentinelNode
        from hbllm.brain.sleep_node import SleepCycleNode
        from hbllm.brain.workspace_node import WorkspaceNode
        from hbllm.brain.world_model_node import WorldModelNode
        from hbllm.memory.memory_node import MemoryNode
        from hbllm.network.node import Node

        # Create PolicyEngine for governance
        policy_engine = None
        if cfg.inject_policy_engine:
            policy_engine = PolicyEngine()
            logger.info("PolicyEngine created for governance")

        nodes = [
            # Core cognitive pipeline
            RouterNode(node_id="router", llm=llm),
            PlannerNode(
                node_id="planner",
                branch_factor=cfg.planner_branch_factor,
                max_depth=cfg.planner_max_depth,
                policy_engine=policy_engine,
            ),
            CriticNode(node_id="critic", llm=llm),
            DecisionNode(node_id="decision", llm=llm, policy_engine=policy_engine),
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
            SleepCycleNode(node_id="sleep", llm=llm),
        ]

        # Proactive governance sentinel
        sentinel_node = None
        if cfg.inject_sentinel and policy_engine:
            sentinel_node = SentinelNode(
                node_id="sentinel",
                policy_engine=policy_engine,
            )
            nodes.append(sentinel_node)

        # Optional nodes based on config
        if cfg.inject_identity:
            nodes.append(IdentityNode(node_id="identity"))

        # Perception nodes (optional — require ML models to be downloaded)
        if cfg.inject_perception:
            from hbllm.perception.audio_in_node import AudioInputNode
            from hbllm.perception.audio_out_node import AudioOutputNode
            from hbllm.perception.vision_node import VisionNode

            nodes.extend(
                [
                    AudioInputNode(node_id="audio_in"),
                    AudioOutputNode(node_id="audio_out"),
                    VisionNode(node_id="vision"),
                ]
            )

        # Reasoning nodes (optional — require extra dependencies)
        if cfg.inject_fuzzy_logic:
            from hbllm.actions.fuzzy_node import FuzzyNode

            nodes.append(FuzzyNode(node_id="fuzzy", llm=llm))
            logger.info("FuzzyNode wired (scikit-fuzzy reasoning)")

        if cfg.inject_symbolic_logic:
            from hbllm.actions.logic_node import LogicNode

            nodes.append(LogicNode(node_id="logic", llm=llm))
            logger.info("LogicNode wired (Z3 theorem prover)")

        # 4. Start all nodes on the bus
        for node in nodes:
            await node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    capabilities=node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(
                    node_id=node.node_id,
                    status=HealthStatus.HEALTHY,
                )
            )

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

        brain = Brain(
            bus=message_bus,
            registry=registry,
            pipeline=pipeline,
            llm=llm,
            nodes=nodes,
            provider=llm_provider,
        )

        # 6. Wire cognitive subsystems
        data_dir = cfg.data_dir

        # Always-on subsystems
        brain.skill_registry = SkillRegistry(data_dir=data_dir)
        brain.tool_memory = ToolMemory(data_dir=data_dir)
        brain.concept_extractor = ConceptExtractor()
        brain.world_simulator = WorldSimulator()
        brain.cognition_router = CognitionRouter()
        brain.reward_model = RewardModel(data_dir=data_dir)
        brain.policy_optimizer = PolicyOptimizer()
        brain.interaction_miner = InteractionMiner(data_dir=data_dir)

        # Configurable subsystems
        if cfg.inject_revision:
            brain.confidence_estimator = ConfidenceEstimator()
            brain.revision_node = RevisionNode()

        if cfg.inject_goals:
            brain.goal_manager = GoalManager(data_dir=data_dir)

        if cfg.inject_self_model:
            brain.self_model = SelfModel(data_dir=data_dir)

        if cfg.inject_metrics:
            brain.cognitive_metrics = CognitiveMetrics(data_dir=data_dir)

        if cfg.inject_cost_optimizer:
            brain.token_optimizer = TokenOptimizer()

        if cfg.inject_policy_engine:
            brain.policy_engine = policy_engine

        if cfg.inject_owner_rules:
            brain.owner_rules = OwnerRuleStore(db_path=str(Path(data_dir) / "owner_rules.db"))

        if cfg.inject_sentinel and sentinel_node:
            brain.sentinel = sentinel_node

        # v2: Intelligence Feedback Loop — wire evaluation, reflection, skill compiler
        if cfg.inject_evaluation:
            eval_node = EvaluationNode(
                node_id="evaluation",
                cognitive_metrics=brain.cognitive_metrics,
                goal_manager=brain.goal_manager,
                self_model=brain.self_model,
                skill_registry=brain.skill_registry,
            )
            await eval_node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=eval_node.node_id,
                    node_type=eval_node.node_type,
                    capabilities=eval_node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(node_id=eval_node.node_id, status=HealthStatus.HEALTHY)
            )
            brain.evaluation_node = eval_node
            nodes.append(eval_node)
            logger.info("v2: EvaluationNode wired (intelligence feedback loop)")

        if cfg.inject_reflection:
            refl_node = ReflectionNode(
                node_id="reflection",
                cognitive_metrics=brain.cognitive_metrics,
                goal_manager=brain.goal_manager,
                self_model=brain.self_model,
                skill_registry=brain.skill_registry,
            )
            await refl_node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=refl_node.node_id,
                    node_type=refl_node.node_type,
                    capabilities=refl_node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(node_id=refl_node.node_id, status=HealthStatus.HEALTHY)
            )
            brain.reflection_node = refl_node
            nodes.append(refl_node)
            logger.info("v2: ReflectionNode wired (periodic batch reflection)")

        if cfg.inject_skill_compiler:
            compiler_node = SkillCompilerNode(
                node_id="skill_compiler",
                skill_registry=brain.skill_registry,
            )
            await compiler_node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=compiler_node.node_id,
                    node_type=compiler_node.node_type,
                    capabilities=compiler_node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(node_id=compiler_node.node_id, status=HealthStatus.HEALTHY)
            )
            brain.skill_compiler_node = compiler_node
            nodes.append(compiler_node)
            logger.info("v2: SkillCompilerNode wired (auto-skill extraction)")

        # v2: Resource Intelligence — wire attention and load managers
        if cfg.inject_attention:
            attn_node = AttentionManager(node_id="attention")
            await attn_node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=attn_node.node_id,
                    node_type=attn_node.node_type,
                    capabilities=attn_node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(node_id=attn_node.node_id, status=HealthStatus.HEALTHY)
            )
            brain.attention_manager = attn_node
            nodes.append(attn_node)
            logger.info("v2: AttentionManager wired (memory budgets & focus)")

        if cfg.inject_load_manager:
            load_node = LoadManager(
                node_id="load_manager",
                monitor_interval=60.0,
            )
            await load_node.start(message_bus)
            from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo

            await registry.register(
                NodeInfo(
                    node_id=load_node.node_id,
                    node_type=load_node.node_type,
                    capabilities=load_node.capabilities,
                )
            )
            await registry.update_health(
                NodeHealth(node_id=load_node.node_id, status=HealthStatus.HEALTHY)
            )
            brain.load_manager = load_node
            nodes.append(load_node)
            logger.info("v2: LoadManager wired (resource monitoring & degradation)")

        logger.info(
            "Cognitive subsystems wired: skills, goals, self-model, metrics, revision, tools, "
            "policy engine, owner rules, sentinel, evaluation, reflection, skill_compiler, "
            "attention, load_manager"
        )

        return brain
