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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.knowledge import KnowledgeBase
    from hbllm.plugin.manager import PluginManager

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
from hbllm.brain.world_state import WorldStateEngine
from hbllm.data.interaction_miner import AsyncInteractionMiner, InteractionMiner
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.cognition_router import CognitionRouter
from hbllm.network.node import HealthStatus, Node, NodeHealth, NodeInfo
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult
from hbllm.serving.provider import LLMProvider, get_provider
from hbllm.serving.token_optimizer import TokenOptimizer
from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import RewardModel

logger = logging.getLogger(__name__)


def _is_slow_cpu() -> bool:
    import os

    try:
        import torch

        # If CUDA or MPS is available, we have a fast GPU/coprocessor
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return not (has_cuda or has_mps)
    except ImportError:
        return True


def _default_api_timeout() -> float:
    import os

    default_val = 300.0 if _is_slow_cpu() else 60.0
    return float(os.getenv("HBLLM_API_TIMEOUT", str(default_val)))


def _default_stream_timeout() -> float:
    import os

    default_val = 300.0 if _is_slow_cpu() else 30.0
    return float(os.getenv("HBLLM_STREAM_TIMEOUT", str(default_val)))


def _default_total_timeout() -> float:
    import os

    default_val = 300.0 if _is_slow_cpu() else 60.0
    return float(os.getenv("HBLLM_TOTAL_TIMEOUT", str(default_val)))


@dataclass
class BrainConfig:
    """Configuration for Brain creation."""

    # ── Composite node flags (v4: consolidated architecture) ──────
    inject_reasoning: bool = True  # ReasoningCore (router+planner+critic+decision+revision+prm)
    inject_memory_system: bool = True  # MemorySystem (memory+experience+sleep)
    inject_meta_cognition: bool = True  # MetaCognition (meta+evaluation+reflection+curiosity)
    inject_skill_engine: bool = True  # SkillEngine (compiler+intelligence+induction+failure+rules)
    inject_governance: bool = True  # GovernanceGuard (sentinel+policy+confidence)
    inject_resources: bool = True  # ResourceManager (workspace+attention+load+scheduler)
    inject_social: bool = True  # SocialLayer (collective+identity)
    inject_learning: bool = True  # LearningLoop (learner+world_model)

    # ── Advanced Capability flags (Phase 3-7) ───────
    inject_embodiment: bool = True
    inject_human_control: bool = True
    inject_causal_graph: bool = True
    inject_compaction: bool = True
    inject_task_graph: bool = True
    inject_mesh: bool = True

    # ── Legacy flags (preserved for backward compatibility) ───────
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
    inject_scheduler: bool = True  # v3: Proactive agent capabilities
    inject_fuzzy_logic: bool = False  # Fuzzy reasoning (requires scikit-fuzzy)
    inject_symbolic_logic: bool = False  # Z3 theorem prover (requires z3-solver)
    inject_browser: bool = True  # Browse web / search via DuckDuckGo
    inject_execution: bool = True  # Python sandboxed code execution
    total_timeout: float = field(default_factory=_default_total_timeout)
    api_timeout: float = field(default_factory=_default_api_timeout)
    stream_timeout: float = field(default_factory=_default_stream_timeout)
    planner_branch_factor: int = 3
    planner_max_depth: int = 2
    data_dir: str = field(default_factory=lambda: os.environ.get("HBLLM_DATA_DIR", "data"))
    inject_sil: bool = True  # Skill Intelligence Layer
    inject_failure_analyzer: bool = True  # Automatic skill repair
    inject_shell: bool = True  # Host shell command executor node
    require_shell_approval: bool = True  # Require manual shell approval
    domain_registry: Any | None = None  # Hierarchical domain registry
    system_prompt: str = (
        "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM modular architecture. "
        "You have access to various cognitive and tool modules, including a BrowserNode (which allows "
        "you to browse the web and search for real-time information), an ExecutionNode (for running "
        "Python code in a secure sandbox), a LogicNode (powered by Z3 for symbolic reasoning), and a "
        "persistent memory node. Be helpful, precise, and accurate."
    )

    # ── Mode selection ────────────────────────────────────────────
    use_composites: bool = True  # Use consolidated composite nodes (v4)

    # Knowledge base
    inject_knowledge: bool = True  # Auto-create knowledge base

    # Persistence
    inject_persistence: bool = True  # Auto-create BrainState

    # Cognitive awareness
    inject_awareness: bool = True  # Brain self-monitoring

    # Plugin system
    inject_plugins: bool = True  # Auto-discover plugins on startup
    plugin_dirs: list[str] | None = None  # Extra plugin scan directories
    watch_plugins: bool = False  # Background watcher for new plugins


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

        # ── v4: Composite nodes ────────────────────────────────────
        self.reasoning_core: Any = None  # ReasoningCore
        self.memory_system: Any = None  # MemorySystem
        self.meta_cognition: Any = None  # MetaCognition
        self.skill_engine: Any = None  # SkillEngine
        self.governance_guard: Any = None  # GovernanceGuard
        self.resource_manager: Any = None  # ResourceManager
        self.social_layer: Any = None  # SocialLayer
        self.learning_loop: Any = None  # LearningLoop

        # Cognitive subsystems (initialized by factory)
        self.skill_registry: SkillRegistry | None = None
        self.goal_manager: GoalManager | None = None
        self.self_model: SelfModel | None = None
        self.cognitive_metrics: CognitiveMetrics | None = None
        self.world_state: WorldStateEngine | None = None
        self.revision_node: RevisionNode | None = None
        self.confidence_estimator: ConfidenceEstimator | None = None
        self.tool_memory: ToolMemory | None = None
        self.concept_extractor: ConceptExtractor | None = None
        self.cognition_router: CognitionRouter | None = None
        self.token_optimizer: TokenOptimizer | None = None
        self.reward_model: RewardModel | None = None
        self.policy_optimizer: PolicyOptimizer | None = None
        self.interaction_miner: AsyncInteractionMiner | None = None
        self.policy_engine: PolicyEngine | None = None
        self.owner_rules: OwnerRuleStore | None = None
        self.sentinel: Any = None  # SentinelNode reference

        # v2: Intelligence Feedback Loop
        self.evaluation_node: EvaluationNode | None = None
        self.reflection_node: ReflectionNode | None = None
        self.skill_compiler_node: SkillCompilerNode | None = None
        self.skill_intelligence_node: Any | None = None
        self.failure_analyzer_node: Any | None = None

        # v2: Resource Intelligence
        self.attention_manager: AttentionManager | None = None
        self.load_manager: LoadManager | None = None

        # v3: Proactive Execution
        self.scheduler_node: Any = None

        # Knowledge base
        self.knowledge_base: KnowledgeBase | None = None

        # Persistence
        self.state: Any = None  # BrainState reference

        # Cognitive awareness
        self.awareness: Any = None  # CognitiveAwareness reference

        # Phase 3-7 Core subsystems
        self.event_log: Any | None = None
        self.os_adapter: Any | None = None
        self.verifier: Any | None = None
        self.task_graph: Any | None = None
        self.causal_graph: Any | None = None
        self.compaction_engine: Any | None = None
        self.permission_registry: Any | None = None
        self.decision_tracer: Any | None = None
        self.security_guard: Any | None = None
        self.mesh_registry: Any | None = None

        # Plugin system
        self.plugin_manager: PluginManager | None = None

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
            await self.interaction_miner.record_interaction(
                query=text,
                response=result.text,
                reward=result.confidence,
                tenant_id=tenant_id,
            )

        return result

    async def _hardware_monitor_loop(self) -> None:
        """Periodic hardware health check for dynamic model offloading."""
        try:
            from hbllm.network.messages import Message, MessageType
        except ImportError:
            return

        while self._hardware_loop_task and not self._hardware_loop_task.cancelled():
            try:
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break

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
            try:
                await self._hardware_loop_task
            except asyncio.CancelledError:
                pass
        # Stop plugin watcher
        if self.plugin_manager:
            await self.plugin_manager.stop_watching()
        # Save knowledge base vectors
        if self.knowledge_base:
            try:
                self.knowledge_base._save_vectors()
            except Exception:
                logger.debug("Error saving knowledge vectors during shutdown", exc_info=True)
        # Close persistence
        if self.state:
            try:
                self.state.close()
            except Exception:
                logger.debug("Error closing brain state during shutdown", exc_info=True)
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

        # Advanced subsystem stats
        if self.task_graph and hasattr(self.task_graph, "stats"):
            stats["task_graph"] = self.task_graph.stats()
        if self.compaction_engine and hasattr(self.compaction_engine, "stats"):
            stats["compaction"] = self.compaction_engine.stats()
        if self.causal_graph and hasattr(self.causal_graph, "stats"):
            stats["causality"] = self.causal_graph.stats()

        return stats


async def _register_node(registry: Any, node: Node) -> None:
    """Helper to register a node and mark it healthy upon startup."""
    await registry.register(node.get_info())
    await registry.update_health(NodeHealth(node_id=node.node_id, status=HealthStatus.HEALTHY))


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

        from hbllm.model.tokenizer import HBLLMTokenizer
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

        # Force float32 precision on CPU to avoid slow emulation overhead,
        # but allow bfloat16 on macOS (Darwin) for HuggingFace models as it is highly accelerated and 8x faster.
        import platform

        from hbllm.model.model_loader import load_model

        is_native_preset = model_size.lower().strip() in {"125m", "500m", "1.5b", "7b", "13b"}
        if dev == "cpu":
            if platform.system() == "Darwin" and not is_native_preset:
                dtype_to_use = "bfloat16"
            else:
                dtype_to_use = "float32"
        else:
            dtype_to_use = "auto"
        model = load_model(source=model_size, device=dev, dtype=dtype_to_use)

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            import os

            from hbllm.model.tokenizer import HBLLMTokenizer

            vocab_paths = [
                "data/training/vocab.json",
                "core/data/training/vocab.json",
                "../data/training/vocab.json",
            ]
            loaded = False
            for p in vocab_paths:
                if os.path.exists(p):
                    logger.info("Loading native tokenizer from %s", p)
                    tokenizer = HBLLMTokenizer.from_vocab(p)
                    loaded = True
                    break
            if not loaded:
                logger.warning("Native vocab not found, using fallback tokenizer")
                tokenizer = HBLLMTokenizer()

        is_native = type(model).__name__ == "HBLLMForCausalLM"

        # Try to load checkpoint (ONLY for native models)
        ckpt_loaded = False
        search_paths = []

        if is_native:
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

        if is_native and not ckpt_loaded:
            logger.warning(
                "No checkpoint found — using randomly initialized native model. "
                "Train a model first with `hbllm sft` or `hbllm train`."
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
            "Local model ready: %s on %s",
            model_size,
            dev,
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
        await registry.start(message_bus)

        # ── Tenant context propagation ───────────────────────────────
        from hbllm.security.tenant_interceptor import TenantInterceptor

        message_bus.add_interceptor(TenantInterceptor())

        # ── v4: Composite node path ──────────────────────────────────
        if cfg.use_composites:
            return await BrainFactory._build_composite_brain(
                llm_provider,
                llm,
                cfg,
                message_bus,
                registry,
            )

        # 3. Create cognitive nodes with LLM injected (legacy path)
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
        from hbllm.brain.world_state import WorldStateEngine
        from hbllm.memory.memory_node import MemoryNode

        # Create PolicyEngine for governance
        policy_engine = None
        if cfg.inject_policy_engine:
            policy_engine = PolicyEngine()
            logger.info("PolicyEngine created for governance")

        # Create hierarchical domain registry
        from hbllm.modules.domain_registry import DomainRegistry

        domain_registry = cfg.domain_registry or DomainRegistry()

        # Auto-discover sub-domain LoRA adapters from data/lora/
        lora_dir = Path(cfg.data_dir) / "lora"
        if lora_dir.is_dir():
            from hbllm.modules.domain_registry import DomainSpec

            for adapter_dir in sorted(lora_dir.iterdir()):
                if adapter_dir.is_dir() and not domain_registry.exists(adapter_dir.name):
                    domain_registry.register(
                        DomainSpec(
                            name=adapter_dir.name,
                            centroid_text=f"Topics relating to {adapter_dir.name.replace('.', ' ')}",
                        )
                    )
                    logger.info("Auto-discovered sub-domain LoRA: %s", adapter_dir.name)

        router_node = RouterNode(node_id="router", llm=llm, domain_registry=domain_registry)
        router_node._centroids_path = Path(cfg.data_dir) / "router_centroids.json"

        skill_registry = SkillRegistry(data_dir=cfg.data_dir)

        nodes = [
            # Core cognitive pipeline
            router_node,
            PlannerNode(
                node_id="planner",
                branch_factor=cfg.planner_branch_factor,
                max_depth=cfg.planner_max_depth,
                policy_engine=policy_engine,
                llm=llm,
            ),
            CriticNode(node_id="critic", llm=llm),
            DecisionNode(node_id="decision", llm=llm, policy_engine=policy_engine),
            WorkspaceNode(node_id="workspace"),
            # Memory (episodic + semantic + procedural + value + knowledge graph)
            MemoryNode(node_id="memory", db_path=Path(cfg.data_dir) / "working_memory.db"),
            # Experience & meta-cognitive layer
            ExperienceNode(node_id="experience", llm=llm),
            MetaReasoningNode(node_id="meta"),
            RuleExtractorNode(node_id="rule_extractor"),
            # Curiosity-driven goal generation
            CuriosityNode(node_id="curiosity"),
            # Collective intelligence (multi-instance knowledge sharing)
            CollectiveNode(node_id="collective", skill_registry=skill_registry),
            # Online learning from feedback (DPO)
            LearnerNode(node_id="learner"),
            # World model (code simulation & sandboxed execution)
            WorldModelNode(node_id="world_model"),
            # Memory consolidation during idle
            SleepCycleNode(node_id="sleep", llm=llm),
        ]

        # Browser Node (DuckDuckGo search + scraping)
        if cfg.inject_browser:
            from hbllm.actions.browser_node import BrowserNode

            nodes.append(BrowserNode(node_id="browser"))
            logger.info("BrowserNode wired (web search & scrape)")

        # Execution Node (sandboxed python execution)
        if cfg.inject_execution:
            from hbllm.actions.execution_node import ExecutionNode

            nodes.append(ExecutionNode(node_id="execution"))
            logger.info("ExecutionNode wired (sandboxed python execution)")

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

        # Host shell execution node
        if cfg.inject_shell:
            from hbllm.actions.shell_node import HostShellNode

            shell_node = HostShellNode(
                node_id="shell_executor",
                workspace_dir=None,
                require_manual_approval=cfg.require_shell_approval,
                policy_engine=policy_engine,
            )
            nodes.append(shell_node)
            logger.info("HostShellNode wired (manual approval=%s)", cfg.require_shell_approval)

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

        # Register and start default DomainModuleNode instances
        if (
            type(llm_provider).__name__ == "LocalProvider"
            and getattr(llm_provider, "_model", None) is not None
        ):
            from hbllm.modules.base_module import DomainModuleNode

            model = llm_provider._model
            tokenizer = llm_provider._tokenizer
            for domain in ["general", "coding", "math"]:
                nodes.append(
                    DomainModuleNode(
                        node_id=f"domain_{domain}",
                        domain_name=domain,
                        model=model,
                        tokenizer=tokenizer,
                        lora_state_dict=None,
                    )
                )
        else:
            from hbllm.modules.base_module import DomainModuleNode

            for domain in ["general", "coding", "math"]:
                nodes.append(
                    DomainModuleNode(
                        node_id=f"domain_{domain}",
                        domain_name=domain,
                        llm=llm,
                    )
                )

        # 4. Start all nodes on the bus
        for node in nodes:
            await node.start(message_bus)
            await _register_node(registry, node)

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
        brain.skill_registry = skill_registry
        brain.tool_memory = ToolMemory(data_dir=data_dir)
        brain.concept_extractor = ConceptExtractor()

        from hbllm.perception.event_log import EventLog

        brain.event_log = EventLog(data_dir=data_dir)
        brain.world_state = WorldStateEngine(event_log=brain.event_log)

        brain.cognition_router = CognitionRouter()
        brain.reward_model = RewardModel(data_dir=data_dir)
        brain.policy_optimizer = PolicyOptimizer()
        brain.interaction_miner = AsyncInteractionMiner(data_dir=data_dir)

        # Advanced Subsystems (Phase 3-7)
        if cfg.inject_task_graph:
            from hbllm.brain.autonomy.task_graph import TaskGraphRuntime

            brain.task_graph = TaskGraphRuntime(data_dir=data_dir)

        if cfg.inject_causal_graph:
            from hbllm.brain.causality.causal_graph import CausalGraph

            brain.causal_graph = CausalGraph(data_dir=data_dir)

        if cfg.inject_compaction:
            from hbllm.brain.compaction.engine import CognitiveCompactionEngine

            brain.compaction_engine = CognitiveCompactionEngine()

        if cfg.inject_embodiment:
            from hbllm.brain.embodiment.os_adapter import OSAdapter
            from hbllm.brain.embodiment.verifier import ExecutionVerifier

            brain.os_adapter = OSAdapter()
            brain.verifier = ExecutionVerifier(brain.os_adapter)

        if cfg.inject_human_control:
            from hbllm.brain.control.guard import SecurityGuard
            from hbllm.brain.control.permissions import PermissionRegistry
            from hbllm.brain.observability.tracer import DecisionTraceLedger

            brain.permission_registry = PermissionRegistry()
            brain.decision_tracer = DecisionTraceLedger(data_dir=data_dir)
            brain.security_guard = SecurityGuard(brain.permission_registry, brain.decision_tracer)

        if cfg.inject_mesh:
            from hbllm.brain.mesh.registry import NodeRegistry, NodeType

            brain.mesh_registry = NodeRegistry(local_node_id="local", local_node_type=NodeType.EDGE)

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
                db_path=Path(cfg.data_dir) / "evaluations.db",
            )
            await _register_node(registry, eval_node)
            await eval_node.start(message_bus)
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
            await _register_node(registry, refl_node)
            await refl_node.start(message_bus)
            brain.reflection_node = refl_node
            nodes.append(refl_node)
            logger.info("v2: ReflectionNode wired (periodic batch reflection)")

        if cfg.inject_skill_compiler:
            compiler_node = SkillCompilerNode(
                node_id="skill_compiler",
                skill_registry=brain.skill_registry,
                llm=llm,
            )
            await _register_node(registry, compiler_node)
            await compiler_node.start(message_bus)
            brain.skill_compiler_node = compiler_node
            nodes.append(compiler_node)
            logger.info("v2: SkillCompilerNode wired (auto-skill extraction)")

        if cfg.inject_failure_analyzer:
            from hbllm.brain.failure_analyzer_node import FailureAnalyzerNode

            fail_node = FailureAnalyzerNode(node_id="failure_analyzer", llm=llm)
            await _register_node(registry, fail_node)
            await fail_node.start(message_bus)
            brain.failure_analyzer_node = fail_node
            nodes.append(fail_node)
            logger.info("FailureAnalyzerNode wired (automated skill repair)")

        if cfg.inject_sil:
            from hbllm.brain.skill_intelligence_node import SkillIntelligenceNode

            sil_node = SkillIntelligenceNode(node_id="sil", skill_registry=brain.skill_registry)
            await _register_node(registry, sil_node)
            await sil_node.start(message_bus)
            brain.skill_intelligence_node = sil_node
            nodes.append(sil_node)
            logger.info("SkillIntelligenceNode wired (execution governor & lifecycle)")

        # v2: Resource Intelligence — wire attention and load managers
        if cfg.inject_attention:
            attn_node = AttentionManager(node_id="attention")
            await _register_node(registry, attn_node)
            await attn_node.start(message_bus)
            brain.attention_manager = attn_node
            nodes.append(attn_node)
            logger.info("v2: AttentionManager wired (memory budgets & focus)")

        if cfg.inject_load_manager:
            load_node = LoadManager(
                node_id="load_manager",
                monitor_interval=60.0,
            )
            await _register_node(registry, load_node)
            await load_node.start(message_bus)
            brain.load_manager = load_node
            nodes.append(load_node)
            logger.info("v2: LoadManager wired (resource monitoring & degradation)")

        # v3: Proactive Execution — wire the scheduler node
        if cfg.inject_scheduler:
            from hbllm.brain.scheduler_node import SchedulerNode

            sched_node = SchedulerNode(
                node_id="scheduler",
                data_dir=data_dir,
            )
            await _register_node(registry, sched_node)
            await sched_node.start(message_bus)
            brain.scheduler_node = sched_node
            nodes.append(sched_node)
            logger.info("v3: SchedulerNode wired (proactive autonomous task execution)")

        logger.info(
            "Cognitive subsystems wired: skills, goals, self-model, metrics, revision, tools, "
            "policy engine, owner rules, sentinel, evaluation, reflection, skill_compiler, "
            "attention, load_manager"
        )

        # ── Cognitive Awareness ───────────────────────────────────────
        if cfg.inject_awareness:
            from hbllm.brain.awareness import CognitiveAwareness

            awareness_node = CognitiveAwareness(node_id="cognitive_awareness")
            await _register_node(registry, awareness_node)
            await awareness_node.start(message_bus)
            brain.awareness = awareness_node
            nodes.append(awareness_node)
            logger.info("CognitiveAwareness wired (brain self-monitoring active)")

        # ── Knowledge Base ────────────────────────────────────────────
        if cfg.inject_knowledge:
            from hbllm.knowledge import KnowledgeBase

            kb_dir = str(Path(data_dir) / "knowledge")
            brain.knowledge_base = KnowledgeBase(data_dir=kb_dir)
            logger.info("KnowledgeBase wired (data_dir=%s)", kb_dir)

        # ── Persistence (BrainState) ─────────────────────────────────
        if cfg.inject_persistence:
            from hbllm.persistence import BrainState

            state_path = str(Path(data_dir) / "brain_state.db")
            brain.state = BrainState(path=state_path)
            logger.info("BrainState wired (path=%s)", state_path)

        # ── Plugin System ────────────────────────────────────────────
        if cfg.inject_plugins:
            from hbllm.plugin.manager import PluginManager

            extra_dirs = [Path(d) for d in (cfg.plugin_dirs or [])]
            brain.plugin_manager = PluginManager(
                plugin_dirs=extra_dirs,
                skill_registry=brain.skill_registry,
                policy_engine=brain.policy_engine,
                knowledge_base=brain.knowledge_base,
            )
            # Auto-discover plugins from all configured paths
            discovered = await brain.plugin_manager.discover_plugins()
            if discovered:
                logger.info(
                    "Plugin system: loaded %d bundles on startup",
                    len(discovered),
                )

            # Optionally start background watcher for hot-loading
            if cfg.watch_plugins:
                await brain.plugin_manager.watch_directories()
                logger.info("Plugin watcher started (runtime hot-loading enabled)")

        return brain

    @staticmethod
    async def _build_composite_brain(
        llm_provider: LLMProvider,
        llm: ProviderLLM,
        cfg: BrainConfig,
        message_bus: MessageBus,
        registry: ServiceRegistry,
    ) -> Brain:
        """
        v4: Build brain using 8 composite nodes instead of 27 individual ones.

        Each composite internally creates and wires its sub-nodes, preserving
        all bus subscriptions for backward compatibility.
        """
        from hbllm.brain.composites import (
            GovernanceGuard,
            LearningLoop,
            MemorySystem,
            MetaCognition,
            ReasoningCore,
            ResourceManager,
            SkillEngine,
            SocialLayer,
        )
        from hbllm.modules.domain_registry import DomainRegistry
        from hbllm.security.trust import TrustInterceptor

        skill_registry = SkillRegistry(data_dir=cfg.data_dir)
        domain_registry = cfg.domain_registry or DomainRegistry()

        # Wire Trust Interceptor (Trust Model Pt 1)
        message_bus.add_interceptor(TrustInterceptor(registry=registry))

        # Wire Tenant Interceptor (ambient context propagation)
        from hbllm.security.tenant_interceptor import TenantInterceptor

        message_bus.add_interceptor(TenantInterceptor())

        # Auto-discover sub-domain LoRA adapters from data/lora/
        lora_dir = Path(cfg.data_dir) / "lora"
        if lora_dir.is_dir():
            from hbllm.modules.domain_registry import DomainSpec

            for adapter_dir in sorted(lora_dir.iterdir()):
                if adapter_dir.is_dir() and not domain_registry.exists(adapter_dir.name):
                    domain_registry.register(
                        DomainSpec(
                            name=adapter_dir.name,
                            centroid_text=f"Topics relating to {adapter_dir.name.replace('.', ' ')}",
                        )
                    )
                    logger.info("Auto-discovered sub-domain LoRA: %s", adapter_dir.name)

        nodes: list[Node] = []

        # 1. ReasoningCore
        reasoning = None
        if cfg.inject_reasoning:
            reasoning = ReasoningCore(
                llm=llm,
                policy_engine=None,  # Set after governance is created
                domain_registry=domain_registry,
                branch_factor=cfg.planner_branch_factor,
                max_depth=cfg.planner_max_depth,
                data_dir=cfg.data_dir,
            )

        # 2. MemorySystem
        memory_sys = None
        if cfg.inject_memory_system:
            memory_sys = MemorySystem(
                llm=llm, registry=registry, db_path=Path(cfg.data_dir) / "working_memory.db"
            )

        # 3. GovernanceGuard (created before MetaCognition so policy_engine is available)
        governance = None
        if cfg.inject_governance:
            governance = GovernanceGuard()

        # 4. MetaCognition
        meta = None
        if cfg.inject_meta_cognition:
            meta = MetaCognition(
                cognitive_metrics=None,  # Wired below
                goal_manager=None,
                self_model=None,
                skill_registry=skill_registry,
                data_dir=cfg.data_dir,
            )

        # 5. SkillEngine
        skills = None
        if cfg.inject_skill_engine:
            skills = SkillEngine(llm=llm, skill_registry=skill_registry)

        # 6. ResourceManager
        resources = None
        if cfg.inject_resources:
            resources = ResourceManager(
                data_dir=cfg.data_dir,
            )

        # 7. SocialLayer
        social = None
        if cfg.inject_social:
            social = SocialLayer(skill_registry=skill_registry)

        # 8. LearningLoop
        learning = None
        if cfg.inject_learning:
            learning = LearningLoop(llm=llm)

        # Start all composite nodes
        composites = [
            reasoning,
            memory_sys,
            governance,
            meta,
            skills,
            resources,
            social,
            learning,
        ]
        for composite in composites:
            if composite is not None:
                await _register_node(registry, composite)
                await composite.start(message_bus)
                nodes.append(composite)

        # Perception nodes (optional — require ML models)
        if cfg.inject_perception:
            from hbllm.perception.audio_in_node import AudioInputNode
            from hbllm.perception.audio_out_node import AudioOutputNode
            from hbllm.perception.vision_node import VisionNode

            for pnode in [
                AudioInputNode(node_id="audio_in"),
                AudioOutputNode(node_id="audio_out"),
                VisionNode(node_id="vision"),
            ]:
                await _register_node(registry, pnode)
                await pnode.start(message_bus)
                nodes.append(pnode)

        # Reasoning nodes (optional — require extra dependencies)
        if cfg.inject_fuzzy_logic:
            from hbllm.actions.fuzzy_node import FuzzyNode

            fnode = FuzzyNode(node_id="fuzzy", llm=llm)
            await _register_node(registry, fnode)
            await fnode.start(message_bus)
            nodes.append(fnode)

        if cfg.inject_symbolic_logic:
            from hbllm.actions.logic_node import LogicNode

            lnode = LogicNode(node_id="logic", llm=llm)
            await _register_node(registry, lnode)
            await lnode.start(message_bus)
            nodes.append(lnode)

        # Browser Node (DuckDuckGo search + scraping)
        if cfg.inject_browser:
            from hbllm.actions.browser_node import BrowserNode

            bnode = BrowserNode(node_id="browser")
            await _register_node(registry, bnode)
            await bnode.start(message_bus)
            nodes.append(bnode)
            logger.info("BrowserNode wired (web search & scrape)")

        # Execution Node (sandboxed python execution)
        if cfg.inject_execution:
            from hbllm.actions.execution_node import ExecutionNode

            enode = ExecutionNode(node_id="execution")
            await _register_node(registry, enode)
            await enode.start(message_bus)
            nodes.append(enode)
            logger.info("ExecutionNode wired (sandboxed python execution)")

        # Host shell execution node
        if cfg.inject_shell:
            from hbllm.actions.shell_node import HostShellNode

            shell_node = HostShellNode(
                node_id="shell_executor",
                workspace_dir=None,
                require_manual_approval=os.getenv("HBLLM_REQUIRE_SHELL_APPROVAL", "true").lower()
                == "true",
                policy_engine=governance.policy_engine if governance else None,
            )
            await _register_node(registry, shell_node)
            await shell_node.start(message_bus)
            nodes.append(shell_node)
            logger.info(
                "HostShellNode wired (manual approval=%s)", shell_node.require_manual_approval
            )

        # Register and start default DomainModuleNode instances
        if (
            type(llm_provider).__name__ == "LocalProvider"
            and getattr(llm_provider, "_model", None) is not None
        ):
            from hbllm.modules.base_module import DomainModuleNode

            model = llm_provider._model
            tokenizer = llm_provider._tokenizer
            for domain in ["general", "coding", "math"]:
                dnode = DomainModuleNode(
                    node_id=f"domain_{domain}",
                    domain_name=domain,
                    model=model,
                    tokenizer=tokenizer,
                    lora_state_dict=None,
                )
                await _register_node(registry, dnode)
                await dnode.start(message_bus)
                nodes.append(dnode)
        else:
            from hbllm.modules.base_module import DomainModuleNode

            for domain in ["general", "coding", "math"]:
                dnode = DomainModuleNode(
                    node_id=f"domain_{domain}",
                    domain_name=domain,
                    llm=llm,
                )
                await _register_node(registry, dnode)
                await dnode.start(message_bus)
                nodes.append(dnode)

        # Create pipeline
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
            "Brain created (composite mode) with %s provider, %d composite nodes",
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

        # Wire composite references
        brain.reasoning_core = reasoning
        brain.memory_system = memory_sys
        brain.governance_guard = governance
        brain.meta_cognition = meta
        brain.skill_engine = skills
        brain.resource_manager = resources
        brain.social_layer = social
        brain.learning_loop = learning

        # Wire backward-compatible attributes from composites
        brain.skill_registry = skill_registry
        brain.tool_memory = ToolMemory(data_dir=cfg.data_dir)
        brain.concept_extractor = ConceptExtractor()
        brain.world_state = WorldStateEngine()
        brain.cognition_router = CognitionRouter()
        brain.reward_model = RewardModel(data_dir=cfg.data_dir)
        brain.policy_optimizer = PolicyOptimizer()
        brain.interaction_miner = AsyncInteractionMiner(data_dir=cfg.data_dir)

        # Map composite sub-components to legacy Brain attributes
        if governance:
            brain.policy_engine = governance.policy_engine
            brain.sentinel = governance.sentinel
            if cfg.inject_revision:
                brain.confidence_estimator = governance.confidence_estimator

        if reasoning and cfg.inject_revision:
            brain.revision_node = reasoning.revision

        if meta:
            if cfg.inject_evaluation:
                brain.evaluation_node = meta.evaluation
            if cfg.inject_reflection:
                brain.reflection_node = meta.reflection

        if skills:
            if cfg.inject_skill_compiler:
                brain.skill_compiler_node = skills.compiler
            brain.skill_intelligence_node = skills.intelligence
            brain.failure_analyzer_node = skills.failure_analyzer

        if resources:
            brain.attention_manager = resources.attention
            brain.load_manager = resources.load_manager
            brain.scheduler_node = resources.scheduler

        if cfg.inject_goals:
            brain.goal_manager = GoalManager(data_dir=cfg.data_dir)

        if cfg.inject_self_model:
            brain.self_model = SelfModel(data_dir=cfg.data_dir)

        if cfg.inject_metrics:
            brain.cognitive_metrics = CognitiveMetrics(data_dir=cfg.data_dir)

        # Wire references in MetaCognition composite and its sub-nodes
        if meta:
            meta._cognitive_metrics = brain.cognitive_metrics
            meta._goal_manager = brain.goal_manager
            meta._self_model = brain.self_model
            if meta.evaluation:
                meta.evaluation.cognitive_metrics = brain.cognitive_metrics
                meta.evaluation.goal_manager = brain.goal_manager
                meta.evaluation.self_model = brain.self_model
            if meta.reflection:
                meta.reflection.cognitive_metrics = brain.cognitive_metrics
                meta.reflection.goal_manager = brain.goal_manager
                meta.reflection.self_model = brain.self_model

        if cfg.inject_cost_optimizer:
            brain.token_optimizer = TokenOptimizer()

        if cfg.inject_owner_rules:
            brain.owner_rules = OwnerRuleStore(db_path=str(Path(cfg.data_dir) / "owner_rules.db"))

        # Cognitive Awareness
        if cfg.inject_awareness:
            from hbllm.brain.awareness import CognitiveAwareness

            awareness_node = CognitiveAwareness(node_id="cognitive_awareness")
            await _register_node(registry, awareness_node)
            await awareness_node.start(message_bus)
            brain.awareness = awareness_node
            nodes.append(awareness_node)

        # Knowledge Base
        if cfg.inject_knowledge:
            from hbllm.knowledge import KnowledgeBase

            kb_dir = str(Path(cfg.data_dir) / "knowledge")
            brain.knowledge_base = KnowledgeBase(data_dir=kb_dir)

        # Persistence
        if cfg.inject_persistence:
            from hbllm.persistence import BrainState

            state_path = str(Path(cfg.data_dir) / "brain_state.db")
            brain.state = BrainState(path=state_path)

        # Plugin System
        if cfg.inject_plugins:
            from hbllm.plugin.manager import PluginManager

            extra_dirs = [Path(d) for d in (cfg.plugin_dirs or [])]
            brain.plugin_manager = PluginManager(
                plugin_dirs=extra_dirs,
                skill_registry=brain.skill_registry,
                policy_engine=brain.policy_engine,
                knowledge_base=brain.knowledge_base,
            )
            discovered = await brain.plugin_manager.discover_plugins()
            if discovered:
                logger.info(
                    "Plugin system: loaded %d bundles on startup",
                    len(discovered),
                )
            if cfg.watch_plugins:
                await brain.plugin_manager.watch_directories()

        logger.info(
            "v4 composite brain ready: %d top-level nodes (was 27+ in legacy mode)",
            len(nodes),
        )

        return brain
