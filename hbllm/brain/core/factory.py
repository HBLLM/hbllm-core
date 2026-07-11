"""
Brain Factory — one-line setup for the full cognitive pipeline.

Creates all brain nodes (Router, Planner, Critic, Decision, Workspace,
Memory) with an LLM provider injected, wires them to a message bus,
and returns a running Brain instance.

Usage::

    from hbllm.brain.core.factory import BrainFactory

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from hbllm.knowledge import KnowledgeBase
    from hbllm.plugin.manager import PluginManager

from hbllm.actions.tool_memory import ToolMemory
from hbllm.brain.control.load_manager import LoadManager
from hbllm.brain.core.provider_adapter import ProviderLLM
from hbllm.brain.emotion.goal_manager import GoalManager
from hbllm.brain.emotion.reflection_node import ReflectionNode

# v2: Intelligence Feedback Loop
from hbllm.brain.evaluation.evaluation_node import EvaluationNode
from hbllm.brain.evaluation.revision_node import RevisionNode
from hbllm.brain.governance.owner_rules import OwnerRuleStore
from hbllm.brain.governance.policy_engine import PolicyEngine

# v2: Resource Intelligence
from hbllm.brain.self_model.attention_manager import AttentionManager
from hbllm.brain.self_model.cognitive_metrics import CognitiveMetrics
from hbllm.brain.self_model.confidence_estimator import ConfidenceEstimator
from hbllm.brain.self_model.self_model import SelfModel
from hbllm.brain.skills.skill_compiler_node import SkillCompilerNode

# New cognitive modules
from hbllm.brain.skills.skill_registry import SkillRegistry
from hbllm.brain.world.world_state import WorldStateEngine
from hbllm.data.interaction_miner import AsyncInteractionMiner
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.cognition_router import CognitionRouter
from hbllm.network.node import HealthStatus, Node, NodeHealth
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.pipeline import CognitivePipeline, PipelineConfig, PipelineResult
from hbllm.serving.provider import LLMProvider, get_provider
from hbllm.serving.token_optimizer import TokenOptimizer
from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import RewardModel

logger = logging.getLogger(__name__)


def _is_slow_cpu() -> bool:

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


class BrainConfig(BaseModel):
    """Configuration for Brain creation.

    Validates all fields and supports env var overrides via HBLLM_* prefix.
    """

    model_config = ConfigDict(
        validate_default=True,
        arbitrary_types_allowed=True,
    )

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

    # ── Cognitive subsystems ─────────────────────────
    inject_user_model: bool = True
    inject_project_graph: bool = True
    inject_executive_cortex: bool = True
    inject_relationship_memory: bool = True
    inject_reality_graph: bool = True
    inject_autonomy_manager: bool = True

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
    inject_audit_trail: bool = True  # Immutable action log with hash chain
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
    total_timeout: float = Field(default_factory=_default_total_timeout, gt=0)
    api_timeout: float = Field(default_factory=_default_api_timeout, gt=0)
    stream_timeout: float = Field(default_factory=_default_stream_timeout, gt=0)
    planner_branch_factor: int = Field(default=3, ge=1, le=10)
    planner_max_depth: int = Field(default=2, ge=1, le=5)
    data_dir: str = Field(
        default_factory=lambda: os.environ.get("HBLLM_DATA_DIR", "data"),
        min_length=1,
    )
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

    # Cognitive Stream: SNN-driven comprehension pipeline
    inject_comprehension: bool = True

    # Autonomy watchers (environment awareness)
    inject_autonomy_watchers: bool = True
    autonomy_watch_dirs: list[str] | None = None  # Directories for filesystem watcher
    autonomy_calendar_dir: str | None = None  # Directory for .ics calendar files

    # IoT / Home Automation (MQTT bridge)
    inject_iot: bool = False  # MqttIoTNode (requires paho-mqtt and MQTT broker)
    iot_mqtt_broker: str = "localhost"
    iot_mqtt_port: int = 1883

    # Autonomous Learning Engine (goal-driven learning)
    inject_autonomous_learning: bool = True

    # Live World State (environment graph fed by perception + IoT)
    inject_world_state: bool = True

    # Dual LLM routing (local + external)
    external_provider: str | None = (
        None  # e.g. "openai/gpt-4o" or "anthropic/claude-sonnet-4-20250514"
    )
    external_provider_kwargs: dict[str, Any] = Field(default_factory=dict)
    dual_llm_complexity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # ── Horizontal Scaling ────────────────────────────────────────
    bus_backend: str = Field(
        default_factory=lambda: os.environ.get("HBLLM_BUS_BACKEND", "memory"),
        description="Message bus backend: 'memory' (default), 'redis', or 'nats'",
    )
    redis_url: str = Field(
        default_factory=lambda: os.environ.get("HBLLM_REDIS_URL", ""),
        description="Redis URL for distributed bus/registry (required when bus_backend='redis')",
    )


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

        # ── Autonomy subsystem (cognitive heartbeat) ──────────────
        self.autonomy_core: Any = None  # AutonomyCore
        self.notification_gateway: Any = None  # NotificationGateway
        self.proactive_processor: Any = None  # ProactiveProcessor
        self.sse_channel: Any = None  # SSEChannel
        self.emotion_engine: Any = None  # EmotionEngine

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
        self.autonomy_manager: Any = None

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

        # Core safety subsystems (Phase 4-5 wiring)
        self.audit_trail: Any | None = None  # AuditTrail (immutable action log)
        self.pii_redactor: Any | None = None  # PIIRedactor (memory write filter)
        self.restraint_engine: Any | None = None  # RestraintEngine (action gating)
        self.offline_manager: Any | None = None  # OfflineManager (network degradation)
        self.voice_auth: Any | None = None  # VoiceAuthenticator (speaker ID)
        self.rollback_registry: Any | None = None  # RollbackRegistry (undo mappings)

        # Plugin system
        self.plugin_manager: PluginManager | None = None

        self._hardware_loop_task: asyncio.Task[None] | None = None

        # Graceful shutdown drain
        self._draining = False
        self._active_requests = 0
        self._drain_event = asyncio.Event()

    @property
    def is_draining(self) -> bool:
        """True if the brain is shutting down and rejecting new requests."""
        return self._draining

    def acquire_request(self) -> bool:
        """Track an in-flight request. Returns False if draining."""
        if self._draining:
            return False
        self._active_requests += 1
        return True

    def release_request(self) -> None:
        """Release an in-flight request. Signals drain if count reaches 0."""
        self._active_requests = max(0, self._active_requests - 1)
        if self._draining and self._active_requests == 0:
            self._drain_event.set()

    async def process(
        self,
        text: str,
        tenant_id: str = "default",
        session_id: str = "default",
    ) -> PipelineResult:
        """Send a query through the full cognitive pipeline."""
        if self._draining:
            return PipelineResult(
                text="Service is shutting down. Please retry shortly.",
                correlation_id="drain",
                error=True,
            )
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

    async def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Stop all nodes, pipeline, and bus with graceful drain.

        Args:
            drain_timeout: Seconds to wait for in-flight requests to complete.
        """
        # Phase 1: Drain — stop accepting new requests, wait for in-flight
        self._draining = True
        if self._active_requests > 0:
            logger.info(
                "Draining %d in-flight requests (timeout=%.0fs)",
                self._active_requests,
                drain_timeout,
            )
            self._drain_event.clear()
            try:
                await asyncio.wait_for(self._drain_event.wait(), timeout=drain_timeout)
                logger.info("All in-flight requests drained")
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "Drain timeout reached with %d requests still active, forcing shutdown",
                    self._active_requests,
                )

        # Phase 2: Shutdown
        if self._hardware_loop_task:
            self._hardware_loop_task.cancel()
            try:
                await self._hardware_loop_task
            except asyncio.CancelledError:
                pass
        # Stop proactive processor
        if self.proactive_processor:
            try:
                await self.proactive_processor.stop()
            except Exception:
                logger.debug("Error stopping proactive processor during shutdown", exc_info=True)
        # Stop autonomy core (cognitive heartbeat)
        if self.autonomy_core:
            try:
                await self.autonomy_core.stop()
            except Exception:
                logger.debug("Error stopping autonomy core during shutdown", exc_info=True)
        # Stop world state engine
        if self.world_state:
            try:
                await self.world_state.stop()
            except Exception:
                logger.debug("Error stopping world state during shutdown", exc_info=True)
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


def _wire_comprehension_stream(
    router_node: Any,
    domain_registry: Any,
    neuromodulator: Any | None = None,
) -> None:
    """Wire the Cognitive Stream comprehension pipeline into a RouterNode.

    Delegated to hbllm.brain.wiring.snn for maintainability.
    """
    from hbllm.brain.wiring.snn import wire_comprehension_stream

    wire_comprehension_stream(router_node, domain_registry, neuromodulator=neuromodulator)


def _wire_expression_stream(
    decision_node: Any,
    router_node: Any | None = None,
    llm: Any | None = None,
    dual_router: Any | None = None,
    neuromodulator: Any | None = None,
) -> None:
    """Wire the expression-side Cognitive Stream into a DecisionNode.

    Delegated to hbllm.brain.wiring.snn for maintainability.
    """
    from hbllm.brain.wiring.snn import wire_expression_stream

    wire_expression_stream(
        decision_node, router_node, llm, dual_router, neuromodulator=neuromodulator
    )


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

        # 1b. Create dual LLM router if external provider is configured
        dual_router = None
        _neuromodulator = None  # Created later if SNN streams are injected
        if cfg.external_provider:
            try:
                external_provider = get_provider(
                    cfg.external_provider, **cfg.external_provider_kwargs
                )
                external_llm = ProviderLLM(external_provider, system_prompt=cfg.system_prompt)

                from hbllm.brain.control.dual_llm_router import DualLLMRouter

                dual_router = DualLLMRouter(
                    local=llm,
                    external=external_llm,
                    complexity_threshold=cfg.dual_llm_complexity_threshold,
                )
                logger.info(
                    "[Factory] Dual LLM Router: local=%s, external=%s",
                    llm_provider.name,
                    external_provider.name,
                )
            except Exception as e:
                logger.warning("[Factory] Failed to create external provider: %s", e)

        # 2. Create bus and registry (configurable backend)
        if bus is not None:
            message_bus = bus
        elif cfg.bus_backend == "redis" and cfg.redis_url:
            from hbllm.network.redis_bus import RedisBus

            message_bus = RedisBus(redis_url=cfg.redis_url)
            logger.info("[Factory] Using RedisBus for horizontal scaling")
        elif cfg.bus_backend == "nats":
            from hbllm.network.nats_bus import NatsBus

            message_bus = NatsBus()
            logger.info("[Factory] Using NatsBus for horizontal scaling")
        else:
            message_bus = InProcessBus()
        await message_bus.start()

        if cfg.bus_backend == "redis" and cfg.redis_url:
            from hbllm.network.redis_registry import RedisRegistry

            registry = RedisRegistry(redis_url=cfg.redis_url)
            logger.info("[Factory] Using RedisRegistry for distributed service discovery")
        else:
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
                dual_router=dual_router,
            )

        # 3. Create cognitive nodes with LLM injected (legacy path)
        from hbllm.brain.wiring.nodes import create_legacy_nodes

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

        skill_registry = SkillRegistry(data_dir=cfg.data_dir)

        nodes = create_legacy_nodes(
            llm=llm,
            llm_provider=llm_provider,
            cfg=cfg,
            policy_engine=policy_engine,
            domain_registry=domain_registry,
            skill_registry=skill_registry,
            dual_router=dual_router,
        )

        # Find sentinel node from created nodes (for subsystem wiring)
        sentinel_node = None
        for n in nodes:
            if getattr(n, "node_id", None) == "sentinel":
                sentinel_node = n
                break

        # 4. Start all nodes on the bus (parallel for faster boot)
        await asyncio.gather(*(node.start(message_bus) for node in nodes))
        for node in nodes:
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

        # 6. Wire cognitive subsystems (extracted to wiring/subsystems.py)
        from hbllm.brain.wiring.subsystems import (
            wire_always_on_subsystems,
            wire_late_subsystems,
            wire_optional_subsystems,
        )

        brain.skill_registry = skill_registry
        await wire_always_on_subsystems(brain, cfg, dual_router=dual_router)
        await wire_optional_subsystems(
            brain,
            cfg,
            nodes,
            registry,
            message_bus,
            policy_engine=policy_engine,
            sentinel_node=sentinel_node,
            llm=llm,
        )
        await wire_late_subsystems(brain, cfg, nodes, registry, message_bus)

        return brain

    @staticmethod
    async def _build_composite_brain(
        llm_provider: LLMProvider,
        llm: ProviderLLM,
        cfg: BrainConfig,
        message_bus: MessageBus,
        registry: ServiceRegistry,
        dual_router: Any = None,
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

        # 5. SkillEngine + MechanismStore
        skills = None
        if cfg.inject_skill_engine:
            from hbllm.brain.emotion.mechanism_store import MechanismStore

            mechanism_store = MechanismStore(data_dir=cfg.data_dir)
            skills = SkillEngine(
                llm=llm,
                skill_registry=skill_registry,
                mechanism_store=mechanism_store,
            )

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

        # Wire Cognitive Stream comprehension into the ReasoningCore's inner RouterNode
        if cfg.inject_comprehension and reasoning is not None and reasoning.router is not None:
            # Create shared NeuromodulationEngine for SNN streams
            from hbllm.brain.snn.neuromodulation import NeuromodulationEngine

            neuromodulator = NeuromodulationEngine()
            # Will be wired to UserModel after it's created (see below)
            _neuromodulator = neuromodulator

            _wire_comprehension_stream(
                reasoning.router,
                domain_registry,
                neuromodulator=neuromodulator,
            )
            # Wire expression-side Cognitive Stream (Layer 5) into the DecisionNode
            if reasoning.decision is not None:
                _wire_expression_stream(
                    reasoning.decision,
                    reasoning.router,
                    llm,
                    dual_router=dual_router,
                    neuromodulator=neuromodulator,
                )

        # ── Autonomous Learning Engine ────────────────────────────────────
        _cognitive_graph_ref = None  # Will be wired to GoalManager later
        _autonomous_learner_ref = None  # Will be wired to GoalManager later
        if cfg.inject_autonomous_learning:
            try:
                from hbllm.brain.causality.causal_model_builder import CausalModelBuilder
                from hbllm.brain.evaluation.failure_analyzer import FailureAnalyzer
                from hbllm.brain.learning.autonomous_learner import AutonomousLearner
                from hbllm.brain.learning.experiment_engine import ExperimentEngine
                from hbllm.brain.learning.learning_subsystem import CognitiveGraph
                from hbllm.brain.learning.meta_learner import MetaLearner
                from hbllm.brain.reasoning.belief_store import BeliefStore
                from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine
                from hbllm.brain.reasoning.contradiction_detector import (
                    BeliefRevisionEngine,
                    ContradictionDetector,
                )

                learning_data_dir = f"{cfg.data_dir}/learning"

                # Get mechanism_store from SkillEngine if available
                mechanism_store = skills.mechanism_store if skills is not None else None

                # Build learning subsystems
                causal_builder = CausalModelBuilder(
                    llm=llm,
                    mechanism_store=mechanism_store,
                    data_dir=learning_data_dir,
                )
                experiment_engine = ExperimentEngine(
                    llm=llm,
                    data_dir=learning_data_dir,
                )
                contradiction_detector = ContradictionDetector(llm=llm)
                belief_engine = BeliefRevisionEngine(data_dir=learning_data_dir)
                meta_learner = MetaLearner(data_dir=learning_data_dir)
                concept_engine = ConceptFormationEngine(
                    llm=llm,
                    causal_model_builder=causal_builder,
                    mechanism_store=mechanism_store,
                    belief_store=BeliefStore(data_dir=learning_data_dir),
                    data_dir=learning_data_dir,
                )

                # Persistent belief storage (Phase 3) — use same instance
                belief_store = concept_engine.belief_store

                # Build shared CognitiveGraph (was LearningSubsystem)
                learning_subsystem = CognitiveGraph(
                    mechanism_store=mechanism_store,
                    failure_analyzer=FailureAnalyzer(),
                    belief_engine=belief_engine,
                    contradiction_detector=contradiction_detector,
                    meta_learner=meta_learner,
                    causal_model_builder=causal_builder,
                    belief_store=belief_store,
                    concept_engine=concept_engine,
                )

                # Store reference for later GoalManager wiring
                _cognitive_graph_ref = learning_subsystem

                # Wire LearningSubsystem into SkillEngine's LearningEventHandler
                if skills is not None:
                    skills.inject_learning_subsystem(learning_subsystem)

                # Create and register the orchestrator node
                autonomous_learner = AutonomousLearner(
                    node_id="autonomous_learner",
                    llm=llm,
                    causal_model_builder=causal_builder,
                    experiment_engine=experiment_engine,
                    contradiction_detector=contradiction_detector,
                    belief_engine=belief_engine,
                    meta_learner=meta_learner,
                    concept_engine=concept_engine,
                )
                _autonomous_learner_ref = autonomous_learner
                await _register_node(registry, autonomous_learner)
                await autonomous_learner.start(message_bus)
                nodes.append(autonomous_learner)

                logger.info(
                    "Autonomous Learning Engine wired: "
                    "CausalModelBuilder, ExperimentEngine, "
                    "ContradictionDetector, BeliefRevision, "
                    "MetaLearner, ConceptFormation, "
                    "CognitiveGraph (shared)"
                )
            except Exception as e:
                logger.warning("Autonomous Learning Engine init failed (non-critical): %s", e)

        # Perception nodes (optional — require ML models)
        if cfg.inject_perception:
            from hbllm.perception.audio_in_node import AudioInputNode
            from hbllm.perception.audio_out_node import AudioOutputNode
            from hbllm.perception.perception_fuser import PerceptionFuser
            from hbllm.perception.vision_node import VisionNode
            from hbllm.perception.voice_config import AudioPipelineConfig

            audio_config = AudioPipelineConfig()
            for pnode in [
                AudioInputNode(
                    node_id="audio_in",
                    config=audio_config,
                ),
                AudioOutputNode(
                    node_id="audio_out",
                    config=audio_config,
                    data_dir=cfg.data_dir,
                ),
                VisionNode(node_id="vision"),
            ]:
                await _register_node(registry, pnode)
                await pnode.start(message_bus)
                nodes.append(pnode)

            # Cross-modal perception fusion
            fuser = PerceptionFuser(bus=message_bus)
            for fusion_topic in [
                "perception.audio",
                "perception.vision",
                "perception.screen",
                "sensory.audio.in",
                "sensory.vision.in",
            ]:
                await message_bus.subscribe(fusion_topic, fuser.on_perception_event)
            logger.info("PerceptionFuser wired for cross-modal fusion")

            # Wake word detection (hands-free activation)
            from hbllm.perception.wake_word import WakeWordDetector

            wake_word = WakeWordDetector(node_id="wake_word_detector")
            await _register_node(registry, wake_word)
            await wake_word.start(message_bus)
            nodes.append(wake_word)

            # Voice streaming bridge (ExpressionStream → AudioOutNode)
            from hbllm.perception.voice_stream_bridge import VoiceStreamBridge

            voice_bridge = VoiceStreamBridge(node_id="voice_stream_bridge")
            await _register_node(registry, voice_bridge)
            await voice_bridge.start(message_bus)
            nodes.append(voice_bridge)

            # Location awareness (geofencing)
            from hbllm.perception.location_adapter import LocationAdapter

            location = LocationAdapter(node_id="location_adapter")
            await _register_node(registry, location)
            await location.start(message_bus)
            nodes.append(location)

            # Conversation turn manager (voice state machine)
            from hbllm.perception.conversation_turn import ConversationTurnManager

            turn_mgr = ConversationTurnManager(node_id="conversation_turn")
            await _register_node(registry, turn_mgr)
            await turn_mgr.start(message_bus)
            nodes.append(turn_mgr)
            logger.info("ConversationTurnManager wired (voice state machine)")

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

        # IoT / Home Automation (MQTT bridge — optional)
        if cfg.inject_iot:
            try:
                from hbllm.actions.iot_mqtt_node import MqttIoTNode

                iot_node = MqttIoTNode(
                    node_id="iot_mqtt",
                    broker_host=cfg.iot_mqtt_broker,
                    broker_port=cfg.iot_mqtt_port,
                )
                await _register_node(registry, iot_node)
                await iot_node.start(message_bus)
                nodes.append(iot_node)
                logger.info(
                    "MqttIoTNode wired (broker=%s:%d)",
                    cfg.iot_mqtt_broker,
                    cfg.iot_mqtt_port,
                )
            except ImportError:
                logger.info("IoT node not available (paho-mqtt not installed)")
            except Exception as e:
                logger.warning("Failed to start IoT node: %s", e)

        # Live World State Engine (environment graph)
        if cfg.inject_world_state:
            from hbllm.brain.world.world_state import WorldStateEngine
            from hbllm.perception.event_log import EventLog

            event_log = EventLog(data_dir=cfg.data_dir)
            world_state = WorldStateEngine(event_log=event_log)
            world_state.start()

            # Wire perception events → WorldStateEngine
            async def _on_perception_for_world(msg: Any) -> None:
                """Route normalized perception events to world state graph."""
                try:
                    from hbllm.perception.reality_bus import PerceptionEvent

                    payload = msg.payload
                    event = PerceptionEvent(
                        entity_id=payload.get("entity_id", msg.source_node_id),
                        event_type=payload.get("event_type", "update"),
                        payload=payload,
                        confidence=float(payload.get("confidence", 0.7)),
                    )
                    await world_state.handle_normalized_event(event)
                except Exception as e:
                    logger.debug("WorldState event ingestion failed: %s", e)

            for ws_topic in [
                "perception.normalized",
                "iot.event",
                "iot.discovery",
                "sensor.reading",
                "device.change",
            ]:
                await message_bus.subscribe(ws_topic, _on_perception_for_world)

            logger.info("WorldStateEngine wired — live environment graph active")
        else:
            world_state = None

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

        # Attach world state if wired
        if world_state is not None:
            brain.world_state = world_state

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

        # Dual LLM Router
        brain.dual_router = dual_router
        if dual_router is not None:
            autonomy = getattr(brain, "autonomy_core", None)
            if autonomy is not None:
                dual_router.state_machine = getattr(autonomy, "state_machine", None)
            logger.info("[Factory] Dual LLM Router attached to brain (composite)")

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

            # Wire GoalManager into CognitiveGraph and AutonomousLearner
            if _cognitive_graph_ref is not None:
                _cognitive_graph_ref.goal_manager = brain.goal_manager
            if _autonomous_learner_ref is not None:
                _autonomous_learner_ref.goal_manager = brain.goal_manager

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

        if cfg.inject_audit_trail:
            from hbllm.security.audit_trail import AuditTrail

            audit_trail = AuditTrail(
                db_path=str(Path(cfg.data_dir) / "audit_trail.db"),
            )
            await audit_trail.init_db()
            brain.audit_trail = audit_trail
            logger.info("AuditTrail wired — immutable action log active")

        # Cognitive Awareness
        if cfg.inject_awareness:
            from hbllm.brain.self_model.awareness import CognitiveAwareness

            awareness_node = CognitiveAwareness(node_id="cognitive_awareness")
            await _register_node(registry, awareness_node)
            await awareness_node.start(message_bus)
            brain.awareness = awareness_node
            nodes.append(awareness_node)

            # EmotionEngine — publishes emotion.state consumed by Awareness & PersonaEngine
            from hbllm.brain.emotion.emotion_engine import EmotionEngine

            emotion_node = EmotionEngine(node_id="emotion_engine")
            await _register_node(registry, emotion_node)
            await emotion_node.start(message_bus)
            brain.emotion_engine = emotion_node
            nodes.append(emotion_node)

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

            extra_dirs: list[Path | str] = [Path(d) for d in (cfg.plugin_dirs or [])]
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

        # ── Autonomy Core (cognitive heartbeat) ────────────────────
        from hbllm.brain.autonomy.loop import AutonomyCore

        autonomy = AutonomyCore(
            fast_path_topics=[
                "user.input",
                "user.action",
                "sensor.anomaly",
                "device.change",
                "system.critical",
                "perception.*",
                "iot.event",
                "iot.discovery",
            ],
        )
        await autonomy.start(message_bus)
        brain.autonomy_core = autonomy
        logger.info("AutonomyCore started — cognitive heartbeat active")

        # ── Proactive Output (autonomy → notifications) ──────────
        from hbllm.serving.notifications import NotificationGateway
        from hbllm.serving.proactive import ProactiveProcessor, SSEChannel

        gateway = NotificationGateway()
        sse = SSEChannel()
        proactive = ProactiveProcessor(
            gateway=gateway,
            pipeline=pipeline,
            sse_channel=sse,
        )
        await proactive.start(message_bus)

        # Route autonomy actions through the bus for proactive processing
        autonomy.set_action_handler(lambda msg: message_bus.publish(msg.topic, msg))

        brain.notification_gateway = gateway
        brain.proactive_processor = proactive
        brain.sse_channel = sse
        logger.info("ProactiveProcessor wired — notifications active")

        # ── Cognitive Subsystems ────────────────────────────────────

        # UserModel — predictive user understanding
        if cfg.inject_user_model:
            from hbllm.brain.social.user_model import UserModelEngine
            from hbllm.brain.social.user_model_node import UserModelNode

            user_model_engine = UserModelEngine(data_dir=cfg.data_dir)
            user_model_node = UserModelNode(
                node_id="user_model",
                user_model_engine=user_model_engine,
                data_dir=cfg.data_dir,
            )
            await _register_node(registry, user_model_node)
            await user_model_node.start(message_bus)
            nodes.append(user_model_node)
            brain.user_model_engine = user_model_engine
            brain.user_model_node = user_model_node
            logger.info("UserModel wired — predictive user understanding active")

            # Wire NeuromodulationEngine to UserModel updates (if SNN streams were created)
            if _neuromodulator is not None:
                try:

                    async def _sync_neuromod(msg: Any) -> None:
                        _neuromodulator.signal_from_user_model(user_model_engine, msg.tenant_id)

                    await message_bus.subscribe("user_model.updated", _sync_neuromod)
                    logger.info("NeuromodulationEngine wired to UserModel updates")
                except Exception as e:
                    logger.debug("Failed to wire neuromodulator to UserModel: %s", e)
        else:
            user_model_engine = None

        # ProjectGraph — persistent project cognition
        if cfg.inject_project_graph:
            from hbllm.brain.world.project_graph import ProjectGraph
            from hbllm.brain.world.project_node import ProjectNode

            project_graph = ProjectGraph(data_dir=cfg.data_dir)
            project_node = ProjectNode(
                node_id="project_graph",
                project_graph=project_graph,
                data_dir=cfg.data_dir,
            )
            await _register_node(registry, project_node)
            await project_node.start(message_bus)
            nodes.append(project_node)
            brain.project_graph = project_graph
            brain.project_node = project_node
            logger.info("ProjectGraph wired — persistent project cognition active")
        else:
            project_graph = None

        # ExecutiveCortex — unified cognitive control
        if cfg.inject_executive_cortex:
            from hbllm.brain.control.executive_cortex import ExecutiveCortex

            _goal_mgr = getattr(brain, "goal_manager", None)
            _load_mgr = getattr(brain, "load_manager", None)
            _attn_mgr = getattr(brain, "attention_manager", None)
            if not _goal_mgr:
                logger.debug("[Factory] ExecutiveCortex: goal_manager not available on brain")
            if not _load_mgr:
                logger.debug("[Factory] ExecutiveCortex: load_manager not available on brain")
            if not _attn_mgr:
                logger.debug("[Factory] ExecutiveCortex: attention_manager not available on brain")

            executive_cortex = ExecutiveCortex(
                goal_manager=_goal_mgr,
                load_manager=_load_mgr,
                attention_system=None,
                attention_manager=_attn_mgr,
                state_machine=None,
                user_model=user_model_engine,
            )
            brain.executive_cortex = executive_cortex
            logger.info("ExecutiveCortex wired — unified cognitive control active")

        # RelationshipMemory — KG-integrated social graph
        if cfg.inject_relationship_memory:
            from hbllm.brain.social.relationship_memory import RelationshipMemory
            from hbllm.brain.social.relationship_node import RelationshipNode

            kg = getattr(brain, "knowledge_base", None)
            kg_graph = getattr(kg, "graph", None) if kg else None
            relationship_memory = RelationshipMemory(
                knowledge_graph=kg_graph,
                data_dir=cfg.data_dir,
            )
            relationship_node = RelationshipNode(
                node_id="relationship_memory",
                relationship_memory=relationship_memory,
                data_dir=cfg.data_dir,
            )
            await _register_node(registry, relationship_node)
            await relationship_node.start(message_bus)
            nodes.append(relationship_node)
            brain.relationship_memory = relationship_memory
            brain.relationship_node = relationship_node
            logger.info("RelationshipMemory wired — social graph active")
        else:
            relationship_memory = None

        # AutonomyManager — proactive opportunity monitoring & routing
        if getattr(cfg, "inject_autonomy_manager", True):
            from hbllm.brain.autonomy.autonomy_manager import AutonomyManager
            from hbllm.brain.autonomy.opportunity_source import BatterySource, SilenceSource

            autonomy_db = str(Path(cfg.data_dir) / "opportunity_history.db")
            autonomy_manager = AutonomyManager(
                node_id="autonomy_manager",
                db_path=autonomy_db,
            )
            # Register opportunity sources
            autonomy_manager.register_source(SilenceSource())
            autonomy_manager.register_source(BatterySource())

            await _register_node(registry, autonomy_manager)
            await autonomy_manager.start(message_bus)
            nodes.append(autonomy_manager)
            brain.autonomy_manager = autonomy_manager
            logger.info("AutonomyManager wired — proactive opportunity routing active")
        else:
            brain.autonomy_manager = None

        # ── Interconnection Wiring ──────────────────────────────────
        # Wire cross-subsystem connections now that all engines exist.
        # This activates the bus topology: UserModel → PersonaEngine,
        # CuriosityNode, SocialTiming, ProactiveInsight, etc.

        # Wire user_model into DecisionNode
        decision_node = getattr(brain, "decision_node", None)
        if decision_node and user_model_engine:
            decision_node._user_model = user_model_engine
            logger.debug("Wired UserModel → DecisionNode (expertise-aware scoring)")

        # Wire user_model into CuriosityNode
        curiosity_node = next((n for n in nodes if getattr(n, "node_id", "") == "curiosity"), None)
        if curiosity_node and user_model_engine:
            curiosity_node._user_model = user_model_engine
            logger.debug("Wired UserModel → CuriosityNode (interest-weighted goals)")

        logger.info(
            "Interconnection wiring complete — user_model=%s, project_graph=%s, "
            "relationship_memory=%s",
            "active" if user_model_engine else "none",
            "active" if project_graph else "none",
            "active" if relationship_memory else "none",
        )

        # RealityGraph — unified world model facade
        if cfg.inject_reality_graph:
            from hbllm.brain.reasoning.reality_graph import RealityGraph

            reality_graph = RealityGraph(
                knowledge_graph=getattr(brain, "knowledge_base", None),
                brain_world_state=getattr(brain, "world_state", None),
                perception_world_state=None,
            )
            brain.reality_graph = reality_graph
            logger.info("RealityGraph wired — unified world model facade active")

        logger.info(
            "v4 composite brain ready: %d top-level nodes, autonomy=ACTIVE",
            len(nodes),
        )

        return brain
