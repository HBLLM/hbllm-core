"""
Cognitive Subsystem Wiring — shared between legacy and composite brain paths.

Extracts the subsystem wiring logic from BrainFactory._build_brain and
_build_composite_brain into reusable functions. Both paths call these
functions identically, eliminating 200+ lines of duplication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hbllm.actions.tool_memory import ToolMemory
from hbllm.brain.emotion.goal_manager import GoalManager
from hbllm.brain.governance.owner_rules import OwnerRuleStore
from hbllm.brain.self_model.cognitive_metrics import CognitiveMetrics
from hbllm.brain.self_model.confidence_estimator import ConfidenceEstimator
from hbllm.brain.self_model.self_model import SelfModel
from hbllm.brain.skills.skill_registry import SkillRegistry
from hbllm.brain.world.world_state import WorldStateEngine
from hbllm.data.interaction_miner import AsyncInteractionMiner
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.network.cognition_router import CognitionRouter
from hbllm.network.node import Node
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.token_optimizer import TokenOptimizer
from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import RewardModel

if TYPE_CHECKING:
    from hbllm.brain.core.factory import Brain, BrainConfig
    from hbllm.network.bus import MessageBus

logger = logging.getLogger(__name__)


async def _register_node(registry: Any, node: Node) -> None:
    """Helper to register a node and mark it healthy upon startup."""
    from hbllm.network.node import HealthStatus, NodeHealth

    await registry.register(node.get_info())
    await registry.update_health(NodeHealth(node_id=node.node_id, status=HealthStatus.HEALTHY))


async def wire_always_on_subsystems(
    brain: Brain,
    cfg: BrainConfig,
    *,
    dual_router: Any | None = None,
) -> None:
    """Wire the always-on subsystems that both legacy and composite paths share."""
    brain.skill_registry = brain.skill_registry or SkillRegistry(data_dir=cfg.data_dir)
    brain.tool_memory = ToolMemory(data_dir=cfg.data_dir)
    brain.concept_extractor = ConceptExtractor()

    from hbllm.perception.event_log import EventLog

    event_log = EventLog(data_dir=cfg.data_dir)
    brain.event_log = event_log
    brain.world_state = WorldStateEngine(event_log=event_log)

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
        logger.info("[Factory] Dual LLM Router attached to brain")


async def wire_optional_subsystems(
    brain: Brain,
    cfg: BrainConfig,
    nodes: list[Node],
    registry: ServiceRegistry,
    message_bus: MessageBus,
    *,
    policy_engine: Any | None = None,
    sentinel_node: Any | None = None,
    llm: Any | None = None,
) -> None:
    """Wire optional cognitive subsystems based on config flags.

    This handles the long chain of `if cfg.inject_*` blocks that are
    identical between the legacy and composite brain paths.
    """
    from hbllm.brain.emotion.reflection_node import ReflectionNode
    from hbllm.brain.evaluation.evaluation_node import EvaluationNode
    from hbllm.brain.evaluation.revision_node import RevisionNode
    from hbllm.brain.skills.skill_compiler_node import SkillCompilerNode

    # Task Graph Runtime
    if cfg.inject_task_graph:
        from hbllm.brain.autonomy.task_graph import TaskGraphRuntime

        brain.task_graph = TaskGraphRuntime(data_dir=cfg.data_dir)

    # Autonomy Watchers
    if cfg.inject_autonomy_watchers:
        await _wire_autonomy_watchers(brain, cfg)

    # Causal Graph
    if cfg.inject_causal_graph:
        from hbllm.brain.causality.causal_graph import CausalGraph

        brain.causal_graph = CausalGraph(data_dir=cfg.data_dir)

    # Cognitive Compaction
    if cfg.inject_compaction:
        from hbllm.brain.compaction.engine import CognitiveCompactionEngine

        brain.compaction_engine = CognitiveCompactionEngine()

    # Embodiment (OS adapter + verifier)
    if cfg.inject_embodiment:
        from hbllm.brain.embodiment.os_adapter import OSAdapter
        from hbllm.brain.embodiment.verifier import ExecutionVerifier

        brain.os_adapter = OSAdapter()
        brain.verifier = ExecutionVerifier(brain.os_adapter)

    # Human Control (permissions + tracer + guard)
    if cfg.inject_human_control:
        from hbllm.brain.control.guard import SecurityGuard
        from hbllm.brain.control.permissions import PermissionRegistry
        from hbllm.brain.observability.tracer import DecisionTraceLedger

        brain.permission_registry = PermissionRegistry()
        brain.decision_tracer = DecisionTraceLedger(data_dir=cfg.data_dir)
        brain.security_guard = SecurityGuard(brain.permission_registry, brain.decision_tracer)

    # Mesh registry
    if cfg.inject_mesh:
        from hbllm.brain.mesh.registry import NodeRegistry, NodeType

        brain.mesh_registry = NodeRegistry(local_node_id="local", local_node_type=NodeType.EDGE)

    # Configurable scalar subsystems
    if cfg.inject_revision:
        brain.confidence_estimator = ConfidenceEstimator()
        brain.revision_node = RevisionNode()

    if cfg.inject_goals:
        brain.goal_manager = GoalManager(data_dir=cfg.data_dir)

    if cfg.inject_self_model:
        brain.self_model = SelfModel(data_dir=cfg.data_dir)

    if cfg.inject_metrics:
        brain.cognitive_metrics = CognitiveMetrics(data_dir=cfg.data_dir)

    if cfg.inject_cost_optimizer:
        brain.token_optimizer = TokenOptimizer()

    if cfg.inject_policy_engine:
        brain.policy_engine = policy_engine

    if cfg.inject_owner_rules:
        brain.owner_rules = OwnerRuleStore(db_path=str(Path(cfg.data_dir) / "owner_rules.db"))

    if cfg.inject_sentinel and sentinel_node:
        brain.sentinel = sentinel_node

    # v2: Intelligence Feedback Loop nodes
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
            user_model=getattr(brain, "user_model_engine", None),
        )
        await _register_node(registry, compiler_node)
        await compiler_node.start(message_bus)
        brain.skill_compiler_node = compiler_node
        nodes.append(compiler_node)
        logger.info("v2: SkillCompilerNode wired (auto-skill extraction)")

    if cfg.inject_failure_analyzer:
        from hbllm.brain.evaluation.failure_analyzer_node import FailureAnalyzerNode

        fail_node = FailureAnalyzerNode(node_id="failure_analyzer", llm=llm)
        await _register_node(registry, fail_node)
        await fail_node.start(message_bus)
        brain.failure_analyzer_node = fail_node
        nodes.append(fail_node)
        logger.info("FailureAnalyzerNode wired (automated skill repair)")

    if cfg.inject_sil:
        from hbllm.brain.skills.skill_intelligence_node import SkillIntelligenceNode

        sil_node = SkillIntelligenceNode(node_id="sil", skill_registry=brain.skill_registry)  # type: ignore[arg-type]
        await _register_node(registry, sil_node)
        await sil_node.start(message_bus)
        brain.skill_intelligence_node = sil_node
        nodes.append(sil_node)
        logger.info("SkillIntelligenceNode wired (execution governor & lifecycle)")

    # v2: Resource Intelligence
    if cfg.inject_attention:
        from hbllm.brain.self_model.attention_manager import AttentionManager

        attn_node = AttentionManager(node_id="attention")
        await _register_node(registry, attn_node)
        await attn_node.start(message_bus)
        brain.attention_manager = attn_node
        nodes.append(attn_node)
        logger.info("v2: AttentionManager wired (memory budgets & focus)")

    if cfg.inject_load_manager:
        from hbllm.brain.control.load_manager import LoadManager

        load_node = LoadManager(node_id="load_manager", monitor_interval=60.0)
        await _register_node(registry, load_node)
        await load_node.start(message_bus)
        brain.load_manager = load_node
        nodes.append(load_node)
        logger.info("v2: LoadManager wired (resource monitoring & degradation)")

    # v3: Proactive Execution
    if cfg.inject_scheduler:
        from hbllm.brain.control.scheduler_node import SchedulerNode

        sched_node = SchedulerNode(node_id="scheduler", data_dir=cfg.data_dir)
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


async def wire_late_subsystems(
    brain: Brain,
    cfg: BrainConfig,
    nodes: list[Node],
    registry: ServiceRegistry,
    message_bus: MessageBus,
) -> None:
    """Wire subsystems that must be created after the main wiring phase.

    These depend on brain attributes set during wire_optional_subsystems.
    """
    # Cognitive Awareness
    if cfg.inject_awareness:
        from hbllm.brain.self_model.awareness import CognitiveAwareness

        awareness_node = CognitiveAwareness(node_id="cognitive_awareness")
        await _register_node(registry, awareness_node)
        await awareness_node.start(message_bus)
        brain.awareness = awareness_node
        nodes.append(awareness_node)
        logger.info("CognitiveAwareness wired (brain self-monitoring active)")

    # Knowledge Base
    if cfg.inject_knowledge:
        from hbllm.knowledge import KnowledgeBase

        kb_dir = str(Path(cfg.data_dir) / "knowledge")
        brain.knowledge_base = KnowledgeBase(data_dir=kb_dir)
        logger.info("KnowledgeBase wired (data_dir=%s)", kb_dir)

    # Persistence (BrainState)
    if cfg.inject_persistence:
        from hbllm.persistence import BrainState

        state_path = str(Path(cfg.data_dir) / "brain_state.db")
        brain.state = BrainState(path=state_path)
        logger.info("BrainState wired (path=%s)", state_path)

    # Plugin System (must come last — depends on skill_registry, policy_engine, knowledge_base)
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
            logger.info("Plugin system: loaded %d bundles on startup", len(discovered))

        if cfg.watch_plugins:
            await brain.plugin_manager.watch_directories()
            logger.info("Plugin watcher started (runtime hot-loading enabled)")


async def _wire_autonomy_watchers(brain: Brain, cfg: BrainConfig) -> None:
    """Wire autonomy watchers (filesystem, health, idle, calendar)."""
    try:
        from hbllm.brain.autonomy.watchers.filesystem_watcher import FilesystemWatcher
        from hbllm.brain.autonomy.watchers.idle_detector import IdleDetector
        from hbllm.brain.autonomy.watchers.system_health_watcher import SystemHealthWatcher

        brain._autonomy_watchers = []

        if cfg.autonomy_watch_dirs:
            fs_watcher = FilesystemWatcher(watch_dirs=list(cfg.autonomy_watch_dirs))
            brain._autonomy_watchers.append(("fs_watcher", fs_watcher))

        health_watcher = SystemHealthWatcher()
        brain._autonomy_watchers.append(("system_health", health_watcher))

        idle_detector = IdleDetector()
        brain._autonomy_watchers.append(("idle_detector", idle_detector))

        if cfg.autonomy_calendar_dir:
            from hbllm.brain.autonomy.watchers.calendar_watcher import CalendarWatcher

            cal_watcher = CalendarWatcher(calendar_dir=cfg.autonomy_calendar_dir)
            brain._autonomy_watchers.append(("calendar", cal_watcher))

        logger.info(
            "[Factory] Registered %d autonomy watchers",
            len(brain._autonomy_watchers),
        )
    except Exception as e:
        logger.warning("[Factory] Failed to wire autonomy watchers: %s", e)
