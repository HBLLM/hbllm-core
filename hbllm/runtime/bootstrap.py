"""Canonical composition root for the HBLLM Cognitive OS.

Provides `build_cognitive_system()` as the single factory for instantiating
a fully-governed, verified cognitive architecture. All entry points
(Mind CLI, Serve, MyAgent, CognitiveBlackbox) must boot through this root.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.runtime.dialogue_service import DialogueService, _format_capability_result

logger = logging.getLogger("hbllm.runtime.bootstrap")

_DEFAULT_DIR_MARKER = object()

__all__ = [
    "CognitiveSystem",
    "CognitiveSystemConfig",
    "DialogueService",
    "_format_capability_result",
    "build_cognitive_system",
    "count_consolidated_skills",
]


def count_consolidated_skills(source: Any) -> int:
    """Count distinct consolidated homoiconic skills across workspace tiers and engine stores."""
    if source is None:
        return 0
    tiered = getattr(source, "tiered_workspace", None)
    engine = getattr(source, "engine", source)
    skill_ids: set[str] = set()
    if tiered is not None:
        for tier_ws in (tiered.persistent, tiered.brain, tiered.working_state):
            for node in tier_ws.graph.all_nodes():
                if str(getattr(node, "node_type", "")) == "skill" and (
                    getattr(node, "bytecode_stream", None) is not None
                    or node.id.startswith("procedural_skill_")
                ):
                    skill_ids.add(str(node.id))
    learner = getattr(engine, "library_learner", None)
    if learner is not None:
        skill_ids.update((getattr(learner, "promoted_skills", None) or {}).keys())
    wm = getattr(engine, "working_memory", None)
    skill_ids.update((getattr(wm, "procedural_library", None) or {}).keys())
    opt_eng = getattr(engine, "option_discovery", None)
    skill_ids.update((getattr(opt_eng, "discovered_options", None) or {}).keys())
    return len(skill_ids)


@dataclass
class CognitiveSystemConfig:
    """Configuration parameters for the unified cognitive system."""

    world_id: str = "default"
    seed: int | None = None
    exploration_budget: int = 150
    enable_logging: bool = True
    instructions: Sequence[str] | str | None = None
    memory_dir: Path | str | None = _DEFAULT_DIR_MARKER  # type: ignore[assignment]
    seed_ontology_path: Path | str | None = None
    tenant_id: str | None = None
    enable_grounding: bool = True
    perception_mode: str = "handcrafted"  # handcrafted | snn | learned
    enable_brain_llm: bool = False
    enable_autonomous_curiosity: bool = False
    load_core_priors: bool = True
    custom_verification_stages: list[Any] | None = None
    max_events_per_store: int | None = 2000


@dataclass
class CognitiveSystem:
    """Container holding all components of a booted cognitive runtime."""

    services: KernelServices
    engine: Any  # AutonomousEpistemicEngine
    bridge: Any | None  # CorticalGroundingBridge
    workspace: HCIRWorkspaceState
    tiered_workspace: TieredWorkspace
    config: CognitiveSystemConfig
    runtime: Any | None = None  # ExecutiveRuntime
    reasoning_runtime: Any | None = None  # UnifiedReasoningRuntime
    brain: Any | None = None
    hydrated_count: int = 0
    innate_priors_count: int = 0
    dialogue_service: DialogueService = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dialogue_service = DialogueService(
            services=self.services,
            workspace=self.workspace,
            tiered_workspace=self.tiered_workspace,
            config=self.config,
            bridge=self.bridge,
            brain=self.brain,
        )

    def attach_ontology_store(self, store: Any) -> None:
        """Attach or update a LexicalOntologyStore across language runtime, parsers, and capability executors."""
        self.dialogue_service.attach_ontology_store(store)

    def _execute_language_turn(self, turn: Any, lang_rt: Any, speaker: str) -> Any:
        """Delegate post-parse cognitive actions to DialogueService."""
        return self.dialogue_service.execute_language_turn(turn, lang_rt, speaker)

    def process_dialogue(
        self,
        dialogue: Any,
        speaker_id: str = "user",
        language: str | None = None,
    ) -> Any:
        """Delegate dialogue processing to DialogueService."""
        return self.dialogue_service.process_dialogue(
            dialogue, speaker_id=speaker_id, language=language
        )

    def consolidated_skills_count(self) -> int:
        """Return total distinct consolidated skills stored across TieredWorkspace and engine."""
        return count_consolidated_skills(self)


def build_cognitive_system(config: CognitiveSystemConfig | None = None) -> CognitiveSystem:
    """Build and wire the single, canonical cognitive runtime.

    Guarantees:
    1. A single verified TransactionManager owning HCIR state mutations.
    2. ConstitutionalVerifier attached to all state modification pipelines.
    3. AutonomousEpistemicEngine and CorticalGroundingBridge sharing the same KernelServices.
    4. TieredWorkspace owning 4-tier memory state.
    5. services.workspace is identical to tiered_workspace.working_state.
    """
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

    cfg = config or CognitiveSystemConfig()

    # 1. State hierarchy & storage (Persistent Tier binding defaults to ~/.hbllm/worlds/<world_id>)
    resolved_mem_dir: Path | None = None
    if cfg.memory_dir is _DEFAULT_DIR_MARKER:
        resolved_mem_dir = Path.home() / ".hbllm" / "worlds" / cfg.world_id
    elif cfg.memory_dir is not None:
        resolved_mem_dir = Path(cfg.memory_dir)

    persistent_store = None
    if resolved_mem_dir is not None:
        from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore

        resolved_mem_dir.mkdir(parents=True, exist_ok=True)
        persistent_store = SQLiteEventStore(
            resolved_mem_dir / "persistent.db",
            max_events=cfg.max_events_per_store,
            synchronous_full=True,
            tenant_id=cfg.tenant_id,
        )

    tiered_workspace = TieredWorkspace(
        persistent_store=persistent_store,
        tenant_id=cfg.tenant_id,
    )
    # Unify workspace: services.workspace IS tiered_workspace.working_state
    workspace = tiered_workspace.working_state

    # 2. Constitutional verification pipeline
    from hbllm.hcir.types import Scope

    authority_registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=authority_registry)
    default_scope = Scope(tenant_id=cfg.tenant_id) if cfg.tenant_id else None
    pipeline = cfg.custom_verification_stages or create_default_pipeline(
        default_scope=default_scope,
        constitutional_verifier=verifier,
    )

    # 3. Provision kernel services container (Single composition root)
    services = KernelServices.create_default(
        workspace=workspace,
        verification_pipeline=pipeline,
        tiered_workspace=tiered_workspace,
        commit_authority_registry=authority_registry,
    )
    services.commit_authority_registry = authority_registry

    innate_priors_count = 0
    if cfg.load_core_priors:
        from hbllm.hcir.seed.loader import hydrate_core_priors

        innate_priors_count = hydrate_core_priors(
            tiered_workspace,
            grant=getattr(tiered_workspace, "_system_grant", None),
            tenant_id=cfg.tenant_id,
        )

    # 4. Instantiate Autonomous Epistemic Engine with shared KernelServices
    engine = AutonomousEpistemicEngine(
        exploration_budget=cfg.exploration_budget,
        enable_logging=cfg.enable_logging,
        instructions=cfg.instructions,
        services=services,
        world_id=cfg.world_id,
        seed=cfg.seed,
        tiered_workspace=tiered_workspace,
    )

    # Auto-hydrate world model & lifelong memory if persistent state exists
    hydrated_count = 0
    if persistent_store is not None or tiered_workspace.persistent.graph.node_count > 0:
        hydrated_count = engine.hydrate_from_workspace(tiered_workspace)
        if hydrated_count > 0:
            logger.info(
                "Hydrated %d nodes into epistemic engine from persistent tier",
                hydrated_count,
            )

    # 5. Wire Grounding Bridge with the exact same KernelServices only when enabled
    bridge: Any | None = None
    if cfg.enable_grounding:
        from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge

        bridge = CorticalGroundingBridge(
            workspace=workspace,
            services=services,
            transaction_manager=services.transaction_manager,
            perception_mode=cfg.perception_mode,
        )
        engine.attach_grounding_bridge(bridge)

    # 6. Instantiate Executive Runtime and wire dual rhythms + classical reasoning CPU + SNN/Language subsystems
    from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
    from hbllm.hcir.language.acquisition import LexiconAcquisitionLoop
    from hbllm.hcir.reasoning.operators.registry import create_default_operator_registry
    from hbllm.hcir.reasoning.unified_runtime import UnifiedReasoningRuntime
    from hbllm.hcir.snn.neuromodulation import NeuromodulationEngine

    reasoning_rt = UnifiedReasoningRuntime(create_default_operator_registry())
    neuromod = NeuromodulationEngine()
    lex_acq = LexiconAcquisitionLoop(workspace.graph)
    services.reasoning_runtime = reasoning_rt
    services.neuromodulation = neuromod
    services.lexicon_acquisition = lex_acq
    services.extensions["reasoning_runtime"] = reasoning_rt
    services.extensions["neuromodulation"] = neuromod
    services.extensions["lexicon_acquisition"] = lex_acq

    if cfg.enable_autonomous_curiosity:
        from hbllm.hcir.capabilities.epistemic_curiosity import EpistemicCuriosityController

        curiosity_ctrl = EpistemicCuriosityController(
            capability_resolver=services.capability_resolver,
            tiered_workspace=tiered_workspace,
        )
        services.curiosity_controller = curiosity_ctrl
        services.epistemic_curiosity = curiosity_ctrl
        services.extensions["epistemic_curiosity"] = curiosity_ctrl
        for cap_name in ("knowledge.explain", "knowledge.explain_why", "knowledge.explain_how"):
            for impl in services.capability_resolver.list_implementations(cap_name):
                if hasattr(impl.executor, "_curiosity_controller"):
                    impl.executor._curiosity_controller = curiosity_ctrl

    runtime = ExecutiveRuntime(services=services)
    runtime._state.tenant_id = cfg.tenant_id or "default"
    runtime.controller.attach_reasoning_runtime(reasoning_rt)
    runtime.attach_engine(engine)
    if bridge is not None:
        runtime.attach_observation_bus(bridge.bus)
        if getattr(bridge, "concept_learner", None) is not None:
            runtime.attach_concept_learner(bridge.concept_learner)
    if getattr(engine, "episodic_cortex", None):
        runtime.attach_episodic_cortex(engine.episodic_cortex)
    if getattr(engine, "library_learner", None):
        runtime.attach_library_learner(engine.library_learner)
        services.extensions["library_learner"] = engine.library_learner

    logger.info(
        "Cognitive system booted successfully (world_id=%s, seed=%s, verifier=ConstitutionalVerifier)",
        cfg.world_id,
        cfg.seed,
    )

    system = CognitiveSystem(
        services=services,
        engine=engine,
        bridge=bridge,
        workspace=services.workspace,
        tiered_workspace=tiered_workspace,
        config=cfg,
        runtime=runtime,
        reasoning_runtime=runtime.controller.reasoning_runtime,
        hydrated_count=hydrated_count,
        innate_priors_count=innate_priors_count,
    )
    if cfg.seed_ontology_path is not None:
        from hbllm.hcir.stores.ontology_store import LexicalOntologyStore

        seed_path = Path(cfg.seed_ontology_path)
        if not seed_path.exists():
            raise FileNotFoundError(f"seed_ontology_path not found: {seed_path}")
        system.attach_ontology_store(LexicalOntologyStore(seed_path))
    return system
