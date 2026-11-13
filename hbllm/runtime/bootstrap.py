"""Canonical composition root for the HBLLM Cognitive OS.

Provides `build_cognitive_system()` as the single factory for instantiating
a fully-governed, verified cognitive architecture. All entry points
(Mind CLI, Serve, MyAgent, CognitiveBlackbox) must boot through this root.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger("hbllm.runtime.bootstrap")

_DEFAULT_DIR_MARKER = object()


@dataclass
class CognitiveSystemConfig:
    """Configuration parameters for the unified cognitive system."""

    world_id: str = "default"
    seed: int | None = None
    exploration_budget: int = 150
    enable_logging: bool = True
    instructions: Sequence[str] | str | None = None
    memory_dir: Path | str | None = _DEFAULT_DIR_MARKER  # type: ignore[assignment]
    enable_grounding: bool = True
    perception_mode: str = "handcrafted"  # handcrafted | snn | learned
    enable_brain_llm: bool = False
    custom_verification_stages: list[Any] | None = None


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

    def process_dialogue(self, dialogue: Any, speaker_id: str = "user") -> Any:
        """Processes a natural language dialogue command or request through the grounded cognitive bridge."""
        if self.bridge is not None:
            return self.bridge.process_dialogue(dialogue, speaker_id=speaker_id)
        from hbllm.hcir.compiler import HCIRCompiler
        from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope

        compiler = HCIRCompiler()
        if isinstance(dialogue, ObservationEnvelope):
            text = str(dialogue.metadata.get("text", ""))
            speaker = str(dialogue.metadata.get("speaker_id", speaker_id))
        else:
            text = str(dialogue)
            speaker = speaker_id
        return compiler.interpret_dialogue(text, speaker_id=speaker)


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
    from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge

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
        persistent_store = SQLiteEventStore(resolved_mem_dir / "persistent.db")

    tiered_workspace = TieredWorkspace(persistent_store=persistent_store)
    # Unify workspace: services.workspace IS tiered_workspace.working_state
    workspace = tiered_workspace.working_state

    # 2. Constitutional verification pipeline
    authority_registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=authority_registry)
    pipeline = cfg.custom_verification_stages or create_default_pipeline(
        constitutional_verifier=verifier
    )

    # 3. Provision kernel services container (Single composition root)
    services = KernelServices.create_default(
        workspace=workspace,
        verification_pipeline=pipeline,
        tiered_workspace=tiered_workspace,
        commit_authority_registry=authority_registry,
    )
    services.commit_authority_registry = authority_registry

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

    # 5. Wire Grounding Bridge with the exact same KernelServices
    bridge: CorticalGroundingBridge | None = None
    if cfg.enable_grounding:
        bridge = CorticalGroundingBridge(
            workspace=workspace,
            services=services,
            transaction_manager=services.transaction_manager,
            perception_mode=cfg.perception_mode,
        )
        engine.attach_grounding_bridge(bridge)

    # 6. Instantiate Executive Runtime and wire dual rhythms
    from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime

    runtime = ExecutiveRuntime(services=services)
    runtime.attach_engine(engine)
    if bridge is not None:
        runtime.attach_observation_bus(bridge.bus)
        if getattr(bridge, "concept_learner", None) is not None:
            runtime.attach_concept_learner(bridge.concept_learner)
    if getattr(engine, "episodic_cortex", None):
        runtime.attach_episodic_cortex(engine.episodic_cortex)
    if getattr(engine, "library_learner", None):
        runtime.attach_library_learner(engine.library_learner)

    logger.info(
        "Cognitive system booted successfully (world_id=%s, seed=%s, verifier=ConstitutionalVerifier)",
        cfg.world_id,
        cfg.seed,
    )

    return CognitiveSystem(
        services=services,
        engine=engine,
        bridge=bridge,
        workspace=services.workspace,
        tiered_workspace=tiered_workspace,
        config=cfg,
        runtime=runtime,
        reasoning_runtime=runtime.controller.reasoning_runtime,
        hydrated_count=hydrated_count,
    )
