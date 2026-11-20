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
from hbllm.hcir.language.core.semantic_frame import FrameType
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
    enable_autonomous_curiosity: bool = False
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

    def process_dialogue(self, dialogue: Any, speaker_id: str = "user") -> Any:
        """Process natural language dialogue via Pure-HCIR first, with optional scaffold fallback."""
        from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope

        if isinstance(dialogue, ObservationEnvelope) and self.bridge is not None:
            return self.bridge.process_dialogue(dialogue, speaker_id=speaker_id)

        text = (
            str(dialogue.metadata.get("text", ""))
            if isinstance(dialogue, ObservationEnvelope)
            else str(dialogue)
        )
        speaker = (
            str(dialogue.metadata.get("speaker_id", speaker_id))
            if isinstance(dialogue, ObservationEnvelope)
            else speaker_id
        )

        # 1. Native non-LLM HCIR Language Runtime (A16/A17)
        from hbllm.hcir.language.runtime import MultilingualLanguageRuntime

        lang_rt = self.services.extensions.get("language_runtime")
        if lang_rt is None:
            lang_rt = MultilingualLanguageRuntime(
                self.workspace.graph,
                workspace=self.workspace,
            )
            # Wire tiered_workspace into the gateway for epistemic prior lookups
            if self.tiered_workspace is not None:
                lang_rt._gateway._tiered_workspace = self.tiered_workspace
            self.services.extensions["language_runtime"] = lang_rt

        turn = lang_rt.process_utterance(text, speaker=speaker)
        metrics = self.services.extensions.setdefault(
            "autonomy_metrics",
            {
                "parse_attempts": 0,
                "parse_hcir_success": 0,
                "realize_attempts": 0,
                "realize_hcir_success": 0,
                "shadow_invocations": 0,
                "distilled_lessons": 0,
            },
        )
        if hasattr(metrics, "record_parse"):
            metrics.record_parse(turn.is_success)
            metrics.record_realize(bool(turn.is_success and turn.response_text))
        else:
            metrics["parse_attempts"] += 1
            if turn.is_success:
                metrics["parse_hcir_success"] += 1
            metrics["realize_attempts"] += 1
            if turn.is_success and turn.response_text:
                metrics["realize_hcir_success"] += 1

        if turn.is_success:
            neuromod = self.services.extensions.get("neuromodulation")
            if neuromod is not None and hasattr(neuromod, "signal_reward"):
                neuromod.signal_reward(0.2)

            # Autonomous epistemic curiosity fallback for unknown QUERY frames
            curiosity = self.services.extensions.get("epistemic_curiosity")
            if (
                curiosity is not None
                and turn.semantic_frame.frame_type == FrameType.QUERY
                and turn.epistemic_state is not None
                and not turn.epistemic_state.is_known
                and turn.epistemic_state.target_subject
            ):
                import asyncio

                q_target = turn.semantic_frame.query_target or "definition"
                q_mode = (
                    "why"
                    if q_target == "cause"
                    else ("how" if q_target == "procedure" else "definition")
                )
                try:
                    try:
                        q_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        q_loop = None

                    if q_loop is not None and q_loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            cur_res = pool.submit(
                                asyncio.run,
                                curiosity.research_concept(
                                    turn.epistemic_state.target_subject,
                                    query_mode=q_mode,
                                    uncertainty=float(turn.epistemic_state.uncertainty),
                                ),
                            ).result(timeout=30.0)
                    else:
                        cur_res = asyncio.run(
                            curiosity.research_concept(
                                turn.epistemic_state.target_subject,
                                query_mode=q_mode,
                                uncertainty=float(turn.epistemic_state.uncertainty),
                            )
                        )
                    if cur_res.researched:
                        # Re-query gateway from freshly populated persistent memory
                        re_state = lang_rt._gateway._lookup_epistemic_prior(
                            turn.epistemic_state.target_subject,
                            query_target=q_target,
                        )
                        if re_state is not None and re_state.is_known:
                            turn.epistemic_state = re_state
                            realizer = lang_rt._realizers.get(turn.language)
                            if realizer is not None:
                                turn.response_text = realizer.realize(
                                    re_state, original_frame=turn.semantic_frame
                                )
                        elif cur_res.summary:
                            turn.response_text = cur_res.summary
                        if cur_res.sources:
                            turn.response_text = (
                                f"{turn.response_text}\n\n"
                                f"[Learned autonomously via web research "
                                f"({cur_res.propositions_ingested} facts ingested); "
                                f"source: {cur_res.sources[0]}]"
                            )
                except Exception as exc:
                    logger.debug("Autonomous curiosity query fallback skipped: %s", exc)

            # If COMMAND frame parsed successfully, dispatch through governed capabilities
            if (
                turn.semantic_frame.frame_type == FrameType.COMMAND
                and turn.semantic_frame.predicate
            ):
                from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher

                dispatcher = self.services.extensions.get("command_dispatcher")
                if dispatcher is None:
                    dispatcher = CommandDispatcher(
                        self.services.capability_resolver,
                        workspace=self.workspace,
                    )
                    self.services.extensions["command_dispatcher"] = dispatcher

                import asyncio

                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop is not None and loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            receipt = pool.submit(
                                asyncio.run, dispatcher.dispatch(turn.semantic_frame)
                            ).result(timeout=30.0)
                    else:
                        receipt = asyncio.run(dispatcher.dispatch(turn.semantic_frame))

                    # Augment the turn result with execution receipt
                    if receipt.success:
                        result_summary = receipt.result.get("content", "")
                        if not result_summary:
                            entries = receipt.result.get("entries", [])
                            if entries:
                                result_summary = "\n".join(
                                    e.get("name", str(e)) for e in entries[:20]
                                )
                        if not result_summary:
                            result_summary = str(receipt.result)

                        turn.response_text = (
                            f"Command executed: {receipt.verb} → {receipt.capability}\n"
                            f"{result_summary}"
                        )
                    else:
                        turn.response_text = f"Command '{receipt.verb}' failed: {receipt.error}"
                        turn.is_success = False
                except Exception as exc:
                    turn.response_text = f"Command dispatch error: {exc}"
                    turn.is_success = False

            return turn

        neuromod = self.services.extensions.get("neuromodulation")
        if neuromod is not None and hasattr(neuromod, "signal_novelty"):
            neuromod.signal_novelty(0.3)

        # 2. Optional isolated LLM teacher scaffold (retired in Pure-HCIR v2.0 graduation)
        scaffold = self.services.extensions.get("llm_scaffold")
        if (
            scaffold is not None
            and not getattr(scaffold, "retired", False)
            and self.config.enable_brain_llm
        ):
            scaffold_res = scaffold.advise_and_distill(text, self.workspace, speaker_id=speaker)
            if not hasattr(metrics, "record_parse"):
                metrics["shadow_invocations"] += 1
                if scaffold_res.get("distilled"):
                    metrics["distilled_lessons"] += 1
            return scaffold_res

        if self.bridge is not None:
            return self.bridge.process_dialogue(dialogue, speaker_id=speaker_id)
        return turn


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
        local_t4 = Path("checkpoints/t4_reading")
        if local_t4.exists() and (local_t4 / "persistent.db").exists():
            resolved_mem_dir = local_t4
        else:
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
        )

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
    services.extensions["reasoning_runtime"] = reasoning_rt
    services.extensions["neuromodulation"] = NeuromodulationEngine()
    services.extensions["lexicon_acquisition"] = LexiconAcquisitionLoop(workspace.graph)

    if cfg.enable_autonomous_curiosity:
        from hbllm.hcir.capabilities.epistemic_curiosity import EpistemicCuriosityController

        curiosity_ctrl = EpistemicCuriosityController(
            capability_resolver=services.capability_resolver,
            tiered_workspace=tiered_workspace,
        )
        services.extensions["epistemic_curiosity"] = curiosity_ctrl
        # Attach to registered knowledge.explain* MemoryCapabilityExecutors
        for cap_name in ("knowledge.explain", "knowledge.explain_why", "knowledge.explain_how"):
            for impl in services.capability_resolver.list_implementations(cap_name):
                if hasattr(impl.executor, "_curiosity_controller"):
                    impl.executor._curiosity_controller = curiosity_ctrl

    runtime = ExecutiveRuntime(services=services)
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
