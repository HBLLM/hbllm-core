"""Autonomous Epistemic World Engine — Domain-Agnostic Exploration, Causal Induction, and Mental Simulation.

Implements the fundamental cognitive loop:
1. Observe raw pixel frames I_t and available actions A.
2. Analyze pixel difference layouts Δ(I_t, I_{t-1}) to ground motor dynamics (avatar self-identification)
   and environmental state mutations (switches, gates, consumables).
3. If current knowledge is insufficient to reach the target (WIN), generate causal hypotheses
   and explore unknown entities via active curiosity-driven trial and error.
4. When sufficient rules are confirmed, simulate candidate action plans purely in mental imagination
   (forward search over learned world transition models).
5. Switch to exploitation and execute the validated winning plan directly.
"""

from __future__ import annotations

import copy
import logging
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

import hbllm.hcir.algorithms as algorithms
from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.graph import (
    CognitiveGraph,
    ConstraintNode,
    EventNode,
    GoalNode,
    ObservationNode,
)
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventBus
from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.stores import EventType
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TaskFrame, TieredWorkspace, WorkspaceTier
from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    CausalHypothesis,
    CausalPredicate,
)
from hbllm.hcir.world.extended_body_schema import (
    ExtendedBodySchema,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.interactive_action_discovery import ActionBudgetGovernor
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.object_state_graph import (
    ObjectStateGraphPlanner,
    RelationalGraphMatcher,
)
from hbllm.hcir.world.percept import (
    BasePerceptionAdapter,
    EmbodiedObservation,
    PerceptionFrame,
    PerceptionRegistry,
    TerminalSignal,
)
from hbllm.hcir.world.prediction_budget import PredictionBudget
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)
from hbllm.hcir.world.spatial_containment import RoomDoor
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from hbllm.perception.saccadic_attention import SaccadicAttentionSystem

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cognitive Phases and Cortical Faculties Modular Imports
# ─────────────────────────────────────────────────────────────────────────────


class EpistemicTruthValue(StrEnum):
    """W122: Three-valued logic distinguishing empirical falsity from epistemic unobserved ignorance."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class EpistemicObservation:
    """W121: Observation representation with calibrated uncertainty bounds and provenance."""

    variable: str
    value: Any
    uncertainty: float
    truth_value: EpistemicTruthValue
    source_faculty: str = "PERCEPTION"
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.confidence == 0.5 and self.uncertainty != 0.5:
            self.confidence = max(0.0, min(1.0, 1.0 - self.uncertainty))

    def to_causal_predicate(self) -> CausalPredicate:
        """Export observation as a declarative causal predicate."""
        return CausalPredicate(
            variable=self.variable,
            operator="==",
            value=self.value,
            relation=f"truth:{self.truth_value.value}:unc:{self.uncertainty:.2f}",
            target_entity=self.source_faculty,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert observation into transactional HCIR graph state mutations."""
        node_id = f"observation_{self.variable}_{abs(hash(str(self.value)))}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "variable": self.variable,
                    "value": str(self.value),
                    "uncertainty": self.uncertainty,
                    "truth_value": str(self.truth_value),
                    "source_faculty": self.source_faculty,
                },
            )
        ]


from hbllm.hcir.world.cortex_assimilator import (
    EpistemicFeedbackAssimilator,
    EpistemicPhase,
)
from hbllm.hcir.world.cortex_causal import (
    ActionAffordance,
    GovernedSetView,
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.cortex_deliberation import (
    BasalGangliaDecision,
    BasalGangliaGatingCircuit,
    NeuromodulatoryState,
    PrefrontalDeliberationEngine,
)
from hbllm.hcir.world.cortex_episodic import (
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.cortex_motor import (
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorCortexEffector,
)
from hbllm.hcir.world.cortex_perception import (
    EpistemicObservationDiff,
    PerceptionEngine,
)
from hbllm.hcir.world.cortex_planner import (
    MentalSimulationPlanner,
    MentalSimulationStep,
)
from hbllm.hcir.world.cortical_stage import WorkingMemoryDelta
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState


class _LazyFaculty:
    """Non-data descriptor that instantiates a faculty on first attribute access.

    Instances may still assign the attribute directly (e.g. to inject a configured
    faculty), which shadows the descriptor for that instance.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        value = self._factory()
        obj.__dict__[self._name] = value
        return value


class AutonomousEpistemicEngine:
    """Domain-agnostic cognitive engine for autonomous exploration, hypothesis testing,
    and forward mental simulation.
    """

    MIN_PROBES_PER_ACTION: int = 3
    consecutive_pacing_steps: int = 0

    # ── Lazily provisioned faculties ─────────────────────────────────────────
    # Faculties that are not on the per-step decision path are instantiated on
    # first access only, so an engine instance does not pay construction and
    # memory cost for capabilities a given episode never exercises.
    relational_matcher: RelationalGraphMatcher = _LazyFaculty(RelationalGraphMatcher)
    visual_symmetry: VisualSymmetryAnalyzer = _LazyFaculty(VisualSymmetryAnalyzer)

    def __init__(
        self,
        exploration_budget: int = 150,
        enable_logging: bool = True,
        instructions: Sequence[str] | str | None = None,
        services: KernelServices | None = None,
        world_id: str = "default",
        seed: int | None = None,
        tiered_workspace: TieredWorkspace | None = None,
    ) -> None:
        if services is None:
            from hbllm.hcir.kernel.services import KernelServices
            from hbllm.hcir.workspace import HCIRWorkspaceState

            services = KernelServices.create_default(HCIRWorkspaceState())

        self.exploration_budget = exploration_budget
        # Deterministic decide-path RNG (WP0.1 replay determinism)
        self._rng = __import__("numpy").random.default_rng(seed)
        self.enable_logging = enable_logging

        # Cognitive state & authoritative EngineState model (WP1)
        self.state: EngineState = EngineState(world_id=world_id)
        self._phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.phase_history: list[tuple[EpistemicPhase, EpistemicPhase, str]] = []
        self.world_id: str = world_id
        self._step_counter: int = 0
        self.environmental_step_limit: int | None = None
        # Anytime cognition budget (WP0.2): per-step deadline for faculties
        self.step_work_units: int = 400
        self.step_budget_ms: float = 150.0
        self.deterministic_budget: bool = True
        self.compute_budget: Any | None = None
        # Discovered lattice quantum (cells per cardinal displacement); 1 until calibrated
        self._spatial_lattice_stride: int = 1
        # WP0.3 motor-effect posterior: per-action counts over observed effect kinds
        self.action_effect_counts: dict[Any, dict[str, int]] = {}

        # Memory of frames & actions
        self._prev_grid: np.ndarray | None = None
        self._last_action: Any | None = None
        self._last_action_data: dict[str, Any] | None = None
        self.grid_history: deque[np.ndarray] = deque(maxlen=6)

        # Avatar self-model (Motor Grounding)
        self._avatar_feature: Any | None = None
        self._avatar_features: set[Any] = set()
        self._avatar_pos: tuple[int, int] | None = None
        self._avatar_size: int = 1
        self._effector_kinetics_mode: int | None = None
        self._last_surprise: float = 0.0

        # Intensional Neuro-Symbolic World Theory (SSOT for domain ontology)
        self.default_bg_feature: Any = 0
        self.symbolic_theory: HCIRSymbolicWorldTheory = HCIRSymbolicWorldTheory()
        self.symbolic_theory.seed_core_knowledge(bg_feature=self.default_bg_feature)

        # Prefrontal Working Memory & Executive Faculties (SSOT for cognitive state)
        self.working_memory: PrefrontalWorkingMemory = PrefrontalWorkingMemory()
        if instructions:
            self.load_instructions(instructions)

        # Bind biological faculties to working memory stores
        self.subgoal_stack: FrontopolarSubgoalStack = self.working_memory.subgoal_stack
        self.body_schema: ExtendedBodySchema = self.working_memory.body_schema
        self.habenular_ior: HabenularEpisodicIOR = self.working_memory.habenular_ior
        self.remote_causal: RemoteCausalAttributor = self.working_memory.remote_causal
        self.acc_conflict: AnteriorCingulateConflictMonitor = self.working_memory.acc_conflict

        # Motor calibration & dynamics
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        self.tested_actions: set[int] = set()
        self.tested_action_positions: set[tuple[int, tuple[int, int] | None]] = set()

        # State Mutation Models (Switches, buttons, door toggles)
        self.state_mutations: list[StateMutationModel] = []

        # Episodic Ground Facts (Context-partitioned spatial memory across attempts)
        self._current_context_id: str = "default"
        self._context_epoch: int = 0
        self.working_memory.visuospatial.context_epoch = 0
        self.habenular_ior.context_epoch = 0

        # Theory of Mind: creature kinds observed to patrol autonomously
        self.mobile_threat_features: set[int] = set()

        # Biological hazard tracker & intuitive physics
        self.hazard_tracker: SpatiotemporalHazardTracker = SpatiotemporalHazardTracker()
        self.saccadic_attention: SaccadicAttentionSystem = SaccadicAttentionSystem()

        # Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        self._bilateral_state: Any | None = None

        # Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment
        self.pending_phase_wait_steps: int = 0
        self.consecutive_pacing_steps: int = 0

        # Faculty: Premotor Spatiotemporal Collision Cones & Trajectory Extrapolation

        # Faculty: Orbitofrontal Cortex (OFC) Counterfactual Deadlock Detector

        # Faculty: Hippocampal & Lateral Habenula Episodic Memory (Trial-and-Error Loop)
        self.episodic_cortex: HippocampalEpisodicCortex = HippocampalEpisodicCortex(
            habenular_ior=self.habenular_ior,
        )

        # Faculty: vmPFC Causal Induction Cortex (Pearl's do(a) intervention & tool resonance)

        # Faculty: Anterior Cingulate Cortex (ACC) Motor Conflict Monitor
        self.acc_conflict_monitor: AnteriorCingulateConflictMonitor = (
            AnteriorCingulateConflictMonitor()
        )

        # Room topology & doorway subgoal reasoning (spatial containment)

        # Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
        self.object_planner: ObjectStateGraphPlanner = ObjectStateGraphPlanner()
        self.perception_adapter: BasePerceptionAdapter | None = None
        self.topology_rooms: dict[int, list[tuple[int, int]]] = {}
        self.topology_doors: list[RoomDoor] = []
        self.room_adjacency: dict[int, list[int]] = {}
        self.current_room_id: int | None = None

        # Confidence-scaled predictive coding surprise engine
        self.last_surprise: float = 0.0
        self.last_surprise_eval: Any | None = None

        # Free energy active inference decision selection
        self.prediction_budget: PredictionBudget = PredictionBudget()

        # Faculty: Prefrontal Deliberation Engine & Cognitive Arbitration
        self.deliberation_engine: PrefrontalDeliberationEngine = PrefrontalDeliberationEngine()

        # Faculty: Non-Rigid Topological Deformation Tracker & Entity Lineage (W059, W014)

        # Track 2: Four Neuromodulatory Systems (Dopamine, Noradrenaline, Acetylcholine, Serotonin)

        # Faculty: Hierarchical Predictive Coding Cortical Columns (Rao & Ballard / Friston)

        # Faculty: BA10 Metacognitive Confidence Calibration & Agency Monitor (Fleming & Dolan)

        # Faculty: Working Concept Store & Analogical Schema Retrieval (AGI Gap 6)
        from hbllm.hcir.grounding.concepts.store import WorkingConceptStore

        self.working_concept_store: WorkingConceptStore = WorkingConceptStore()

        # Faculty: Causal Schema Inducer & Latent Variable Discovery (AGI Gap 7)
        from hbllm.hcir.grounding.causal.schema import SchemaInducer

        self.schema_inducer: SchemaInducer = SchemaInducer()

        # Faculty: Temporal Option Discovery & Bottleneck Cuts (AGI Gap 11)
        from hbllm.hcir.world.option_discovery import OptionDiscoveryEngine

        self.option_discovery: OptionDiscoveryEngine = OptionDiscoveryEngine()

        # Faculty: Corrigibility Monitor (AGI Gap 15)
        from hbllm.hcir.kernel.safety_corrigibility import CorrigibilityMonitor

        self.corrigibility_monitor: CorrigibilityMonitor = CorrigibilityMonitor()

        # Faculty: Temporal State Buffer & State Difference Computer (W041-W047)

        # Faculty: Transition Sequence Model & Reversibility Engine (W048-W050)

        # Faculty: Interventional Causal Discovery Engine (W061-W070)

        # Faculty: Rule Induction & Latent Structure Engine (W071-W080)

        # Faculty: Whole-Grid Outcome Predictor & Rollout Engine (W081-W089)

        # Domain 13: Relational Confidence & Evidence Provenance (W121-W130)
        # Delegated to formal internal memory stores (symbolic_theory and working_memory)

        # ── Interactive World-Modeling Faculties (W161–W190) ──────────────────
        # Module 1: Ungrounded Action Discovery & Constrained Experimentation (W161-W165, W171, W187, W188)
        self.budget_governor: ActionBudgetGovernor = ActionBudgetGovernor(
            max_budget=self.exploration_budget
        )

        # Module 2: Open-Set Goal Induction & Reachability Deadlock Pruning (W168-W170, W174, W175, W181, W182)

        # Module 3: Temporal Dynamics, Dual-Store Memory & Objective Efficiency (W167, W172, W173, W176-W180, W189, W190)

        self.bg_feature: int = 0
        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []

        # External domain & benchmark extension registries (e.g. plugin adapters)
        self.target_scorers: list[Any] = []
        self.deliberation_policies: list[Any] = []

        # Tiered Cognitive Workspace & Transaction Manager / Kernel Services live mirroring
        self.services: KernelServices | None = services
        self.event_bus: KernelEventBus | None = (
            getattr(services, "event_bus", None) if services is not None else None
        )
        self.capability_resolver: CapabilityResolver | None = (
            getattr(services, "capability_resolver", None) if services is not None else None
        )
        self.tiered_workspace: TieredWorkspace | None = (
            tiered_workspace
            if tiered_workspace is not None
            else (getattr(services, "tiered_workspace", None) if services is not None else None)
        )
        self.transaction_manager: TransactionManager | None = (
            getattr(services, "transaction_manager", None) if services is not None else None
        )
        self.active_task_frame: TaskFrame | None = None
        if self.tiered_workspace is not None:
            self.attach_tiered_workspace(self.tiered_workspace)

        # Canonical 5-Stage Active Inference Pipeline (AGENTS.md §2, WP1)
        from hbllm.hcir.world.cortical_pipeline import CorticalPipeline

        self.cortical_pipeline: CorticalPipeline = CorticalPipeline(
            transaction_manager=self.transaction_manager,
            engine=self,
        )
        # Goal Arbiter (WP10, Gap 5)

        self.active_arbitrated_goal: Any | None = None
        self.grounding_bridge: Any | None = None
        self._entity_cache: dict[tuple[Any, ...], list[SpatialEntity]] = {}
        self._last_surprise_eval: Any | None = None
        self.rejected_delta_commits: int = 0
        self.rejected_faculty_commits: int = 0

    def attach_grounding_bridge(
        self,
        bridge: Any | None = None,
    ) -> Any:
        """Attaches an SGE CorticalGroundingBridge to bind SGE into the active inference cycle."""
        if bridge is not None:
            self.grounding_bridge = bridge
        else:
            from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge

            ws = self.services.workspace if self.services is not None else None
            self.grounding_bridge = CorticalGroundingBridge(
                workspace=ws,
                services=self.services,
                transaction_manager=self.transaction_manager,
                symbolic_theory=self.symbolic_theory,
            )
        self.cortical_pipeline.grounding_bridge = self.grounding_bridge
        return self.grounding_bridge

    # ─────────────────────────────────────────────────────────────────────────
    # Working Memory & Knowledge Base Properties (Single Source of Truth)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def verified_safe_features(self) -> GovernedSetView:
        """Empirically verified safe/walkable features in symbolic world theory."""
        return GovernedSetView(
            self.symbolic_theory.walkable_features,
            on_add=lambda f: self.symbolic_theory.induce_walkable(f, evidence="manual"),
            on_discard=lambda f: self.symbolic_theory.retract_walkable(f, reason="manual_discard"),
        )

    @verified_safe_features.setter
    def verified_safe_features(self, val: Any) -> None:
        if val:
            for f in val:
                self.symbolic_theory.induce_walkable(f, evidence="verified_safe_features_setter")

    @property
    def mental_plan(self) -> deque[MentalSimulationStep]:
        """Active mental simulation plan in prefrontal working memory."""
        return self.working_memory.mental_plan

    @mental_plan.setter
    def mental_plan(self, val: Any) -> None:
        d = deque(val) if not isinstance(val, deque) else val
        self.working_memory.mental_plan = d
        self.state.mental_plan = d

    @property
    def active_hypothesis(self) -> Any | None:
        """Active causal hypothesis evaluated in prefrontal working memory."""
        return self.working_memory.active_hypothesis

    @active_hypothesis.setter
    def active_hypothesis(self, val: Any | None) -> None:
        self.working_memory.active_hypothesis = val
        self.state.active_hypothesis = val

    @property
    def consecutive_simulation_failures(self) -> int:
        return self.working_memory.consecutive_simulation_failures

    @consecutive_simulation_failures.setter
    def consecutive_simulation_failures(self, val: int) -> None:
        self.working_memory.consecutive_simulation_failures = val
        self.state.consecutive_simulation_failures = val

    @property
    def simulation_cooldown(self) -> int:
        return self.working_memory.simulation_cooldown

    @simulation_cooldown.setter
    def simulation_cooldown(self, val: int) -> None:
        self.working_memory.simulation_cooldown = val
        self.state.simulation_cooldown = val

    @property
    def consecutive_plan_failures(self) -> int:
        return self.working_memory.consecutive_plan_failures

    @consecutive_plan_failures.setter
    def consecutive_plan_failures(self, val: int) -> None:
        self.working_memory.consecutive_plan_failures = val
        self.state.consecutive_plan_failures = val

    @property
    def exploration_cooldown(self) -> int:
        return self.working_memory.exploration_cooldown

    @exploration_cooldown.setter
    def exploration_cooldown(self, val: int) -> None:
        self.working_memory.exploration_cooldown = val
        self.state.exploration_cooldown = val

    @property
    def consecutive_stuck_steps(self) -> int:
        return self.working_memory.consecutive_stuck_steps

    @consecutive_stuck_steps.setter
    def consecutive_stuck_steps(self, val: int) -> None:
        self.working_memory.consecutive_stuck_steps = val
        self.state.consecutive_stuck_steps = val

    @property
    def context_barriers(self) -> dict[int, set[tuple[int, int]]]:
        """Spatial barrier coordinates mapped per context epoch in visuospatial working memory."""
        return self.working_memory.visuospatial.context_barriers

    @context_barriers.setter
    def context_barriers(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.working_memory.visuospatial.context_barriers = val

    @property
    def level_learned_barriers(self) -> dict[int, set[tuple[int, int]]]:
        return self.context_barriers

    @level_learned_barriers.setter
    def level_learned_barriers(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.context_barriers = val

    @property
    def learned_barriers(self) -> set[tuple[int, int]]:
        """Barrier coordinates on active level in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_barriers

    @learned_barriers.setter
    def learned_barriers(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_barriers = val

    @property
    def context_hazards(self) -> dict[int, set[tuple[int, int]]]:
        """Static hazard loci mapped per context epoch in visuospatial working memory."""
        return self.working_memory.visuospatial.context_hazards

    @context_hazards.setter
    def context_hazards(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.working_memory.visuospatial.context_hazards = val

    @property
    def level_lethal_positions(self) -> dict[int, set[tuple[int, int]]]:
        return self.context_hazards

    @level_lethal_positions.setter
    def level_lethal_positions(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.context_hazards = val

    @property
    def learned_goal_positions(self) -> set[tuple[int, int]]:
        """Goal coordinates mapped in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_goal_positions

    @learned_goal_positions.setter
    def learned_goal_positions(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_goal_positions = val
        if hasattr(self, "_state") and self._state is not None:
            self._state.learned_goal_positions = set(val)

    # ── Homeostatic Motivation & Social Directives (WP10) ────────────

    @property
    def homeostatic_state(self) -> Any:
        """Homeostatic allostatic state in prefrontal working memory."""
        return self.working_memory.homeostatic_state

    def consume_energy(self, amount: float) -> float:
        """Consume computational/physical energy budget."""
        return self.working_memory.consume_energy(amount)

    def replenish_energy(self, amount: float) -> float:
        """Replenish computational/physical energy budget."""
        return self.working_memory.replenish_energy(amount)

    def add_social_instruction(self, instruction: dict[str, Any]) -> None:
        """Add an explicit social or environment instruction."""
        self.working_memory.add_social_instruction(instruction)

    def clear_social_instructions(self) -> None:
        """Clear all social instructions."""
        self.working_memory.clear_social_instructions()

    @property
    def learned_receptacle_positions(self) -> set[tuple[int, int]]:
        """Receptacle coordinates mapped in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_receptacle_positions

    @learned_receptacle_positions.setter
    def learned_receptacle_positions(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_receptacle_positions = val
        if hasattr(self, "_state") and self._state is not None:
            self._state.learned_receptacle_positions = set(val)

    @property
    def context_failed_transitions(self) -> dict[int, set[tuple[tuple[int, int], int]]]:
        """Negative state-action transitions archived per context epoch in Habenular memory."""
        return self.habenular_ior.context_failed_transitions

    @context_failed_transitions.setter
    def context_failed_transitions(self, val: dict[int, set[tuple[tuple[int, int], int]]]) -> None:
        self.habenular_ior.context_failed_transitions = val

    @property
    def level_failed_transitions(self) -> dict[int, set[tuple[tuple[int, int], int]]]:
        return self.context_failed_transitions

    @level_failed_transitions.setter
    def level_failed_transitions(self, val: dict[int, set[tuple[tuple[int, int], int]]]) -> None:
        self.context_failed_transitions = val

    @property
    def failed_transitions(self) -> set[tuple[tuple[int, int], int]]:
        """Negative state-action transitions on active level in Habenular memory."""
        return self.habenular_ior.failed_transitions

    @failed_transitions.setter
    def failed_transitions(self, val: set[tuple[tuple[int, int], int]]) -> None:
        self.habenular_ior.failed_transitions = val

    @property
    def current_simulated_goal(self) -> tuple[int, int] | None:
        """Active goal coordinate tracked in prefrontal working memory."""
        return self.working_memory.simulated_goal

    @current_simulated_goal.setter
    def current_simulated_goal(self, val: tuple[int, int] | None) -> None:
        self.working_memory.simulated_goal = val

    @property
    def last_predicted_pos(self) -> tuple[int, int] | None:
        """Last predicted avatar position (efference copy) in prefrontal working memory."""
        return self.working_memory.last_predicted_pos

    @last_predicted_pos.setter
    def last_predicted_pos(self, val: tuple[int, int] | None) -> None:
        self.working_memory.last_predicted_pos = val

    @property
    def _avatar_controllability_evidence(self) -> dict[int, list[tuple[int, tuple[int, int]]]]:
        """Motor grounding controllability evidence held in prefrontal working memory."""
        return self.working_memory.avatar_controllability_evidence

    @_avatar_controllability_evidence.setter
    def _avatar_controllability_evidence(
        self, val: dict[int, list[tuple[int, tuple[int, int]]]]
    ) -> None:
        self.working_memory.avatar_controllability_evidence = val

    @property
    def active_probe_target(self) -> tuple[int, int] | None:
        return self.working_memory.active_probe_target

    @active_probe_target.setter
    def active_probe_target(self, val: tuple[int, int] | None) -> None:
        self.working_memory.active_probe_target = val
        self.state.active_probe_target = val

    @property
    def active_probe_id(self) -> str | None:
        return self.working_memory.active_probe_id

    @active_probe_id.setter
    def active_probe_id(self, val: str | None) -> None:
        self.working_memory.active_probe_id = val
        self.state.active_probe_id = val

    @property
    def probe_target_steps(self) -> int:
        return self.working_memory.probe_target_steps

    @probe_target_steps.setter
    def probe_target_steps(self, val: int) -> None:
        self.working_memory.probe_target_steps = val
        self.state.probe_target_steps = val

    @property
    def probed_entity_ids(self) -> set[str]:
        return self.working_memory.probed_entity_ids

    @probed_entity_ids.setter
    def probed_entity_ids(self, val: set[str]) -> None:
        self.working_memory.probed_entity_ids = val

    @property
    def entity_visit_counts(self) -> dict[str, int]:
        return self.working_memory.entity_visit_counts

    @entity_visit_counts.setter
    def entity_visit_counts(self, val: dict[str, int]) -> None:
        self.working_memory.entity_visit_counts = val

    @property
    def position_visit_counts(self) -> dict[tuple[int, int], int]:
        return self.working_memory.position_visit_counts

    @position_visit_counts.setter
    def position_visit_counts(self, val: dict[tuple[int, int], int]) -> None:
        self.working_memory.position_visit_counts = val

    @property
    def recent_positions(self) -> deque[tuple[int, int]]:
        return self.working_memory.recent_positions

    @recent_positions.setter
    def recent_positions(self, val: deque[tuple[int, int]]) -> None:
        self.working_memory.recent_positions = val

    @property
    def recent_actions(self) -> deque[Any]:
        return self.working_memory.recent_actions

    @recent_actions.setter
    def recent_actions(self, val: deque[Any]) -> None:
        self.working_memory.recent_actions = val

    @property
    def exhausted_candidate_goals(self) -> set[tuple[int, int]]:
        return self.working_memory.exhausted_candidate_goals

    @exhausted_candidate_goals.setter
    def exhausted_candidate_goals(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.exhausted_candidate_goals = val

    @property
    def quiescent_actuation_targets(self) -> set[tuple[int, int]]:
        return self.working_memory.quiescent_actuation_targets

    @quiescent_actuation_targets.setter
    def quiescent_actuation_targets(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.quiescent_actuation_targets = val

    @property
    def effective_actuation_targets(self) -> set[tuple[int, int]]:
        return self.working_memory.effective_actuation_targets

    @effective_actuation_targets.setter
    def effective_actuation_targets(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.effective_actuation_targets = val

    @property
    def last_effective_actuation_coord(self) -> tuple[int, int] | None:
        return self.working_memory.last_effective_actuation_coord

    @last_effective_actuation_coord.setter
    def last_effective_actuation_coord(self, val: tuple[int, int] | None) -> None:
        self.working_memory.last_effective_actuation_coord = val

    @property
    def consecutive_effective_actuations(self) -> int:
        return self.working_memory.consecutive_effective_actuations

    @consecutive_effective_actuations.setter
    def consecutive_effective_actuations(self, val: int) -> None:
        self.working_memory.consecutive_effective_actuations = val

    @property
    def actuation_affordances(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        return self.working_memory.actuation_affordances

    @actuation_affordances.setter
    def actuation_affordances(self, val: dict[tuple[int, int], list[tuple[int, int]]]) -> None:
        self.working_memory.actuation_affordances = val

    @property
    def last_effector_target_step(self) -> dict[tuple[int, int], int]:
        return self.working_memory.last_effector_target_step

    @last_effector_target_step.setter
    def last_effector_target_step(self, val: dict[tuple[int, int], int]) -> None:
        self.working_memory.last_effector_target_step = val

    @property
    def active_goal_converging_coord(self) -> tuple[int, int] | None:
        return self.working_memory.active_goal_converging_coord

    @active_goal_converging_coord.setter
    def active_goal_converging_coord(self, val: tuple[int, int] | None) -> None:
        self.working_memory.active_goal_converging_coord = val

    @property
    def consecutive_goal_converging_actuations(self) -> int:
        return self.working_memory.consecutive_goal_converging_actuations

    @consecutive_goal_converging_actuations.setter
    def consecutive_goal_converging_actuations(self, val: int) -> None:
        self.working_memory.consecutive_goal_converging_actuations = val

    @property
    def active_causal_actuator_coord(self) -> tuple[int, int] | None:
        if self.state.active_causal_actuator_coord is not None:
            return self.state.active_causal_actuator_coord
        return self.working_memory.active_causal_actuator_coord

    @active_causal_actuator_coord.setter
    def active_causal_actuator_coord(self, val: tuple[int, int] | None) -> None:
        self.working_memory.active_causal_actuator_coord = val
        self.state.active_causal_actuator_coord = val

    @property
    def consecutive_causal_probes(self) -> int:
        if self.state.consecutive_causal_probes is not None:
            return self.state.consecutive_causal_probes
        return self.working_memory.consecutive_causal_probes

    @consecutive_causal_probes.setter
    def consecutive_causal_probes(self, val: int) -> None:
        self.working_memory.consecutive_causal_probes = val
        self.state.consecutive_causal_probes = val

    @property
    def causal_actuator_streak_limit(self) -> int:
        return self.working_memory.causal_actuator_streak_limit

    @causal_actuator_streak_limit.setter
    def causal_actuator_streak_limit(self, val: int) -> None:
        self.working_memory.causal_actuator_streak_limit = val

    @property
    def quiescent_features(self) -> set[int]:
        return self.working_memory.quiescent_features

    @quiescent_features.setter
    def quiescent_features(self, val: set[int]) -> None:
        self.working_memory.quiescent_features = val

    @property
    def effective_features(self) -> set[int]:
        return self.working_memory.effective_features

    @effective_features.setter
    def effective_features(self, val: set[int]) -> None:
        self.working_memory.effective_features = val

    @property
    def feature_quiescent_counts(self) -> dict[int, int]:
        if self.state.feature_quiescent_counts:
            return self.state.feature_quiescent_counts
        return self.working_memory.feature_quiescent_counts

    @feature_quiescent_counts.setter
    def feature_quiescent_counts(self, val: dict[int, int]) -> None:
        self.working_memory.feature_quiescent_counts = val
        self.state.feature_quiescent_counts = val

    @property
    def _feature_quiescent_counts(self) -> dict[int, int]:
        if self.state.feature_quiescent_counts:
            return self.state.feature_quiescent_counts
        return self.working_memory.feature_quiescent_counts

    @_feature_quiescent_counts.setter
    def _feature_quiescent_counts(self, val: dict[int, int]) -> None:
        self.working_memory.feature_quiescent_counts = val
        self.state.feature_quiescent_counts = val

    @property
    def inhibited_actions(self) -> dict[Any, int]:
        return self.working_memory.inhibited_actions

    @inhibited_actions.setter
    def inhibited_actions(self, val: dict[Any, int]) -> None:
        self.working_memory.inhibited_actions = val

    @property
    def prior_avatar_feature(self) -> Any | None:
        return self.working_memory.prior_avatar_feature

    @prior_avatar_feature.setter
    def prior_avatar_feature(self, val: Any | None) -> None:
        self.working_memory.prior_avatar_feature = val

    @property
    def entity_action_failed_positions(self) -> dict[Any, set[tuple[int, int]]]:
        return self.working_memory.entity_action_failed_positions

    @entity_action_failed_positions.setter
    def entity_action_failed_positions(self, val: dict[Any, set[tuple[int, int]]]) -> None:
        self.working_memory.entity_action_failed_positions = val

    @property
    def immobile_entity_actions(self) -> set[Any]:
        return self.working_memory.immobile_entity_actions

    @immobile_entity_actions.setter
    def immobile_entity_actions(self, val: set[Any]) -> None:
        self.working_memory.immobile_entity_actions = val

    @property
    def action_affordances(self) -> dict[Any, ActionAffordance]:
        return self.working_memory.action_affordances

    @action_affordances.setter
    def action_affordances(self, val: dict[Any, ActionAffordance]) -> None:
        self.working_memory.action_affordances = val

    @property
    def focus_switch_actions(self) -> set[Any]:
        return self.working_memory.focus_switch_actions

    @focus_switch_actions.setter
    def focus_switch_actions(self, val: set[Any]) -> None:
        self.working_memory.focus_switch_actions = val

    @property
    def undo_actions(self) -> set[Any]:
        return self.working_memory.undo_actions

    @undo_actions.setter
    def undo_actions(self, val: set[Any]) -> None:
        self.working_memory.undo_actions = val

    @property
    def index_selector_actions(self) -> set[Any]:
        return self.working_memory.index_selector_actions

    @index_selector_actions.setter
    def index_selector_actions(self, val: set[Any]) -> None:
        self.working_memory.index_selector_actions = val

    @property
    def value_mutator_actions(self) -> set[Any]:
        return self.working_memory.value_mutator_actions

    @value_mutator_actions.setter
    def value_mutator_actions(self, val: set[Any]) -> None:
        self.working_memory.value_mutator_actions = val

    @property
    def last_non_displacement_action(self) -> Any | None:
        return self.working_memory.last_non_displacement_action

    @last_non_displacement_action.setter
    def last_non_displacement_action(self, val: Any | None) -> None:
        self.working_memory.last_non_displacement_action = val

    @property
    def context_epistemic_probes(self) -> int:
        return self.working_memory.context_epistemic_probes

    @context_epistemic_probes.setter
    def context_epistemic_probes(self, val: int) -> None:
        self.working_memory.context_epistemic_probes = val

    @property
    def level_epistemic_probes(self) -> int:
        return self.context_epistemic_probes

    @level_epistemic_probes.setter
    def level_epistemic_probes(self, val: int) -> None:
        self.context_epistemic_probes = val

    @property
    def total_epistemic_probes(self) -> int:
        return self.working_memory.total_epistemic_probes

    @total_epistemic_probes.setter
    def total_epistemic_probes(self, val: int) -> None:
        self.working_memory.total_epistemic_probes = val

    def register_target_scorer(self, scorer: Any) -> None:
        """Register an external or domain-specific target candidate scorer."""
        if scorer not in self.target_scorers:
            self.target_scorers.append(scorer)
        MotorCortexEffector.register_target_scorer(scorer)

    def unregister_target_scorer(self, scorer: Any) -> None:
        """Unregister an external target candidate scorer."""
        if scorer in self.target_scorers:
            self.target_scorers.remove(scorer)
        MotorCortexEffector.unregister_target_scorer(scorer)

    def attach_kernel_services(
        self,
        services: KernelServices,
        goal_id: str = "main_goal",
        hydrate_persistent: bool = True,
    ) -> None:
        """Connect the epistemic engine to a KernelServices container."""
        self.services = services
        self.event_bus = getattr(services, "event_bus", None)
        self.capability_resolver = getattr(services, "capability_resolver", None)
        if getattr(services, "transaction_manager", None) is not None:
            self.transaction_manager = services.transaction_manager
        if services.tiered_workspace is not None:
            self.attach_tiered_workspace(
                services.tiered_workspace,
                goal_id=goal_id,
                hydrate_persistent=hydrate_persistent,
            )

    def attach_tiered_workspace(
        self,
        tiered: TieredWorkspace,
        goal_id: str = "main_goal",
        hydrate_persistent: bool = True,
    ) -> TaskFrame:
        """Connect the epistemic engine to an event-sourced 4-tier cognitive workspace.

        Establishes an active task frame in the Working tier and binds the TransactionManager
        as the single commit authority for atomic graph mutations.
        Optionally hydrates learned skills, goals, barriers, and axioms from the Persistent tier.
        """
        self.tiered_workspace = tiered
        frame = tiered.create_task_frame(goal_id=goal_id)
        self.active_task_frame = frame

        # Ensure active task frame has a TransactionManager and bind it to engine
        verifier = (
            getattr(self.services, "constitutional_verifier", None)
            if self.services is not None
            else None
        )
        stages = [verifier] if verifier is not None else None
        event_bus = self.services.event_bus if self.services is not None else self.event_bus

        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        # A task frame's working scratch workspace needs a TM bound to that exact frame.workspace
        if (
            self.transaction_manager is not None
            and getattr(self.transaction_manager, "workspace", None) is frame.workspace
        ):
            tm = self.transaction_manager
        elif (
            frame.workspace.transaction_manager is not None
            and getattr(frame.workspace.transaction_manager, "workspace", None) is frame.workspace
        ):
            tm = frame.workspace.transaction_manager
        else:
            tm = TransactionManager(
                workspace=frame.workspace,
                verification_stages=stages,
                event_bus=event_bus,
            )
            frame.workspace.attach_transaction_manager(tm, replace=True)

        self.transaction_manager = tm
        if (
            self.services is not None
            and getattr(self.services, "workspace", None) is not frame.workspace
        ):
            self.services.workspace = frame.workspace
            self.services.transaction_manager = tm

        logger.info(
            "AutonomousEpistemicEngine: Attached TieredWorkspace (Frame: %s, Goal: %s)",
            frame.frame_id,
            goal_id,
        )
        if hydrate_persistent:
            self.hydrate_from_persistent_tier()
        return frame

    def hydrate_from_persistent_tier(self) -> int:
        """Hydrate internal memory stores (working memory, symbolic theory, visuospatial memory)
        from nodes stored in the persistent tier of the TieredWorkspace.

        Returns the number of restored knowledge items.
        """
        if self.tiered_workspace is None:
            return 0

        persistent_ws = self.tiered_workspace.persistent
        res = persistent_ws.query(GraphQuery(limit=9999))
        hydrated_count, kb_hydrated_feats = (
            self.symbolic_theory.transition_kb.hydrate_from_nodes_with_keys(res.nodes)
        )

        for node in res.nodes:
            nid = node.id
            ntype = getattr(node, "node_type", "")
            props = getattr(node, "properties", {}) or {}

            # 1. Restore Procedural Skills
            if nid.startswith("procedural_skill_") or ntype == "skill":
                schema_name = (
                    props.get("schema_name")
                    or getattr(node, "name", "")
                    or getattr(node, "label", "")
                    or nid.removeprefix("procedural_skill_")
                )
                if schema_name not in self.working_memory.procedural_library:
                    from hbllm.hcir.skills.declarative import DeclarativeNeuroSymbolicSkill

                    class RestoredSkill(DeclarativeNeuroSymbolicSkill):
                        skill_name = schema_name
                        description = (
                            getattr(node, "description", "")
                            or f"Restored persistent skill {schema_name}"
                        )

                    self.working_memory.register_procedural_skill(schema_name, RestoredSkill())
                    hydrated_count += 1
                    logger.info(
                        "AutonomousEpistemicEngine: Hydrated persistent skill '%s'", schema_name
                    )

            # 2. Restore Semantic Goal Ontology
            elif nid.startswith("ontology_goal_"):
                feat = props.get("feature_id")
                if feat is None:
                    try:
                        feat = int(nid.removeprefix("ontology_goal_"))
                    except ValueError:
                        logger.debug("Non-integer ontology_goal suffix ignored: %s", nid)
                if feat is not None and not (
                    feat in kb_hydrated_feats
                    or (str(feat).lstrip("-").isdigit() and int(feat) in kb_hydrated_feats)
                ):
                    has_reward = (
                        props.get("has_terminal_reward") is True
                        or props.get("provenance")
                        in (
                            "terminal_reward",
                            "terminal_reward:success",
                            str(TerminalSignal.SUCCESS),
                        )
                        or props.get("evidence_rank", 0) >= 100
                        or (
                            props.get("has_terminal_reward") is not False
                            and props.get("evidence_rank") is None
                            and not any(
                                w in str(props.get("provenance", "")).lower()
                                for w in ("unverified", "candidate", "heuristic")
                            )
                            and (
                                "goal" in nid.lower()
                                or "winning" in str(getattr(node, "definition", "")).lower()
                                or "win" in str(props.get("description", "")).lower()
                            )
                        )
                    )
                    if has_reward:
                        self.symbolic_theory.induce_goal(
                            int(feat), evidence="hydration:terminal_reward"
                        )
                        hydrated_count += 1
                        logger.info(
                            "AutonomousEpistemicEngine: Hydrated persistent goal feature %s", feat
                        )
                    else:
                        logger.info(
                            "AutonomousEpistemicEngine: Hydration skipped unverified goal feature %s (no terminal reward provenance)",
                            feat,
                        )
                        self.symbolic_theory.induce_candidate_goal(
                            int(feat), evidence="hydration:unverified"
                        )

            # 3. Restore Learned Spatial Goals
            elif nid.startswith("learned_goal_") or ntype == "goal":
                pos = props.get("position")
                if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    gp = (int(pos[0]), int(pos[1]))
                else:
                    parts = nid.removeprefix("learned_goal_").split("_")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        gp = (int(parts[0]), int(parts[1]))
                    else:
                        gp = None
                if gp is not None:
                    self.record_learned_goal(gp)
                    hydrated_count += 1

            # 4. Restore Learned Barriers
            elif nid.startswith("learned_barrier_") or ntype == "constraint":
                pos = props.get("position")
                if pos and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    bp = (int(pos[0]), int(pos[1]))
                else:
                    parts = nid.removeprefix("learned_barrier_").split("_")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        bp = (int(parts[0]), int(parts[1]))
                    else:
                        bp = None
                if bp is not None:
                    self.record_learned_barrier(bp)
                    hydrated_count += 1

            # 5. Restore Spelke Physical Axioms
            elif nid.startswith("spelke_axiom_"):
                from hbllm.hcir.world.cortex_causal import CausalPredicate, SpelkePhysicalAxiom

                axiom_name = props.get("axiom_name") or nid.removeprefix("spelke_axiom_")
                for ax in SpelkePhysicalAxiom:
                    if ax.value == axiom_name:
                        if ax not in self.symbolic_theory.spelke_axioms:
                            self.symbolic_theory.spelke_axioms[ax] = CausalPredicate(
                                variable=f"spelke_axiom_{ax.value}",
                                operator="==",
                                value="INVARIANT",
                                relation="core_knowledge_physics",
                                target_entity=ax.value,
                            )
                            hydrated_count += 1
                        break

        logger.info(
            "AutonomousEpistemicEngine: Hydrated %d persistent knowledge items", hydrated_count
        )
        return hydrated_count

    def commit_working_memory_transactions(self) -> HCIRTransaction | None:
        """Mirror current epistemic state and internal memory to the active task frame via TransactionManager.

        Atomic operations produced by to_transaction_operations() are bundled into an HCIRTransaction
        and committed to the active TaskFrame workspace.
        """
        if self.transaction_manager is None or self.active_task_frame is None:
            return None

        ops = self.to_transaction_operations()
        if not ops:
            return None

        dest_ws = getattr(self.transaction_manager, "workspace", None) or getattr(
            self.active_task_frame, "workspace", None
        )
        filtered = self._filter_unchanged_upserts(dest_ws, ops)
        if not filtered:
            filtered = [ops[0]]  # Ensure atomic telemetry heartbeat when tx requested

        batch_size = 400
        committed_tx = None
        for i in range(0, len(filtered), batch_size):
            tx = HCIRTransaction(
                author="AutonomousEpistemicEngine",
                operations=filtered[i : i + batch_size],
            )
            committed_tx = self.transaction_manager.commit(tx)

        if self.tiered_workspace is not None:
            self.tiered_workspace.notify_commit()

        if self.event_bus is not None:
            self.event_bus.publish(
                KernelEvent(
                    event_type="epistemic.transaction_committed",
                    source="AutonomousEpistemicEngine",
                    data={
                        "tx_id": committed_tx.tx_id,
                        "op_count": len(ops),
                        "world_id": self.world_id,
                    },
                )
            )

        # Clear transient pending buffer
        return committed_tx

    async def execute_subgoal_program_with_checkpoint(
        self,
        stream: InstructionStream,
        workspace: HCIRWorkspaceState,
        services: KernelServices | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Execute frontopolar subgoal instruction stream with transactional checkpoint and rollback VM."""
        sim_branch = f"subgoal_sim_{uuid.uuid4().hex[:8]}"
        workspace.fork(sim_branch)
        target_version = workspace.snapshot_manager.current_version
        sim_ws = workspace.get_branch(sim_branch) or workspace

        tx_mgr = sim_ws.transaction_manager
        if tx_mgr is None:
            from hbllm.hcir.kernel.transaction_manager import TransactionManager

            tx_mgr = TransactionManager(workspace=sim_ws)
            sim_ws.attach_transaction_manager(tx_mgr, replace=True)

        if services is not None and getattr(services, "workspace", None) is sim_ws:
            branch_services = services
        else:
            base_services = services or self.services
            resolver = (
                base_services.capability_resolver
                if base_services is not None
                else CapabilityResolver()
            )
            scheduler = (
                base_services.scheduler
                if base_services is not None
                else KernelInstructionScheduler()
            )
            from hbllm.hcir.kernel.event_bus import KernelEventBus

            event_bus = base_services.event_bus if base_services is not None else KernelEventBus()
            branch_services = KernelServices(
                workspace=sim_ws,
                transaction_manager=tx_mgr,
                capability_resolver=resolver,
                scheduler=scheduler,
                event_bus=event_bus,
            )

        interpreter = HCIRInterpreter(sim_ws, branch_services)
        exec_result = await interpreter.execute(stream)
        has_errors = (
            any(a.severity == "error" for a in exec_result.annotations) or not exec_result.success
        )

        if not has_errors:
            if sim_ws is not workspace:
                tm = (
                    workspace.transaction_manager
                    or (
                        self.services.transaction_manager
                        if self.services is not None
                        and getattr(self.services.transaction_manager, "workspace", None)
                        is workspace
                        else None
                    )
                    or (
                        self.transaction_manager
                        if self.transaction_manager is not None
                        and getattr(self.transaction_manager, "workspace", None) is workspace
                        else None
                    )
                )
                if tm is None:
                    from hbllm.hcir.kernel.transaction_manager import TransactionManager

                    tm = TransactionManager(workspace=workspace)
                    workspace.attach_transaction_manager(tm, replace=True)

                merged = workspace.merge_branch(
                    sim_branch,
                    transaction_manager=tm,
                    author="subgoal_exec",
                )
                if not merged:
                    workspace.drop_branch(sim_branch)
                    return (
                        {
                            "status": "failure",
                            "error": "subgoal merge rejected by transaction pipeline",
                            "branch": sim_branch,
                        },
                        False,
                    )
            return ({"status": "success", "result": exec_result, "branch": sim_branch}, True)
        else:
            if sim_ws is not workspace:
                workspace.drop_branch(sim_branch)
            workspace.snapshot_manager.record_kernel_event(
                EventType.ROLLBACK,
                {"branch": sim_branch, "target_version": target_version},
            )
            err_msg = exec_result.error or (
                exec_result.annotations[0].assertion
                if exec_result.annotations
                else "execution_error"
            )
            self.acc_conflict_monitor.record_conflict(f"subgoal_failed_{err_msg}", severity=0.6)
            self.transition_to_phase(EpistemicPhase.REPLANNING)
            return ({"status": "failed", "error": err_msg}, False)

    def run_subgoal_program_sync(
        self,
        stream: InstructionStream,
        workspace: HCIRWorkspaceState,
        services: KernelServices | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Synchronously execute subgoal instruction stream with transactional checkpoint and rollback VM."""
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.execute_subgoal_program_with_checkpoint(stream, workspace, services),
                ).result()
        else:
            return asyncio.run(
                self.execute_subgoal_program_with_checkpoint(stream, workspace, services)
            )

    def evaluate_truth_value(
        self,
        variable_name: str,
        observed_state: dict[str, Any],
        expected_value: Any,
    ) -> EpistemicTruthValue:
        """W122: Distinguishes between unknown (unobserved) and false (observed negative)."""
        if variable_name not in observed_state:
            return EpistemicTruthValue.UNKNOWN
        actual = observed_state[variable_name]
        return EpistemicTruthValue.TRUE if actual == expected_value else EpistemicTruthValue.FALSE

    def track_relational_confidence(self, relation_id: str, new_confidence: float) -> float:
        """W125: Confidence tracking per relation in the world model."""
        self.relational_confidence_map[relation_id] = max(0.0, min(1.0, new_confidence))
        return self.relational_confidence_map[relation_id]

    def record_evidence_provenance(
        self,
        claim: str,
        source_faculty: str,
        confidence: float,
    ) -> dict[str, Any]:
        """W127: Records evidence provenance and observational lineage."""
        record = {
            "claim": claim,
            "source_faculty": source_faculty,
            "confidence": confidence,
            "timestep": self.step_counter,
        }
        self.evidence_provenance_log.append(record)
        return record

    @property
    def episodic_memory(self) -> HippocampalEpisodicCortex:
        """Hippocampal episodic cortex managing experiences and attractor networks (alias)."""
        return self.episodic_cortex

    @property
    def basal_ganglia(self) -> BasalGangliaGatingCircuit:
        """Basal Ganglia tri-pathway action arbitration circuit."""
        return self.deliberation_engine.basal_ganglia

    def arbitrate_basal_ganglia_action(
        self,
        candidate_utilities: dict[Any, float],
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
        risk_penalties: dict[Any, float] | None = None,
    ) -> tuple[BasalGangliaDecision | None, list[BasalGangliaDecision]]:
        """Arbitrate candidate actions through the Basal Ganglia tri-pathway circuit."""
        return self.deliberation_engine.arbitrate_basal_ganglia_action(
            self,
            candidate_utilities=candidate_utilities,
            acc_conflict=acc_conflict,
            threat_looming=threat_looming,
            surprise_entropy=surprise_entropy,
            risk_penalties=risk_penalties,
        )

    def step_neuromodulatory_dynamics(
        self,
        reward: float = 0.0,
        predicted_value: float = 0.0,
        actual_value: float = 0.0,
        conflict_level: float = 0.0,
    ) -> NeuromodulatoryState:
        """Update VTA-LC coupled neuromodulatory state."""
        return self.deliberation_engine.step_neuromodulatory_dynamics(
            self,
            reward=reward,
            predicted_value=predicted_value,
            actual_value=actual_value,
            conflict_level=conflict_level,
        )

    def extract_minimal_causal_skeleton(
        self, graph: CognitiveGraph | None = None
    ) -> CognitiveGraph:
        """Extract the minimal transitive reduction DAG of the causal or workspace graph."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return algorithms.transitive_reduction_subgraph(target_graph)

    def extract_geodesic_spine(
        self, graph: CognitiveGraph | None = None
    ) -> tuple[CognitiveGraph, float]:
        """Extract the minimal-cost connectivity spine across graph vertices using Kruskal's MST."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return algorithms.minimum_spanning_tree_subgraph(target_graph)

    @property
    def phase(self) -> EpistemicPhase:
        """Active cognitive lifecycle phase."""
        if self.working_memory.phase is not None:
            return self.working_memory.phase
        if self.state.phase is not None:
            return self.state.phase
        return self._phase

    @phase.setter
    def phase(self, value: EpistemicPhase) -> None:
        self.working_memory.phase = value
        self.state.phase = value
        self.transition_to_phase(value, reason="state_assignment")

    @property
    def current_phase(self) -> EpistemicPhase:
        """Active cognitive lifecycle phase (alias)."""
        return self.phase

    @property
    def relational_confidence_map(self) -> dict[str, float]:
        """Relational confidence map managed in symbolic world theory (Invariant 8)."""
        return self.symbolic_theory.relational_confidence_map

    @relational_confidence_map.setter
    def relational_confidence_map(self, val: dict[str, float]) -> None:
        self.symbolic_theory.relational_confidence_map = val

    @property
    def evidence_provenance_log(self) -> list[dict[str, Any]]:
        """Evidence provenance log managed in prefrontal working memory (Invariant 8)."""
        return self.working_memory.evidence_provenance_log

    @evidence_provenance_log.setter
    def evidence_provenance_log(self, val: list[dict[str, Any]]) -> None:
        self.working_memory.evidence_provenance_log = val

    @property
    def _pipeline_runs(self) -> int:
        return self.working_memory.pipeline_runs

    @_pipeline_runs.setter
    def _pipeline_runs(self, val: int) -> None:
        self.working_memory.pipeline_runs = val

    @property
    def pipeline_runs(self) -> int:
        return self.working_memory.pipeline_runs

    @pipeline_runs.setter
    def pipeline_runs(self, val: int) -> None:
        self.working_memory.pipeline_runs = val

    @property
    def _last_decision_branch(self) -> str | None:
        return self.working_memory.last_decision_branch

    @_last_decision_branch.setter
    def _last_decision_branch(self, val: str | None) -> None:
        self.working_memory.last_decision_branch = val

    @property
    def last_decision_branch(self) -> str | None:
        return self.working_memory.last_decision_branch

    @last_decision_branch.setter
    def last_decision_branch(self, val: str | None) -> None:
        self.working_memory.last_decision_branch = val

    def transition_to_phase(self, new_phase: EpistemicPhase, reason: str = "") -> None:
        """Formal event-sourced phase transition."""
        self.working_memory.phase = new_phase
        self.state.phase = new_phase
        if self._phase != new_phase:
            old = self._phase
            logger.info(
                "AutonomousEpistemicEngine: Phase transition [%s -> %s] (reason: %s)",
                old,
                new_phase,
                reason,
            )
            self.phase_history.append((old, new_phase, reason))
            self._phase = new_phase
            if self.event_bus is not None:
                self.event_bus.publish(
                    KernelEvent(
                        event_type="epistemic.phase_transition",
                        source="AutonomousEpistemicEngine",
                        data={
                            "old_phase": str(old),
                            "new_phase": str(new_phase),
                            "reason": reason,
                            "world_id": self.world_id,
                        },
                    )
                )
            if self.transaction_manager is not None:
                try:
                    phase_node = EventNode(
                        id=f"phase_{self.world_id}",
                        event_kind="epistemic.phase_transition",
                        event_data={
                            "phase": str(new_phase),
                            "previous_phase": str(old),
                            "reason": reason,
                            "timestep": self.step_counter,
                        },
                    )
                    self.transaction_manager.commit(
                        HCIRTransaction(
                            author="AutonomousEpistemicEngine",
                            operations=[
                                TransactionOperation(
                                    op=TransactionOp.UPSERT_NODE,
                                    node_id=phase_node.id,
                                    node_data=phase_node.model_dump(),
                                )
                            ],
                        )
                    )
                except Exception as e:
                    logger.debug("Phase transition transaction notice: %s", e)

    def assess_evidence_sufficiency(
        self,
        observations: list[Any],
        min_observations: int = 3,
        min_confidence: float = 0.80,
    ) -> tuple[bool, float]:
        """W130: Evidence sufficiency assessment before epistemic commitment."""
        n = len(observations)
        if n == 0:
            return False, 0.0
        confs = [o.confidence for o in observations]
        avg_conf = sum(confs) / n
        is_sufficient = (n >= min_observations) and (avg_conf >= min_confidence)
        return is_sufficient, avg_conf

    def register_action_space(self, action_specs: Sequence[Any]) -> None:
        """Register external action definitions dynamically from peripheral driver or device descriptor.

        Accepts:
        - List of dicts (from SynapticDeviceDescriptor.action_schema):
          e.g. [{'action_id': 1, 'name': 'MOVE_UP'}, {'action_id': 6, 'name': 'ACTUATE_CELL', 'parameters': {'x': 'int', 'y': 'int'}}]
        - List of ActionAffordance instances
        - List of bare action IDs (int, str, Enum)
        """
        _dir_priors: dict[str, tuple[int, int]] = {
            "UP": (-1, 0),
            "NORTH": (-1, 0),
            "DOWN": (1, 0),
            "SOUTH": (1, 0),
            "LEFT": (0, -1),
            "WEST": (0, -1),
            "RIGHT": (0, 1),
            "EAST": (0, 1),
        }
        for spec in action_specs:
            aff = self._parse_action_spec(spec)
            self.action_affordances[aff.action_id] = aff
            aid_key = (
                int(aff.action_id)
                if isinstance(aff.action_id, str) and aff.action_id.isdigit()
                else aff.action_id
            )
            if aid_key != aff.action_id:
                self.action_affordances[aid_key] = aff

            # Bayesian Directional Prior Initialization
            if aid_key not in self.action_dynamics and aff.action_id not in self.action_dynamics:
                name_u = aff.name.upper()
                prior_delta: tuple[int, int] | None = None
                for k, d in _dir_priors.items():
                    if k in name_u:
                        prior_delta = d
                        break
                if prior_delta is None and isinstance(aid_key, int):
                    # Standard discrete 2D grid direction convention: 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
                    _idx_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                    prior_delta = _idx_map.get(aid_key)

                if prior_delta is not None:
                    dr, dc = prior_delta
                    aff.is_displacement = True
                    aff.delta = (dr, dc)
                    dyn_model = ActionDynamicsModel(
                        action_id=aid_key,
                        delta_r=dr,
                        delta_c=dc,
                        confidence=0.75,
                        probes_tested=0,
                        displacement=(float(dr), float(dc)),
                    )
                    self.action_dynamics[aid_key] = dyn_model
                    if aid_key != aff.action_id:
                        self.action_dynamics[aff.action_id] = dyn_model

    def _parse_action_spec(self, spec: Any) -> ActionAffordance:
        return MotorCortexEffector.parse_action_spec(spec)

    def is_spatial_effector(self, action: Any) -> bool:
        """Returns True if the action is configured or observed to accept spatial coordinates."""
        aff = self.action_affordances.get(action)
        if aff is not None:
            return aff.requires_spatial_target
        return False

    def is_displacement_action(self, action: Any) -> bool:
        """Returns True if this action causes physical self/ego-motion displacement."""
        dyn = self.action_dynamics.get(action)
        if dyn is not None and dyn.is_displacement_action():
            return True
        aff = self.action_affordances.get(action)
        if aff is not None and aff.is_displacement:
            return True
        return action in self.action_deltas

    def has_displacement_capabilities(self, available_actions: Sequence[Any] | None = None) -> bool:
        """Returns True if the environment/agent possesses physical displacement actions."""
        actions = (
            available_actions
            if available_actions is not None
            else list(self.action_affordances.keys())
        )
        return any(self.is_displacement_action(a) for a in actions)

    @property
    def action_deltas(self) -> dict[Any, tuple[int, int]]:
        """Canonical mapping of action_id -> (delta_r, delta_c) displacement vector.

        Synthesizes calibrated empirical action dynamics, declared affordance models,
        and standard embodied directional conventions.
        """
        deltas: dict[Any, tuple[int, int]] = {}
        for act, aff in self.action_affordances.items():
            if aff.is_displacement and aff.delta is not None and len(aff.delta) >= 2:
                deltas[act] = (int(round(float(aff.delta[0]))), int(round(float(aff.delta[1]))))
        for act, dyn in self.action_dynamics.items():
            if dyn.is_displacement_action():
                deltas[act] = dyn.get_displacement()
        # Cardinal motor stride harmonization via axis-aware consensus:
        # Reconcile unequal antiparallel magnitudes along an axis (e.g. -5 vs truncated +4 -> 5)
        # and propagate modal cardinal stride to uncalibrated or unconfirmed singleton axes.
        cardinal_acts = [(act, d) for act, d in deltas.items() if (d[0] == 0) != (d[1] == 0)]
        stride_counts: dict[int, int] = {}
        axis_max: dict[int, int] = {0: 0, 1: 0}
        axis_signs: dict[tuple[int, int], int] = {}
        for act, (dr, dc) in cardinal_acts:
            axis = 0 if dr != 0 else 1
            sign = 1 if (dr + dc) > 0 else -1
            mag = abs(dr) + abs(dc)
            axis_signs[(axis, sign)] = max(axis_signs.get((axis, sign), 0), mag)
            dyn = self.action_dynamics.get(act)
            if dyn is not None and dyn.probes_tested >= 1:
                axis_max[axis] = max(axis_max[axis], mag)
                stride_counts[mag] = stride_counts.get(mag, 0) + 1
        consensus = max(stride_counts, key=lambda s: (stride_counts[s], s)) if stride_counts else 0
        for act, (dr, dc) in cardinal_acts:
            dyn = self.action_dynamics.get(act)
            probes = dyn.probes_tested if dyn is not None else 0
            axis = 0 if dr != 0 else 1
            sign = 1 if (dr + dc) > 0 else -1
            mag = abs(dr) + abs(dc)
            twin_mag = axis_signs.get((axis, -sign), 0)
            target_mag = axis_max[axis]
            if consensus > 1 and (
                (probes == 0 and mag <= 1)
                or (twin_mag != mag and stride_counts.get(consensus, 0) >= 2 and probes <= 2)
            ):
                target_mag = max(target_mag, consensus)
            if target_mag > mag:
                deltas[act] = (sign * target_mag, 0) if axis == 0 else (0, sign * target_mag)

        _discrete_grid_convention = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        for act, d in _discrete_grid_convention.items():
            if act not in deltas and (
                act in self.action_affordances or not self.action_affordances
            ):
                axis_c = 0 if d[0] != 0 else 1
                lq = axis_max[axis_c] if axis_max[axis_c] > 1 else consensus
                deltas[act] = (d[0] * lq, d[1] * lq) if lq > 1 else d
        return deltas

    @action_deltas.setter
    def action_deltas(self, val: dict[Any, tuple[int, int]]) -> None:
        """Allow explicit initialization of calibrated displacement actions."""
        for act, d in val.items():
            self.action_dynamics[act] = ActionDynamicsModel(
                action_id=act,
                delta_r=d[0],
                delta_c=d[1],
                probes_tested=1,
            )

    def is_action_calibrated(self, action: Any) -> bool:
        """Returns True if empirical dynamics for this action have been observed."""
        if self.is_spatial_effector(action):
            aff = self.action_affordances.get(action)
            return aff is not None and aff.requires_spatial_target
        dyn = self.action_dynamics.get(action)
        if dyn is None:
            return False
        return dyn.probes_tested > 0 or dyn.confidence >= 0.9

    def is_focus_switch_action(self, action: Any) -> bool:
        """Returns True if this action switches active control/focus between entities."""
        if action in self.focus_switch_actions:
            return True
        aff = self.action_affordances.get(action)
        return bool(aff is not None and getattr(aff, "is_focus_switch", False))

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array."""
        return PerceptionEngine.normalize_sensory_input(raw)

    # ── Property Pass-Throughs to HCIRSymbolicWorldTheory ─────────────────────

    @property
    def avatar_feature(self) -> Any | None:
        if self.state.avatar_feature is not None:
            return self.state.avatar_feature
        if self.state.avatar_features:
            return next(iter(self.state.avatar_features))
        if self._avatar_feature is not None:
            return self._avatar_feature
        if self.avatar_features:
            return next(iter(self.avatar_features))
        return None

    @avatar_feature.setter
    def avatar_feature(self, val: Any | None) -> None:
        self._avatar_feature = val
        self.state.avatar_feature = val
        if val is not None:
            if not self.avatar_features or val not in self.avatar_features:
                self.avatar_features = {val}
        else:
            self.avatar_features.clear()

    @property
    def avatar_features(self) -> set[Any]:
        return self.state.avatar_features

    @avatar_features.setter
    def avatar_features(self, val: Any) -> None:
        s = set(val) if val is not None else set()
        self.state.avatar_features = s
        self._avatar_features = s

    @property
    def avatar_pos(self) -> tuple[int, int] | None:
        if self.state.avatar_pos is not None:
            return self.state.avatar_pos
        return self._avatar_pos

    @avatar_pos.setter
    def avatar_pos(self, val: tuple[int, int] | None) -> None:
        self.state.avatar_pos = val
        self._avatar_pos = val

    @property
    def avatar_size(self) -> int:
        return self.state.avatar_size

    @avatar_size.setter
    def avatar_size(self, val: int) -> None:
        self.state.avatar_size = val
        self._avatar_size = val

    @property
    def prev_grid(self) -> np.ndarray | None:
        return self.state.prev_grid

    @prev_grid.setter
    def prev_grid(self, val: np.ndarray | None) -> None:
        self.state.prev_grid = val
        self._prev_grid = val

    @property
    def last_action(self) -> int | None:
        return self.state.last_action

    @last_action.setter
    def last_action(self, val: int | None) -> None:
        self.state.last_action = val
        self._last_action = val

    @property
    def last_action_data(self) -> dict[str, Any] | None:
        return self.state.last_action_data

    @last_action_data.setter
    def last_action_data(self, val: dict[str, Any] | None) -> None:
        self.state.last_action_data = val
        self._last_action_data = val

    @property
    def step_counter(self) -> int:
        return self.state.step_counter

    @step_counter.setter
    def step_counter(self, val: int) -> None:
        self.state.step_counter = val
        self._step_counter = val

    @property
    def spatial_lattice_stride(self) -> int:
        return self.state.spatial_lattice_stride

    @spatial_lattice_stride.setter
    def spatial_lattice_stride(self, val: int) -> None:
        self.state.spatial_lattice_stride = val
        self._spatial_lattice_stride = val

    @property
    def effector_kinetics_mode(self) -> int | None:
        return self.state.effector_kinetics_mode

    @effector_kinetics_mode.setter
    def effector_kinetics_mode(self, val: int | None) -> None:
        self.state.effector_kinetics_mode = val
        self._effector_kinetics_mode = val

    @property
    def last_surprise(self) -> float:
        return self.state.last_surprise

    @last_surprise.setter
    def last_surprise(self, val: float) -> None:
        self.state.last_surprise = val
        self._last_surprise = val

    @property
    def last_surprise_eval(self) -> Any | None:
        return self.state.last_surprise_eval

    @last_surprise_eval.setter
    def last_surprise_eval(self, val: Any | None) -> None:
        self.state.last_surprise_eval = val
        self._last_surprise_eval = val

    @property
    def consecutive_quiescent_actions(self) -> int:
        return self.state.consecutive_quiescent_actions

    @consecutive_quiescent_actions.setter
    def consecutive_quiescent_actions(self, val: int) -> None:
        self.state.consecutive_quiescent_actions = val
        self.working_memory.consecutive_quiescent_actions = val

    @property
    def bilateral_state(self) -> Any | None:
        if self.state.bilateral_state is not None:
            return self.state.bilateral_state
        return self._bilateral_state

    @bilateral_state.setter
    def bilateral_state(self, val: Any | None) -> None:
        self.state.bilateral_state = val
        self._bilateral_state = val

    @property
    def current_actor_pos(self) -> tuple[int, int] | None:
        """Canonical alias for avatar_pos to satisfy actor introspection contracts."""
        return self.avatar_pos

    @current_actor_pos.setter
    def current_actor_pos(self, val: tuple[int, int] | None) -> None:
        self.avatar_pos = val

    @property
    def current_context_hazards(self) -> set[tuple[int, int]]:
        """Hazard loci grounded for the current context epoch."""
        return (
            self.context_hazards.get(self.context_epoch, set())
            | self.hazard_tracker.static_lethal_positions
        )

    @property
    def current_level_lethal_positions(self) -> set[tuple[int, int]]:
        return self.current_context_hazards

    @property
    def learned_goal_features(self) -> GovernedSetView:
        return GovernedSetView(
            self.symbolic_theory.goal_features,
            on_add=lambda f: self.symbolic_theory.induce_goal(f, evidence="terminal_reward:manual"),
            on_discard=lambda f: self.symbolic_theory.retract_goal(f, reason="manual_discard"),
            on_clear=self.symbolic_theory.clear_goals,
        )

    @learned_goal_features.setter
    def learned_goal_features(self, val: Any) -> None:
        if not val:
            self.symbolic_theory.clear_goals()
        else:
            for f in val:
                self.symbolic_theory.induce_goal(f, evidence="learned_goal_features_setter")

    @property
    def learned_barrier_features(self) -> GovernedSetView:
        return GovernedSetView(
            self.symbolic_theory.barrier_features,
            on_add=lambda f: self.symbolic_theory.induce_barrier(f, evidence="manual"),
            on_discard=lambda f: self.symbolic_theory.retract_barrier(f, reason="manual_discard"),
            on_clear=self.symbolic_theory.clear_barriers,
        )

    @learned_barrier_features.setter
    def learned_barrier_features(self, val: Any) -> None:
        if not val:
            self.symbolic_theory.clear_barriers()
        else:
            for f in val:
                self.symbolic_theory.induce_barrier(f, evidence="learned_barrier_features_setter")

    @property
    def learned_walkable_features(self) -> GovernedSetView:
        return GovernedSetView(
            self.symbolic_theory.walkable_features,
            on_add=lambda f: self.symbolic_theory.induce_walkable(f, evidence="manual"),
            on_discard=lambda f: self.symbolic_theory.retract_walkable(f, reason="manual_discard"),
        )

    @learned_walkable_features.setter
    def learned_walkable_features(self, val: Any) -> None:
        if val:
            for f in val:
                self.symbolic_theory.induce_walkable(f, evidence="learned_walkable_features_setter")

    @property
    def learned_cargo_features(self) -> GovernedSetView:
        return GovernedSetView(
            self.symbolic_theory.cargo_features,
            on_add=lambda f: self.symbolic_theory.induce_cargo(f, evidence="manual"),
            on_discard=lambda f: self.symbolic_theory.retract_cargo(f, reason="manual_discard"),
        )

    @learned_cargo_features.setter
    def learned_cargo_features(self, val: Any) -> None:
        if val:
            for f in val:
                self.symbolic_theory.induce_cargo(f, evidence="learned_cargo_features_setter")

    @property
    def learned_receptacle_features(self) -> GovernedSetView:
        return GovernedSetView(
            self.symbolic_theory.receptacle_features,
            on_add=lambda f: self.symbolic_theory.induce_receptacle(f, evidence="manual"),
            on_discard=lambda f: self.symbolic_theory.retract_receptacle(
                f, reason="manual_discard"
            ),
        )

    @learned_receptacle_features.setter
    def learned_receptacle_features(self, val: Any) -> None:
        if val:
            for f in val:
                self.symbolic_theory.induce_receptacle(
                    f, evidence="learned_receptacle_features_setter"
                )

    # ── Epistemic status of feature values ──────────────────────────────────────
    UNVERIFIED_FEATURE_COST: float = 8.0

    def is_feature_unverified(self, feat: int, bg: int) -> bool:
        """True if the agent has neither experience nor empirical TransitionKB belief for ``feat``."""
        av = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else ()
        )
        if feat == bg or feat in av:
            return False
        return self.symbolic_theory.evaluate_transition(feat).confidence <= 0.0

    def _epistemic_uncertainty_cost(self, feat: int, bg: int) -> float:
        """Soft exploration cost for stepping onto an unverified feature."""
        return self.UNVERIFIED_FEATURE_COST if self.is_feature_unverified(feat, bg) else 0.0

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Load executive cognitive directives into working memory to guide epistemic policies."""
        self.working_memory.load_instructions(instructions)

    @property
    def current_context_id(self) -> str:
        return self._current_context_id

    @current_context_id.setter
    def current_context_id(self, val: str) -> None:
        self._current_context_id = str(val)
        try:
            epoch = int(val)
        except (ValueError, TypeError):
            logger.debug("Non-integer context_id %r retains epoch %d", val, self._context_epoch)
        else:
            self.set_context_epoch(epoch)

    @property
    def context_epoch(self) -> int:
        return self._context_epoch

    @context_epoch.setter
    def context_epoch(self, epoch: int) -> None:
        self.set_context_epoch(epoch)

    @property
    def current_level(self) -> int:
        return self.context_epoch

    @current_level.setter
    def current_level(self, level: int) -> None:
        self.set_context_epoch(level)

    def set_context_epoch(self, epoch: int) -> None:
        if self._context_epoch != epoch:
            self._context_epoch = epoch
            self.working_memory.visuospatial.context_epoch = epoch
            self.habenular_ior.context_epoch = epoch
            self.hazard_tracker.static_lethal_positions = self.context_hazards.setdefault(
                epoch, set()
            )

    # ── Episodic State Management ─────────────────────────────────────────────

    def on_boundary(self, boundary: Any) -> None:
        """Episode lifecycle boundary dispatch (WP0.5).

        Kinds: DEATH (same level retry), LEVEL_COMPLETE (advance),
        CONTEXT_RESET (lattice/context change mid-stream). Motor posteriors
        are context-local (retained across DEATH/LEVEL_COMPLETE); level-local
        state (goals, inhibitions, object ids, mental plans) is cleared;
        long-term skills are lifelong.
        """
        from hbllm.hcir.world.percept import EpisodeBoundaryKind, resolve_boundary_flags

        kind, retain_dyn, is_same_context = resolve_boundary_flags(boundary)
        if kind in (
            EpisodeBoundaryKind.GOAL_REACHED,
            EpisodeBoundaryKind.LEVEL_COMPLETE,
            "goal_reached",
            "level_complete",
        ):
            if (
                not self.state.feedback_assimilated
                and self.prev_grid is not None
                and self.last_action is not None
            ):
                self.assimilate_feedback(self.prev_grid, [], is_success=True)
            ep = self.episodic_cortex
            has_unconsolidated_win = bool(ep.current_episode) or (
                len(ep.successful_trajectories) > getattr(ep, "consolidated_trajectory_count", 0)
            )
            if has_unconsolidated_win:
                try:
                    self.consolidate_episodic_memory(terminal_reward=1.0)
                except Exception as exc:  # consolidation must never break the lifecycle
                    logger.debug("Episodic consolidation failed at boundary: %s", exc)
            self.reset_episode(retain_dynamics=retain_dyn, is_new_context=not is_same_context)
        elif kind == EpisodeBoundaryKind.DEATH or kind == "death":
            if (
                not self.state.feedback_assimilated
                and self.prev_grid is not None
                and self.last_action is not None
            ):
                self.assimilate_feedback(self.prev_grid, [], is_failure=True)
            self.reset_episode(retain_dynamics=retain_dyn, is_new_context=not is_same_context)
        else:
            self.reset_episode(retain_dynamics=retain_dyn, is_new_context=not is_same_context)
        if not is_same_context and self.symbolic_theory.labels_non_stationary:
            from hbllm.hcir.world.cortex_causal import invalidate_stale_context_ontology

            invalidate_stale_context_ontology(
                self.symbolic_theory,
                hazard_tracker=self.hazard_tracker,
                episodic_cortex=self.episodic_cortex,
                working_memory=self.working_memory,
                engine_or_view=self,
                bg_feature=self.bg_feature,
                preserve_current_context=False,
                reason="non_stationary_episode_boundary",
            )
        # Fan out to every faculty exposing a boundary handler (WP0.5).
        for attr in vars(self).values():
            handler = getattr(attr, "on_boundary", None)
            if callable(handler) and attr is not self and attr is not self.object_planner:
                try:
                    handler(boundary)
                except Exception as exc:
                    logger.debug("Boundary handler %r failed: %s", type(attr).__name__, exc)

    def reset_episode(
        self,
        retain_dynamics: bool = True,
        is_new_context: bool | None = None,
        context_epoch: int | None = None,
        *,
        is_new_level: bool | None = None,
        level: int | None = None,
    ) -> None:
        """Reset episodic state upon context transition or catastrophe.
        Delegated to HippocampalEpisodicCortex for schema consolidation.

        Deprecated: prefer on_boundary(EpisodeBoundary) (WP0.5).
        """
        new_context = (
            is_new_context
            if is_new_context is not None
            else (is_new_level if is_new_level is not None else False)
        )
        target_epoch = context_epoch if context_epoch is not None else level
        from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta

        delta = StateDelta()
        self.episodic_cortex.consolidate_schema_and_reset(
            ReadOnlyEngineView(self),
            delta,
            retain_dynamics=retain_dynamics,
            is_new_context=new_context,
            context_epoch=target_epoch,
        )
        self.apply_delta(delta, author="episodic_cortex")
        if isinstance(self._entity_cache, dict):
            self._entity_cache.clear()
        self.working_memory.structural_goals_cache = None
        if new_context or not retain_dynamics:
            self.active_arbitrated_goal = None
        self.step_counter = 0
        if new_context:
            self.environmental_step_limit = None
            self.budget_governor.reset(max_budget=self.exploration_budget)
        else:
            eff_budget = (
                self.environmental_step_limit
                if self.environmental_step_limit is not None
                else self.exploration_budget
            )
            self.budget_governor.reset(max_budget=eff_budget)

    def consolidate_episodic_memory(self, terminal_reward: float = 1.0) -> dict[str, Any]:
        """Consolidate accumulated experiences via Hippocampal SWR replay & schema distillation.

        Biologically invokes sharp-wave ripple forward and reverse replay over the
        active episodic buffer, distills invariant schemas into declarative skills,
        induces program DSL routines, registers resulting causal predicates in symbolic
        world theory, and commits mutations through TransactionManager if available.
        """
        res = self.episodic_cortex.consolidate_sleep_replay(terminal_reward=terminal_reward)
        schemas = res.get("schemas", [])
        for schema in schemas:
            self.symbolic_theory.register_predicate(schema.causal_predicate)
            if hasattr(self, "procedural_macros"):
                self.procedural_macros[schema.schema_id] = schema.action_motif

        # Propose state mutations through TransactionManager if present
        all_ops = list(res.get("operations", []))

        if self.transaction_manager is not None and all_ops:
            from hbllm.hcir.transactions import HCIRTransaction

            for i in range(0, len(all_ops), 400):
                tx = HCIRTransaction(
                    author="HippocampalReplayConsolidator",
                    operations=all_ops[i : i + 400],
                )
                self.transaction_manager.commit(tx)

        return res

    def retrieve_relational_analogies(
        self,
        grid: np.ndarray,
        min_match_score: float = 0.5,
    ) -> list[tuple[Any, float, list[Any]]]:
        """Extract object scene graph from grid and retrieve analogous relational concepts (G6)."""
        from hbllm.hcir.grounding.concepts.relational_graph_matcher import (
            RelationalGraphMatcher,
        )

        query_graph = RelationalGraphMatcher.build_scene_graph(grid)
        return self.working_concept_store.retrieve_analogous_concepts(
            query_graph=query_graph,
            min_match_score=min_match_score,
        )

    def query_latent_counterfactual(
        self,
        action_type: str,
        target_role: str,
        hypothetical_latent_state: bool,
    ) -> dict[str, Any] | None:
        """Evaluate Pearl Level 3 counterfactual prediction over hypothetical latent state (G7)."""
        return self.schema_inducer.evaluate_latent_counterfactual(
            action_type=action_type,
            target_role=target_role,
            hypothetical_latent_state=hypothetical_latent_state,
        )

    def explain_last_decision(self) -> str:
        """Generate human-interpretable natural language explanation of the last decision (G8 / §15)."""
        act = self.last_action
        if act is None:
            return "No decision has been made yet: agent is in an uninitialized idle state."

        phase = getattr(self, "epistemic_phase", "Active Deliberation")
        target = getattr(self, "active_subgoal", None) or getattr(self, "current_target", None)

        parts = [f"Chose motor action {act} during epistemic phase '{phase}'."]
        if target is not None:
            parts.append(f"Pursuing target/subgoal at {target}.")

        if self.symbolic_theory is not None:
            active_causes = list(self.symbolic_theory.causal_predicates)
            if active_causes:
                parts.append(
                    f"Conditioned on active causal hypothesis '{active_causes[-1].variable} {active_causes[-1].operator} {active_causes[-1].value}'."
                )

        return " ".join(parts)

    def record_domain_outcome(
        self,
        domain_id: str,
        success: bool,
        mean_error: float = 0.20,
        predicted_confidence: float = 0.80,
    ) -> Any:
        """Record domain outcome in prefrontal working memory metacognitive competence model (G9)."""
        return self.working_memory.record_domain_episode(
            domain_id=domain_id,
            success=success,
            mean_error=mean_error,
            predicted_confidence=predicted_confidence,
        )

    def get_highest_priority_practice_domain(self) -> str | None:
        """Get the domain currently in the Zone of Proximal Development with highest practice priority."""
        return self.working_memory.get_highest_priority_practice_domain()

    def discover_and_induce_options(self, grid: Any | None = None) -> list[Any]:
        """Discover topological bottleneck states and induce temporal options (G11)."""
        current_grid = grid if grid is not None else getattr(self, "grid", None)
        if current_grid is None or not isinstance(current_grid, np.ndarray):
            return []

        bottlenecks = self.option_discovery.discover_bottlenecks_from_grid(current_grid)
        options = []
        for b in bottlenecks:
            opt = self.option_discovery.induce_option(bottleneck=b)
            options.append(opt)
        return options

    def record_option_usage(self, option_id: str, steps: int, success: bool = True) -> None:
        """Record option execution steps and success to track reuse rate (G11)."""
        self.option_discovery.record_option_execution(
            option_id=option_id, steps=steps, success=success
        )

    def get_option_reuse_rate(self) -> float:
        """Get the current option reuse rate across episodes (G11)."""
        return self.option_discovery.get_option_reuse_rate()

    def record_action_effect(self, action: Any, kind: str) -> None:
        """Increment the per-action effect-kind count (WP0.3)."""
        counts = self.action_effect_counts.setdefault(action, {})
        counts[kind] = counts.get(kind, 0) + 1
        if kind in ("no_op", "blocked"):
            self.working_memory.consecutive_null_actions[action] = (
                self.working_memory.consecutive_null_actions.get(action, 0) + 1
            )
        else:
            self.working_memory.consecutive_null_actions[action] = 0

    def motor_posterior(self, action: Any) -> dict[str, float]:
        """Posterior P(effect_kind | action) from observed samples (WP0.3)."""
        counts = self.action_effect_counts.get(action)
        if not counts:
            return {}
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def expected_effect(self, action: Any) -> str | None:
        """Most probable effect kind for an action (WP0.3)."""
        post = self.motor_posterior(action)
        return max(post, key=lambda k: post[k]) if post else None

    def effect_entropy(self, action: Any) -> float:
        """Shannon entropy over the action's effect posterior (WP0.3)."""
        import math

        post = self.motor_posterior(action)
        if not post:
            return 1.0  # unprobed action -> maximally uncertain
        return -sum(p * math.log2(p) for p in post.values() if p > 0)

    def consecutive_null_count(self, action: Any) -> int:
        return self.working_memory.consecutive_null_actions.get(action, 0)

    def set_environmental_step_limit(self, limit: int) -> None:
        """Set strict environmental step countdown or step limit."""
        self.environmental_step_limit = limit
        self.exploration_budget = limit
        self.budget_governor.reset(max_budget=limit)

    def get_remaining_step_budget(self) -> int:
        """Return remaining step budget for the current level/episode."""
        if self.environmental_step_limit is not None:
            return max(0, self.environmental_step_limit - self.step_counter)
        return self.budget_governor.get_remaining_budget()

    def is_under_strict_step_budget(self) -> bool:
        """True if operating under a tight step countdown or low remaining budget."""
        rem = self.get_remaining_step_budget()
        return rem <= 25 or (
            self.environmental_step_limit is not None and self.environmental_step_limit <= 35
        )

    def is_motor_grounded(self) -> bool:
        """True if the agent has identified its avatar and calibrated directional actions,
        OR if the environment is allocentric/actuation-only (no displacement actions exist) and enough
        exploratory actuations have been made to characterize the environment."""
        # Standard motor grounding: avatar + calibrated displacement actions
        if (self.avatar_feature is not None or bool(self.avatar_features)) and any(
            dyn.confidence >= 0.7 for dyn in self.action_dynamics.values()
        ):
            return True
        # Point-actuation-only environment grounding: if no displacement dynamics were ever
        # calibrated and only spatial effectors are known, treat as grounded after
        # enough exploration to let the object planner engage.
        has_displacement = any(
            dyn.is_displacement_action() for dyn in self.action_dynamics.values()
        )
        if not has_displacement and self.action_dynamics and self.total_epistemic_probes >= 5:
            return True
        return False

    # ── Delegated Public Interfaces ───────────────────────────────────────────

    def perceive(self, raw_observation: Any) -> PerceptionFrame:
        """Lift raw sensory input into an abstract PerceptionFrame using the registered sensory adapter."""
        if isinstance(raw_observation, PerceptionFrame):
            return raw_observation
        adapter = self.perception_adapter or PerceptionRegistry.get_default()
        if adapter is not None:
            return adapter.lift_percept(raw_observation, context=self)
        if not (hasattr(raw_observation, "shape") and len(raw_observation.shape) >= 2):
            raise ValueError("lattice extent unknown")
        H, W = raw_observation.shape[0], raw_observation.shape[1]
        return PerceptionFrame(shape=(H, W))

    def estimate_background(self, grid: np.ndarray) -> Any:
        theory = self.symbolic_theory
        goals = set(theory.goal_features)
        walkables = {
            w
            for w in theory.walkable_features
            if w not in goals and theory.has_verified_walkable(w)
        }
        barriers = set(theory.barrier_features)
        barriers |= set(self.hazard_tracker.known_lethal_features)
        barriers_arg: set[int] | None = barriers or None
        bg = PerceptionEngine.estimate_background(
            grid,
            barrier_features=barriers_arg,
            avatar_features=self.avatar_features,
            bg_feature=self.bg_feature,
            walkable_features=walkables,
        )
        if self.bg_feature is None or self.bg_feature != bg:
            old_bg = self.bg_feature
            # A monochrome win/loss transition flash should not shift background ontology.
            if int(np.unique(grid).size) <= 1 and old_bg is not None:
                return old_bg
            self.bg_feature = bg
            if theory is not None:
                if old_bg is not None and old_bg != bg and not theory.has_verified_walkable(old_bg):
                    theory.retract_walkable(old_bg, reason="background_shift")
                elif (
                    old_bg is None
                    and self.default_bg_feature != bg
                    and not theory.has_verified_walkable(self.default_bg_feature)
                ):
                    theory.retract_walkable(self.default_bg_feature, reason="background_shift")
                if (
                    bg not in theory.barrier_features
                    and bg not in theory.avatar_features
                    and bg not in theory.goal_features
                    and bg not in theory.cargo_features
                    and not theory.has_verified_walkable(bg)
                ):
                    theory.induce_walkable(bg, evidence="background_shift")
        return bg

    def _perception_cache_key(self, grid: np.ndarray, bg: int) -> tuple[Any, ...]:
        """Key capturing every input that can change entity segmentation or role assignment."""
        theory = self.symbolic_theory
        theory_sig: tuple[Any, ...] = ()
        if theory is not None:
            theory_sig = tuple(
                frozenset(getattr(theory, attr, ()) or ())
                for attr in (
                    "goal_features",
                    "candidate_goal_features",
                    "receptacle_features",
                    "walkable_features",
                    "barrier_features",
                    "cargo_features",
                )
            )
        return (
            grid.shape,
            grid.tobytes(),
            bg,
            self.avatar_pos,
            frozenset(self.avatar_features),
            len(self.state_mutations),
            frozenset(self.learned_goal_positions),
            frozenset(self.learned_receptacle_positions),
            theory_sig,
        )

    def extract_entities(self, grid: np.ndarray, bg: int) -> list[SpatialEntity]:
        """Segment the frame into typed entities.

        Memoised per frame: perception is invoked by several faculties within one
        cognitive step on the same observation, so the (expensive) connected-component
        lifting is computed once per distinct (frame, theory) key. Callers receive
        fresh ``SpatialEntity`` copies so role reassignment never leaks between callers.
        """
        key = self._perception_cache_key(grid, bg)
        cached = self._entity_cache.get(key)
        if cached is None:
            cached = PerceptionEngine.extract_entities(
                grid,
                bg,
                symbolic_theory=self.symbolic_theory,
                avatar_features=self.avatar_features,
                state_mutations=self.state_mutations,
                learned_barriers=self.learned_barriers,
                learned_goal_positions=self.learned_goal_positions,
                learned_receptacle_positions=self.learned_receptacle_positions,
                avatar_pos=self.avatar_pos,
            )
            if len(self._entity_cache) >= 4:
                self._entity_cache.pop(next(iter(self._entity_cache)))
            self._entity_cache[key] = cached
            post_key = self._perception_cache_key(grid, bg)
            if post_key != key:
                self._entity_cache[post_key] = cached
        return [copy.copy(e) for e in cached]

    def compute_frame_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        return PerceptionEngine.compute_frame_diff(prev_grid, curr_grid)

    def detect_structural_goals(
        self, grid: np.ndarray, bg: int | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        bg_val = bg if bg is not None else self.bg_feature
        # Anytime cognition (WP0.2): cache by frame-content hash — identical
        # frames yield identical structural goals; skip re-analysis.
        try:
            import hashlib

            _h = int.from_bytes(
                hashlib.blake2b(np.ascontiguousarray(grid).tobytes(), digest_size=8).digest(),
                "little",
            )
            if (
                self.working_memory.structural_goals_cache
                and self.working_memory.structural_goals_cache[0] == _h
            ):
                return list(self.working_memory.structural_goals_cache[1])
        except Exception:
            _h = None
        result = PerceptionEngine.detect_structural_goals(
            grid,
            bg=bg_val,
            avatar_features=self.avatar_features,
            avatar_feature=self.avatar_feature,
            barrier_features=set(self.symbolic_theory.barrier_features),
            known_lethal_features=self.hazard_tracker.known_lethal_features,
            avatar_pos=self.avatar_pos,
            **kwargs,
        )
        if _h is not None:
            self.working_memory.structural_goals_cache = (_h, list(result))
        return result

    def assimilate_feedback(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        terminal: TerminalSignal | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action: int | None = None,
        action_data: dict[str, Any] | None = None,
        actuation_rejected: bool = False,
        **kwargs: Any,
    ) -> None:
        if actuation_rejected:
            # The environment rejected the action — this sample teaches
            # nothing about dynamics, so skip binding it (WP0.4).
            return
        is_success = is_success or terminal == TerminalSignal.SUCCESS
        is_failure = is_failure or terminal in (TerminalSignal.FAILURE, TerminalSignal.TIMEOUT)
        curr_grid = self.normalize_sensory_input(curr_grid)
        if action is None:
            action = self.last_action
        if action_data is None:
            action_data = self.last_action_data
        prev_av_pos = self.avatar_pos

        # Synthesize active subgoal schema into reusable procedural skill upon level win
        if is_success and self.working_memory.active_schema is not None:
            self.working_memory.synthesize_skill_from_schema(self.working_memory.active_schema)

        from hbllm.hcir.world.engine_state import (
            FacultyWriteJournal,
            JournaledEngineView,
        )

        journal = FacultyWriteJournal(author="assimilator")
        EpistemicFeedbackAssimilator.assimilate(
            JournaledEngineView(self, journal),
            curr_grid,
            available_actions,
            terminal=terminal,
            is_success=is_success,
            is_failure=is_failure,
            action=action,
            action_data=action_data,
        )
        self.commit_faculty_journal(journal)

        # Biological Faculty: Popperian Inductive Hypothesis Consolidation & Retraction (Track 2)
        self.working_memory.hypothesis_engine.export_to_world_theory(self.symbolic_theory)
        self.working_memory.hypothesis_engine.retract_refuted_rules(self.symbolic_theory)

        if action is not None:
            self.budget_governor.record_step()
        if is_success or is_failure:
            self.working_memory.prev_oriented_threats = []
            if self.grounding_bridge is not None:
                sync_fn = getattr(self.grounding_bridge, "sync_theory_roles", None)
                if callable(sync_fn):
                    sync_fn(
                        self.symbolic_theory,
                        hazard_features=self.hazard_tracker.known_lethal_features,
                    )
            if self.tiered_workspace is not None:
                try:
                    self.persist_to_workspace(author="episode_boundary")
                except Exception as exc:
                    logger.debug("Workspace persistence at episode boundary failed: %s", exc)

        if is_success and self.tiered_workspace is not None and self.active_task_frame is not None:
            # Commit only delta/missing working memory state to active task frame
            self.commit_working_memory_transactions()

            # Promote verified goals, ontology, and procedural skills when present in WORKING task frames
            active_frames = getattr(self.tiered_workspace, "active_task_frames", ())
            if active_frames:

                def _in_working(nid: str) -> bool:
                    return any(f.workspace.get_node(nid) is not None for f in active_frames)

                all_goals: set[tuple[int, int]] = set(self.learned_goal_positions)
                for g_set in self.working_memory.visuospatial.context_goals.values():
                    all_goals.update(g_set)
                promote_ids = list(self.symbolic_theory.transition_kb.exported_node_ids())
                promote_ids.extend(
                    f"ontology_goal_{goal_feat}" for goal_feat in self.symbolic_theory.goal_features
                )
                promote_ids.extend(f"learned_goal_{gp[0]}_{gp[1]}" for gp in all_goals)
                promote_ids.extend(
                    f"procedural_skill_{schema_name}"
                    for schema_name in self.working_memory.procedural_library
                )
                working_promote_ids = [nid for nid in promote_ids if _in_working(nid)]
                if working_promote_ids:
                    self.tiered_workspace.promote_batch(
                        working_promote_ids,
                        from_tier=WorkspaceTier.WORKING,
                        to_tier=WorkspaceTier.PERSISTENT,
                    )

        # Attribute environmental mutations to active object interaction
        active_obj_id = getattr(self.object_planner, "active_target_object_id", None)
        if active_obj_id and self.prev_grid is not None and self.prev_grid.shape == curr_grid.shape:
            diff_mask = self.prev_grid != curr_grid
            if (
                prev_av_pos
                and 0 <= prev_av_pos[0] < curr_grid.shape[0]
                and 0 <= prev_av_pos[1] < curr_grid.shape[1]
            ):
                diff_mask[prev_av_pos[0], prev_av_pos[1]] = False
            if (
                self.avatar_pos
                and 0 <= self.avatar_pos[0] < curr_grid.shape[0]
                and 0 <= self.avatar_pos[1] < curr_grid.shape[1]
            ):
                diff_mask[self.avatar_pos[0], self.avatar_pos[1]] = False
            diff_coords = np.argwhere(diff_mask)
            if len(diff_coords) > 0:
                mutated_cells = {(int(r), int(c)) for r, c in diff_coords}
                self.object_planner.ledger.record_causal_effect(
                    object_id=active_obj_id,
                    effect_type="MUTATED_CELLS",
                    affected_coords=mutated_cells,
                )

    def simulate_in_mind(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        return MentalSimulationPlanner.simulate_in_mind(
            self,
            curr_grid,
            available_actions,
            target_role=target_role,
            blocked_cells=blocked_cells,
        )

    def schedule_causal_subgoal_chain(
        self,
        target_destination: tuple[int, int],
        target_type: SubgoalType,
        causal_prerequisites: list[tuple[str, tuple[int, int], SubgoalType, int | None]],
        target_id: str = "master_causal_target",
        priority: float = 2.0,
    ) -> list[FrontopolarSubgoal]:
        """Schedule a chained sequence of prerequisite subgoals leading to a master goal."""
        master_sg = FrontopolarSubgoal(
            subgoal_id=target_id,
            subgoal_type=target_type,
            target_entity_pos=target_destination,
            target_destination=target_destination,
            priority=priority,
            is_terminal=True,
        )
        return self.subgoal_stack.decompose_causal_chain(master_sg, causal_prerequisites)

    def plan_epistemic_probe(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        return EpistemicCuriosityExplorer.plan_epistemic_probe(self, curr_grid, available_actions)

    def apply_probe_proposal(self, proposal: EpistemicProbeProposal) -> None:
        """Commit an EpistemicProbeProposal's state updates atomically."""
        from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta

        delta = StateDelta()
        EpistemicCuriosityExplorer.apply_proposal_to_engine(
            view=ReadOnlyEngineView(self), delta=delta, proposal=proposal
        )
        self.apply_delta(delta, author="epistemic_curiosity")

    def _update_avatar_position_from_grid(
        self, curr_grid: np.ndarray, known_av_feats: set[int]
    ) -> None:
        """Visually localize the avatar position on the current grid."""
        H, W = curr_grid.shape
        from scipy.ndimage import label

        mask = np.isin(curr_grid, list(known_av_feats))
        if not np.any(mask):
            return

        labeled, num_features = label(mask)
        if num_features == 0:
            return

        # If avatar was already precisely localized by assimilate_feedback via motion tracking,
        # and current avatar_pos contains an avatar feature and matches avatar_size, verify and keep it
        if self.avatar_pos is not None:
            r, c = self.avatar_pos
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                comp_lbl = labeled[r, c]
                if comp_lbl > 0:
                    coords = np.argwhere(labeled == comp_lbl)
                    min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                    min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                    is_enclosing_frame = (
                        min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                    ) or (
                        min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4)
                    )
                    if not is_enclosing_frame and not self.working_memory.is_exogenous_locus(
                        (r, c), lattice=curr_grid, bg=self.bg_feature, include_outer_edge=True
                    ):
                        comp_area = len(coords)
                        if self.avatar_size == 0 or (
                            0.25 * self.avatar_size <= comp_area <= 2.5 * self.avatar_size
                        ):
                            return

        best_pos = None
        min_score = float("inf")

        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            area = len(coords)
            if 1 <= area <= max(49, int(H * W * 0.15)):
                min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                # Filter out enclosing outer boundary stripes / structural frames
                is_enclosing_frame = (
                    min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))
                if is_enclosing_frame:
                    continue

                centroid = (
                    int(round(float(np.mean(coords[:, 0])))),
                    int(round(float(np.mean(coords[:, 1])))),
                )
                if self.working_memory.is_exogenous_locus(
                    centroid, lattice=curr_grid, bg=self.bg_feature, include_outer_edge=True
                ):
                    continue
                if self.avatar_pos is not None:
                    dist = abs(centroid[0] - self.avatar_pos[0]) + abs(
                        centroid[1] - self.avatar_pos[1]
                    )
                    size_diff = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                    score = dist + size_diff * 0.5
                else:
                    # Fresh localization: choose component best matching known avatar_size
                    score = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                if score < min_score:
                    min_score = score
                    best_pos = centroid

        if best_pos is not None:
            self.avatar_pos = best_pos

    def ground_effector_action(
        self, curr_grid: np.ndarray, action: Any = None
    ) -> dict[str, Any] | None:
        """Spatially ground an allocentric effector command onto salient affordances."""
        return MotorCortexEffector.ground_effector_action(self, curr_grid, action)

    # ── Theory of Mind: Creature Motion Observation ──────────────────────────

    def infer_motor_step_size(self, available_actions: Sequence[Any]) -> int:
        """Largest calibrated displacement quantum (the world's 'cell' size)."""
        return MotorCortexEffector.infer_motor_step_size(available_actions, self.action_dynamics)

    def _observe_oriented_threat_motion(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        av_feats: set[int],
    ) -> None:
        """Learn which creature kinds patrol by watching them translate.

        A human watching a creature notices: "that one walked one cell along its
        facing direction while I moved — it patrols". Stationary sentries never
        translate. This is a feature-level concept, so it generalizes to other
        creatures of the same kind in later levels.
        """
        effective_av_feats = set(av_feats) | set(self.avatar_features)
        if self.avatar_feature is not None:
            effective_av_feats.add(self.avatar_feature)
        if self.prior_avatar_feature is not None:
            effective_av_feats.add(self.prior_avatar_feature)

        if not effective_av_feats:
            self.working_memory.prev_oriented_threats = []
            return
        try:
            bg = self.estimate_background(curr_grid)
            entities = self.extract_entities(curr_grid, bg)
            threats = PerceptionEngine.detect_oriented_threats(
                curr_grid,
                entities,
                bg=bg,
                step_size=self.infer_motor_step_size(available_actions),
                avatar_pos=self.avatar_pos,
                avatar_features=effective_av_feats,
            )
        except Exception:
            self.working_memory.prev_oriented_threats = []
            return

        if self.step_counter <= 1 or self.prev_grid is None:
            self.working_memory.prev_oriented_threats = threats
            return

        step_sz = self.infer_motor_step_size(available_actions)
        prev = self.working_memory.prev_oriented_threats
        if prev:
            prev_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in prev:
                prev_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            curr_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in threats:
                curr_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            for feat, cur_positions in curr_by_feat.items():
                if (
                    feat in effective_av_feats
                    or feat in self.mobile_threat_features
                    or feat in self.symbolic_theory.cargo_features
                ):
                    continue
                old_positions = prev_by_feat.get(feat, set())
                appeared = cur_positions - old_positions
                vanished = old_positions - cur_positions
                if appeared and len(appeared) == len(vanished):
                    for a in appeared:
                        if any(
                            (a[0] == v[0]) != (a[1] == v[1])
                            and 0 < (abs(a[0] - v[0]) + abs(a[1] - v[1])) <= step_sz * 2
                            for v in vanished
                        ):
                            self.mobile_threat_features.add(feat)
                            self.consecutive_simulation_failures = 0
                            self.simulation_cooldown = 0
                            self.mental_plan.clear()
                            self.transition_to_phase(
                                EpistemicPhase.MENTAL_SIMULATION,
                                reason="observed_creature_patrolling",
                            )
                            logger.info(
                                "AutonomousEpistemicEngine: Observed creature kind %d patrolling "
                                "autonomously — modeling its trajectory.",
                                feat,
                            )
                            break
        self.working_memory.prev_oriented_threats = threats

    # ── Master Decision Loop ──────────────────────────────────────────────────

    def step(
        self, obs: EmbodiedObservation, last_action: Any | None = None
    ) -> tuple[Any, dict[str, Any] | None]:
        """Canonical domain-agnostic step entry point for the epistemic engine.

        Consumes an EmbodiedObservation and executes active inference deliberation.
        """
        if last_action is not None:
            self.last_action = last_action
        elif obs.metadata and "last_action" in obs.metadata:
            self.last_action = obs.metadata["last_action"]
        elif obs.metadata and "action_taken" in obs.metadata:
            self.last_action = obs.metadata["action_taken"]

        if obs.metadata:
            self.working_memory.last_observation_metadata = dict(obs.metadata)
            if "reference_frame" in obs.metadata:
                self.working_memory.reference_frame = obs.metadata["reference_frame"]
            if "agent_pos" in obs.metadata:
                curr_cont = np.asarray(obs.metadata["agent_pos"], dtype=float)
                prev_cont = self.working_memory.continuous_agent_pos
                if prev_cont is not None:
                    d_pos = curr_cont - prev_cont
                    disp = float(np.linalg.norm(d_pos))
                    self.working_memory.continuous_displacement = disp
                    self.working_memory.last_continuous_moved = disp > 1e-4
                    self.working_memory.continuous_delta = (float(d_pos[0]), float(d_pos[1]))
                else:
                    self.working_memory.continuous_displacement = 0.0
                    self.working_memory.last_continuous_moved = False
                    self.working_memory.continuous_delta = (0.0, 0.0)
                self.working_memory.continuous_agent_pos = curr_cont.copy()

        self.working_memory.update_continuous_goal_from_lattice(
            obs.lattice,
            self.symbolic_theory,
            self.avatar_features,
            self.avatar_feature,
            self.bg_feature,
        )

        return self.decide(
            curr_grid=obs.lattice,
            available_actions=obs.available_actions,
            terminal=obs.terminal,
            context_id=obs.context_id,
            episode_id=obs.episode_id,
            objects=list(obs.entities) if obs.entities else None,
            action_schemas=(
                [
                    {
                        "action_id": (
                            int(a.actuator_id)
                            if isinstance(a.actuator_id, str) and a.actuator_id.isdigit()
                            else a.actuator_id
                        ),
                        "kind": a.kind,
                        "valid_targets": list(a.valid_targets) if a.valid_targets else None,
                        **(dict(a.params) if a.params else {}),
                    }
                    for a in obs.actuators
                ]
                or None
            ),
            **{
                k: obs.metadata[k]
                for k in ("actuation_rejected", "step_limit")
                if k in obs.metadata
            },
        )

    def decide(
        self,
        curr_grid: Any,
        available_actions: Sequence[Any],
        terminal: TerminalSignal | None = None,
        is_success: bool = False,
        is_failure: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        Delegated to PrefrontalDeliberationEngine.
        """
        if (lim := kwargs.pop("step_limit", None)) is not None:
            self.set_environmental_step_limit(int(lim))
        if (ctx_id := kwargs.get("context_id")) is not None and str(ctx_id):
            if str(ctx_id) != "default" or self._current_context_id == "default":
                self.current_context_id = str(ctx_id)
        from hbllm.hcir.world.compute_budget import ComputeBudget

        self.compute_budget = ComputeBudget.for_step(
            self.step_work_units,
            remaining_actions=self.get_remaining_step_budget(),
            deterministic=self.deterministic_budget,
            step_budget_ms=None if self.deterministic_budget else self.step_budget_ms,
        )
        if curr_grid is not None and terminal is None:
            av_set = set(self.avatar_features or ()) | (
                {self.avatar_feature} if self.avatar_feature is not None else set()
            )
            curr_grid = self.working_memory.integrate_sensory_observation(
                curr_grid, av_set, self.bg_feature
            )
        is_success = is_success or terminal == TerminalSignal.SUCCESS
        is_failure = is_failure or terminal in (TerminalSignal.FAILURE, TerminalSignal.TIMEOUT)
        from hbllm.hcir.world.engine_state import CognitiveContext

        self._pipeline_runs += 1
        from types import MappingProxyType

        ctx = CognitiveContext(
            curr_grid=curr_grid,
            step_counter=self.step_counter,
            signals=MappingProxyType(
                {
                    "_curr_grid": curr_grid,
                    "_available_actions": list(available_actions),
                    "_terminal": terminal,
                    "_is_success": is_success,
                    "_is_failure": is_failure,
                    "_action_schemas": action_schemas,
                    "_objects": objects,
                    "_micro_actions": micro_actions,
                    "_kwargs": dict(kwargs),
                }
            ),
        )
        result = self.cortical_pipeline.run(ctx)
        self._last_decision_branch = result.signals.get("branch")
        if result.rejected:
            r_msg = result.rejection_reason
            return None, {
                "pipeline_rejected": True,
                "reason": r_msg,
                "goal_target": None,
                "goal_source": "none",
                "subgoal_stack": [],
                "plan": None,
                "plan_source": "arbitration",
                "candidates": {},
                "chosen_action": None,
                "veto_reasons": [r_msg or "rejected"],
                "phase": str(self._phase),
                "branch": "rejected",
                "action_data": None,
            }

        act = (
            result.final_action[0]
            if isinstance(result.final_action, (tuple, list))
            else result.final_action
        )
        meta = (
            result.final_action[1]
            if isinstance(result.final_action, (tuple, list))
            and len(result.final_action) > 1
            and isinstance(result.final_action[1], dict)
            else dict(result.signals.get("meta", {}))
        )
        for k_def, v_def in (
            ("goal_target", None),
            ("goal_source", "none"),
            ("subgoal_stack", []),
            ("plan", None),
            ("plan_source", "arbitration"),
            ("candidates", {int(a): (1.0 if a == act else 0.0) for a in available_actions}),
            ("chosen_action", act),
            ("veto_reasons", []),
            ("phase", str(self._phase)),
            ("branch", result.signals.get("branch", "unknown")),
            ("action_data", None),
        ):
            meta.setdefault(k_def, v_def)

        if terminal is not None or is_success or is_failure:
            from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

            self.state.feedback_assimilated = True
            b_kind = EpisodeBoundaryKind.GOAL_REACHED if is_success else EpisodeBoundaryKind.DEATH
            self.on_boundary(EpisodeBoundary(kind=b_kind, same_level=not is_success))

        return act, meta

    def to_causal_predicate(self) -> CausalPredicate:
        """Export master epistemic engine state as a declarative causal predicate."""
        avatar_str = f"({self.avatar_pos[0]},{self.avatar_pos[1]})" if self.avatar_pos else "none"
        return CausalPredicate(
            variable="autonomous_epistemic_engine",
            operator="==",
            value=str(self._phase),
            relation=f"step:{self.step_counter}:avatar:{avatar_str}:goals:{len(self.learned_goal_positions)}",
            target_entity="epistemic_world_engine",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export composite causal predicates across all coordinated cortical faculties."""
        predicates = [self.to_causal_predicate()]
        if self.symbolic_theory:
            for p in self.symbolic_theory.goal_predicates:
                predicates.append(p)
            for p in self.symbolic_theory.barrier_predicates:
                predicates.append(p)
            for p in self.symbolic_theory.walkable_predicates:
                predicates.append(p)
        predicates.append(self.episodic_cortex.to_causal_predicate())
        predicates.extend(self.subgoal_stack.export_causal_predicates())
        return predicates

    def record_learned_barrier(self, barrier_pos: tuple[int, int]) -> None:
        """Record learned barrier locally and propose atomic state transaction."""
        if barrier_pos not in self.learned_barriers:
            self.learned_barriers.add(barrier_pos)
            kb = self.symbolic_theory.transition_kb
            if kb.get_belief(barrier_pos, scope="locus") is None:
                from hbllm.hcir.world.causal.transition_kb import TransitionOutcome

                kb.record_outcome(
                    barrier_pos,
                    TransitionOutcome.BLOCKED,
                    scope="locus",
                    evidence="learned_barrier",
                )
            bid = f"learned_barrier_{barrier_pos[0]}_{barrier_pos[1]}"
            bnode = ConstraintNode(
                id=bid,
                name=f"barrier_{barrier_pos[0]}_{barrier_pos[1]}",
                description=f"Learned impenetrable barrier at {barrier_pos}",
                target=f"position_{barrier_pos[0]}_{barrier_pos[1]}",
                properties={"position": list(barrier_pos), "role": "BARRIER"},
            )
            self._commit_operations(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=bid,
                        node_data=bnode.model_dump(),
                    )
                ],
                author="record_learned_barrier",
            )

    def record_learned_goal(self, goal_pos: tuple[int, int]) -> None:
        """Record learned goal locally and propose atomic state transaction."""
        self.working_memory.visuospatial.context_goals.setdefault(self.context_epoch, set()).add(
            goal_pos
        )
        if goal_pos not in self.learned_goal_positions:
            self.learned_goal_positions.add(goal_pos)
            self.state.learned_goal_positions.add(goal_pos)
            if self.symbolic_theory is not None:
                kb = self.symbolic_theory.transition_kb
                if kb.get_belief(goal_pos, scope="locus") is None:
                    from hbllm.hcir.world.causal.transition_kb import TransitionOutcome

                    kb.record_outcome(
                        goal_pos,
                        TransitionOutcome.TERMINAL_WIN,
                        scope="locus",
                        evidence="learned_goal",
                    )
            gid = f"learned_goal_{goal_pos[0]}_{goal_pos[1]}"

            gnode = GoalNode(
                id=gid,
                description=f"Learned goal location at {goal_pos}",
                properties={"position": list(goal_pos), "role": "GOAL"},
            )
            self._commit_operations(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=gid,
                        node_data=gnode.model_dump(),
                    )
                ],
                author="record_learned_goal",
            )

    def _record_predicate(self, obj: Any) -> None:
        if (
            self.symbolic_theory is not None
            and obj is not None
            and hasattr(obj, "to_causal_predicate")
        ):
            self.symbolic_theory.record_causal_predicate(obj.to_causal_predicate())

    def _commit_workspace(self) -> Any | None:
        """Resolve the TransactionManager-governed workspace (task-frame fallback)."""
        if self.transaction_manager is not None and self.transaction_manager.workspace is not None:
            return self.transaction_manager.workspace
        if self.active_task_frame is not None:
            return self.active_task_frame.workspace
        return None

    def _commit_operations(self, ops: list[TransactionOperation], author: str) -> None:
        """Commit ops to the engine's TM-governed workspace (task-frame fallback)."""
        if not ops:
            return
        ws = self._commit_workspace()
        if ws is not None:
            ws.commit_operations(ops, author=author)

    def _telemetry_operation(self) -> TransactionOperation:
        """Single op mirroring _MIRRORED executive fields as one ObservationNode."""
        node_id = "autonomous_epistemic_engine"
        engine_node = ObservationNode(
            id=node_id,
            description="Autonomous epistemic engine executive telemetry",
            sensor_source="autonomous_epistemic_engine",
            payload={
                "phase": str(self._phase),
                "step_counter": self.step_counter,
                "avatar_pos": list(self.avatar_pos) if self.avatar_pos else None,
                "avatar_feature": self._avatar_feature,
                "avatar_features": sorted(self.avatar_features),
                "tested_actions": sorted(self.tested_actions),
                "exploration_budget": self.exploration_budget,
                "consecutive_plan_failures": self.consecutive_plan_failures,
            },
        )
        return TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id=node_id,
            node_data=engine_node.model_dump(),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert epistemic engine state, internal memory, and knowledge base into transactional HCIR mutations."""
        ops = [self._telemetry_operation()]
        ops.extend(self.symbolic_theory.to_transaction_operations())
        ops.extend(self.working_memory.to_transaction_operations())
        all_goal_positions = set(self.learned_goal_positions)
        for g_set in self.working_memory.visuospatial.context_goals.values():
            all_goal_positions.update(g_set)
        for goal_pos in all_goal_positions:
            gid = f"learned_goal_{goal_pos[0]}_{goal_pos[1]}"
            gnode = GoalNode(
                id=gid,
                description=f"Learned goal location at {goal_pos}",
                properties={"position": list(goal_pos), "role": "GOAL"},
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=gid,
                    node_data=gnode.model_dump(),
                )
            )

        for barrier_pos in self.learned_barriers:
            bid = f"learned_barrier_{barrier_pos[0]}_{barrier_pos[1]}"
            bnode = ConstraintNode(
                id=bid,
                name=f"barrier_{barrier_pos[0]}_{barrier_pos[1]}",
                description=f"Learned impenetrable barrier at {barrier_pos}",
                target=f"position_{barrier_pos[0]}_{barrier_pos[1]}",
                properties={"position": list(barrier_pos), "role": "BARRIER"},
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=bid,
                    node_data=bnode.model_dump(),
                )
            )
        return ops

    @staticmethod
    def _filter_unchanged_upserts(
        dest_ws: Any, ops: list[TransactionOperation]
    ) -> list[TransactionOperation]:
        """Filter out UPSERT_NODE operations whose target node already exists with identical fields."""
        if dest_ws is None or not ops:
            return ops
        get_node = getattr(dest_ws, "get_node", None)
        if not callable(get_node):
            return ops
        sentinel = object()
        skip_keys = frozenset({"id", "provenance", "created_at", "updated_at"})
        filtered: list[TransactionOperation] = []
        for op in ops:
            if (
                op.op != TransactionOp.UPSERT_NODE
                or not op.node_id
                or not isinstance(op.node_data, dict)
            ):
                filtered.append(op)
                continue
            existing = get_node(op.node_id)
            if existing is None:
                filtered.append(op)
                continue
            same = True
            for k, v in op.node_data.items():
                if k in skip_keys:
                    continue
                if hasattr(v, "value"):
                    v = v.value
                ev_field = getattr(existing, k, sentinel)
                if ev_field is sentinel:
                    same = False
                    break
                if hasattr(ev_field, "value"):
                    ev_cmp = ev_field.value
                elif hasattr(ev_field, "model_dump") and isinstance(v, dict):
                    ev_cmp = ev_field.model_dump(mode="python")
                else:
                    ev_cmp = ev_field
                if ev_cmp != v:
                    same = False
                    break
            if not same:
                filtered.append(op)
        return filtered

    def persist_to_workspace(
        self,
        target: Any = None,
        author: str = "epistemic_engine",
    ) -> list[TransactionOperation]:
        """Propose and commit all world theory, concepts, and internal memory to persistent workspace."""
        from hbllm.hcir.workspace import HCIRWorkspaceState
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        ops = self.to_transaction_operations()
        dest_ws: HCIRWorkspaceState | None = None
        grant: Any = None

        if target is not None:
            if isinstance(target, TieredWorkspace):
                dest_ws = target.persistent
                grant = getattr(target, "_system_grant", None)
            elif isinstance(target, HCIRWorkspaceState):
                dest_ws = target
        elif self.services is not None:
            if self.services.tiered_workspace is not None:
                dest_ws = self.services.tiered_workspace.persistent
                grant = getattr(self.services.tiered_workspace, "_system_grant", None)
            else:
                dest_ws = self.services.workspace

        if grant is None and self.services is not None:
            reg = self.services.commit_authority_registry
            if reg is not None and dest_ws is not None:
                grant = reg.issue(author, {dest_ws.tier})

        if dest_ws is not None and ops:
            delta_ops = self._filter_unchanged_upserts(dest_ws, ops)
            batch_size = 400
            for i in range(0, len(delta_ops), batch_size):
                dest_ws.commit_operations(delta_ops[i : i + batch_size], author=author, grant=grant)
        return ops

    def hydrate_from_workspace(
        self,
        source: Any = None,
    ) -> int:
        """Hydrate world theory, concepts, and internal memory from workspace nodes.

        Returns the count of nodes hydrated.
        """
        from hbllm.hcir.workspace import HCIRWorkspaceState
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        src_ws: HCIRWorkspaceState | None = None

        if source is not None:
            if isinstance(source, TieredWorkspace):
                src_ws = source.persistent
            elif isinstance(source, HCIRWorkspaceState):
                src_ws = source
        elif self.services is not None:
            if self.services.tiered_workspace is not None:
                src_ws = self.services.tiered_workspace.persistent
            else:
                src_ws = self.services.workspace

        if src_ws is None:
            return 0

        nodes = list(src_ws.graph.all_nodes())
        if not nodes:
            return 0

        self.symbolic_theory.hydrate_from_nodes(nodes)

        # Hydrate goal positions, barriers, concepts, homeostatic state, and telemetry if present
        for node in nodes:
            props = node.properties or {}
            if node.id.startswith("learned_goal_") or node.id.startswith("learned_barrier_"):
                pos = props.get("position")
                if pos and isinstance(pos, (list, tuple)):
                    pt: Any = tuple(
                        int(x)
                        if isinstance(x, (int, float)) and float(x).is_integer()
                        else float(x)
                        for x in pos
                    )
                    target_set = (
                        self.learned_goal_positions
                        if node.id.startswith("learned_goal_")
                        else self.learned_barriers
                    )
                    target_set.add(pt)
            elif node.id == "engine_telemetry":
                avatar_feat = props.get("avatar_feature")
                if avatar_feat is not None:
                    self._avatar_feature = avatar_feat
                for af in props.get("avatar_features") or []:
                    self.avatar_features.add(af)
                for ta in props.get("tested_actions") or []:
                    self.tested_actions.add(ta)
            elif node.id == "homeostatic_state":
                homeo = getattr(self.working_memory, "homeostatic_state", None) or getattr(
                    self.working_memory, "homeostatic", None
                )
                if homeo is not None:
                    for k in (
                        "energy_budget",
                        "safety_margin",
                        "competence_progress",
                        "fatigue",
                    ):
                        if k in props and hasattr(homeo, k):
                            setattr(homeo, k, props[k])
            elif (
                getattr(node, "node_type", None) == "concept"
                or getattr(node, "category", None) == "memory"
            ):
                concept_store = getattr(self.working_memory, "concept_store", None)
                restore_fn = getattr(concept_store, "restore_concept", None)
                if callable(restore_fn):
                    restore_fn(node)

        return len(nodes)

    def freeze(self) -> CognitiveContext:
        """Capture an immutable CognitiveContext snapshot of engine state for pure cortical stages.

        Faculties emit intent; engine owns no mutable truth.
        Cortical faculties observe this frozen snapshot and emit transactional operations.
        """
        return EngineState.from_engine(self).freeze()

    def apply_delta(self, delta: Any, author: str = "unknown") -> None:
        """The ONLY faculty write path: apply a StateDelta to engine state.

        Assigns each pending field on self/self.state and emits HCIR
        transaction operations for changed fields (one commit per call).
        """
        if delta is None or getattr(delta, "is_empty", lambda: True)():
            return
        cow = getattr(delta, "_cow", {})
        for field, value in delta.items().items():
            if (
                field in cow
                and cow[field] is not None
                and (hasattr(value, "__len__") or field in cow)
                and cow[field] is value
                and (
                    # live-container mutations are already in place — skip reassign
                    getattr(self, field, None) is value or getattr(self.state, field, None) is value
                )
            ):
                continue
            # Engine-first: properties proxy correctly to their true owner
            # (working_memory / state / trackers). state is the fallback for
            # fields not exposed on the engine.
            if hasattr(self, field):
                setattr(self, field, value)
            elif hasattr(self.state, field):
                setattr(self.state, field, value)
            else:
                setattr(self, field, value)
        # HCIR emission: fields mirrored into the world-theory snapshot get a
        # single transaction commit; unmapped fields are engine-transient.
        mirrored = self.working_memory.get_mirrored_fields()
        if mirrored & set(delta.items()):
            try:
                ops = [self._telemetry_operation()]
                ws = self._commit_workspace()
                if ws is not None:
                    ws.commit_operations(ops, author=author)
            except Exception as exc:
                self.rejected_delta_commits += 1
                logger.error("apply_delta commit rejected (author=%s): %s", author, exc)

    def apply_wm_delta(self, delta: WorkingMemoryDelta) -> None:
        """Apply a declarative WorkingMemoryDelta to working memory and engine state."""
        field = delta.field
        val = delta.value
        op = delta.op

        if field == "inhibit_action":
            action, duration = val if isinstance(val, (tuple, list)) else (val, 4)
            self.working_memory.inhibit_action(action, duration)
            return

        from hbllm.hcir.world.engine_state import apply_wm_field_delta

        apply_wm_field_delta(self, self.working_memory, self.state, field, val, op)

    def rollback_faculty_journal(self, journal: Any) -> None:
        """Roll back all mutations recorded in a FacultyWriteJournal."""
        if journal is None or not getattr(journal, "touched", None):
            return
        for key in journal.touched:
            before = journal.before.get(key)
            if repr(before) == "<object object>":
                continue
            if key.startswith("working_memory."):
                target: Any = self.working_memory
                fname = key[15:]
            elif key.startswith("state."):
                target = self.state
                fname = key[6:]
            else:
                target = self
                fname = key
            live = getattr(target, fname, None)
            if isinstance(live, dict) and isinstance(before, dict):
                live.clear()
                live.update(before)
            elif isinstance(live, (set, frozenset)) and isinstance(before, (set, frozenset, list)):
                live.clear()
                live.update(before)
            elif isinstance(live, list) and isinstance(before, list):
                live[:] = before
            elif hasattr(live, "clear") and hasattr(live, "extend") and hasattr(before, "__iter__"):
                live.clear()
                live.extend(before)
            else:
                try:
                    setattr(target, fname, before)
                except Exception as exc:
                    logger.debug("Faculty journal rollback setattr(%s) skipped: %s", fname, exc)

    @staticmethod
    def _journal_values_equal(before: Any, current: Any) -> bool:
        """Fast recursive equality check that avoids np.array2string/repr() on lattice arrays."""
        if before is current:
            return True
        if type(before) is not type(current):
            return False
        if isinstance(before, np.ndarray):
            return (
                before.shape == current.shape
                and before.dtype == current.dtype
                and bool(np.array_equal(before, current))
            )
        if isinstance(before, dict):
            if len(before) != len(current) or before.keys() != current.keys():
                return False
            eq = AutonomousEpistemicEngine._journal_values_equal
            return all(eq(v, current[k]) for k, v in before.items())
        if isinstance(before, (list, tuple)):
            if len(before) != len(current):
                return False
            eq = AutonomousEpistemicEngine._journal_values_equal
            return all(eq(a, b) for a, b in zip(before, current))
        try:
            res = before == current
            if isinstance(res, np.ndarray):
                return bool(res.all())
            return bool(res)
        except Exception:
            return False

    def commit_faculty_journal(self, journal: Any) -> bool:
        """Commit a journaled faculty write set as one HCIR transaction (WP1).

        Writes already landed (write-through); on rejection, before-images are
        restored (containers via in-place clear+update so external refs hold).
        """
        if journal is None or not journal.touched:
            return True
        changed: dict[str, Any] = {}
        for key in journal.touched:
            before = journal.before.get(key)
            if key.startswith("working_memory."):
                current = getattr(self.working_memory, key[15:], None)
            elif key.startswith("state."):
                current = getattr(self.state, key[6:], None)
            else:
                current = getattr(self, key, None)
            if not self._journal_values_equal(before, current):
                changed[key] = before
        if not changed:
            return True
        try:
            mirrored = self.working_memory.get_mirrored_fields()
            raw_fields = {k.removeprefix("state.").removeprefix("working_memory.") for k in changed}
            if mirrored & raw_fields:
                ops = [self._telemetry_operation()]
                ws = self._commit_workspace()
                if ws is not None:
                    ws.commit_operations(ops, author=journal.author)
            return True
        except Exception as exc:
            self.rejected_faculty_commits += 1
            logger.error("faculty journal commit rejected (author=%s): %s", journal.author, exc)
            self.rollback_faculty_journal(journal)
            return False
