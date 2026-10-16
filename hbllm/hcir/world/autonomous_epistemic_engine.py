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

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.graph import (
    ConstraintNode,
    GoalNode,
    ObservationNode,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace_tiers import TaskFrame, TieredWorkspace, WorkspaceTier
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.bilateral_coordination import (
    BilateralCoordinateIntegrator,
    BilateralState,
)
from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    BeliefTransitionEvent,
    CausalHypothesis,
    CausalPredicate,
)
from hbllm.hcir.world.cerebellar_phase_clock import (
    CerebellarPhaseClock,
)
from hbllm.hcir.world.confidence_calibrator import ConfidenceCalibrator
from hbllm.hcir.world.counterfactual_simulation import (
    CounterfactualDeadlockDetector,
)
from hbllm.hcir.world.extended_body_schema import (
    ExtendedBodySchema,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoalStack,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.inferotemporal_segmentation import (
    InferotemporalSegmentationEngine,
)
from hbllm.hcir.world.interactive_action_discovery import (
    ActionBudgetGovernor,
    ActionPreconditionLearner,
    ConstrainedActiveExperimenter,
    UngroundedActionDiscoveryEngine,
    UnknownMechanicsLedger,
)
from hbllm.hcir.world.interactive_goal_evaluator import (
    BayesianGoalInductionEngine,
    DynamicProgressEstimator,
    GoalConfirmationGate,
    ModelBasedDeadlockDetector,
)
from hbllm.hcir.world.intuitive_physics import IntuitivePhysicsEngine
from hbllm.hcir.world.kinetic_stream import (
    DorsalKineticStream,
)
from hbllm.hcir.world.motor_calibration import (
    ActionDynamicsModel,
    StateMutationModel,
)
from hbllm.hcir.world.object_state_graph import (
    ObjectStateGraphPlanner,
    RelationalGraphMatcher,
)
from hbllm.hcir.world.optical_ray_projection import (
    OpticalRayProjector,
)
from hbllm.hcir.world.parietal_coordinate_transform import (
    GeometricModelSelector,
    ParietalCoordinateTransformer,
)
from hbllm.hcir.world.percept import (
    BasePerceptionAdapter,
    PerceptionFrame,
    PerceptionRegistry,
)
from hbllm.hcir.world.prediction_budget import PredictionBudget
from hbllm.hcir.world.predictors.whole_grid import WholeGridPredictor
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)
from hbllm.hcir.world.representation_expansion import ConceptFormationEngine
from hbllm.hcir.world.rule_induction import (
    RuleInductionEngine,
)
from hbllm.hcir.world.spatial_containment import RoomDoor, RoomTopologyExtractor
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.spatiotemporal_tracker import (
    EntityLineageNode,
    LifecycleTrackingResult,
    MorphologicalDeformationTracker,
    SpatiotemporalHazardTracker,
)
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.hcir.world.temporal_dependency_tracker import (
    DelayedEffectModeler,
    DualStoreMemoryManager,
    EnvironmentChangeDetector,
    LongHorizonDependencyGraph,
    ObjectiveLearningEfficiencyTracker,
)
from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer
from hbllm.hcir.world.world_state_snapshot import (
    ReversibilityEngine,
    StateDifferenceComputer,
    TemporalStateBuffer,
    TransitionSequenceModel,
    WorldStateSnapshot,
)
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
    CausalInductionCortex,
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.cortex_deliberation import PrefrontalDeliberationEngine
from hbllm.hcir.world.cortex_episodic import (
    FalsifiableAnalogyEngine,
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.cortex_motor import (
    ACCConflictMonitor,
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorCortexEffector,
)
from hbllm.hcir.world.cortex_perception import (
    EpistemicObservationDiff,
    OrientedThreat,
    PerceptionEngine,
)
from hbllm.hcir.world.cortex_planner import (
    MentalSimulationPlanner,
    MentalSimulationStep,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Master Autonomous Epistemic Engine Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class AutonomousEpistemicEngine:
    """Domain-agnostic cognitive engine for autonomous exploration, hypothesis testing,
    and forward mental simulation.
    """

    MIN_PROBES_PER_ACTION: int = 3
    consecutive_pacing_steps: int = 0

    def __init__(
        self,
        exploration_budget: int = 150,
        enable_logging: bool = True,
        instructions: Sequence[str] | str | None = None,
    ) -> None:
        self.exploration_budget = exploration_budget
        self.enable_logging = enable_logging

        # Cognitive state
        self._phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.phase_history: list[tuple[EpistemicPhase, EpistemicPhase, str]] = []
        self.world_id: str = "default"
        self.step_counter: int = 0
        self.environmental_step_limit: int | None = None

        # Memory of frames & actions
        self.prev_grid: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_action_data: dict[str, Any] | None = None
        self.consecutive_quiescent_actions: int = 0
        self.grid_history: deque[np.ndarray] = deque(maxlen=6)

        # Avatar self-model (Motor Grounding)
        self._avatar_feature: int | None = None
        self.avatar_features: set[int] = set()
        self.avatar_pos: tuple[int, int] | None = None
        self.avatar_size: int = 1

        # Intensional Neuro-Symbolic World Theory (SSOT for domain ontology)
        self.symbolic_theory: HCIRSymbolicWorldTheory = HCIRSymbolicWorldTheory()
        self.symbolic_theory.seed_core_knowledge(bg_feature=0)

        # Prefrontal Working Memory & Executive Faculties (SSOT for cognitive state)
        self.working_memory: PrefrontalWorkingMemory = PrefrontalWorkingMemory()
        if instructions:
            self.load_instructions(instructions)

        # Bind biological faculties to working memory stores
        self.subgoal_stack: FrontopolarSubgoalStack = self.working_memory.subgoal_stack
        self.body_schema: ExtendedBodySchema = self.working_memory.body_schema
        self.habenular_ior: HabenularEpisodicIOR = self.working_memory.habenular_ior
        self.remote_causal: RemoteCausalAttributor = self.working_memory.remote_causal
        self.optical_projector: OpticalRayProjector = self.working_memory.optical_projector
        self.it_segmenter: InferotemporalSegmentationEngine = self.working_memory.it_segmenter

        # Motor calibration & dynamics
        self.action_dynamics: dict[int, ActionDynamicsModel] = {}
        self.tested_actions: set[int] = set()
        self.tested_action_positions: set[tuple[int, tuple[int, int] | None]] = set()
        self.effector_kinetics_mode: int | None = (
            None  # Active inference: +1 pull/attraction, -1 push/repulsion
        )

        # State Mutation Models (Switches, buttons, door toggles)
        self.state_mutations: list[StateMutationModel] = []

        # Episodic Ground Facts (Layout coordinates, preserved per-level across attempts)
        self._current_level: int = 0
        self.working_memory.visuospatial.current_level = 0
        self.habenular_ior.current_level = 0

        # Theory of Mind: creature kinds observed to patrol autonomously
        self.mobile_threat_features: set[int] = set()

        # Biological hazard tracker & intuitive physics
        self.hazard_tracker: SpatiotemporalHazardTracker = SpatiotemporalHazardTracker()
        self.saccadic_attention: SaccadicAttentionSystem = SaccadicAttentionSystem()
        self.physics_engine: IntuitivePhysicsEngine = IntuitivePhysicsEngine()

        # Faculty C: Dorsal Visual Stream (Area MT/V5) Kinetic Figure-Ground Segregation
        self.dorsal_kinetic_stream: DorsalKineticStream = DorsalKineticStream()

        # Faculty D: Bilateral Convergent Coordinate Frames (Split-Hemisphere / Dual-Agent Mirroring)
        self.bilateral_integrator: BilateralCoordinateIntegrator = BilateralCoordinateIntegrator()
        self.bilateral_state: BilateralState | None = None

        # Faculty E: Basal Ganglia & Cerebellar Predictive Phase Entrainment
        self.cerebellar_clock: CerebellarPhaseClock = CerebellarPhaseClock()
        self.pending_phase_wait_steps: int = 0
        self.consecutive_pacing_steps: int = 0

        # Faculty: Premotor Spatiotemporal Collision Cones & Trajectory Extrapolation
        self.collision_cones: SpatiotemporalCollisionCones = SpatiotemporalCollisionCones()

        # Faculty: Orbitofrontal Cortex (OFC) Counterfactual Deadlock Detector
        self.deadlock_detector: CounterfactualDeadlockDetector = CounterfactualDeadlockDetector()

        # Faculty: Hippocampal & Lateral Habenula Episodic Memory (Trial-and-Error Loop)
        self.episodic_cortex: HippocampalEpisodicCortex = HippocampalEpisodicCortex(
            habenular_ior=self.habenular_ior,
        )

        # Faculty: vmPFC Causal Induction Cortex (Pearl's do(a) intervention & tool resonance)
        self.causal_cortex: CausalInductionCortex = CausalInductionCortex(
            symbolic_theory=self.symbolic_theory,
            causal_attributor=self.remote_causal,
            body_schema=self.body_schema,
        )

        # Faculty: Anterior Cingulate Cortex (ACC) Motor Conflict Monitor
        self.acc_conflict_monitor: ACCConflictMonitor = ACCConflictMonitor()

        # Room topology & doorway subgoal reasoning (spatial containment)
        self.room_topology: RoomTopologyExtractor = RoomTopologyExtractor()

        # Object-Centric Macro-Action State Graph Planner (5 Executive Directives)
        self.object_planner: ObjectStateGraphPlanner = ObjectStateGraphPlanner()
        self.perception_adapter: BasePerceptionAdapter | None = None
        self.topology_rooms: dict[int, list[tuple[int, int]]] = {}
        self.topology_doors: list[RoomDoor] = []
        self.room_adjacency: dict[int, list[int]] = {}
        self.current_room_id: int | None = None

        # Confidence-scaled predictive coding surprise engine
        self.surprise_engine: SurpriseEngine = SurpriseEngine(surprise_threshold=0.15)
        self.last_surprise: float = 0.0
        self.last_surprise_eval: SurpriseEvaluation | None = None

        # Free energy active inference decision selection
        self.active_inference: ActiveInferenceEngine = ActiveInferenceEngine(
            w_reward=0.35,
            w_info_gain=0.30,
            w_future_val=0.15,
            w_risk=0.10,
            w_cost=0.10,
        )
        self.prediction_budget: PredictionBudget = PredictionBudget()
        self.pending_transactions: list[TransactionOperation] = []

        # Faculty: Prefrontal Deliberation Engine & Cognitive Arbitration
        self.deliberation_engine: PrefrontalDeliberationEngine = PrefrontalDeliberationEngine()

        # Faculty: Non-Rigid Topological Deformation Tracker & Entity Lineage (W059, W014)
        self.deformation_tracker: MorphologicalDeformationTracker = (
            MorphologicalDeformationTracker()
        )
        self.latest_lifecycle_result: LifecycleTrackingResult | None = None
        self.entity_lineage_graph: dict[str, EntityLineageNode] = {}

        # Faculty: Parietal Coordinate Transformer & Invariant Geometric Selector (W040)
        self.coordinate_transformer: ParietalCoordinateTransformer = ParietalCoordinateTransformer()
        self.geometric_selector: GeometricModelSelector = GeometricModelSelector()

        # Faculty: Constraint-Aware Relational Graph Matcher (W038)
        self.relational_matcher: RelationalGraphMatcher = RelationalGraphMatcher()

        # Faculty: Falsifiable Cross-Task Analogy Engine (W148)
        self.analogy_engine: FalsifiableAnalogyEngine = FalsifiableAnalogyEngine()

        # Faculty: Temporal State Buffer & State Difference Computer (W041-W047)
        self.temporal_buffer: TemporalStateBuffer = TemporalStateBuffer()
        self.difference_computer: StateDifferenceComputer = StateDifferenceComputer()

        # Faculty: Transition Sequence Model & Reversibility Engine (W048-W050)
        self.transition_sequence_model: TransitionSequenceModel = TransitionSequenceModel()
        self.reversibility_engine: ReversibilityEngine = ReversibilityEngine()

        # Faculty: Interventional Causal Discovery Engine (W061-W070)
        self.causal_discovery_engine: BaseCausalDiscoveryEngine = BaseCausalDiscoveryEngine()

        # Faculty: Rule Induction & Latent Structure Engine (W071-W080)
        self.rule_induction_engine: RuleInductionEngine = RuleInductionEngine()

        # Faculty: Whole-Grid Outcome Predictor & Rollout Engine (W081-W089)
        self.whole_grid_predictor: WholeGridPredictor = WholeGridPredictor()

        # Faculty: Predictive Confidence Calibrator (W090)
        self.confidence_calibrator: ConfidenceCalibrator = ConfidenceCalibrator()

        # Faculty: Abstract Concept Formation & Compositional Engine (W091-W100)
        self.concept_engine: ConceptFormationEngine = ConceptFormationEngine()

        # Faculty: Visual Symmetry & Mathematical Regularity Analyzer (W101-W110)
        self.visual_symmetry: VisualSymmetryAnalyzer = VisualSymmetryAnalyzer()

        # Domain 13: Relational Confidence & Evidence Provenance (W121-W130)
        self.relational_confidence_map: dict[str, float] = {}
        self.evidence_provenance_log: list[dict[str, Any]] = []

        # ── Interactive World-Modeling Faculties (W161–W190) ──────────────────
        # Module 1: Ungrounded Action Discovery & Constrained Experimentation (W161-W165, W171, W187, W188)
        self.action_discovery: UngroundedActionDiscoveryEngine = UngroundedActionDiscoveryEngine()
        self.unknown_mechanics: UnknownMechanicsLedger = UnknownMechanicsLedger()
        self.active_experimenter: ConstrainedActiveExperimenter = ConstrainedActiveExperimenter()
        self.precondition_learner: ActionPreconditionLearner = ActionPreconditionLearner()
        self.budget_governor: ActionBudgetGovernor = ActionBudgetGovernor(
            max_budget=self.exploration_budget
        )

        # Module 2: Open-Set Goal Induction & Reachability Deadlock Pruning (W168-W170, W174, W175, W181, W182)
        self.goal_induction: BayesianGoalInductionEngine = BayesianGoalInductionEngine()
        self.goal_gate: GoalConfirmationGate = GoalConfirmationGate()
        self.progress_estimator: DynamicProgressEstimator = DynamicProgressEstimator()
        self.model_deadlock_detector: ModelBasedDeadlockDetector = ModelBasedDeadlockDetector()

        # Module 3: Temporal Dynamics, Dual-Store Memory & Objective Efficiency (W167, W172, W173, W176-W180, W189, W190)
        self.delayed_effects: DelayedEffectModeler = DelayedEffectModeler()
        self.dependency_graph: LongHorizonDependencyGraph = LongHorizonDependencyGraph()
        self.dual_memory: DualStoreMemoryManager = DualStoreMemoryManager()
        self.change_detector: EnvironmentChangeDetector = EnvironmentChangeDetector()
        self.efficiency_tracker: ObjectiveLearningEfficiencyTracker = (
            ObjectiveLearningEfficiencyTracker()
        )

        self.bg_feature: int = 0
        self.hypotheses: list[CausalHypothesis] = []
        self.belief_history: list[BeliefTransitionEvent] = []

        # External domain & benchmark extension registries (e.g. plugin adapters)
        self.target_scorers: list[Any] = []
        self.deliberation_policies: list[Any] = []

        # Tiered Cognitive Workspace & Transaction Manager live mirroring
        self.tiered_workspace: TieredWorkspace | None = None
        self.transaction_manager: TransactionManager | None = None
        self.active_task_frame: TaskFrame | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Working Memory & Knowledge Base Properties (Single Source of Truth)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def verified_safe_features(self) -> set[int]:
        """Empirically verified safe/walkable features in symbolic world theory."""
        return self.symbolic_theory.walkable_features

    @verified_safe_features.setter
    def verified_safe_features(self, val: set[int]) -> None:
        self.symbolic_theory.walkable_features = val

    @property
    def mental_plan(self) -> deque[MentalSimulationStep]:
        """Active mental simulation plan in prefrontal working memory."""
        return self.working_memory.mental_plan

    @mental_plan.setter
    def mental_plan(self, val: deque[MentalSimulationStep]) -> None:
        self.working_memory.mental_plan = val

    @property
    def active_hypothesis(self) -> Any | None:
        """Active causal hypothesis evaluated in prefrontal working memory."""
        return self.working_memory.active_hypothesis

    @active_hypothesis.setter
    def active_hypothesis(self, val: Any | None) -> None:
        self.working_memory.active_hypothesis = val

    @property
    def consecutive_simulation_failures(self) -> int:
        return self.working_memory.consecutive_simulation_failures

    @consecutive_simulation_failures.setter
    def consecutive_simulation_failures(self, val: int) -> None:
        self.working_memory.consecutive_simulation_failures = val

    @property
    def simulation_cooldown(self) -> int:
        return self.working_memory.simulation_cooldown

    @simulation_cooldown.setter
    def simulation_cooldown(self, val: int) -> None:
        self.working_memory.simulation_cooldown = val

    @property
    def consecutive_plan_failures(self) -> int:
        return self.working_memory.consecutive_plan_failures

    @consecutive_plan_failures.setter
    def consecutive_plan_failures(self, val: int) -> None:
        self.working_memory.consecutive_plan_failures = val

    @property
    def exploration_cooldown(self) -> int:
        return self.working_memory.exploration_cooldown

    @exploration_cooldown.setter
    def exploration_cooldown(self, val: int) -> None:
        self.working_memory.exploration_cooldown = val

    @property
    def consecutive_stuck_steps(self) -> int:
        return self.working_memory.consecutive_stuck_steps

    @consecutive_stuck_steps.setter
    def consecutive_stuck_steps(self, val: int) -> None:
        self.working_memory.consecutive_stuck_steps = val

    @property
    def level_learned_barriers(self) -> dict[int, set[tuple[int, int]]]:
        """Spatial barrier coordinates mapped per level in visuospatial working memory."""
        return self.working_memory.visuospatial.level_learned_barriers

    @level_learned_barriers.setter
    def level_learned_barriers(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.working_memory.visuospatial.level_learned_barriers = val

    @property
    def learned_barriers(self) -> set[tuple[int, int]]:
        """Barrier coordinates on active level in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_barriers

    @learned_barriers.setter
    def learned_barriers(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_barriers = val

    @property
    def level_lethal_positions(self) -> dict[int, set[tuple[int, int]]]:
        """Lethal obstacle coordinates mapped per level in visuospatial working memory."""
        return self.working_memory.visuospatial.level_lethal_positions

    @level_lethal_positions.setter
    def level_lethal_positions(self, val: dict[int, set[tuple[int, int]]]) -> None:
        self.working_memory.visuospatial.level_lethal_positions = val

    @property
    def learned_goal_positions(self) -> set[tuple[int, int]]:
        """Goal coordinates mapped in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_goal_positions

    @learned_goal_positions.setter
    def learned_goal_positions(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_goal_positions = val

    @property
    def learned_receptacle_positions(self) -> set[tuple[int, int]]:
        """Receptacle coordinates mapped in visuospatial working memory."""
        return self.working_memory.visuospatial.learned_receptacle_positions

    @learned_receptacle_positions.setter
    def learned_receptacle_positions(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.visuospatial.learned_receptacle_positions = val

    @property
    def level_failed_transitions(self) -> dict[int, set[tuple[tuple[int, int], int]]]:
        """Negative state-action transitions archived in Habenular episodic memory."""
        return self.habenular_ior.level_failed_transitions

    @level_failed_transitions.setter
    def level_failed_transitions(self, val: dict[int, set[tuple[tuple[int, int], int]]]) -> None:
        self.habenular_ior.level_failed_transitions = val

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
        return self.working_memory.focal_attention_point

    @current_simulated_goal.setter
    def current_simulated_goal(self, val: tuple[int, int] | None) -> None:
        self.working_memory.focal_attention_point = val

    @property
    def last_predicted_pos(self) -> tuple[int, int] | None:
        """Last predicted avatar position (efference copy) in prefrontal working memory."""
        return self.working_memory.last_predicted_pos

    @last_predicted_pos.setter
    def last_predicted_pos(self, val: tuple[int, int] | None) -> None:
        self.working_memory.last_predicted_pos = val

    @property
    def _feedback_assimilated(self) -> bool:
        """Sensory feedback assimilation state in prefrontal working memory."""
        return self.working_memory.feedback_assimilated

    @_feedback_assimilated.setter
    def _feedback_assimilated(self, val: bool) -> None:
        self.working_memory.feedback_assimilated = val

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
    def _prev_oriented_threats(self) -> list[OrientedThreat]:
        """Dynamic threat kinematics held in prefrontal working memory."""
        return self.working_memory.prev_oriented_threats

    @_prev_oriented_threats.setter
    def _prev_oriented_threats(self, val: list[OrientedThreat]) -> None:
        self.working_memory.prev_oriented_threats = val

    @property
    def active_probe_target(self) -> tuple[int, int] | None:
        return self.working_memory.active_probe_target

    @active_probe_target.setter
    def active_probe_target(self, val: tuple[int, int] | None) -> None:
        self.working_memory.active_probe_target = val

    @property
    def active_probe_id(self) -> str | None:
        return self.working_memory.active_probe_id

    @active_probe_id.setter
    def active_probe_id(self, val: str | None) -> None:
        self.working_memory.active_probe_id = val

    @property
    def probe_target_steps(self) -> int:
        return self.working_memory.probe_target_steps

    @probe_target_steps.setter
    def probe_target_steps(self, val: int) -> None:
        self.working_memory.probe_target_steps = val

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
    def goals(self) -> list[tuple[int, int]]:
        """Return unified list of active candidate goal coordinates."""
        res: list[tuple[int, int]] = list(self.learned_goal_positions)
        sgs = getattr(self.working_memory, "structural_goals", None)
        if not sgs and self.prev_grid is not None:
            sgs = self.detect_structural_goals(self.prev_grid, bg=self.bg_feature)
            if hasattr(self.working_memory, "structural_goals"):
                self.working_memory.structural_goals = sgs
        if sgs:
            for sg in sgs:
                if sg.get("type") == "target_constellation":
                    f = sg.get("feature")
                    pos = sg.get("position")
                    if (
                        f is not None
                        and pos is not None
                        and hasattr(self.working_memory, "target_constellations")
                    ):
                        self.working_memory.target_constellations.setdefault(f, pos)

            av_feats = self.avatar_features or (
                {self.avatar_feature} if self.avatar_feature is not None else set()
            )
            constellations = [
                sg
                for sg in sgs
                if sg.get("type") == "target_constellation"
                and (sg.get("feature") in av_feats if av_feats else True)
            ]
            if constellations:
                c_goals: list[tuple[int, int]] = []
                for c in constellations:
                    pos = c.get("position")
                    if pos and pos not in c_goals:
                        c_goals.append(pos)
                return c_goals

            if hasattr(self.working_memory, "target_constellations"):
                for av_f in av_feats:
                    if av_f in self.working_memory.target_constellations:
                        return [self.working_memory.target_constellations[av_f]]

            for sg in sgs:
                pos = sg.get("position")
                if pos and pos not in res:
                    res.append(pos)
                if sg.get("type") != "target_constellation":
                    for cp in sg.get("cells", []):
                        if cp not in res:
                            res.append(cp)
        return res

    @property
    def exhausted_candidate_goals(self) -> set[tuple[int, int]]:
        return self.working_memory.exhausted_candidate_goals

    @exhausted_candidate_goals.setter
    def exhausted_candidate_goals(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.exhausted_candidate_goals = val

    @property
    def quiescent_click_targets(self) -> set[tuple[int, int]]:
        return self.working_memory.quiescent_click_targets

    @quiescent_click_targets.setter
    def quiescent_click_targets(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.quiescent_click_targets = val

    @property
    def effective_click_targets(self) -> set[tuple[int, int]]:
        return self.working_memory.effective_click_targets

    @effective_click_targets.setter
    def effective_click_targets(self, val: set[tuple[int, int]]) -> None:
        self.working_memory.effective_click_targets = val

    @property
    def last_effective_click_coord(self) -> tuple[int, int] | None:
        return self.working_memory.last_effective_click_coord

    @last_effective_click_coord.setter
    def last_effective_click_coord(self, val: tuple[int, int] | None) -> None:
        self.working_memory.last_effective_click_coord = val

    @property
    def consecutive_effective_clicks(self) -> int:
        return self.working_memory.consecutive_effective_clicks

    @consecutive_effective_clicks.setter
    def consecutive_effective_clicks(self, val: int) -> None:
        self.working_memory.consecutive_effective_clicks = val

    @property
    def click_affordances(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        return self.working_memory.click_affordances

    @click_affordances.setter
    def click_affordances(self, val: dict[tuple[int, int], list[tuple[int, int]]]) -> None:
        self.working_memory.click_affordances = val

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
    def consecutive_goal_converging_clicks(self) -> int:
        return self.working_memory.consecutive_goal_converging_clicks

    @consecutive_goal_converging_clicks.setter
    def consecutive_goal_converging_clicks(self, val: int) -> None:
        self.working_memory.consecutive_goal_converging_clicks = val

    @property
    def active_causal_actuator_coord(self) -> tuple[int, int] | None:
        return self.working_memory.active_causal_actuator_coord

    @active_causal_actuator_coord.setter
    def active_causal_actuator_coord(self, val: tuple[int, int] | None) -> None:
        self.working_memory.active_causal_actuator_coord = val

    @property
    def consecutive_causal_probes(self) -> int:
        return self.working_memory.consecutive_causal_probes

    @consecutive_causal_probes.setter
    def consecutive_causal_probes(self, val: int) -> None:
        self.working_memory.consecutive_causal_probes = val

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
        return self.working_memory.feature_quiescent_counts

    @feature_quiescent_counts.setter
    def feature_quiescent_counts(self, val: dict[int, int]) -> None:
        self.working_memory.feature_quiescent_counts = val

    @property
    def _feature_quiescent_counts(self) -> dict[int, int]:
        return self.working_memory.feature_quiescent_counts

    @_feature_quiescent_counts.setter
    def _feature_quiescent_counts(self, val: dict[int, int]) -> None:
        self.working_memory.feature_quiescent_counts = val

    @property
    def inhibited_actions(self) -> dict[Any, int]:
        return self.working_memory.inhibited_actions

    @inhibited_actions.setter
    def inhibited_actions(self, val: dict[Any, int]) -> None:
        self.working_memory.inhibited_actions = val

    @property
    def prior_avatar_feature(self) -> int | None:
        return self.working_memory.prior_avatar_feature

    @prior_avatar_feature.setter
    def prior_avatar_feature(self, val: int | None) -> None:
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
        return self.symbolic_theory.focus_switch_actions

    @focus_switch_actions.setter
    def focus_switch_actions(self, val: set[Any]) -> None:
        self.symbolic_theory.focus_switch_actions = val
        self.working_memory.focus_switch_actions = val

    @property
    def undo_actions(self) -> set[Any]:
        return self.symbolic_theory.undo_actions

    @undo_actions.setter
    def undo_actions(self, val: set[Any]) -> None:
        self.symbolic_theory.undo_actions = val
        self.working_memory.undo_actions = val

    @property
    def index_selector_actions(self) -> set[Any]:
        return self.symbolic_theory.index_selector_actions

    @index_selector_actions.setter
    def index_selector_actions(self, val: set[Any]) -> None:
        self.symbolic_theory.index_selector_actions = val
        self.working_memory.index_selector_actions = val

    @property
    def value_mutator_actions(self) -> set[Any]:
        return self.symbolic_theory.value_mutator_actions

    @value_mutator_actions.setter
    def value_mutator_actions(self, val: set[Any]) -> None:
        self.symbolic_theory.value_mutator_actions = val
        self.working_memory.value_mutator_actions = val

    @property
    def last_non_displacement_action(self) -> Any | None:
        return self.working_memory.last_non_displacement_action

    @last_non_displacement_action.setter
    def last_non_displacement_action(self, val: Any | None) -> None:
        self.working_memory.last_non_displacement_action = val

    @property
    def level_epistemic_probes(self) -> int:
        return self.working_memory.level_epistemic_probes

    @level_epistemic_probes.setter
    def level_epistemic_probes(self, val: int) -> None:
        self.working_memory.level_epistemic_probes = val

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

    def register_deliberation_policy(self, policy: Any) -> None:
        """Register an external deliberation arbitration policy."""
        if policy not in self.deliberation_policies:
            self.deliberation_policies.append(policy)

    def unregister_deliberation_policy(self, policy: Any) -> None:
        """Unregister an external deliberation arbitration policy."""
        if policy in self.deliberation_policies:
            self.deliberation_policies.remove(policy)

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
        self.transaction_manager = TransactionManager(workspace=frame.workspace)
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
        hydrated_count = 0

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
                        pass
                if feat is not None:
                    self.symbolic_theory.goal_features.add(int(feat))
                    self.symbolic_theory.candidate_goal_features.add(int(feat))
                    hydrated_count += 1
                    logger.info(
                        "AutonomousEpistemicEngine: Hydrated persistent goal feature %s", feat
                    )

            # 3. Restore Learned Spatial Goals
            elif nid.startswith("learned_goal_") or ntype == "goal":
                pos = props.get("position")
                if pos and len(pos) == 2:
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
                if pos and len(pos) == 2:
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

        tx = HCIRTransaction(
            author="AutonomousEpistemicEngine",
            operations=ops,
        )
        committed_tx = self.transaction_manager.commit(tx)

        if self.tiered_workspace is not None:
            self.tiered_workspace.notify_commit()

        # Clear transient pending buffer
        self.pending_transactions.clear()
        return committed_tx

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
    def phase(self) -> EpistemicPhase:
        """Active cognitive lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, value: EpistemicPhase) -> None:
        self.transition_to_phase(value, reason="state_assignment")

    def transition_to_phase(self, new_phase: EpistemicPhase, reason: str = "") -> None:
        """Formal event-sourced phase transition."""
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

    def shift_phase(self, new_phase: EpistemicPhase, reason: str = "") -> None:
        """Alias for transition_to_phase."""
        self.transition_to_phase(new_phase, reason=reason)

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

    def is_action_sufficiently_probed(self, action_id: Any) -> bool:
        """Check if an action has been tested enough times to have a reliable dynamics model."""
        dyn = self.action_dynamics.get(action_id)
        if dyn is None:
            return False
        probes = getattr(dyn, "probes_tested", 0)
        return probes >= self.MIN_PROBES_PER_ACTION

    def register_action_space(self, action_specs: Sequence[Any]) -> None:
        """Register external action definitions dynamically from peripheral driver or device descriptor.

        Accepts:
        - List of dicts (from SynapticDeviceDescriptor.action_schema):
          e.g. [{'action_id': 1, 'name': 'MOVE_UP'}, {'action_id': 6, 'name': 'CLICK_CELL', 'parameters': {'x': 'int', 'y': 'int'}}]
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

            # Bayesian Directional Prior Initialization
            if aff.action_id not in self.action_dynamics:
                name_u = aff.name.upper()
                prior_delta: tuple[int, int] | None = None
                for k, d in _dir_priors.items():
                    if k in name_u:
                        prior_delta = d
                        break
                if prior_delta is None and isinstance(aff.action_id, int):
                    # Standard discrete 2D grid direction convention: 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
                    _idx_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                    prior_delta = _idx_map.get(aff.action_id)

                if prior_delta is not None:
                    dr, dc = prior_delta
                    aff.is_displacement = True
                    aff.delta = (dr, dc)
                    self.action_dynamics[aff.action_id] = ActionDynamicsModel(
                        action_id=aff.action_id,
                        delta_r=dr,
                        delta_c=dc,
                        confidence=0.75,
                        probes_tested=0,
                        displacement=(float(dr), float(dc)),
                    )

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
        # Symmetrical motor stride reconciliation:
        # Reconcile antiparallel displacement vectors along cardinal axes to the maximum verified stride
        # if one direction was constrained by an obstacle during calibration (e.g. (0, -6) vs (0, 3)).
        for a1, d1 in list(deltas.items()):
            for a2, d2 in list(deltas.items()):
                if a1 != a2:
                    if d1[0] == 0 and d2[0] == 0 and d1[1] * d2[1] < 0:
                        max_stride = max(abs(d1[1]), abs(d2[1]))
                        if abs(d1[1]) < max_stride and max_stride % abs(d1[1]) == 0:
                            deltas[a1] = (0, -max_stride if d1[1] < 0 else max_stride)
                    elif d1[1] == 0 and d2[1] == 0 and d1[0] * d2[0] < 0:
                        max_stride = max(abs(d1[0]), abs(d2[0]))
                        if abs(d1[0]) < max_stride and max_stride % abs(d1[0]) == 0:
                            deltas[a1] = (-max_stride if d1[0] < 0 else max_stride, 0)

        # Isotropic spatial lattice stride generalization:
        # If any cardinal displacement action demonstrates an unconstrained stride S > 1,
        # propagate S to uncalibrated or unit-prior cardinal actions (probes_tested == 0 or |d| <= 1).
        calibrated_strides = [
            max(abs(dyn.delta_r), abs(dyn.delta_c))
            for dyn in self.action_dynamics.values()
            if getattr(dyn, "probes_tested", 0) >= 1 and dyn.is_displacement_action()
        ]
        if calibrated_strides:
            lattice_quantum = max(calibrated_strides)
            if lattice_quantum > 1:
                for act in list(deltas.keys()):
                    dyn = self.action_dynamics.get(act)
                    if dyn is None or getattr(dyn, "probes_tested", 0) == 0:
                        dr, dc = deltas[act]
                        if abs(dr) == 1 and dc == 0:
                            deltas[act] = (dr * lattice_quantum, 0)
                        elif abs(dc) == 1 and dr == 0:
                            deltas[act] = (0, dc * lattice_quantum)

        _discrete_grid_convention = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        for act, d in _discrete_grid_convention.items():
            if act not in deltas and (
                act in self.action_affordances or not self.action_affordances
            ):
                if calibrated_strides and max(calibrated_strides) > 1:
                    lq = max(calibrated_strides)
                    deltas[act] = (d[0] * lq, d[1] * lq)
                else:
                    deltas[act] = d
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
        return action in self.action_dynamics

    def is_focus_switch_action(self, action: Any) -> bool:
        """Returns True if this action switches active control/focus between entities."""
        if action in self.focus_switch_actions:
            return True
        aff = self.action_affordances.get(action)
        return bool(aff is not None and getattr(aff, "is_focus_switch", False))

    def mark_focus_switch_action(self, action: Any) -> None:
        """Empirically register an action as an entity focus switcher."""
        self.focus_switch_actions.add(action)
        if action in self.action_affordances:
            self.action_affordances[action].is_focus_switch = True
        else:
            self.action_affordances[action] = ActionAffordance(
                action_id=action,
                is_focus_switch=True,
            )
        logger.info(
            "AutonomousEpistemicEngine: Grounded empirical FOCUS_SWITCH affordance for action %s",
            action,
        )

    @staticmethod
    def normalize_sensory_input(raw: Any) -> np.ndarray:
        """Normalize arbitrary sensory observations (camera, depth, lidar, 2D grid) into a spatial 2D array."""
        return PerceptionEngine.normalize_sensory_input(raw)

    # ── Property Pass-Throughs to HCIRSymbolicWorldTheory ─────────────────────

    @property
    def avatar_feature(self) -> int | None:
        if self._avatar_feature is not None:
            return self._avatar_feature
        if self.avatar_features:
            return next(iter(self.avatar_features))
        return None

    @avatar_feature.setter
    def avatar_feature(self, val: int | None) -> None:
        self._avatar_feature = val
        if val is not None:
            if not self.avatar_features or val not in self.avatar_features:
                self.avatar_features = {val}
        else:
            self.avatar_features.clear()

    @property
    def current_actor_pos(self) -> tuple[int, int] | None:
        """Canonical alias for avatar_pos to satisfy actor introspection contracts."""
        return self.avatar_pos

    @current_actor_pos.setter
    def current_actor_pos(self, val: tuple[int, int] | None) -> None:
        self.avatar_pos = val

    @property
    def current_target_pos(self) -> tuple[int, int] | None:
        """Canonical alias for active target or goal position."""
        if self.active_probe_target is not None:
            return self.active_probe_target
        if self.learned_goal_positions:
            return next(iter(self.learned_goal_positions))
        return None

    @property
    def current_level_lethal_positions(self) -> set[tuple[int, int]]:
        """Lethal positions grounded for the current active level."""
        return (
            self.level_lethal_positions.get(self.current_level, set())
            | self.hazard_tracker.static_lethal_positions
        )

    @property
    def learned_goal_features(self) -> set[int]:
        return self.symbolic_theory.goal_features

    @learned_goal_features.setter
    def learned_goal_features(self, val: set[int]) -> None:
        self.symbolic_theory.goal_features = val

    @property
    def learned_barrier_features(self) -> set[int]:
        return self.symbolic_theory.barrier_features

    @learned_barrier_features.setter
    def learned_barrier_features(self, val: set[int]) -> None:
        self.symbolic_theory.barrier_features = val

    @property
    def learned_walkable_features(self) -> set[int]:
        return self.symbolic_theory.walkable_features

    @learned_walkable_features.setter
    def learned_walkable_features(self, val: set[int]) -> None:
        self.symbolic_theory.walkable_features = val

    @property
    def learned_cargo_features(self) -> set[int]:
        return self.symbolic_theory.cargo_features

    @learned_cargo_features.setter
    def learned_cargo_features(self, val: set[int]) -> None:
        self.symbolic_theory.cargo_features = val

    @property
    def learned_receptacle_features(self) -> set[int]:
        return self.symbolic_theory.receptacle_features

    @learned_receptacle_features.setter
    def learned_receptacle_features(self, val: set[int]) -> None:
        self.symbolic_theory.receptacle_features = val

    # ── Epistemic status of feature values ──────────────────────────────────────
    UNVERIFIED_FEATURE_COST: float = 8.0

    def is_feature_unverified(self, feat: int, bg: int) -> bool:
        """True if the agent has neither experience nor theory about ``feat``.

        A feature is 'known' if it is background, part of the avatar, walked on
        and survived, or classified by the symbolic theory (walkable, barrier,
        goal, candidate goal, cargo). Anything else is epistemically unknown.
        """
        if feat == bg or self.symbolic_theory.is_walkable(feat):
            return False
        av = self.avatar_features or (
            {self.avatar_feature} if self.avatar_feature is not None else set()
        )
        if feat in av:
            return False
        th = self.symbolic_theory
        if th.is_walkable(feat) or th.is_barrier(feat):
            return False
        if (
            feat in th.goal_features
            or feat in th.candidate_goal_features
            or feat in th.cargo_features
        ):
            return False
        return True

    def _epistemic_uncertainty_cost(self, feat: int, bg: int) -> float:
        """Soft exploration cost for stepping onto an unverified feature."""
        return self.UNVERIFIED_FEATURE_COST if self.is_feature_unverified(feat, bg) else 0.0

    def load_instructions(self, instructions: Sequence[str] | str | None) -> None:
        """Load executive cognitive directives into working memory to guide epistemic policies."""
        self.working_memory.load_instructions(instructions)

    @property
    def current_level(self) -> int:
        return self._current_level

    @current_level.setter
    def current_level(self, level: int) -> None:
        self.set_current_level(level)

    def set_current_level(self, level: int) -> None:
        if self._current_level != level:
            self._current_level = level
            self.working_memory.visuospatial.current_level = level
            self.habenular_ior.current_level = level
            self.hazard_tracker.static_lethal_positions = self.level_lethal_positions.setdefault(
                level, set()
            )

    # ── Episodic State Management ─────────────────────────────────────────────

    def reset_episode(
        self,
        retain_dynamics: bool = True,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        """Reset episodic state upon level transition or death.
        Delegated to HippocampalEpisodicCortex for schema consolidation.
        """
        self.episodic_cortex.consolidate_schema_and_reset(
            self,
            retain_dynamics=retain_dynamics,
            is_new_level=is_new_level,
            level=level,
        )
        self.step_counter = 0
        if is_new_level:
            self.environmental_step_limit = None
            self.budget_governor.reset(max_budget=self.exploration_budget)
            self.dual_memory.reset_episodic_memory()
            self.effective_click_targets.clear()
        else:
            eff_budget = (
                self.environmental_step_limit
                if self.environmental_step_limit is not None
                else self.exploration_budget
            )
            self.budget_governor.reset(max_budget=eff_budget)

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
        OR if the environment is allocentric/click-only (no displacement actions exist) and enough
        exploratory clicks have been made to characterize the environment."""
        # Standard motor grounding: avatar + calibrated displacement actions
        if (self.avatar_feature is not None or bool(self.avatar_features)) and any(
            dyn.confidence >= 0.7 for dyn in self.action_dynamics.values()
        ):
            return True
        # Click-only environment grounding: if no displacement dynamics were ever
        # calibrated and only spatial effectors are known, treat as grounded after
        # enough exploration to let the object planner engage.
        has_displacement = any(
            dyn.is_displacement_action() for dyn in self.action_dynamics.values()
        )
        if not has_displacement and self.action_dynamics and self.total_epistemic_probes >= 5:
            return True
        return False

    def compute_exploration_budget(self, n_actions: int) -> int:
        """Adaptive budget: more probing when more unknowns exist."""
        if not self.is_motor_grounded():
            calibrated = sum(1 for d in self.action_dynamics.values() if d.confidence >= 0.7)
            uncalibrated = max(0, n_actions - calibrated)
            return max(
                self.exploration_budget,
                uncalibrated * self.MIN_PROBES_PER_ACTION * 3,
            )
        confirmed_count = (
            len(self.symbolic_theory.goal_features)
            + len(self.symbolic_theory.barrier_features)
            + len(self.state_mutations)
        )
        scaling = max(0.2, 1.0 - 0.15 * confirmed_count)
        return max(15, int(self.exploration_budget * scaling))

    # ── Delegated Public Interfaces ───────────────────────────────────────────

    def perceive(self, raw_observation: Any) -> PerceptionFrame:
        """Lift raw sensory input into an abstract PerceptionFrame using the registered sensory adapter."""
        if isinstance(raw_observation, PerceptionFrame):
            return raw_observation
        adapter = self.perception_adapter or PerceptionRegistry.get_default()
        if adapter is not None:
            return adapter.lift_percept(raw_observation, context=self)
        H, W = (
            (raw_observation.shape[0], raw_observation.shape[1])
            if hasattr(raw_observation, "shape") and len(raw_observation.shape) >= 2
            else (16, 16)
        )
        return PerceptionFrame(shape=(H, W))

    def estimate_background(self, grid: np.ndarray) -> int:
        return PerceptionEngine.estimate_background(
            grid,
            barrier_features=self.symbolic_theory.barrier_features,
            avatar_features=self.avatar_features,
            bg_feature=self.bg_feature,
        )

    def extract_entities(self, grid: np.ndarray, bg: int) -> list[SpatialEntity]:
        return PerceptionEngine.extract_entities(
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

    def compute_frame_diff(
        self, prev_grid: np.ndarray, curr_grid: np.ndarray
    ) -> EpistemicObservationDiff:
        return PerceptionEngine.compute_frame_diff(prev_grid, curr_grid)

    def detect_structural_goals(
        self, grid: np.ndarray, bg: int | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        bg_val = bg if bg is not None else self.bg_feature
        return PerceptionEngine.detect_structural_goals(
            grid,
            bg=bg_val,
            avatar_features=self.avatar_features,
            avatar_feature=self.avatar_feature,
            barrier_features=self.symbolic_theory.barrier_features,
            known_lethal_features=self.hazard_tracker.known_lethal_features,
            avatar_pos=self.avatar_pos,
            **kwargs,
        )

    def assimilate_feedback(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        is_win: bool = False,
        is_lost: bool = False,
        action: int | None = None,
        action_data: dict[str, Any] | None = None,
    ) -> None:
        if action is None:
            action = self.last_action
        if action_data is None:
            action_data = self.last_action_data
        prev_av_pos = self.avatar_pos

        # Synthesize active subgoal schema into reusable procedural skill upon level win
        if is_win and self.working_memory.active_schema is not None:
            self.working_memory.synthesize_skill_from_schema(self.working_memory.active_schema)

        EpistemicFeedbackAssimilator.assimilate(
            self,
            curr_grid,
            available_actions,
            is_win=is_win,
            is_lost=is_lost,
            action=action,
            action_data=action_data,
        )

        # Wire W041-W050 temporal buffer & state transition sequence modeling
        new_snap = WorldStateSnapshot(
            world_id=self.world_id,
            variables={"grid": curr_grid, "avatar_pos": self.avatar_pos},
        )
        prev_snap = self.temporal_buffer.current_state
        self.temporal_buffer.push(new_snap)
        if prev_snap is not None and action is not None:
            self.transition_sequence_model.record_transition(
                pre_state=prev_snap,
                action=str(action),
                post_state=new_snap,
            )

        # Wire interactive world-modeling feedback updates (W161-W190)
        if action is not None:
            self.budget_governor.record_step()
            prior_st = {"avatar_pos": prev_av_pos}
            next_st = {"avatar_pos": self.avatar_pos}
            self.action_discovery.record_transition(
                action_id=action,
                prior_state=prior_st,
                next_state=next_st,
            )
            self.delayed_effects.record_step(
                step=self.step_counter,
                action_id=action,
                state_features=next_st,
            )
        if is_win or is_lost:
            self.goal_induction.observe_feedback(
                current_state=curr_grid,
                level_incremented=is_win,
                is_death_or_reset=is_lost,
            )
            self._prev_oriented_threats = []

        if is_win and self.tiered_workspace is not None and self.active_task_frame is not None:
            # Commit working memory state to the active task frame
            self.commit_working_memory_transactions()

            # Promote verified goals, ontology, and procedural skills to WorkspaceTier.PERSISTENT
            for goal_feat in self.symbolic_theory.goal_features:
                node_id = f"ontology_goal_{goal_feat}"
                self.tiered_workspace.promote(
                    node_id=node_id,
                    from_tier=WorkspaceTier.WORKING,
                    to_tier=WorkspaceTier.PERSISTENT,
                )

            all_goals: set[tuple[int, int]] = set(self.learned_goal_positions)
            for g_set in self.working_memory.visuospatial.level_learned_goals.values():
                all_goals.update(g_set)
            for goal_pos in all_goals:
                node_id = f"learned_goal_{goal_pos[0]}_{goal_pos[1]}"
                self.tiered_workspace.promote(
                    node_id=node_id,
                    from_tier=WorkspaceTier.WORKING,
                    to_tier=WorkspaceTier.PERSISTENT,
                )

            for schema_name in self.working_memory.procedural_library:
                node_id = f"procedural_skill_{schema_name}"
                self.tiered_workspace.promote(
                    node_id=node_id,
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

    def plan_epistemic_probe(
        self,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
    ) -> tuple[int, dict[str, Any] | None]:
        return EpistemicCuriosityExplorer.plan_epistemic_probe(self, curr_grid, available_actions)

    def apply_probe_proposal(self, proposal: EpistemicProbeProposal) -> None:
        """Commit an EpistemicProbeProposal's state updates atomically."""
        EpistemicCuriosityExplorer.apply_proposal_to_engine(self, proposal)

    def update_room_topology(self, curr_grid: np.ndarray) -> None:
        """Extract rooms and doorways from current occupancy grid knowledge."""
        H, W = curr_grid.shape
        occupancy = np.ones((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if (
                    (r, c) in self.learned_barriers
                    or self.symbolic_theory.is_barrier(val)
                    or val in self.hazard_tracker.known_lethal_features
                ):
                    occupancy[r, c] = False

        rooms, doors = self.room_topology.extract_rooms_and_doors(occupancy, min_room_size=4)
        self.topology_rooms = rooms
        self.topology_doors = doors
        self.room_adjacency = self.room_topology.build_adjacency_graph(rooms, doors)

        # Localize current room
        self.current_room_id = None
        if self.avatar_pos is not None:
            for rid, coords in rooms.items():
                if self.avatar_pos in coords:
                    self.current_room_id = rid
                    break

    def _update_avatar_position_from_grid(
        self,
        curr_grid: np.ndarray,
        known_av_feats: set[int],
        preferred_pos: tuple[int, int] | None = None,
    ) -> None:
        """Visually localize the avatar position on the current grid."""
        H, W = curr_grid.shape
        from scipy.ndimage import binary_closing, label

        mask = np.isin(curr_grid, list(known_av_feats))
        if not np.any(mask):
            return

        # Perform 3x3 topological morphological closing to integrate multi-part bodies or cutouts with internal markers
        closed_mask = binary_closing(mask, structure=np.ones((3, 3)))
        labeled, num_features = label(closed_mask)
        if num_features == 0:
            return

        # If avatar was already precisely localized by assimilate_feedback via motion tracking,
        # and current avatar_pos contains an avatar feature and matches avatar_size, verify and keep it
        if self.avatar_pos is not None and preferred_pos is None:
            r, c = self.avatar_pos
            comp_lbl = 0
            if 0 <= r < H and 0 <= c < W and labeled[r, c] > 0:
                comp_lbl = labeled[r, c]
            else:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and labeled[nr, nc] > 0:
                            cand_lbl = labeled[nr, nc]
                            if cand_lbl > 0:
                                comp_lbl = cand_lbl
                                break
                    if comp_lbl > 0:
                        break
            if comp_lbl > 0:
                coords = np.argwhere(labeled == comp_lbl)
                min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                is_edge_hud = (
                    min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))
                if not is_edge_hud:
                    comp_area = len(coords)
                    if self.avatar_size == 0 or (
                        0.15 * self.avatar_size <= comp_area <= 6.0 * self.avatar_size
                    ):
                        self.avatar_size = comp_area
                        return

        # Collect candidates
        candidates = []
        max_candidate_area = 0
        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            area = len(coords)
            if 1 <= area <= max(64, int(H * W * 0.15)):
                min_r, max_r = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
                min_c, max_c = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))
                # Filter out peripheral border lines / HUD gauges
                is_edge_hud = (
                    min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                ) or (min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4))
                if is_edge_hud:
                    continue

                centroid = (
                    int(round(float(np.mean(coords[:, 0])))),
                    int(round(float(np.mean(coords[:, 1])))),
                )
                candidates.append((centroid, area, coords))
                if area > max_candidate_area:
                    max_candidate_area = area

        if not candidates:
            return

        # Check if current avatar_pos actually tracks this feature
        is_tracking_active = False
        if self.avatar_pos is not None:
            ar, ac = self.avatar_pos
            if 0 <= ar < H and 0 <= ac < W and curr_grid[ar, ac] in known_av_feats:
                is_tracking_active = True

        best_pos = None
        best_size = self.avatar_size
        min_score = float("inf")

        for centroid, area, coords in candidates:
            # If substantial bodies exist (area >= 4), filter out single-pixel peripheral guide marks/notches
            if max_candidate_area >= 4 and area <= 2:
                continue

            if preferred_pos is not None:
                dist = abs(centroid[0] - preferred_pos[0]) + abs(centroid[1] - preferred_pos[1])
                score = dist * 2.0 - area * 0.01
            elif is_tracking_active and self.avatar_pos is not None:
                dist = abs(centroid[0] - self.avatar_pos[0]) + abs(centroid[1] - self.avatar_pos[1])
                size_diff = abs(area - self.avatar_size) if self.avatar_size > 0 else 0
                score = dist + size_diff * 0.5
            elif self.avatar_size > 0:
                score = abs(area - self.avatar_size) - area * 0.01
            else:
                # Fresh localization or focus switch: select dominant cohesive body
                score = -float(area)

            if score < min_score:
                min_score = score
                best_pos = centroid
                best_size = area

        if best_pos is not None:
            self.avatar_pos = best_pos
            if not is_tracking_active or preferred_pos is not None:
                self.avatar_size = best_size

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
            self._prev_oriented_threats = []
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
                goals=set(self.goals),
                cargo_features=self.learned_cargo_features,
                barrier_features=self.symbolic_theory.barrier_features,
                walkable_features=self.symbolic_theory.walkable_features,
            )
        except Exception:
            self._prev_oriented_threats = []
            return

        if (
            self.step_counter <= 1
            or self.prev_grid is None
            or getattr(self, "_last_threat_obs_level", -1) != self._current_level
        ):
            self._last_threat_obs_level = self._current_level
            self._prev_oriented_threats = threats
            return

        step_sz = self.infer_motor_step_size(available_actions)
        prev = self._prev_oriented_threats
        if prev:
            prev_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in prev:
                prev_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            curr_by_feat: dict[int, set[tuple[int, int]]] = {}
            for t in threats:
                curr_by_feat.setdefault(t.feature_id, set()).add(t.pos)
            for feat, cur_positions in curr_by_feat.items():
                if (
                    feat == bg
                    or feat == self.bg_feature
                    or feat in effective_av_feats
                    or feat in self.mobile_threat_features
                    or feat in self.learned_cargo_features
                    or feat in self.symbolic_theory.cargo_features
                    or feat in self.symbolic_theory.goal_features
                    or feat in self.learned_goal_features
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
                            self.symbolic_theory.walkable_features.discard(feat)
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
        self._prev_oriented_threats = threats

    # ── Master Decision Loop ──────────────────────────────────────────────────

    def decide(
        self,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Unified cognitive decision function:
        Perceive -> Assimilate -> Simulate in Mind -> Exploit / Epistemically Probe.
        Delegated to PrefrontalDeliberationEngine.
        """
        return self.deliberation_engine.decide(
            self,
            curr_grid=curr_grid,
            available_actions=available_actions,
            is_win=is_win,
            is_lost=is_lost,
            action_schemas=action_schemas,
        )

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
            bid = f"learned_barrier_{barrier_pos[0]}_{barrier_pos[1]}"
            bnode = ConstraintNode(
                id=bid,
                name=f"barrier_{barrier_pos[0]}_{barrier_pos[1]}",
                description=f"Learned impenetrable barrier at {barrier_pos}",
                target=f"position_{barrier_pos[0]}_{barrier_pos[1]}",
                properties={"position": list(barrier_pos), "role": "BARRIER"},
            )
            self.pending_transactions.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=bid,
                    node_data=bnode.model_dump(),
                )
            )

    def record_learned_goal(self, goal_pos: tuple[int, int]) -> None:
        """Record learned goal locally and propose atomic state transaction."""
        self.working_memory.visuospatial.level_learned_goals.setdefault(
            self.current_level, set()
        ).add(goal_pos)
        if goal_pos not in self.learned_goal_positions:
            self.learned_goal_positions.add(goal_pos)
            gid = f"learned_goal_{goal_pos[0]}_{goal_pos[1]}"

            gnode = GoalNode(
                id=gid,
                description=f"Learned goal location at {goal_pos}",
                properties={"position": list(goal_pos), "role": "GOAL"},
            )
            self.pending_transactions.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=gid,
                    node_data=gnode.model_dump(),
                )
            )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert epistemic engine state, internal memory, and knowledge base into transactional HCIR mutations."""
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
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data=engine_node.model_dump(),
            )
        ]
        ops.extend(self.pending_transactions)
        ops.extend(self.symbolic_theory.to_transaction_operations())
        ops.extend(self.working_memory.to_transaction_operations())
        all_goal_positions = set(self.learned_goal_positions)
        for g_set in self.working_memory.visuospatial.level_learned_goals.values():
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
