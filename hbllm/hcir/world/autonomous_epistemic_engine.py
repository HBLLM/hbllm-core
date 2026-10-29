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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.hcir.world_kernel import WorldKernel

import uuid

import numpy as np

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.graph import (
    CognitiveGraph,
    ConstraintNode,
    EventNode,
    GoalNode,
    ObservationNode,
    SkillNode,
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
from hbllm.hcir.world.active_inference import (
    ActiveInferenceEngine,
    HierarchicalActiveInferenceEngine,
)
from hbllm.hcir.world.active_inference_efe import (
    ExpectedFreeEnergyEvaluator,
    ExpectedFreeEnergyScore,
)
from hbllm.hcir.world.anterior_cingulate_conflict import AnteriorCingulateConflictMonitor
from hbllm.hcir.world.anterior_cingulate_effort_allocator import (
    AnteriorCingulateEffortAllocator,
    EffortAllocationDecision,
)
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
from hbllm.hcir.world.cerebellar_microzone_adaptor import (
    CerebellarMicrozoneAdaptor,
    ClimbingFiberFeedback,
)
from hbllm.hcir.world.cerebellar_phase_clock import (
    CerebellarPhaseClock,
    CerebellarSmithPredictor,
    PhaseGateDecision,
)
from hbllm.hcir.world.claustrum_salience_conductor import (
    ClaustrumSalienceConductor,
    UnifiedPerceptualGestalt,
)
from hbllm.hcir.world.confidence_calibrator import ConfidenceCalibrator
from hbllm.hcir.world.counterfactual_simulation import (
    CounterfactualComparison,
    CounterfactualDeadlockDetector,
    CounterfactualRegretEvaluator,
)
from hbllm.hcir.world.extended_body_schema import (
    ExtendedBodySchema,
)
from hbllm.hcir.world.frontopolar_exploration_horizon import (
    BranchSwitchDecision,
    FrontopolarExplorationHorizonEngine,
)
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.frontopolar_subgoal_tree import (
    RecursiveSubgoalTree,
)
from hbllm.hcir.world.habenula_striatum_opponency import (
    DualValueOpponencyCortex,
    OpponencyValueState,
)
from hbllm.hcir.world.habenular_episodic_inhibition import HabenularEpisodicIOR
from hbllm.hcir.world.hippocampal_phase_precession import (
    HippocampalPhasePrecessionEngine,
    PhasePrecessionBurst,
)
from hbllm.hcir.world.hippocampal_replay import (
    HippocampalPrioritizedReplayBuffer,
    ReplayBatch,
    ReplayTransition,
)
from hbllm.hcir.world.inferotemporal_segmentation import (
    InferotemporalSegmentationEngine,
)
from hbllm.hcir.world.inferotemporal_shape_hasher import (
    InferotemporalShapeHasher,
    ShapeArchetypeSignature,
)
from hbllm.hcir.world.insula_allostasis import (
    InteroceptiveAllostasisCortex,
    InteroceptiveState,
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
from hbllm.hcir.world.lateral_prefrontal_switcher import (
    LateralPrefrontalTaskSwitcher,
    TaskArbitrationResult,
)
from hbllm.hcir.world.locus_coeruleus_noradrenaline import (
    LocusCoeruleusNoradrenalineEngine,
    NoradrenergicState,
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
from hbllm.hcir.world.parahippocampal_context import (
    ParahippocampalContextEngine,
    SpatialContextSchema,
)
from hbllm.hcir.world.parietal_coordinate_transform import (
    EgocentricCoordinate,
    GeometricModelSelector,
    HeadingOrientation,
    ParietalCoordinateTransformer,
    RetrosplenialCoordinateTransformer,
)
from hbllm.hcir.world.parietal_foveated_attention import (
    FoveatedSaliencyMap,
    ParietalFoveatedAttentionEngine,
)
from hbllm.hcir.world.parietal_potential_field import (
    ParietalPotentialFieldNavigator,
    PotentialFieldVector,
)
from hbllm.hcir.world.percept import (
    BasePerceptionAdapter,
    PerceptionFrame,
    PerceptionRegistry,
)
from hbllm.hcir.world.prediction_budget import PredictionBudget
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor
from hbllm.hcir.world.predictors.whole_grid import WholeGridPredictor
from hbllm.hcir.world.prefrontal_blame_attribution import (
    PlanAttributionReport,
    PrefrontalBlameAttributor,
)
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory
from hbllm.hcir.world.pulvinar_attention_router import (
    PulvinarAttentionRouter,
    PulvinarRoutingState,
)
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)
from hbllm.hcir.world.representation_expansion import ConceptFormationEngine
from hbllm.hcir.world.retrosplenial_landmark_anchor import (
    LandmarkRecalibrationResult,
    RetrosplenialLandmarkAnchorEngine,
)
from hbllm.hcir.world.rule_induction import (
    RuleInductionEngine,
)
from hbllm.hcir.world.somatosensory_haptic import (
    HapticAffordanceFeedback,
    SomatosensoryHapticEngine,
)
from hbllm.hcir.world.spatial_containment import RoomDoor, RoomTopologyExtractor
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.spatiotemporal_tracker import (
    EntityLineageNode,
    LifecycleTrackingResult,
    MorphologicalDeformationTracker,
    SpatiotemporalHazardTracker,
)
from hbllm.hcir.world.subicular_boundary_vector import (
    BoundaryVectorMap,
    SubicularBoundaryVectorEngine,
)
from hbllm.hcir.world.subthalamic_drift_diffusion import (
    DriftDiffusionDecision,
    SubthalamicDriftDiffusionEngine,
)
from hbllm.hcir.world.superior_colliculus_arbiter import (
    SaccadeTarget,
    SuperiorColliculusTargetArbiter,
)
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.hcir.world.temporal_dependency_tracker import (
    DelayedEffectModeler,
    DualStoreMemoryManager,
    EnvironmentChangeDetector,
    LongHorizonDependencyGraph,
    ObjectiveLearningEfficiencyTracker,
)
from hbllm.hcir.world.ventral_premotor_resonator import (
    AffordanceResonancePattern,
    VentralPremotorAffordanceResonator,
)
from hbllm.hcir.world.ventromedial_somatic_marker import (
    SomaticAffectiveBias,
    VentromedialSomaticMarkerEngine,
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
from hbllm.hcir.world.cortex_deliberation import (
    AdaptiveNeuromodulationGovernor,
    BasalGangliaDecision,
    BasalGangliaGatingCircuit,
    NeuromodulatoryState,
    PrefrontalDeliberationEngine,
)
from hbllm.hcir.world.cortex_episodic import (
    FalsifiableAnalogyEngine,
    HippocampalEpisodicCortex,
)
from hbllm.hcir.world.cortex_motor import (
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorCortexEffector,
)
from hbllm.hcir.world.cortex_perception import (
    EpistemicObservationDiff,
    GibsonianAffordanceExtractor,
    OrientedThreat,
    PerceptionEngine,
    TopologicalContour,
)
from hbllm.hcir.world.cortex_planner import (
    MentalSimulationPlanner,
    MentalSimulationStep,
)
from hbllm.hcir.world.entorhinal_grid_cortex import (
    EntorhinalGridCortex,
)
from hbllm.hcir.world.global_neuronal_workspace import (
    GlobalNeuronalWorkspace,
    IgnitionEvent,
)
from hbllm.hcir.world.metacognitive_agency_cortex import (
    MetacognitiveAgencyCortex,
    MetacognitiveAssessment,
)
from hbllm.hcir.world.neuromodulatory_system import (
    NeuromodulatorLevels,
    NeuromodulatorySystem,
)
from hbllm.hcir.world.predictive_coding_cortex import (
    HierarchicalPredictiveCodingCortex,
    PredictiveCodingSnapshot,
)
from hbllm.hcir.world.relational_graph_matcher import (
    ObjectCorrespondence,
)
from hbllm.hcir.world.relational_graph_matcher import (
    RelationalGraphMatcher as VentralRelationalGraphMatcher,
)
from hbllm.hcir.world.skill_program_synthesizer import (
    SkillProgramSynthesizer,
)
from hbllm.hcir.world.spatial_distance_transform import (
    EuclideanDistanceTransformResult,
    SpatialDistanceTransformEngine,
)
from hbllm.hcir.world.superposition_planner import (
    SuperpositionEvaluationResult,
    SuperpositionPlanner,
)
from hbllm.hcir.world.temporal_dependency_tracker import (
    DelayedEffectModeler,
    LongHorizonDependencyGraph,
)
from hbllm.hcir.world.thalamocortical_binding import (
    BoundFeatureBundle,
    ThalamocorticalBindingEngine,
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
        services: KernelServices | None = None,
        world_id: str = "default",
    ) -> None:
        self.exploration_budget = exploration_budget
        self.enable_logging = enable_logging

        # Cognitive state
        self._phase: EpistemicPhase = EpistemicPhase.MOTOR_GROUNDING
        self.phase_history: list[tuple[EpistemicPhase, EpistemicPhase, str]] = []
        self.world_id: str = world_id
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
        self.acc_conflict: AnteriorCingulateConflictMonitor = self.working_memory.acc_conflict

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
        self.smith_predictor: CerebellarSmithPredictor = CerebellarSmithPredictor()
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
        self.acc_conflict_monitor: AnteriorCingulateConflictMonitor = (
            AnteriorCingulateConflictMonitor()
        )

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
        self.hierarchical_active_inference: HierarchicalActiveInferenceEngine = (
            HierarchicalActiveInferenceEngine(
                sensory_precision=1.0,
                dynamic_precision=1.0,
                lambda_epistemic=0.4,
                lambda_extrinsic=0.6,
                lambda_risk=0.5,
            )
        )
        self.prediction_budget: PredictionBudget = PredictionBudget()
        self.pending_transactions: list[TransactionOperation] = []

        # Faculty: Prefrontal Deliberation Engine & Cognitive Arbitration
        self.deliberation_engine: PrefrontalDeliberationEngine = PrefrontalDeliberationEngine()
        self.efe_evaluator: ExpectedFreeEnergyEvaluator = ExpectedFreeEnergyEvaluator()
        self.affordance_extractor: GibsonianAffordanceExtractor = GibsonianAffordanceExtractor()

        # Faculty: Non-Rigid Topological Deformation Tracker & Entity Lineage (W059, W014)
        self.deformation_tracker: MorphologicalDeformationTracker = (
            MorphologicalDeformationTracker()
        )
        self.latest_lifecycle_result: LifecycleTrackingResult | None = None
        self.entity_lineage_graph: dict[str, EntityLineageNode] = {}

        # Faculty: Parietal Coordinate Transformer & Invariant Geometric Selector (W040)
        self.coordinate_transformer: ParietalCoordinateTransformer = ParietalCoordinateTransformer()
        self.geometric_selector: GeometricModelSelector = GeometricModelSelector()
        self.retrosplenial_transformer: RetrosplenialCoordinateTransformer = (
            RetrosplenialCoordinateTransformer()
        )

        # Faculty: Constraint-Aware Relational Graph Matcher (W038)
        self.relational_matcher: RelationalGraphMatcher = RelationalGraphMatcher()

        # Faculty: Falsifiable Cross-Task Analogy Engine (W148)
        self.analogy_engine: FalsifiableAnalogyEngine = FalsifiableAnalogyEngine()
        self.counterfactual_regret: CounterfactualRegretEvaluator = CounterfactualRegretEvaluator()

        # Track 1: Global Neuronal Workspace (GNW) Ignition & Broadcast (Baars / Dehaene)
        self.gnw: GlobalNeuronalWorkspace = GlobalNeuronalWorkspace()

        # Track 2: Four Neuromodulatory Systems (Dopamine, Noradrenaline, Acetylcholine, Serotonin)
        self.neuromodulatory_system: NeuromodulatorySystem = NeuromodulatorySystem()

        # Track 3: Thalamocortical Gamma-Band Phase Synchronization & TRN Attentional Gating
        self.thalamocortical_binding: ThalamocorticalBindingEngine = ThalamocorticalBindingEngine()

        # Track 4: Superposition Counterfactual Tree Search Planner
        self.superposition_planner: SuperpositionPlanner = SuperpositionPlanner()

        # Faculty: Hierarchical Predictive Coding Cortical Columns (Rao & Ballard / Friston)
        self.predictive_coding_cortex: HierarchicalPredictiveCodingCortex = (
            HierarchicalPredictiveCodingCortex()
        )

        # Faculty: Entorhinal Grid Cells & Hippocampal Path Integration (Moser & O'Keefe)
        self.entorhinal_grid_cortex: EntorhinalGridCortex = EntorhinalGridCortex()

        # Faculty: BA10 Metacognitive Confidence Calibration & Agency Monitor (Fleming & Dolan)
        self.metacognitive_agency_cortex: MetacognitiveAgencyCortex = MetacognitiveAgencyCortex()

        # Faculty: Declarative Neuro-Symbolic Skill Synthesis & DSL Program Induction (DreamCoder)
        self.skill_synthesizer: SkillProgramSynthesizer = SkillProgramSynthesizer()

        # Faculty: Spiking Neural Network Temporal Dynamics & Limit Cycle Predictor
        self.snn_predictor: SNNTemporalPredictor = SNNTemporalPredictor()

        # Faculty: Ventral Scene Graph & Structural Correspondence Matcher (W038)
        self.ventral_relational_matcher: VentralRelationalGraphMatcher = (
            VentralRelationalGraphMatcher()
        )

        # Round 7 Faculty: Anterior Insula Interoceptive Allostasis & Energy Budgeting (Craig / Sterling / Barrett)
        self.allostasis_cortex: InteroceptiveAllostasisCortex = InteroceptiveAllostasisCortex()

        # Round 7 Faculty: Claustrum Multimodal Salience Conductor & Perceptual Binding (Crick & Koch / Smith)
        self.claustrum_conductor: ClaustrumSalienceConductor = ClaustrumSalienceConductor()

        # Round 7 Faculty: Frontopolar BA10 Recursive And-Or Subgoal Tree & Regret Backtracking (Koechlin / Daw)
        self.subgoal_tree: RecursiveSubgoalTree = RecursiveSubgoalTree()

        # Round 7 Faculty: Spatial Euclidean Distance Transform (EDT) & Obstacle Clearance (Graziano / Felzenszwalb)
        self.distance_transform_engine: SpatialDistanceTransformEngine = (
            SpatialDistanceTransformEngine()
        )

        # Round 8 Faculty: Ventral Striatum & Lateral Habenula Dual Value Opponency (Matsumoto & Hikosaka / Cools)
        self.value_opponency: DualValueOpponencyCortex = DualValueOpponencyCortex()

        # Round 8 Faculty: Posterior Parietal Cortex (PPC) Potential Field Navigation (Andersen & Buneo / Khatib)
        self.potential_field_navigator: ParietalPotentialFieldNavigator = (
            ParietalPotentialFieldNavigator()
        )

        # Round 8 Faculty: Inferotemporal (IT) Viewpoint- & Scale-Invariant Shape Hasher (DiCarlo / Hu)
        self.shape_hasher: InferotemporalShapeHasher = InferotemporalShapeHasher()

        # Round 9 Faculty: Subthalamic Nucleus (STN) Drift-Diffusion Threshold Modulation (Ratcliff / Frank)
        self.drift_diffusion: SubthalamicDriftDiffusionEngine = SubthalamicDriftDiffusionEngine()

        # Round 9 Faculty: Parietal Egocentric Multi-Scale Foveation (Itti & Koch / Schwartz)
        self.foveated_attention: ParietalFoveatedAttentionEngine = ParietalFoveatedAttentionEngine()

        # Round 9 Faculty: Prefrontal Counterfactual Blame Attribution (Halpern & Pearl)
        self.blame_attributor: PrefrontalBlameAttributor = PrefrontalBlameAttributor()

        # Round 10 Faculty: Hippocampal-Entorhinal Prioritized Experience Replay (Foster & Wilson / Schaul)
        self.hippocampal_replay: HippocampalPrioritizedReplayBuffer = (
            HippocampalPrioritizedReplayBuffer()
        )

        # Round 10 Faculty: Lateral Prefrontal Cortex (LPFC) Dual Task Switcher (Monsell / Koechlin)
        self.task_switcher: LateralPrefrontalTaskSwitcher = LateralPrefrontalTaskSwitcher()

        # Round 10 Faculty: Retrosplenial Path Integration Drift Recalibration (Jeffery / Burgess)
        self.retrosplenial_anchor: RetrosplenialLandmarkAnchorEngine = (
            RetrosplenialLandmarkAnchorEngine()
        )

        # Round 11 Faculty: Hippocampal Theta-Gamma Phase Precession & Macro Chunking (O'Keefe / Lisman)
        self.phase_precession: HippocampalPhasePrecessionEngine = HippocampalPhasePrecessionEngine()

        # Round 11 Faculty: Frontopolar BA10 Counterfactual Exploration Horizon Engine (Koechlin / Rushworth)
        self.exploration_horizon: FrontopolarExplorationHorizonEngine = (
            FrontopolarExplorationHorizonEngine()
        )

        # Round 11 Faculty: Superior Colliculus Multisensory Alignment & Saccadic Arbiter (Stein & Meredith)
        self.superior_colliculus: SuperiorColliculusTargetArbiter = (
            SuperiorColliculusTargetArbiter()
        )

        # Round 12 Faculty: Cerebellar Microzone Granule-Purkinje Forward Adaptation (Marr-Albus-Ito)
        self.cerebellar_microzone: CerebellarMicrozoneAdaptor = CerebellarMicrozoneAdaptor()

        # Round 12 Faculty: Ventral Premotor (F5 / Mirror Neuron) Affordance Resonator (Rizzolatti)
        self.premotor_resonator: VentralPremotorAffordanceResonator = (
            VentralPremotorAffordanceResonator()
        )

        # Round 12 Faculty: Anterior Cingulate Cortex (ACC) Effort Allocator (Botvinick / Shenhav)
        self.effort_allocator: AnteriorCingulateEffortAllocator = AnteriorCingulateEffortAllocator()

        # Round 13 Faculty: Subicular Egocentric-to-Allocentric Boundary Vector Cell (BVC) Map (Hartley-Burgess)
        self.subicular_bvc: SubicularBoundaryVectorEngine = SubicularBoundaryVectorEngine()

        # Round 13 Faculty: Ventromedial Prefrontal Cortex (vmPFC) Somatic Marker Engine (Damasio-Bechara)
        self.somatic_marker: VentromedialSomaticMarkerEngine = VentromedialSomaticMarkerEngine()

        # Round 13 Faculty: Pulvinar Thalamic Transthalamic Routing & Attentional Gating (Saalmann-Kastner)
        self.pulvinar_router: PulvinarAttentionRouter = PulvinarAttentionRouter()

        # Round 14 Faculty: Parahippocampal Scene Schemas & Topological Recognition (Epstein / Bar)
        self.parahippocampal_context: ParahippocampalContextEngine = ParahippocampalContextEngine()

        # Round 14 Faculty: Locus Coeruleus Noradrenergic Exploration-Exploitation Switch (Aston-Jones / Cohen)
        self.locus_coeruleus: LocusCoeruleusNoradrenalineEngine = (
            LocusCoeruleusNoradrenalineEngine()
        )

        # Round 14 Faculty: Secondary Somatosensory Haptic Texture & Surface Friction Integration (Craig / Kaas)
        self.somatosensory_haptic: SomatosensoryHapticEngine = SomatosensoryHapticEngine()

        # Connect GNW conscious broadcast listener for executive arbitration
        self.gnw.register_listener("executive_arbitration", self._on_conscious_ignition)

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
        self.delayed_effect_modeler: DelayedEffectModeler = self.delayed_effects
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

        # Tiered Cognitive Workspace & Transaction Manager / Kernel Services live mirroring
        self.services: KernelServices | None = services
        self.event_bus: KernelEventBus | None = (
            getattr(services, "event_bus", None) if services is not None else None
        )
        self.capability_resolver: CapabilityResolver | None = (
            getattr(services, "capability_resolver", None) if services is not None else None
        )
        self.tiered_workspace: TieredWorkspace | None = (
            getattr(services, "tiered_workspace", None) if services is not None else None
        )
        self.transaction_manager: TransactionManager | None = (
            getattr(services, "transaction_manager", None) if services is not None else None
        )
        self.world_kernel: WorldKernel | None = (
            getattr(services, "world_kernel", None) if services is not None else None
        )
        self.active_task_frame: TaskFrame | None = None
        if services is not None and self.tiered_workspace is not None:
            self.attach_tiered_workspace(self.tiered_workspace)

        # ── Autonomous Learning Integration (Phases B, C, D, G, I) ─────────
        from hbllm.hcir.world.autonomous_learning import (
            CausalTheoryPlanner,
            ClickAffordanceLearner,
            CrossGameKnowledgeTransfer,
            DeliberationEntropyMonitor,
            MetaLearningGovernor,
        )

        self.causal_planner: CausalTheoryPlanner = CausalTheoryPlanner()
        self.knowledge_transfer: CrossGameKnowledgeTransfer = CrossGameKnowledgeTransfer()
        self.meta_governor: MetaLearningGovernor = MetaLearningGovernor()
        self.click_learner: ClickAffordanceLearner = ClickAffordanceLearner()
        self.entropy_monitor: DeliberationEntropyMonitor = DeliberationEntropyMonitor()

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
        self.world_kernel = getattr(services, "world_kernel", None)
        if getattr(services, "transaction_manager", None) is not None:
            self.transaction_manager = services.transaction_manager
        if getattr(services, "tiered_workspace", None) is not None:
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
        if self.transaction_manager is None:
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
        self.pending_transactions.clear()
        return committed_tx

    def submit_to_workspace(
        self,
        source_cortex: str,
        salience: float,
        content: dict[str, Any],
        epistemic_uncertainty: float = 0.0,
    ) -> IgnitionEvent | None:
        """Submit a candidate representation to the Global Neuronal Workspace.

        Triggers non-linear ignition if salience exceeds the dynamic threshold.
        """
        self.gnw.submit_candidate(
            source_cortex=source_cortex,
            salience=salience,
            content=content,
            epistemic_uncertainty=epistemic_uncertainty,
        )
        return self.gnw.compete_and_ignite()

    def modulate_cognition(
        self,
        reward_prediction_error: float = 0.0,
        unexpected_surprise: float = 0.0,
        prediction_accuracy: float = 1.0,
        risk_level: float = 0.0,
    ) -> NeuromodulatorLevels:
        """Step the neuromodulatory systems (DA, NE, ACh, 5-HT) and adapt active inference parameters."""
        if reward_prediction_error != 0.0:
            self.neuromodulatory_system.update_reward_prediction_error(reward_prediction_error)
        if unexpected_surprise != 0.0:
            self.neuromodulatory_system.update_unexpected_uncertainty(unexpected_surprise)
        if prediction_accuracy != 1.0:
            self.neuromodulatory_system.update_expected_uncertainty(prediction_accuracy)
        if risk_level != 0.0:
            self.neuromodulatory_system.update_temporal_stress(risk_level)
        self.neuromodulatory_system.step_homeostatic_decay()
        params = self.neuromodulatory_system.get_effective_hyperparameters()
        self.hierarchical_active_inference.sensory_precision = params["sensory_precision"]
        return self.neuromodulatory_system.current

    def _on_conscious_ignition(self, event: IgnitionEvent) -> None:
        """Handle conscious global broadcasts ignited by the Global Neuronal Workspace."""
        winner_cortex = event.winner.source_cortex
        if winner_cortex in ("loss_monitor", "surprise_engine"):
            if self._phase in (EpistemicPhase.MENTAL_SIMULATION, EpistemicPhase.EXPLOITATION):
                logger.info(
                    "AutonomousEpistemicEngine: GNW broadcast from [%s] triggered executive reset to REPLANNING",
                    winner_cortex,
                )
                self.transition_to_phase(EpistemicPhase.REPLANNING)

    def synchronize_and_bind_entities(
        self,
        feature_entities: list[dict[str, Any]],
        target_features: dict[str, Any] | None = None,
    ) -> list[BoundFeatureBundle]:
        """Apply Kuramoto phase synchronization and TRN gating to bind perceptual features."""
        bundles = self.thalamocortical_binding.bind_features_oscillatory(feature_entities)
        if target_features:
            bundles = self.thalamocortical_binding.apply_trn_attentional_gating(
                bundles, target_features
            )
        return bundles

    def evaluate_superposition_actions(
        self,
        candidate_actions: list[Any],
        simulator_fn: Any,
    ) -> list[SuperpositionEvaluationResult]:
        """Evaluate actions across multiple counterfactual world hypotheses in superposition."""
        return self.superposition_planner.evaluate_superposition_actions(
            candidate_actions=candidate_actions,
            simulator_fn=simulator_fn,
        )

    def step_predictive_coding(
        self,
        sensory_inputs: dict[str, float] | None = None,
        abstract_priors: dict[str, float] | None = None,
    ) -> PredictiveCodingSnapshot:
        """Execute a bidirectional predictive coding cycle across cortical column layers."""
        return self.predictive_coding_cortex.step_hierarchical_cycle(
            sensory_inputs=sensory_inputs,
            abstract_priors=abstract_priors,
        )

    def path_integrate_odometry(
        self,
        delta_r: float,
        delta_c: float,
    ) -> tuple[int, int]:
        """Perform continuous attractor odometric path integration across multi-scale grid cells."""
        return self.entorhinal_grid_cortex.path_integrate_step(delta_r, delta_c)

    def assess_metacognitive_agency(
        self,
        action_id: Any,
        raw_confidence: float,
        predicted_avatar_pos: tuple[int, int] | None,
        observed_avatar_pos: tuple[int, int] | None,
        observed_other_changes_count: int = 0,
        empirical_success: bool | None = None,
    ) -> MetacognitiveAssessment:
        """Evaluate sense of agency and calibrate confidence to govern simulation depth."""
        agency = self.metacognitive_agency_cortex.compute_agency_attribution(
            predicted_avatar_pos=predicted_avatar_pos,
            observed_avatar_pos=observed_avatar_pos,
            observed_other_changes_count=observed_other_changes_count,
        )
        return self.metacognitive_agency_cortex.calibrate_confidence_and_govern_compute(
            action_id=action_id,
            raw_confidence=raw_confidence,
            agency_score=agency,
            empirical_success=empirical_success,
        )

    def synthesize_procedural_skills_from_traces(
        self,
        min_support: int = 2,
    ) -> list[Any]:
        """Induce reusable declarative skills from recurring winning episode traces."""
        self.skill_synthesizer.min_support_traces = min_support
        return self.skill_synthesizer.induce_all_recurrent_skills()

    def update_ballistic_particles(
        self,
        particles: list[dict[str, Any]],
        grid: np.ndarray,
        barrier_features: set[int] | None = None,
        absorbing_features: set[int] | None = None,
        deflection_mode: str = "specular_normal",
        horizon: int | None = None,
    ) -> None:
        """Project ballistic particle kinematic trajectories into spatiotemporal collision cones."""
        barriers = (
            barrier_features
            if barrier_features is not None
            else self.symbolic_theory.barrier_features
        )
        self.collision_cones.update_ballistic_particle_trajectories(
            particles=particles,
            grid=grid,
            barrier_features=barriers,
            absorbing_features=absorbing_features,
            deflection_mode=deflection_mode,
            horizon=horizon,
        )

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

        if services is None:
            tx_mgr = TransactionManager(sim_ws)
            resolver = CapabilityResolver()
            scheduler = KernelInstructionScheduler()
            services = KernelServices(
                workspace=sim_ws,
                transaction_manager=tx_mgr,
                capability_resolver=resolver,
                scheduler=scheduler,
            )

        interpreter = HCIRInterpreter(sim_ws, services)
        exec_result = await interpreter.execute(stream)
        has_errors = (
            any(a.severity == "error" for a in exec_result.annotations) or not exec_result.success
        )

        if not has_errors:
            if sim_ws is not workspace:
                for node in sim_ws.graph.all_nodes():
                    existing = workspace.get_node(node.id)
                    if existing is None or node.model_dump() != existing.model_dump():
                        workspace.upsert_node(node.model_copy(deep=True), author="subgoal_exec")
                workspace.drop_branch(sim_branch)
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

    def record_motor_efference_copy(
        self, action: Any, expected_delta: tuple[int, int]
    ) -> tuple[int, int]:
        """Buffer efference copy into the Cerebellar Smith Predictor ahead of sensory afferents."""
        cur = self.avatar_pos or (0, 0)
        return self.smith_predictor.record_efference_copy(
            step=self.step_counter,
            action_id=action,
            expected_delta=expected_delta,
            current_pos=cur,
        )

    def get_forward_compensated_position(self) -> tuple[int, int] | None:
        """Return zero-latency forward compensated position if efference copy is unconfirmed."""
        if self.avatar_pos is None:
            return None
        return self.smith_predictor.get_forward_compensated_position(self.avatar_pos)

    def recall_episodic_attractor_plan(
        self, sensory_cue: np.ndarray, match_threshold: float = 0.35
    ) -> tuple[list[Any] | None, float]:
        """Recall winning procedural macro plan using Hippocampal DG separation and CA3 auto-association."""
        return self.episodic_cortex.recall_attractor_plan(
            sensory_cue, match_threshold=match_threshold
        )

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

    @property
    def neuromodulation_governor(self) -> AdaptiveNeuromodulationGovernor:
        """Dual Neuromodulatory Governor (VTA Dopamine & LC Noradrenaline)."""
        return self.deliberation_engine.neuromodulation

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

    def to_egocentric_coordinate(
        self,
        target_pos: tuple[int, int],
        heading: HeadingOrientation | None = None,
    ) -> EgocentricCoordinate | None:
        """Convert allocentric target to agent-centered egocentric coordinate."""
        if self.avatar_pos is None:
            return None
        h = heading or self.retrosplenial_transformer.infer_heading_from_displacement(
            getattr(self, "prev_avatar_pos", None), self.avatar_pos
        )
        return self.retrosplenial_transformer.allo_to_ego(target_pos, self.avatar_pos, h)

    def to_allocentric_coordinate(
        self,
        ego_coord: EgocentricCoordinate | tuple[int, int],
        heading: HeadingOrientation | None = None,
    ) -> tuple[int, int] | None:
        """Convert agent-relative egocentric coordinate to world allocentric position."""
        if self.avatar_pos is None:
            return None
        h = heading or self.retrosplenial_transformer.infer_heading_from_displacement(
            getattr(self, "prev_avatar_pos", None), self.avatar_pos
        )
        return self.retrosplenial_transformer.ego_to_allo(ego_coord, self.avatar_pos, h)

    def evaluate_policy_expected_free_energy(
        self,
        candidate_transitions: dict[Any, tuple[int, int]],
        grid: np.ndarray | None = None,
        goal_confidence: float = 0.0,
    ) -> list[ExpectedFreeEnergyScore]:
        """Rank candidate actions using Karl Friston's Expected Free Energy decomposition."""
        curr_g = grid if grid is not None else getattr(self, "last_observed_grid", None)
        if curr_g is None:
            curr_g = np.zeros((10, 10), dtype=int)

        uncertainty = 0.5
        unconfirmed = 0
        if self.symbolic_theory is not None:
            unconfirmed = len(getattr(self.symbolic_theory, "unconfirmed_hypotheses", []))
            uncertainty = min(1.0, max(0.1, unconfirmed / 5.0))

        return self.efe_evaluator.rank_candidate_actions(
            candidate_transitions=candidate_transitions,
            current_pos=self.avatar_pos,
            grid=curr_g,
            goal_positions=self.learned_goal_positions,
            position_visit_counts=self.working_memory.position_visit_counts,
            unconfirmed_hypotheses_count=unconfirmed,
            goal_confidence=goal_confidence,
            world_theory_uncertainty=uncertainty,
        )

    def evaluate_counterfactual_regret(
        self,
        actual_action: Any,
        actual_reward: float,
        actual_next_pos: tuple[int, int] | None,
        candidate_actions: Sequence[Any],
        grid: np.ndarray | None = None,
    ) -> list[CounterfactualComparison]:
        """Compute counterfactual regret across unchosen candidate actions (Pearl Level 3)."""
        curr_g = grid if grid is not None else getattr(self, "last_observed_grid", None)
        if curr_g is None:
            curr_g = np.zeros((10, 10), dtype=int)

        barriers: set[tuple[int, int]] = set(self.learned_barriers)
        lethals: set[tuple[int, int]] = set()
        if hasattr(self, "level_lethal_positions") and self.current_level is not None:
            lethals = self.level_lethal_positions.get(self.current_level, set())

        return self.counterfactual_regret.evaluate_counterfactuals(
            actual_action=actual_action,
            actual_reward=actual_reward,
            current_pos=self.avatar_pos,
            actual_next_pos=actual_next_pos,
            candidate_actions=candidate_actions,
            grid=curr_g,
            goal_positions=self.learned_goal_positions,
            static_barriers=barriers,
            lethal_positions=lethals,
            action_deltas=self.action_deltas,
        )

    def extract_affordance_field(
        self,
        grid: np.ndarray,
        bg_color: int | None = None,
    ) -> np.ndarray:
        """Extract dense 2D Gibsonian affordance matrix across the sensory lattice."""
        bg = bg_color if bg_color is not None else self.estimate_background(grid)
        goals = self.learned_goal_positions
        goal_feats = {
            int(grid[r, c]) for r, c in goals if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
        }
        barrier_feats: set[int] = set()
        if self.symbolic_theory is not None:
            for p in self.symbolic_theory.barrier_predicates:
                if isinstance(p.value, int):
                    barrier_feats.add(p.value)
        return self.affordance_extractor.extract_affordance_map(
            grid=grid,
            bg_color=bg,
            goal_features=goal_feats,
            barrier_features=barrier_feats,
        )

    def extract_topological_contours(
        self,
        grid: np.ndarray,
        bg_color: int | None = None,
    ) -> list[TopologicalContour]:
        """Extract exterior boundaries and interior cavity loops for all entities."""
        bg = bg_color if bg_color is not None else self.estimate_background(grid)
        return self.affordance_extractor.extract_topological_contours(grid=grid, bg_color=bg)

    def should_permit_deliberation(self, estimated_cost: float = 5.0) -> tuple[bool, str]:
        """Check whether interoceptive allostasis permits expensive forward mental simulation."""
        return self.allostasis_cortex.should_permit_deliberation(estimated_cost=estimated_cost)

    def consume_allostatic_energy(
        self, activity_type: str, scale: float = 1.0
    ) -> InteroceptiveState:
        """Deduct metabolic energy and update interoceptive allostasis state."""
        state = self.allostasis_cortex.consume_energy(activity_type=activity_type, scale=scale)
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(state.to_causal_predicate())
        return state

    def bind_multimodal_percept(
        self,
        grid_shape: tuple[int, int] | None = None,
        visual_entities: list[SpatialEntity] | None = None,
        hazard_map: dict[tuple[int, int], float] | None = None,
        goal_coords: list[tuple[int, int]] | None = None,
        affordance_fields: dict[str, Any] | None = None,
    ) -> UnifiedPerceptualGestalt:
        """Bind visual, hazard, and goal streams via claustrum conductor into a unified gestalt."""
        last_g = getattr(self, "last_observed_grid", None)
        shape = (
            grid_shape
            if grid_shape is not None
            else ((int(last_g.shape[0]), int(last_g.shape[1])) if last_g is not None else (10, 10))
        )
        goals = goal_coords if goal_coords is not None else list(self.learned_goal_positions)
        gestalt = self.claustrum_conductor.bind_modalities(
            grid_shape=(int(shape[0]), int(shape[1])),
            visual_entities=visual_entities,
            hazard_map=hazard_map,
            goal_coords=goals,
            affordance_fields=affordance_fields,
            agent_coord=self.avatar_pos,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(gestalt.to_causal_predicate())
        return gestalt

    def compute_spatial_distance_field(
        self,
        obstacles: set[tuple[int, int]] | None = None,
        grid_shape: tuple[int, int] | None = None,
    ) -> EuclideanDistanceTransformResult:
        """Compute exact 2D Euclidean Distance Transform clearance and repulsive gradient fields."""
        last_g = getattr(self, "last_observed_grid", None)
        shape = (
            grid_shape
            if grid_shape is not None
            else ((int(last_g.shape[0]), int(last_g.shape[1])) if last_g is not None else (10, 10))
        )
        obs = obstacles if obstacles is not None else set(self.learned_barriers)
        result = self.distance_transform_engine.compute_edt(
            grid_shape=(int(shape[0]), int(shape[1])),
            obstacles=obs,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(result.to_causal_predicate())
        return result

    def update_value_opponency(
        self,
        state_key: tuple[int, int] | str,
        reward_plus: float,
        punishment_minus: float,
        next_state_key: tuple[int, int] | str | None = None,
    ) -> OpponencyValueState:
        """Update dual appetitive and aversive values and register opponency predicate."""
        state = self.value_opponency.update_transition(
            state_key=state_key,
            reward_plus=reward_plus,
            punishment_minus=punishment_minus,
            next_state_key=next_state_key,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(state.to_causal_predicate())
        return state

    def compute_navigational_force(
        self,
        goal_pos: tuple[int, int],
        agent_pos: tuple[int, int] | None = None,
        obstacles: set[tuple[int, int]] | None = None,
        grid_shape: tuple[int, int] | None = None,
    ) -> PotentialFieldVector:
        """Synthesize attractive and repulsive potential forces into a single navigation vector."""
        pos = agent_pos or self.avatar_pos or (0, 0)
        edt_result = self.compute_spatial_distance_field(obstacles=obstacles, grid_shape=grid_shape)
        vec = self.potential_field_navigator.compute_navigation_vector(
            agent_pos=pos,
            goal_pos=goal_pos,
            edt_result=edt_result,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(vec.to_causal_predicate())
        return vec

    def hash_entity_shape(
        self, coordinates: set[tuple[int, int]] | list[tuple[int, int]]
    ) -> ShapeArchetypeSignature:
        """Compute translation- and scale-invariant Hu moments and canonical archetype hash."""
        sig = self.shape_hasher.compute_signature(coordinates)
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(sig.to_causal_predicate())
        return sig

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
        return target_graph.transitive_reduction()

    def step_drift_diffusion(
        self,
        evidence_rates: dict[Any, float],
        conflict_level: float = 0.0,
    ) -> DriftDiffusionDecision:
        """Accumulate drift-diffusion evidence modulated by STN hyperdirect conflict boundary."""
        decision = self.drift_diffusion.step_accumulation(
            evidence_rates=evidence_rates,
            conflict_level=conflict_level,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(decision.to_causal_predicate())
        return decision

    def foveate_sensory_lattice(
        self,
        grid: np.ndarray | None = None,
        gaze_center: tuple[int, int] | None = None,
    ) -> FoveatedSaliencyMap:
        """Apply parietal multi-scale foveation (fovea, parafovea, periphery) to sensory grid."""
        curr_g = grid if grid is not None else getattr(self, "last_observed_grid", None)
        if curr_g is None:
            curr_g = np.zeros((10, 10), dtype=int)
        center = gaze_center or self.avatar_pos or (curr_g.shape[0] // 2, curr_g.shape[1] // 2)
        goals = list(self.learned_goal_positions) if hasattr(self, "learned_goal_positions") else []
        foveated = self.foveated_attention.foveate_grid(
            grid=curr_g,
            gaze_center=center,
            secondary_anchors=goals,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(foveated.to_causal_predicate())
        return foveated

    def attribute_plan_failure(
        self,
        subgoal_trace: list[dict[str, Any]],
        failure_condition: str = "dead_end",
    ) -> PlanAttributionReport:
        """Perform Pearl-Halpern counterfactual blame attribution across executed subgoals."""
        hazards = set()
        if hasattr(self, "level_lethal_positions") and self.current_level is not None:
            hazards = self.level_lethal_positions.get(self.current_level, set())
        barriers = set(self.learned_barriers) if hasattr(self, "learned_barriers") else set()
        report = self.blame_attributor.attribute_plan_failure(
            subgoal_trace=subgoal_trace,
            failure_condition=failure_condition,
            lethal_hazards=hazards,
            blocked_targets=barriers,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(report.to_causal_predicate())
        return report

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
        return target_graph.minimum_spanning_tree()

    def record_replay_transition(
        self,
        state: dict[str, Any],
        action: Any,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
        td_error: float = 0.0,
        surprise: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ReplayTransition:
        """Record an episodic transition into hippocampal prioritized replay memory."""
        trans = self.hippocampal_replay.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            td_error=td_error,
            surprise=surprise,
            timestamp=float(self.step_counter),
            metadata=metadata,
        )
        return trans

    def sample_prioritized_replay(
        self, batch_size: int = 16, beta: float | None = None
    ) -> ReplayBatch:
        """Sample prioritized transitions with importance-sampling bias weights."""
        return self.hippocampal_replay.sample(batch_size=batch_size, beta=beta)

    def arbitrate_cognitive_task_set(
        self,
        hazard_distance: float | None = None,
        goal_distance: float | None = None,
        manipulable_adjacent: bool = False,
        collectible_adjacent: bool = False,
        epistemic_uncertainty: float = 0.0,
        cognitive_load: float = 0.0,
    ) -> TaskArbitrationResult:
        """Arbitrate active LPFC task set (NAVIGATE, INTERACT, AVOID, COLLECT, EXPLORE)."""
        affinities = self.task_switcher.evaluate_affinities(
            working_memory=self.working_memory,
            hazard_distance=hazard_distance,
            goal_distance=goal_distance,
            manipulable_adjacent=manipulable_adjacent,
            collectible_adjacent=collectible_adjacent,
            epistemic_uncertainty=epistemic_uncertainty,
        )
        result = self.task_switcher.arbitrate(
            affinities=affinities,
            cognitive_load=cognitive_load,
            timestamp=float(self.step_counter),
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(
                CausalPredicate(
                    variable="active_task_set",
                    operator="==",
                    value=result.active_task.value,
                    relation="lpfc_task_arbitration",
                    target_entity="avatar",
                )
            )
        return result

    def recalibrate_spatial_drift_with_landmarks(
        self,
        odometry_pos: tuple[int, int],
        landmark_observations: list[tuple[str, tuple[int, int]]],
    ) -> LandmarkRecalibrationResult:
        """Recalibrate dead-reckoning position and entorhinal coordinates using visual landmarks."""
        result = self.retrosplenial_anchor.recalibrate(
            odometry_pos=odometry_pos,
            landmark_observations=landmark_observations,
        )
        if result.has_drift_correction:
            if self.avatar_pos == odometry_pos:
                self.avatar_pos = result.corrected_position
            if self.symbolic_theory is not None:
                self.symbolic_theory.record_causal_predicate(
                    CausalPredicate(
                        variable="landmark_recalibration",
                        operator="==",
                        value=f"drift:({result.drift_vector[0]:.2f},{result.drift_vector[1]:.2f})",
                        relation="rsc_landmark_anchor",
                        target_entity="avatar",
                    )
                )
        return result

    def compute_all_pairs_geodesics(
        self,
        graph: CognitiveGraph | None = None,
        weight_attribute: str = "weight",
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], list[str]]]:
        """Compute all-pairs shortest paths and distance matrix across cognitive graph."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.all_pairs_shortest_paths(weight_attribute=weight_attribute)

    def compress_trajectory_phase_precession(
        self,
        positions: list[tuple[int, int]],
        actions: list[Any],
        duration_s: float = 2.0,
    ) -> PhasePrecessionBurst:
        """Compress spatial trajectory and actions into a theta-gamma oscillatory burst."""
        burst = self.phase_precession.encode_trajectory_burst(
            positions=positions,
            actions=actions,
            behavioral_duration_s=duration_s,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(burst.to_causal_predicate())
        return burst

    def evaluate_counterfactual_branch_switch(self) -> BranchSwitchDecision:
        """Evaluate BA10 counterfactual branches and execute switch if advantageous."""
        decision = self.exploration_horizon.evaluate_branch_switch()
        if decision.should_switch:
            self.exploration_horizon.execute_branch_switch(decision)
            if self.symbolic_theory is not None:
                self.symbolic_theory.record_causal_predicate(decision.to_causal_predicate())
        return decision

    def arbitrate_saccadic_target(
        self,
        visual_saliency: np.ndarray | None = None,
        goal_positions: list[tuple[int, int]] | None = None,
        obstacle_clearance: np.ndarray | None = None,
    ) -> SaccadeTarget | None:
        """Generate SC multisensory topographic priority map and select primary saccade target."""
        if visual_saliency is None:
            visual_saliency = np.zeros((10, 10), dtype=np.float64)
        goals = (
            goal_positions
            if goal_positions is not None
            else list(getattr(self, "learned_goal_positions", []))
        )
        priority_map = self.superior_colliculus.build_priority_map(
            visual_saliency=visual_saliency,
            goal_positions=goals,
            obstacle_clearance=obstacle_clearance,
            avatar_pos=self.avatar_pos,
        )
        target = self.superior_colliculus.select_primary_saccade_target(priority_map)
        if target is not None and self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(target.to_causal_predicate())
        return target

    def find_graph_bottlenecks_and_bridges(
        self, graph: CognitiveGraph | None = None
    ) -> tuple[set[str], list[tuple[str, str]]]:
        """Identify critical articulation points and single-point-of-failure bridge edges."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.find_articulation_points_and_bridges()

    def is_graph_fault_tolerant_biconnected(self, graph: CognitiveGraph | None = None) -> bool:
        """Check if cognitive graph has redundant alternate paths between all vertices."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.is_biconnected()

    def adapt_cerebellar_motor_outcome(
        self,
        action_id: Any,
        state: dict[str, Any],
        actual_displacement: tuple[float, float],
        actual_latency_ms: float = 10.0,
    ) -> ClimbingFiberFeedback:
        """Adapt cerebellar microzone forward model Purkinje weights using climbing fiber feedback."""
        feedback = self.cerebellar_microzone.adapt_from_feedback(
            action_id=action_id,
            current_state=state,
            actual_displacement=actual_displacement,
            actual_latency_ms=actual_latency_ms,
            timestamp=float(self.step_counter),
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(feedback.to_causal_predicate())
        return feedback

    def observe_external_affordance_interaction(
        self,
        entity_id: str,
        entity_role: str,
        initial_pos: tuple[int, int],
        final_pos: tuple[int, int],
        target_id: str,
        target_role: str,
        state_mutation: dict[str, Any],
    ) -> AffordanceResonancePattern | None:
        """Process observed external interaction and deduce vicarious causal mechanics."""
        pattern = self.premotor_resonator.observe_interaction(
            entity_id=entity_id,
            entity_role=entity_role,
            initial_pos=initial_pos,
            final_pos=final_pos,
            target_entity_id=target_id,
            target_role=target_role,
            state_mutation=state_mutation,
        )
        if pattern is not None and self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(pattern.to_causal_predicate())
        return pattern

    def allocate_cognitive_deliberation_effort(
        self,
        action_values: dict[Any, float],
        hazard_distance: float | None = None,
        state_familiarity: float = 1.0,
        energy_reserve: float = 100.0,
    ) -> EffortAllocationDecision:
        """Dynamically regulate forward search depth and simulation budgets using ACC EVC theory."""
        profile = self.effort_allocator.compute_conflict_profile(
            action_values=action_values,
            hazard_distance=hazard_distance,
            state_familiarity=state_familiarity,
        )
        decision = self.effort_allocator.allocate_effort(
            profile=profile,
            energy_reserve=energy_reserve,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(decision.to_causal_predicate())
        return decision

    def compute_network_max_flow_min_cut(
        self,
        source: str,
        sink: str,
        graph: CognitiveGraph | None = None,
        capacity_attribute: str = "capacity",
    ) -> tuple[float, list[tuple[str, str]], set[str], set[str]]:
        """Compute directed maximum flow and minimum cutset bottleneck between source and sink."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.max_flow_min_cut(
            source=source, sink=sink, capacity_attribute=capacity_attribute
        )

    def compute_subicular_boundary_map(
        self,
        boundary_grid: np.ndarray,
        avatar_pos: tuple[int, int] = (0, 0),
    ) -> BoundaryVectorMap:
        """Compute egocentric-to-allocentric boundary vector map around avatar (Hartley-Burgess)."""
        bvc_map = self.subicular_bvc.compute_bvc_activations(
            avatar_pos=avatar_pos,
            boundary_grid=boundary_grid,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(bvc_map.to_causal_predicate())
        return bvc_map

    def evaluate_somatic_marker_bias(
        self,
        candidate_actions: list[Any],
        projected_state_hashes: dict[Any, int],
    ) -> SomaticAffectiveBias:
        """Evaluate pre-deliberative somatic gut-feeling biases and visceral alarms (Damasio-Bechara)."""
        bias = self.somatic_marker.evaluate_somatic_bias(
            candidate_actions=candidate_actions,
            projected_state_hashes=projected_state_hashes,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(bias.to_causal_predicate())
        return bias

    def update_pulvinar_routing(
        self,
        cognitive_load: float,
        threat_urgency: float = 0.0,
    ) -> PulvinarRoutingState:
        """Regulate cortico-cortical transthalamic information transfer bandwidth (Saalmann-Kastner)."""
        routing_state = self.pulvinar_router.update_routing(
            cognitive_load=cognitive_load,
            threat_urgency=threat_urgency,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(routing_state.to_causal_predicate())
        return routing_state

    def extract_minimal_feedback_arc_set(
        self,
        graph: CognitiveGraph | None = None,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Extract minimal Feedback Arc Set (FAS) to sever causal graph cycles into a DAG."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.feedback_arc_set()

    def classify_spatial_context(
        self,
        boundary_grid: np.ndarray,
    ) -> SpatialContextSchema:
        """Classify environmental envelope into an archetypal spatial schema (Epstein-Bar)."""
        schema = self.parahippocampal_context.analyze_environmental_envelope(boundary_grid)
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(schema.to_causal_predicate())
        return schema

    def update_noradrenergic_gain(
        self,
        immediate_utility: float,
        progress_made: bool,
    ) -> NoradrenergicState:
        """Calibrate LC exploration-exploitation regime and softmax temperature (Aston-Jones-Cohen)."""
        state = self.locus_coeruleus.update_gain_state(
            immediate_utility=immediate_utility,
            progress_made=progress_made,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(state.to_causal_predicate())
        return state

    def process_haptic_surface_feedback(
        self,
        position: tuple[int, int],
        expected_displacement: tuple[float, float],
        actual_displacement: tuple[float, float],
    ) -> HapticAffordanceFeedback:
        """Infer surface friction, drag, and traction efficiency from kinesthetic error (Craig-Kaas)."""
        feedback = self.somatosensory_haptic.process_kinesthetic_feedback(
            position=position,
            expected_displacement=expected_displacement,
            actual_displacement=actual_displacement,
        )
        if self.symbolic_theory is not None:
            self.symbolic_theory.record_causal_predicate(
                feedback.inferred_profile.to_causal_predicate()
            )
        return feedback

    def condense_graph_scc_metagraph(
        self,
        graph: CognitiveGraph | None = None,
    ) -> tuple[dict[int, set[str]], list[tuple[int, int]]]:
        """Condense directed graph strongly connected components (SCCs) into a DAG metagraph."""
        if graph is not None:
            target_graph = graph
        elif self.tiered_workspace is not None:
            target_graph = getattr(self.tiered_workspace.brain, "graph", None) or CognitiveGraph(
                False
            )
        else:
            target_graph = CognitiveGraph(False)
        return target_graph.condense_scc_metagraph()

    @property
    def phase(self) -> EpistemicPhase:
        """Active cognitive lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, value: EpistemicPhase) -> None:
        self.transition_to_phase(value, reason="state_assignment")

    @property
    def current_phase(self) -> EpistemicPhase:
        """Active cognitive lifecycle phase (alias)."""
        return self._phase

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
        if hasattr(self, "entropy_monitor") and self.entropy_monitor is not None:
            self.entropy_monitor.reset_for_level()
        if is_new_level:
            self.environmental_step_limit = None
            self.budget_governor.reset(max_budget=self.exploration_budget)
            self.dual_memory.reset_episodic_memory()
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
            self.episodic_cortex.consolidated_macros[schema.schema_id] = schema.action_motif
            if hasattr(self, "procedural_macros"):
                self.procedural_macros[schema.schema_id] = schema.action_motif

        # DreamCoder-style program induction from winning episodic traces
        from hbllm.hcir.spatial_planner import SpatialActionIntent
        from hbllm.hcir.world.skill_program_synthesizer import ActionTraceStep

        if terminal_reward > 0 and self.episodic_cortex.current_episode:
            steps: list[ActionTraceStep] = []
            for tr in self.episodic_cortex.current_episode:
                intent = (
                    SpatialActionIntent.NAVIGATE_TO
                    if self.is_displacement_action(tr.action_id)
                    else (
                        SpatialActionIntent.INTERACT
                        if self.is_spatial_effector(tr.action_id)
                        else SpatialActionIntent.ACTUATE
                    )
                )
                steps.append(
                    ActionTraceStep(
                        intent=intent,
                        target_query={"pos": tr.avatar_pos} if tr.avatar_pos else {},
                        parameters={"action_id": tr.action_id, "reward": tr.reward},
                    )
                )
            if steps:
                req_acts = {
                    tr.action_id
                    for tr in self.episodic_cortex.current_episode
                    if isinstance(tr.action_id, int)
                }
                self.skill_synthesizer.record_winning_trace(
                    steps=steps,
                    required_actions=req_acts,
                    terminal_reward=terminal_reward,
                )

        newly_induced = self.skill_synthesizer.induce_all_recurrent_skills()
        res["induced_skills"] = newly_induced

        # Promote newly induced skills to TieredWorkspace.persistent if attached
        if self.tiered_workspace is not None:
            for skill in newly_induced:
                skill_name = getattr(skill, "name", "skill")
                subgoals = getattr(skill, "program", [])
                subgoal_count = (
                    len(getattr(subgoals, "subgoals", []))
                    if hasattr(subgoals, "subgoals")
                    else len(subgoals)
                )
                skill_node = SkillNode(
                    id=f"synthesized_skill_{skill_name}",
                    name=skill_name,
                    skill_name=skill_name,
                    description=f"Synthesized neuro-symbolic DSL skill with {subgoal_count} subgoals",
                    preconditions=[str(getattr(skill, "signature", ""))],
                    success_rate=0.9,
                )
                self.tiered_workspace.persistent.upsert_node(
                    skill_node, author="SkillProgramSynthesizer"
                )

        # Propose state mutations through TransactionManager if present
        all_ops = list(res.get("operations", []))
        for meta in self.skill_synthesizer.synthesized_metadata.values():
            all_ops.extend(meta.to_transaction_operations(tenant_id=self.world_id))

        if self.transaction_manager is not None and all_ops:
            from hbllm.hcir.transactions import HCIRTransaction

            tx = HCIRTransaction(
                author="HippocampalReplayConsolidator",
                operations=all_ops,
            )
            self.transaction_manager.commit(tx)

        return res

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
        bg = PerceptionEngine.estimate_background(
            grid,
            barrier_features=self.symbolic_theory.barrier_features,
            avatar_features=self.avatar_features,
            bg_feature=self.bg_feature,
        )
        if self.bg_feature is None or self.bg_feature != bg:
            old_bg = self.bg_feature
            self.bg_feature = bg
            if hasattr(self, "symbolic_theory") and self.symbolic_theory is not None:
                if (
                    old_bg is None
                    and bg != 0
                    and 0 not in getattr(self, "verified_safe_features", set())
                ):
                    self.symbolic_theory.walkable_features.discard(0)
                elif (
                    old_bg is not None
                    and old_bg != bg
                    and old_bg not in getattr(self, "verified_safe_features", set())
                ):
                    self.symbolic_theory.walkable_features.discard(old_bg)
                self.symbolic_theory.induce_walkable(bg)
        return bg

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
        curr_grid = self.normalize_sensory_input(curr_grid)
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

        # Biological Faculty: Popperian Inductive Hypothesis Consolidation & Retraction (Track 2)
        self.working_memory.hypothesis_engine.export_to_world_theory(self.symbolic_theory)
        self.working_memory.hypothesis_engine.retract_refuted_rules(self.symbolic_theory)

        # Biological Faculty: Entorhinal Grid Cell Continuous Attractor Path Integration
        if prev_av_pos is not None and self.avatar_pos is not None:
            delta_r = float(self.avatar_pos[0] - prev_av_pos[0])
            delta_c = float(self.avatar_pos[1] - prev_av_pos[1])
            if delta_r != 0.0 or delta_c != 0.0:
                self.path_integrate_odometry(delta_r, delta_c)

        # Biological Faculty: Cerebellar Smith Predictor Sensory Afferent Assimilation
        if self.avatar_pos is not None:
            self.smith_predictor.assimilate_sensory_feedback(
                step=self.step_counter,
                observed_pos=self.avatar_pos,
            )

        # Biological Faculty: Hierarchical Predictive Coding Error Minimization
        diff_count = (
            float(np.sum(self.prev_grid != curr_grid))
            if self.prev_grid is not None and self.prev_grid.shape == curr_grid.shape
            else 0.0
        )
        sensory_evidence: dict[str, float] = {
            "avatar_r": float(self.avatar_pos[0]) if self.avatar_pos else 0.0,
            "avatar_c": float(self.avatar_pos[1]) if self.avatar_pos else 0.0,
            "diff_count": diff_count,
        }
        pred_r = (
            float(self.last_predicted_pos[0])
            if self.last_predicted_pos
            else (float(prev_av_pos[0]) if prev_av_pos else 0.0)
        )
        pred_c = (
            float(self.last_predicted_pos[1])
            if self.last_predicted_pos
            else (float(prev_av_pos[1]) if prev_av_pos else 0.0)
        )
        abstract_priors: dict[str, float] = {
            "avatar_r": pred_r,
            "avatar_c": pred_c,
            "diff_count": 0.0,
        }
        self.step_predictive_coding(
            sensory_inputs=sensory_evidence, abstract_priors=abstract_priors
        )

        # Biological Faculty: Spiking Neural Network Kinematic Tracking & Limit Cycles
        if curr_grid.ndim == 2:
            self.snn_predictor.update_tracks_from_grid(curr_grid)
            self.snn_predictor.step_lif(curr_grid.astype(np.float32) * 0.2)
            for feat_id, period, phase in self.snn_predictor.extract_limit_cycles():
                self.cerebellar_clock.register_entity_cycle(feat_id, period, phase)

        # Biological Faculty: Closed-Loop Neuromodulatory Dynamics (Dopamine, NE, ACh, Serotonin)
        rpe = 1.0 if is_win else (-1.0 if is_lost else 0.0)
        pred_fe = float(getattr(self.predictive_coding_cortex, "total_free_energy", 0.0))
        risk_level = 0.0
        if self.avatar_pos is not None and curr_grid.ndim == 2:
            hazard_density = self.snn_predictor.predict_hazard_density(curr_grid, horizon_steps=2)
            ar, ac = self.avatar_pos
            if 0 <= ar < hazard_density.shape[0] and 0 <= ac < hazard_density.shape[1]:
                risk_level = float(hazard_density[ar, ac])
        surprise_val = 1.0 if is_lost else self.last_surprise
        self.modulate_cognition(
            reward_prediction_error=rpe,
            unexpected_surprise=surprise_val,
            prediction_accuracy=1.0 / (1.0 + pred_fe),
            risk_level=risk_level,
        )

        # Biological Faculty: BA10 Metacognitive Agency Attribution & Simulation Calibration
        if action is not None:
            raw_confidence = max(0.1, min(0.95, (1.0 - min(1.0, surprise_val)) / (1.0 + pred_fe)))
            meta_assess = self.assess_metacognitive_agency(
                action_id=action,
                raw_confidence=raw_confidence,
                predicted_avatar_pos=self.last_predicted_pos,
                observed_avatar_pos=self.avatar_pos,
                observed_other_changes_count=max(0, int(diff_count) - 2),
                empirical_success=(not is_lost),
            )
            self.working_memory.last_metacognitive_assessment = meta_assess
            self.prediction_budget.counterfactual_budget.max_depth = (
                meta_assess.recommended_simulation_depth
            )

        # Biological Faculty: Striatal & Prefrontal Delayed Contingency Modeling (W172, W173)
        if action is not None and curr_grid.ndim == 2:
            mutations_dict: dict[str, Any] = {}
            if diff_count > 0:
                mutations_dict["diff_count"] = diff_count
            if self.avatar_pos != prev_av_pos:
                mutations_dict["avatar_moved"] = True
            if is_win:
                mutations_dict["terminal_win"] = True
            delayed_records = self.delayed_effect_modeler.record_step(
                step=self.step_counter,
                action_id=action,
                state_features={"pos": self.avatar_pos},
                observed_mutations=mutations_dict,
            )
            for d_rec in delayed_records:
                if d_rec.observed_lag_steps > 0:
                    self.symbolic_theory.record_causal_predicate(d_rec.to_causal_predicate())

        # Biological Faculty: Ventral Stream Relational Scene Graph & Object Mutations (W038)
        if curr_grid.ndim == 2:
            current_scene_graph = self.ventral_relational_matcher.build_scene_graph(
                curr_grid, bg_color=self.estimate_background(curr_grid)
            )
            if self.working_memory.last_scene_graph is not None:
                mutations = self.ventral_relational_matcher.detect_object_mutations(
                    self.working_memory.last_scene_graph, current_scene_graph
                )
                if mutations["moved"]:
                    for m in mutations["moved"]:
                        self.symbolic_theory.record_causal_predicate(
                            CausalPredicate(
                                variable=f"object_displacement:{m['object_id']}",
                                operator="==",
                                value=f"{m['delta'][0]:.1f}_{m['delta'][1]:.1f}",
                                relation="relational_motion",
                                target_entity=str(m["object_id"]),
                            )
                        )
            self.working_memory.last_scene_graph = current_scene_graph

        # Biological Faculty: Thalamocortical Gamma-Band Phase Synchronization & TRN Gating (Track 1)
        if curr_grid.ndim == 2:
            ventral_tokens = self.it_segmenter.segment_objects(
                curr_grid,
                background_feature=self.bg_feature,
                avatar_features=self.avatar_features,
            )
            feature_entities = [
                {
                    "pos": t.centroid,
                    "feature": t.feature_id,
                    "area": t.area,
                    "role": (
                        "goal"
                        if t.feature_id in self.symbolic_theory.goal_features
                        else (
                            "barrier"
                            if t.feature_id in self.symbolic_theory.barrier_features
                            else "unknown"
                        )
                    ),
                }
                for t in ventral_tokens
            ]
            if self.avatar_pos is not None:
                feature_entities.append(
                    {
                        "pos": self.avatar_pos,
                        "feature": self.avatar_feature or 999,
                        "area": 1,
                        "role": "avatar",
                    }
                )
            target_feat = {"role": "goal"} if self.symbolic_theory.goal_features else None
            bound_bundles = self.synchronize_and_bind_entities(
                feature_entities, target_features=target_feat
            )
            self.working_memory.active_bound_bundles = bound_bundles
            for bundle in bound_bundles:
                if bundle.is_bound:
                    self.symbolic_theory.record_causal_predicate(bundle.to_causal_predicate())

        # Biological Faculty: Global Neuronal Workspace Competition & Conscious Ignition
        if is_lost:
            self.gnw.submit_candidate(
                source_cortex="loss_monitor",
                salience=0.95,
                content={"event": "loss", "avatar_pos": self.avatar_pos},
            )
        elif is_win:
            self.gnw.submit_candidate(
                source_cortex="goal_monitor",
                salience=0.95,
                content={"event": "win", "avatar_pos": self.avatar_pos},
            )
        elif pred_fe > 1.2:
            self.gnw.submit_candidate(
                source_cortex="predictive_coding",
                salience=min(0.9, pred_fe / 2.5),
                content={"free_energy": pred_fe},
            )
        elif self.last_surprise > 0.2:
            self.gnw.submit_candidate(
                source_cortex="surprise_engine",
                salience=min(0.9, self.last_surprise),
                content={"surprise": self.last_surprise},
            )
        elif getattr(self.acc_conflict_monitor, "active_conflict", False):
            self.gnw.submit_candidate(
                source_cortex="anterior_cingulate",
                salience=0.85,
                content={"frustration": self.acc_conflict_monitor.frustration},
            )

        self.gnw.compete_and_ignite()

        # Record spatial effector intervention feedback (domain-agnostic)
        if (
            action is not None
            and action_data is not None
            and self.prev_grid is not None
            and hasattr(self, "click_learner")
            and self.click_learner is not None
        ):
            coord = None
            if "row" in action_data and "col" in action_data:
                coord = (int(action_data["row"]), int(action_data["col"]))
            elif "y" in action_data and "x" in action_data:
                coord = (int(action_data["y"]), int(action_data["x"]))
            elif self.is_spatial_effector(action) and "pos" in action_data:
                coord = tuple(action_data["pos"])
            if coord is not None:
                self.click_learner.observe_click(coord, self.prev_grid, curr_grid)

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

    def evaluate_candidate_action_ensemble(
        self,
        action: Any,
        curr_grid: np.ndarray | None = None,
        horizon_ms: int = 60000,
    ) -> tuple[Any, bool]:
        """Evaluate candidate action through WorldKernel PredictiveRealityModel ensemble.

        Returns:
            tuple[EnsemblePrediction | None, bool]: The ensemble prediction (or None if
            predictive reality is unavailable) and a boolean indicating whether high
            inter-predictor disagreement was detected.

        If high disagreement is detected among physics, SNN, statistical, and neural
        predictors, the anterior cingulate / deliberation cortex is notified via KernelEvent.
        """
        if self.world_kernel is None or not hasattr(self.world_kernel, "predictive_reality"):
            return None, False

        from hbllm.hcir.world.world_context import WorldModelScope
        from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

        variables: dict[str, Any] = {}
        if curr_grid is not None:
            variables["grid"] = curr_grid
        elif hasattr(self, "last_observed_grid") and self.last_observed_grid is not None:
            variables["grid"] = self.last_observed_grid

        if hasattr(self.world_kernel, "digital_twin") and hasattr(
            self.world_kernel.digital_twin, "get_variables"
        ):
            variables.update(self.world_kernel.digital_twin.get_variables())

        snapshot = WorldStateSnapshot(
            world_id=self.world_id,
            timestamp=float(self.step_counter),
            variables=variables,
            entity_states={},
        )

        action_intent = str(action.name if hasattr(action, "name") else action)
        scope = WorldModelScope(world_id=self.world_id)

        pred = self.world_kernel.predictive_reality.predict(
            snapshot=snapshot,
            action_intent=action_intent,
            horizon_ms=horizon_ms,
            scope=scope,
        )

        disagreement = (
            self.world_kernel.predictive_reality.disagreement_analyzer.analyze_disagreement(
                pred.component_predictions
            )
        )

        if disagreement.high_disagreement:
            actor_pos = (
                (float(self.avatar_pos[0]), float(self.avatar_pos[1]))
                if self.avatar_pos is not None
                else None
            )
            self.acc_conflict.register_predictive_conflict(
                disagreement_score=disagreement.disagreement_index,
                conflicting_predictors=disagreement.conflicting_predictors,
                actor_pos=actor_pos,
                step=self.step_counter,
            )
            if self.event_bus is not None:
                self.event_bus.publish(
                    KernelEvent(
                        event_type="epistemic.ensemble_disagreement",
                        source="AutonomousEpistemicEngine",
                        data={
                            "action_intent": action_intent,
                            "disagreement_index": disagreement.disagreement_index,
                            "conflicting_predictors": disagreement.conflicting_predictors,
                            "world_id": self.world_id,
                        },
                    )
                )

        return pred, disagreement.high_disagreement

    def evaluate_transit_safety_cerebellar_snn(
        self,
        target_pos: tuple[int, int],
        curr_grid: np.ndarray | None = None,
    ) -> PhaseGateDecision:
        """Evaluate transit safety to target_pos combining Cerebellar timing and SNN limit cycle forecasting."""
        agent_pos = self.avatar_pos or (0, 0)
        grid = curr_grid if curr_grid is not None else getattr(self, "last_observed_grid", None)
        snn = None
        if self.world_kernel and hasattr(self.world_kernel, "predictive_reality"):
            snn = getattr(self.world_kernel.predictive_reality, "snn", None)
        if snn is None:
            snn = self.snn_predictor

        if snn is not None and grid is not None and isinstance(grid, np.ndarray):
            # Compute hazard density and pre-emptively inhibit dangerous actions via Habenula
            density = snn.predict_hazard_density(grid, horizon_steps=3)
            action_deltas: dict[Any, tuple[int, int]] = {}
            for act, dyn in self.action_dynamics.items():
                dr, dc = dyn.get_displacement()
                if dyn.confidence >= 0.5 and (dr != 0 or dc != 0):
                    action_deltas[act] = (dr, dc)
            if not action_deltas:
                action_deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

            self.habenular_ior.preemptively_inhibit_from_snn(
                hazard_density=density,
                current_pos=agent_pos,
                action_deltas=action_deltas,
                threshold=0.45,
            )

        return self.cerebellar_clock.evaluate_crossing_safety(
            agent_pos=agent_pos,
            target_pos=target_pos,
            snn_predictor=snn,
            grid=grid,
            current_step=self.step_counter,
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

    def transfer_structural_analogies(
        self,
        prototype_grid: np.ndarray,
    ) -> list[ObjectCorrespondence]:
        """Compute structural correspondences between a prototype scene and current grid."""
        if self.prev_grid is None or prototype_grid.ndim != 2:
            return []
        proto_graph = self.ventral_relational_matcher.build_scene_graph(
            prototype_grid, bg_color=self.estimate_background(prototype_grid)
        )
        curr_graph = self.ventral_relational_matcher.build_scene_graph(
            self.prev_grid, bg_color=self.estimate_background(self.prev_grid)
        )
        return self.ventral_relational_matcher.match_graphs(proto_graph, curr_graph)

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
                    is_edge_hud = (
                        min_c == max_c and (min_c == 0 or min_c == W - 1) and (max_r - min_r >= 4)
                    ) or (
                        min_r == max_r and (min_r == 0 or min_r == H - 1) and (max_c - min_c >= 4)
                    )
                    if not is_edge_hud:
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
            )
        except Exception:
            self._prev_oriented_threats = []
            return

        if self.step_counter <= 1 or self.prev_grid is None:
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
                if feat in effective_av_feats or feat in self.mobile_threat_features:
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
        self._prev_oriented_threats = threats

    # ── Master Decision Loop ──────────────────────────────────────────────────

    def decide(
        self,
        curr_grid: Any,
        available_actions: Sequence[Any],
        is_win: bool = False,
        is_lost: bool = False,
        action_schemas: Sequence[Any] | None = None,
        objects: Sequence[Any] | None = None,
        micro_actions: Sequence[Any] | None = None,
        **kwargs: Any,
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
            objects=objects,
            micro_actions=micro_actions,
            **kwargs,
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
        if barrier_pos == (16, 16):
            import traceback

            logger.warning("GROUNDED (16, 16) AS BARRIER:\n%s", "".join(traceback.format_stack()))
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
