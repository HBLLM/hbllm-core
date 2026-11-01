"""HCIR World — Embodied, Spatial, and Epistemic World Modeling Subsystem."""

from __future__ import annotations

from hbllm.hcir.world.amygdalar_threat_conditioning import (
    ConditionedThreatStimulus,
    DefensiveVigilanceMode,
    DefensiveVigilanceState,
)
from hbllm.hcir.world.anterior_cingulate_effort_allocator import (
    ConflictProfile,
    EffortAllocationDecision,
)
from hbllm.hcir.world.anterior_insular_timing import (
    InteroceptivePacingState,
    PacingMode,
)
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EpistemicObservation,
    EpistemicTruthValue,
)
from hbllm.hcir.world.boundary_grid_phase_aligner import (
    BoundaryPhaseAlignmentState,
    PhaseInnovationVector,
)
from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    BeliefTransitionEvent,
    CausalHypothesis,
    CausalPredicate,
)
from hbllm.hcir.world.causal_scm import (
    CausalVariable,
    CounterfactualOutcome,
    StructuralCausalModel,
)
from hbllm.hcir.world.cerebellar_microzone_adaptor import (
    ClimbingFiberFeedback,
    MotorPrediction,
)
from hbllm.hcir.world.claustrum_interhemispheric import (
    CrossModalSynchronyPacket,
    InterHemisphericBindingState,
)
from hbllm.hcir.world.claustrum_salience_conductor import (
    UnifiedPerceptualGestalt,
)
from hbllm.hcir.world.cortex_assimilator import (
    EpistemicFeedbackAssimilator,
    EpistemicPhase,
)
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortical_pipeline import (
    CorticalPipeline,
    PipelineResult,
)
from hbllm.hcir.world.cortical_stage import (
    CorticalStage,
    StageResult,
)
from hbllm.hcir.world.disagreement_analyzer import (
    DisagreementReport,
    PredictorDisagreementAnalyzer,
)
from hbllm.hcir.world.dorsolateral_rule_abstraction import (
    AbstractRuleSchema,
    RuleTransferResult,
)
from hbllm.hcir.world.engine_state import (
    CognitiveContext,
    EngineState,
)
from hbllm.hcir.world.entorhinal_border_cell import (
    BorderCell,
    GridModuleRescalingResult,
)
from hbllm.hcir.world.frontopolar_exploration_horizon import (
    BranchSwitchDecision,
    CounterfactualBranch,
)
from hbllm.hcir.world.frontopolar_subgoal_tree import (
    RecursiveSubgoalTree,
    SubgoalNodeType,
    SubgoalTreeNode,
    SubgoalTreeStatus,
)
from hbllm.hcir.world.global_workspace_ignition import (
    CandidateIgnitionSignal,
    ConsciousIgnitionPacket,
    GlobalWorkspaceState,
    IgnitionPriority,
)
from hbllm.hcir.world.habenula_striatum_opponency import (
    OpponencyRegime,
    OpponencyValueState,
)
from hbllm.hcir.world.hippocampal_phase_precession import (
    PhasePrecessionBurst,
    ThetaPhase,
)
from hbllm.hcir.world.hippocampal_replay import (
    ReplayBatch,
    ReplayTransition,
)
from hbllm.hcir.world.hippocampal_time_cell import (
    TemporalTimeCell,
    TimeIntervalState,
)
from hbllm.hcir.world.inferotemporal_shape_hasher import (
    ShapeArchetypeSignature,
)
from hbllm.hcir.world.insula_allostasis import (
    AllostaticRegime,
    InteroceptiveState,
)
from hbllm.hcir.world.lateral_prefrontal_switcher import (
    CognitiveTaskSet,
    TaskArbitrationResult,
    TaskSwitchEvent,
)
from hbllm.hcir.world.locus_coeruleus_noradrenaline import (
    ExplorationRegime,
    NoradrenergicState,
)
from hbllm.hcir.world.medial_prefrontal_schema import (
    GeneralizedTaskSchema,
    SchemaGeneralizationReport,
    SchemaMatchResult,
)
from hbllm.hcir.world.object_state_graph import (
    ObjectCategory,
    ObjectStateGraphPlanner,
    RelationalGraphMatcher,
    SalientObject,
)
from hbllm.hcir.world.orbitofrontal_credit_assignment import (
    CounterfactualCreditReport,
    InferredBranchValue,
)
from hbllm.hcir.world.parahippocampal_context import (
    ContextTransitionEvent,
    SpatialContextSchema,
)
from hbllm.hcir.world.parietal_foveated_attention import (
    FoveatedSaliencyMap,
    FoveationResolution,
)
from hbllm.hcir.world.parietal_potential_field import (
    PotentialFieldVector,
)
from hbllm.hcir.world.pedunculopontine_locomotor import (
    LocomotorDriveCommand,
    LocomotorPacingRegime,
)
from hbllm.hcir.world.percept import (
    BasePerceptionAdapter,
    DefaultPerceptionAdapter,
    EntityRole,
    PerceptionFrame,
    PerceptionRegistry,
    SpatialEntity,
)
from hbllm.hcir.world.prediction_types import (
    EnsemblePrediction,
    PredictionProvenance,
)
from hbllm.hcir.world.predictive_reality import (
    PredictiveRealityModel,
)
from hbllm.hcir.world.predictor_weights import (
    PredictorWeightManager,
    PredictorWeightState,
)
from hbllm.hcir.world.prefrontal_blame_attribution import (
    PlanAttributionReport,
    SubgoalBlameScore,
)
from hbllm.hcir.world.prefrontal_working_memory import (
    PrefrontalWorkingMemory,
    WorkingMemorySalienceState,
)
from hbllm.hcir.world.pulvinar_attention_router import (
    PulvinarRoutingState,
    TransthalamicRoute,
)
from hbllm.hcir.world.retrosplenial_heading import (
    AllocentricHeadingVector,
    HeadDirectionAttractorState,
)
from hbllm.hcir.world.retrosplenial_landmark_anchor import (
    AllocentricLandmark,
    LandmarkRecalibrationResult,
)
from hbllm.hcir.world.somatosensory_haptic import (
    HapticAffordanceFeedback,
    SurfaceFrictionProfile,
)
from hbllm.hcir.world.spatial_distance_transform import (
    EuclideanDistanceTransformResult,
)
from hbllm.hcir.world.subicular_boundary_vector import (
    BoundaryVectorCell,
    BoundaryVectorMap,
)
from hbllm.hcir.world.subthalamic_drift_diffusion import (
    DriftDiffusionDecision,
)
from hbllm.hcir.world.subthalamic_threshold_escalator import (
    STNBrakingStatus,
    ThresholdModulationDecision,
)
from hbllm.hcir.world.superior_colliculus_arbiter import (
    MultisensoryPriorityMap,
    SaccadeTarget,
)
from hbllm.hcir.world.surprise_engine import (
    SurpriseEngine,
    SurpriseEvaluation,
)
from hbllm.hcir.world.thalamic_reticular_gating import (
    SensoryShieldingState,
    TRNGateChannel,
)
from hbllm.hcir.world.ventral_premotor_resonator import (
    AffordanceResonancePattern,
    VicariousCausalInference,
)
from hbllm.hcir.world.ventral_tegmental_curiosity import (
    CuriosityDriveSignal,
    EpistemicInformationGain,
)
from hbllm.hcir.world.ventrolateral_motor_inhibition import (
    InhibitionRegime,
    ResponseInhibitionCommand,
)
from hbllm.hcir.world.ventromedial_somatic_marker import (
    SomaticAffectiveBias,
    SomaticMarker,
)
from hbllm.hcir.world.visuospatial_working_memory import (
    VisuospatialWorkingMemory,
    WorkingMemorySlot,
)
from hbllm.hcir.world.world_context import (
    BranchMode,
    WorldContext,
    WorldModelScope,
)
from hbllm.hcir.world.world_state_snapshot import (
    ReversibilityEngine,
    StateDifferenceComputer,
    TemporalStateBuffer,
    WorldStateSnapshot,
)
from hbllm.hcir.world.world_transaction import (
    WorldKernelTransaction,
)

__all__ = [
    "AbstractRuleSchema",
    "AffordanceResonancePattern",
    "AllocentricHeadingVector",
    "AllocentricLandmark",
    "AllostaticRegime",
    "AutonomousEpistemicEngine",
    "BaseCausalDiscoveryEngine",
    "BasePerceptionAdapter",
    "BeliefTransitionEvent",
    "BorderCell",
    "BoundaryPhaseAlignmentState",
    "BoundaryVectorCell",
    "BoundaryVectorMap",
    "BranchMode",
    "BranchSwitchDecision",
    "CandidateIgnitionSignal",
    "CausalHypothesis",
    "CausalPredicate",
    "CausalVariable",
    "ClimbingFiberFeedback",
    "CognitiveContext",
    "CognitiveTaskSet",
    "ConditionedThreatStimulus",
    "ConflictProfile",
    "ConsciousIgnitionPacket",
    "ContextTransitionEvent",
    "CorticalPipeline",
    "CorticalStage",
    "CounterfactualBranch",
    "CounterfactualCreditReport",
    "CounterfactualOutcome",
    "CrossModalSynchronyPacket",
    "CuriosityDriveSignal",
    "DefaultPerceptionAdapter",
    "DefensiveVigilanceMode",
    "DefensiveVigilanceState",
    "DisagreementReport",
    "DriftDiffusionDecision",
    "EffortAllocationDecision",
    "EngineState",
    "EnsemblePrediction",
    "EntityRole",
    "EpistemicFeedbackAssimilator",
    "EpistemicInformationGain",
    "EpistemicObservation",
    "EpistemicPhase",
    "EpistemicTruthValue",
    "ExplorationRegime",
    "EuclideanDistanceTransformResult",
    "FoveatedSaliencyMap",
    "FoveationResolution",
    "GeneralizedTaskSchema",
    "GlobalWorkspaceState",
    "GridModuleRescalingResult",
    "HapticAffordanceFeedback",
    "HCIRSymbolicWorldTheory",
    "HeadDirectionAttractorState",
    "IgnitionPriority",
    "InferredBranchValue",
    "InhibitionRegime",
    "InterHemisphericBindingState",
    "InteroceptivePacingState",
    "InteroceptiveState",
    "LandmarkRecalibrationResult",
    "LocomotorDriveCommand",
    "LocomotorPacingRegime",
    "MotorPrediction",
    "MultisensoryPriorityMap",
    "NoradrenergicState",
    "ObjectCategory",
    "ObjectStateGraphPlanner",
    "OpponencyRegime",
    "OpponencyValueState",
    "PacingMode",
    "PerceptionFrame",
    "PerceptionRegistry",
    "PhaseInnovationVector",
    "PipelineResult",
    "PhasePrecessionBurst",
    "PlanAttributionReport",
    "PotentialFieldVector",
    "PredictiveRealityModel",
    "PredictorDisagreementAnalyzer",
    "PredictorWeightManager",
    "PredictorWeightState",
    "PredictionProvenance",
    "PrefrontalWorkingMemory",
    "PulvinarRoutingState",
    "RecursiveSubgoalTree",
    "RelationalGraphMatcher",
    "ReplayBatch",
    "ReplayTransition",
    "ResponseInhibitionCommand",
    "ReversibilityEngine",
    "RuleTransferResult",
    "STNBrakingStatus",
    "SaccadeTarget",
    "SalientObject",
    "SchemaGeneralizationReport",
    "SchemaMatchResult",
    "SensoryShieldingState",
    "ShapeArchetypeSignature",
    "SomaticAffectiveBias",
    "SomaticMarker",
    "SpatialContextSchema",
    "SpatialEntity",
    "StageResult",
    "StateDifferenceComputer",
    "StructuralCausalModel",
    "SubgoalBlameScore",
    "SubgoalNodeType",
    "SubgoalTreeNode",
    "SubgoalTreeStatus",
    "SurfaceFrictionProfile",
    "SurpriseEngine",
    "SurpriseEvaluation",
    "TaskArbitrationResult",
    "TaskSwitchEvent",
    "TemporalStateBuffer",
    "TemporalTimeCell",
    "ThetaPhase",
    "ThresholdModulationDecision",
    "TimeIntervalState",
    "TRNGateChannel",
    "TransthalamicRoute",
    "UnifiedPerceptualGestalt",
    "VicariousCausalInference",
    "VisuospatialWorkingMemory",
    "WorkingMemorySalienceState",
    "WorkingMemorySlot",
    "WorldContext",
    "WorldKernelTransaction",
    "WorldModelScope",
    "WorldStateSnapshot",
]
