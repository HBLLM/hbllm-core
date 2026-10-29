"""HCIR World — Embodied, Spatial, and Epistemic World Modeling Subsystem."""

from __future__ import annotations

from hbllm.hcir.world.anterior_cingulate_effort_allocator import (
    AnteriorCingulateEffortAllocator,
    ConflictProfile,
    EffortAllocationDecision,
)
from hbllm.hcir.world.autonomous_epistemic_engine import (
    AutonomousEpistemicEngine,
    EpistemicObservation,
    EpistemicTruthValue,
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
    CerebellarMicrozoneAdaptor,
    ClimbingFiberFeedback,
    MotorPrediction,
)
from hbllm.hcir.world.claustrum_salience_conductor import (
    ClaustrumSalienceConductor,
    UnifiedPerceptualGestalt,
)
from hbllm.hcir.world.cortex_assimilator import (
    EpistemicFeedbackAssimilator,
    EpistemicPhase,
)
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.disagreement_analyzer import (
    DisagreementReport,
    PredictorDisagreementAnalyzer,
)
from hbllm.hcir.world.frontopolar_exploration_horizon import (
    BranchSwitchDecision,
    CounterfactualBranch,
    FrontopolarExplorationHorizonEngine,
)
from hbllm.hcir.world.frontopolar_subgoal_tree import (
    RecursiveSubgoalTree,
    SubgoalNodeType,
    SubgoalTreeNode,
    SubgoalTreeStatus,
)
from hbllm.hcir.world.habenula_striatum_opponency import (
    DualValueOpponencyCortex,
    OpponencyRegime,
    OpponencyValueState,
)
from hbllm.hcir.world.hippocampal_phase_precession import (
    HippocampalPhasePrecessionEngine,
    PhasePrecessionBurst,
    ThetaPhase,
)
from hbllm.hcir.world.hippocampal_replay import (
    HippocampalPrioritizedReplayBuffer,
    ReplayBatch,
    ReplayTransition,
)
from hbllm.hcir.world.inferotemporal_shape_hasher import (
    InferotemporalShapeHasher,
    ShapeArchetypeSignature,
)
from hbllm.hcir.world.insula_allostasis import (
    AllostaticRegime,
    InteroceptiveAllostasisCortex,
    InteroceptiveState,
)
from hbllm.hcir.world.lateral_prefrontal_switcher import (
    CognitiveTaskSet,
    LateralPrefrontalTaskSwitcher,
    TaskArbitrationResult,
    TaskSwitchEvent,
)
from hbllm.hcir.world.object_state_graph import (
    ObjectCategory,
    ObjectStateGraphPlanner,
    RelationalGraphMatcher,
    SalientObject,
)
from hbllm.hcir.world.parietal_foveated_attention import (
    FoveatedSaliencyMap,
    FoveationResolution,
    ParietalFoveatedAttentionEngine,
)
from hbllm.hcir.world.parietal_potential_field import (
    ParietalPotentialFieldNavigator,
    PotentialFieldVector,
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
    PrefrontalBlameAttributor,
    SubgoalBlameScore,
)
from hbllm.hcir.world.prefrontal_working_memory import (
    PrefrontalWorkingMemory,
    WorkingMemorySalienceState,
)
from hbllm.hcir.world.pulvinar_attention_router import (
    PulvinarAttentionRouter,
    PulvinarRoutingState,
    TransthalamicRoute,
)
from hbllm.hcir.world.retrosplenial_landmark_anchor import (
    AllocentricLandmark,
    LandmarkRecalibrationResult,
    RetrosplenialLandmarkAnchorEngine,
)
from hbllm.hcir.world.spatial_distance_transform import (
    EuclideanDistanceTransformResult,
    SpatialDistanceTransformEngine,
)
from hbllm.hcir.world.subicular_boundary_vector import (
    BoundaryVectorCell,
    BoundaryVectorMap,
    SubicularBoundaryVectorEngine,
)
from hbllm.hcir.world.subthalamic_drift_diffusion import (
    DriftDiffusionDecision,
    SubthalamicDriftDiffusionEngine,
)
from hbllm.hcir.world.superior_colliculus_arbiter import (
    MultisensoryPriorityMap,
    SaccadeTarget,
    SuperiorColliculusTargetArbiter,
)
from hbllm.hcir.world.surprise_engine import (
    SurpriseEngine,
    SurpriseEvaluation,
)
from hbllm.hcir.world.ventral_premotor_resonator import (
    AffordanceResonancePattern,
    VentralPremotorAffordanceResonator,
    VicariousCausalInference,
)
from hbllm.hcir.world.ventromedial_somatic_marker import (
    SomaticAffectiveBias,
    SomaticMarker,
    VentromedialSomaticMarkerEngine,
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
    "AffordanceResonancePattern",
    "AllocentricLandmark",
    "AllostaticRegime",
    "AnteriorCingulateEffortAllocator",
    "AutonomousEpistemicEngine",
    "BaseCausalDiscoveryEngine",
    "BasePerceptionAdapter",
    "BeliefTransitionEvent",
    "BoundaryVectorCell",
    "BoundaryVectorMap",
    "BranchMode",
    "BranchSwitchDecision",
    "CausalHypothesis",
    "CausalPredicate",
    "CausalVariable",
    "CerebellarMicrozoneAdaptor",
    "ClaustrumSalienceConductor",
    "ClimbingFiberFeedback",
    "CognitiveTaskSet",
    "ConflictProfile",
    "CounterfactualBranch",
    "CounterfactualOutcome",
    "DefaultPerceptionAdapter",
    "DisagreementReport",
    "DriftDiffusionDecision",
    "DualValueOpponencyCortex",
    "EffortAllocationDecision",
    "EnsemblePrediction",
    "EntityRole",
    "EpistemicFeedbackAssimilator",
    "EpistemicObservation",
    "EpistemicPhase",
    "EpistemicTruthValue",
    "EuclideanDistanceTransformResult",
    "FoveatedSaliencyMap",
    "FoveationResolution",
    "FrontopolarExplorationHorizonEngine",
    "HCIRSymbolicWorldTheory",
    "HippocampalPhasePrecessionEngine",
    "HippocampalPrioritizedReplayBuffer",
    "InferotemporalShapeHasher",
    "InteroceptiveAllostasisCortex",
    "InteroceptiveState",
    "LandmarkRecalibrationResult",
    "LateralPrefrontalTaskSwitcher",
    "MotorPrediction",
    "MultisensoryPriorityMap",
    "ObjectCategory",
    "ObjectStateGraphPlanner",
    "OpponencyRegime",
    "OpponencyValueState",
    "ParietalFoveatedAttentionEngine",
    "ParietalPotentialFieldNavigator",
    "PerceptionFrame",
    "PerceptionRegistry",
    "PhasePrecessionBurst",
    "PlanAttributionReport",
    "PotentialFieldVector",
    "PredictiveRealityModel",
    "PredictorDisagreementAnalyzer",
    "PredictorWeightManager",
    "PredictorWeightState",
    "PredictionProvenance",
    "PrefrontalBlameAttributor",
    "PrefrontalWorkingMemory",
    "PulvinarAttentionRouter",
    "PulvinarRoutingState",
    "RecursiveSubgoalTree",
    "RelationalGraphMatcher",
    "ReplayBatch",
    "ReplayTransition",
    "RetrosplenialLandmarkAnchorEngine",
    "ReversibilityEngine",
    "SaccadeTarget",
    "SalientObject",
    "ShapeArchetypeSignature",
    "SomaticAffectiveBias",
    "SomaticMarker",
    "SpatialDistanceTransformEngine",
    "SpatialEntity",
    "StateDifferenceComputer",
    "StructuralCausalModel",
    "SubgoalBlameScore",
    "SubgoalNodeType",
    "SubgoalTreeNode",
    "SubgoalTreeStatus",
    "SubicularBoundaryVectorEngine",
    "SubthalamicDriftDiffusionEngine",
    "SuperiorColliculusTargetArbiter",
    "SurpriseEngine",
    "SurpriseEvaluation",
    "TaskArbitrationResult",
    "TaskSwitchEvent",
    "TemporalStateBuffer",
    "ThetaPhase",
    "TransthalamicRoute",
    "UnifiedPerceptualGestalt",
    "VentralPremotorAffordanceResonator",
    "VentromedialSomaticMarkerEngine",
    "VicariousCausalInference",
    "VisuospatialWorkingMemory",
    "WorkingMemorySalienceState",
    "WorkingMemorySlot",
    "WorldContext",
    "WorldKernelTransaction",
    "WorldModelScope",
    "WorldStateSnapshot",
]
