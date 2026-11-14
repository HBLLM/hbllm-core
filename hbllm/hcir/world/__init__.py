"""HCIR World — Embodied, Spatial, and Epistemic World Modeling Subsystem."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.hcir.world.experimental.amygdalar_threat_conditioning import (
        ConditionedThreatStimulus,
        DefensiveVigilanceMode,
        DefensiveVigilanceState,
    )
    from hbllm.hcir.world.experimental.anterior_cingulate_effort_allocator import (
        ConflictProfile,
        EffortAllocationDecision,
    )
    from hbllm.hcir.world.experimental.anterior_insular_timing import (
        InteroceptivePacingState,
        PacingMode,
    )
    from hbllm.hcir.world.experimental.boundary_grid_phase_aligner import (
        BoundaryPhaseAlignmentState,
        PhaseInnovationVector,
    )
    from hbllm.hcir.world.experimental.cerebellar_microzone_adaptor import (
        ClimbingFiberFeedback,
        MotorPrediction,
    )
    from hbllm.hcir.world.experimental.claustrum_interhemispheric import (
        CrossModalSynchronyPacket,
        InterHemisphericBindingState,
    )
    from hbllm.hcir.world.experimental.dorsolateral_rule_abstraction import (
        AbstractRuleSchema,
        RuleTransferResult,
    )
    from hbllm.hcir.world.experimental.entorhinal_border_cell import (
        BorderCell,
        GridModuleRescalingResult,
    )
    from hbllm.hcir.world.experimental.frontopolar_exploration_horizon import (
        BranchSwitchDecision,
        CounterfactualBranch,
    )
    from hbllm.hcir.world.experimental.global_workspace_ignition import (
        CandidateIgnitionSignal,
        ConsciousIgnitionPacket,
        GlobalWorkspaceState,
        IgnitionPriority,
    )
    from hbllm.hcir.world.experimental.hippocampal_phase_precession import (
        PhasePrecessionBurst,
        ThetaPhase,
    )
    from hbllm.hcir.world.experimental.hippocampal_replay import ReplayBatch, ReplayTransition
    from hbllm.hcir.world.experimental.hippocampal_time_cell import (
        TemporalTimeCell,
        TimeIntervalState,
    )
    from hbllm.hcir.world.experimental.lateral_prefrontal_switcher import (
        CognitiveTaskSet,
        TaskArbitrationResult,
        TaskSwitchEvent,
    )
    from hbllm.hcir.world.experimental.locus_coeruleus_noradrenaline import (
        ExplorationRegime,
        NoradrenergicState,
    )
    from hbllm.hcir.world.experimental.medial_prefrontal_schema import (
        GeneralizedTaskSchema,
        SchemaGeneralizationReport,
        SchemaMatchResult,
    )
    from hbllm.hcir.world.experimental.orbitofrontal_credit_assignment import (
        CounterfactualCreditReport,
        InferredBranchValue,
    )
    from hbllm.hcir.world.experimental.parahippocampal_context import (
        ContextTransitionEvent,
        SpatialContextSchema,
    )
    from hbllm.hcir.world.experimental.pedunculopontine_locomotor import (
        LocomotorDriveCommand,
        LocomotorPacingRegime,
    )
    from hbllm.hcir.world.experimental.prefrontal_blame_attribution import (
        PlanAttributionReport,
        SubgoalBlameScore,
    )
    from hbllm.hcir.world.experimental.pulvinar_attention_router import (
        PulvinarRoutingState,
        TransthalamicRoute,
    )
    from hbllm.hcir.world.experimental.retrosplenial_heading import (
        AllocentricHeadingVector,
        HeadDirectionAttractorState,
    )
    from hbllm.hcir.world.experimental.retrosplenial_landmark_anchor import (
        AllocentricLandmark,
        LandmarkRecalibrationResult,
    )
    from hbllm.hcir.world.experimental.somatosensory_haptic import (
        HapticAffordanceFeedback,
        SurfaceFrictionProfile,
    )
    from hbllm.hcir.world.experimental.spatial_distance_transform import (
        EuclideanDistanceTransformResult,
    )
    from hbllm.hcir.world.experimental.subicular_boundary_vector import (
        BoundaryVectorCell,
        BoundaryVectorMap,
    )
    from hbllm.hcir.world.experimental.subthalamic_threshold_escalator import (
        STNBrakingStatus,
        ThresholdModulationDecision,
    )
    from hbllm.hcir.world.experimental.superior_colliculus_arbiter import (
        MultisensoryPriorityMap,
        SaccadeTarget,
    )
    from hbllm.hcir.world.experimental.thalamic_reticular_gating import (
        SensoryShieldingState,
        TRNGateChannel,
    )
    from hbllm.hcir.world.experimental.ventral_premotor_resonator import (
        AffordanceResonancePattern,
        VicariousCausalInference,
    )
    from hbllm.hcir.world.experimental.ventral_tegmental_curiosity import (
        CuriosityDriveSignal,
        EpistemicInformationGain,
    )
    from hbllm.hcir.world.experimental.ventrolateral_motor_inhibition import (
        InhibitionRegime,
        ResponseInhibitionCommand,
    )
    from hbllm.hcir.world.experimental.ventromedial_somatic_marker import (
        SomaticAffectiveBias,
        SomaticMarker,
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
from hbllm.hcir.world.causal_scm import CausalVariable, CounterfactualOutcome, StructuralCausalModel
from hbllm.hcir.world.cortex_assimilator import EpistemicFeedbackAssimilator, EpistemicPhase
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortical_pipeline import CorticalPipeline, PipelineResult
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.disagreement_analyzer import DisagreementReport, PredictorDisagreementAnalyzer
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState
from hbllm.hcir.world.frontopolar_subgoal_tree import (
    RecursiveSubgoalTree,
    SubgoalNodeType,
    SubgoalTreeNode,
    SubgoalTreeStatus,
)
from hbllm.hcir.world.object_state_graph import (
    ObjectCategory,
    ObjectStateGraphPlanner,
    RelationalGraphMatcher,
    SalientObject,
)
from hbllm.hcir.world.percept import (
    BasePerceptionAdapter,
    DefaultPerceptionAdapter,
    EntityRole,
    PerceptionFrame,
    PerceptionRegistry,
    SpatialEntity,
)
from hbllm.hcir.world.prediction_types import EnsemblePrediction, PredictionProvenance
from hbllm.hcir.world.predictive_reality import PredictiveRealityModel
from hbllm.hcir.world.predictor_weights import PredictorWeightManager, PredictorWeightState
from hbllm.hcir.world.prefrontal_working_memory import (
    PrefrontalWorkingMemory,
    WorkingMemorySalienceState,
)
from hbllm.hcir.world.surprise_engine import SurpriseEngine, SurpriseEvaluation
from hbllm.hcir.world.visuospatial_working_memory import (
    VisuospatialWorkingMemory,
    WorkingMemorySlot,
)
from hbllm.hcir.world.world_context import BranchMode, WorldContext, WorldModelScope
from hbllm.hcir.world.world_state_snapshot import (
    ReversibilityEngine,
    StateDifferenceComputer,
    TemporalStateBuffer,
    WorldStateSnapshot,
)
from hbllm.hcir.world.world_transaction import WorldKernelTransaction

_LAZY_EXPORTS: dict[str, str] = {
    "AbstractRuleSchema": "hbllm.hcir.world.dorsolateral_rule_abstraction",
    "AffordanceResonancePattern": "hbllm.hcir.world.ventral_premotor_resonator",
    "AllocentricHeadingVector": "hbllm.hcir.world.retrosplenial_heading",
    "AllocentricLandmark": "hbllm.hcir.world.retrosplenial_landmark_anchor",
    "BorderCell": "hbllm.hcir.world.entorhinal_border_cell",
    "BoundaryPhaseAlignmentState": "hbllm.hcir.world.boundary_grid_phase_aligner",
    "BoundaryVectorCell": "hbllm.hcir.world.subicular_boundary_vector",
    "BoundaryVectorMap": "hbllm.hcir.world.subicular_boundary_vector",
    "BranchSwitchDecision": "hbllm.hcir.world.frontopolar_exploration_horizon",
    "CandidateIgnitionSignal": "hbllm.hcir.world.global_workspace_ignition",
    "ClimbingFiberFeedback": "hbllm.hcir.world.cerebellar_microzone_adaptor",
    "CognitiveTaskSet": "hbllm.hcir.world.lateral_prefrontal_switcher",
    "ConditionedThreatStimulus": "hbllm.hcir.world.amygdalar_threat_conditioning",
    "ConflictProfile": "hbllm.hcir.world.anterior_cingulate_effort_allocator",
    "ConsciousIgnitionPacket": "hbllm.hcir.world.global_workspace_ignition",
    "ContextTransitionEvent": "hbllm.hcir.world.parahippocampal_context",
    "CounterfactualBranch": "hbllm.hcir.world.frontopolar_exploration_horizon",
    "CounterfactualCreditReport": "hbllm.hcir.world.orbitofrontal_credit_assignment",
    "CrossModalSynchronyPacket": "hbllm.hcir.world.claustrum_interhemispheric",
    "CuriosityDriveSignal": "hbllm.hcir.world.ventral_tegmental_curiosity",
    "DefensiveVigilanceMode": "hbllm.hcir.world.amygdalar_threat_conditioning",
    "DefensiveVigilanceState": "hbllm.hcir.world.amygdalar_threat_conditioning",
    "EffortAllocationDecision": "hbllm.hcir.world.anterior_cingulate_effort_allocator",
    "EpistemicInformationGain": "hbllm.hcir.world.ventral_tegmental_curiosity",
    "EuclideanDistanceTransformResult": "hbllm.hcir.world.spatial_distance_transform",
    "ExplorationRegime": "hbllm.hcir.world.locus_coeruleus_noradrenaline",
    "GeneralizedTaskSchema": "hbllm.hcir.world.medial_prefrontal_schema",
    "GlobalWorkspaceState": "hbllm.hcir.world.global_workspace_ignition",
    "GridModuleRescalingResult": "hbllm.hcir.world.entorhinal_border_cell",
    "HapticAffordanceFeedback": "hbllm.hcir.world.somatosensory_haptic",
    "HeadDirectionAttractorState": "hbllm.hcir.world.retrosplenial_heading",
    "IgnitionPriority": "hbllm.hcir.world.global_workspace_ignition",
    "InferredBranchValue": "hbllm.hcir.world.orbitofrontal_credit_assignment",
    "InhibitionRegime": "hbllm.hcir.world.ventrolateral_motor_inhibition",
    "InterHemisphericBindingState": "hbllm.hcir.world.claustrum_interhemispheric",
    "InteroceptivePacingState": "hbllm.hcir.world.anterior_insular_timing",
    "LandmarkRecalibrationResult": "hbllm.hcir.world.retrosplenial_landmark_anchor",
    "LearnedProposalDistribution": "hbllm.hcir.world.library_learning",
    "LibraryLearner": "hbllm.hcir.world.library_learning",
    "LocomotorDriveCommand": "hbllm.hcir.world.pedunculopontine_locomotor",
    "LocomotorPacingRegime": "hbllm.hcir.world.pedunculopontine_locomotor",
    "MotorPrediction": "hbllm.hcir.world.cerebellar_microzone_adaptor",
    "MultisensoryPriorityMap": "hbllm.hcir.world.superior_colliculus_arbiter",
    "NoradrenergicState": "hbllm.hcir.world.locus_coeruleus_noradrenaline",
    "PacingMode": "hbllm.hcir.world.anterior_insular_timing",
    "PhaseInnovationVector": "hbllm.hcir.world.boundary_grid_phase_aligner",
    "PhasePrecessionBurst": "hbllm.hcir.world.hippocampal_phase_precession",
    "PlanAttributionReport": "hbllm.hcir.world.prefrontal_blame_attribution",
    "PulvinarRoutingState": "hbllm.hcir.world.pulvinar_attention_router",
    "ReplayBatch": "hbllm.hcir.world.hippocampal_replay",
    "ReplayTransition": "hbllm.hcir.world.hippocampal_replay",
    "RecursiveValueInducer": "hbllm.hcir.world.recursive_value_induction",
    "ResponseInhibitionCommand": "hbllm.hcir.world.ventrolateral_motor_inhibition",
    "RuleTransferResult": "hbllm.hcir.world.dorsolateral_rule_abstraction",
    "STNBrakingStatus": "hbllm.hcir.world.subthalamic_threshold_escalator",
    "SaccadeTarget": "hbllm.hcir.world.superior_colliculus_arbiter",
    "SchemaGeneralizationReport": "hbllm.hcir.world.medial_prefrontal_schema",
    "SchemaMatchResult": "hbllm.hcir.world.medial_prefrontal_schema",
    "SensoryShieldingState": "hbllm.hcir.world.thalamic_reticular_gating",
    "SomaticAffectiveBias": "hbllm.hcir.world.ventromedial_somatic_marker",
    "SomaticMarker": "hbllm.hcir.world.ventromedial_somatic_marker",
    "SpatialContextSchema": "hbllm.hcir.world.parahippocampal_context",
    "SubgoalBlameScore": "hbllm.hcir.world.prefrontal_blame_attribution",
    "SurfaceFrictionProfile": "hbllm.hcir.world.somatosensory_haptic",
    "TRNGateChannel": "hbllm.hcir.world.thalamic_reticular_gating",
    "TaskArbitrationResult": "hbllm.hcir.world.lateral_prefrontal_switcher",
    "TaskSwitchEvent": "hbllm.hcir.world.lateral_prefrontal_switcher",
    "TemporalTimeCell": "hbllm.hcir.world.hippocampal_time_cell",
    "ThetaPhase": "hbllm.hcir.world.hippocampal_phase_precession",
    "ThresholdModulationDecision": "hbllm.hcir.world.subthalamic_threshold_escalator",
    "TimeIntervalState": "hbllm.hcir.world.hippocampal_time_cell",
    "TransthalamicRoute": "hbllm.hcir.world.pulvinar_attention_router",
    "VicariousCausalInference": "hbllm.hcir.world.ventral_premotor_resonator",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_path = _LAZY_EXPORTS[name]
        try:
            mod = importlib.import_module(mod_path)
        except ModuleNotFoundError:
            if not mod_path.startswith("hbllm.hcir.world.experimental."):
                exp_path = mod_path.replace("hbllm.hcir.world.", "hbllm.hcir.world.experimental.")
                mod = importlib.import_module(exp_path)
            else:
                raise
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = [
    "AbstractRuleSchema",
    "AffordanceResonancePattern",
    "AllocentricHeadingVector",
    "AllocentricLandmark",
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
    "LandmarkRecalibrationResult",
    "LearnedProposalDistribution",
    "LibraryLearner",
    "LocomotorDriveCommand",
    "LocomotorPacingRegime",
    "MotorPrediction",
    "MultisensoryPriorityMap",
    "NoradrenergicState",
    "ObjectCategory",
    "ObjectStateGraphPlanner",
    "PacingMode",
    "PerceptionFrame",
    "PerceptionRegistry",
    "PhaseInnovationVector",
    "PipelineResult",
    "PhasePrecessionBurst",
    "PlanAttributionReport",
    "PredictiveRealityModel",
    "PredictorDisagreementAnalyzer",
    "PredictorWeightManager",
    "PredictorWeightState",
    "PredictionProvenance",
    "PrefrontalWorkingMemory",
    "PulvinarRoutingState",
    "RecursiveSubgoalTree",
    "RecursiveValueInducer",
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
    "VicariousCausalInference",
    "VisuospatialWorkingMemory",
    "WorkingMemorySalienceState",
    "WorkingMemorySlot",
    "WorldContext",
    "WorldKernelTransaction",
    "WorldModelScope",
    "WorldStateSnapshot",
]
