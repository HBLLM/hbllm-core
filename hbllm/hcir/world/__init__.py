"""HCIR World — Embodied, Spatial, and Epistemic World Modeling Subsystem."""

from __future__ import annotations

import importlib

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
from hbllm.hcir.world.cortex_assimilator import EpistemicFeedbackAssimilator, EpistemicPhase
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortical_pipeline import CorticalPipeline, PipelineResult
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.disagreement_analyzer import DisagreementReport, PredictorDisagreementAnalyzer
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState
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

__all__ = [
    "AutonomousEpistemicEngine",
    "BaseCausalDiscoveryEngine",
    "BasePerceptionAdapter",
    "BeliefTransitionEvent",
    "BranchMode",
    "CausalHypothesis",
    "CausalPredicate",
    "CognitiveContext",
    "CorticalPipeline",
    "CorticalStage",
    "DefaultPerceptionAdapter",
    "DisagreementReport",
    "EngineState",
    "EnsemblePrediction",
    "EntityRole",
    "EpistemicFeedbackAssimilator",
    "EpistemicObservation",
    "EpistemicPhase",
    "EpistemicTruthValue",
    "HCIRSymbolicWorldTheory",
    "ObjectCategory",
    "ObjectStateGraphPlanner",
    "PerceptionFrame",
    "PerceptionRegistry",
    "PipelineResult",
    "PredictionProvenance",
    "PredictiveRealityModel",
    "PredictorDisagreementAnalyzer",
    "PredictorWeightManager",
    "PredictorWeightState",
    "PrefrontalWorkingMemory",
    "RelationalGraphMatcher",
    "ReversibilityEngine",
    "SalientObject",
    "SpatialEntity",
    "StageResult",
    "StateDifferenceComputer",
    "SurpriseEngine",
    "SurpriseEvaluation",
    "TemporalStateBuffer",
    "VisuospatialWorkingMemory",
    "WorkingMemorySalienceState",
    "WorkingMemorySlot",
    "WorldContext",
    "WorldKernelTransaction",
    "WorldModelScope",
    "WorldStateSnapshot",
    "annotations",
]
