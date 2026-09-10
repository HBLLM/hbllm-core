"""
Developmental Learning Adapter Plugin for HBLLM Core (Milestone A23).

Bridges the BabyWorld simulator and Piagetian developmental cognitive curriculum
(D0–D16) to HCIR, featuring active interventional causal discovery under confounding.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import DevelopmentalActionAdapter
from .affordance_discovery import AffordanceDiscoveryEngine
from .benchmark import run_a23_5_benchmark
from .blank_brain import BlankBrainSubstrate, create_blank_brain_substrate
from .causal_discovery import InterventionalCausalDiscoveryEngine
from .cohorts import (
    ActiveDevelopmentalHCIRCohort,
    BaseDevelopmentalCohort,
    CohortDiscoveryResult,
    MatureHCIRCohort,
    NeuralLearnerCohort,
    PassiveDevelopmentalHCIRCohort,
    ScriptedCohort,
)
from .compositional_language import CompositionalLanguageEngine
from .concept_abstraction import ConceptAbstractionEngine
from .continual_development import ContinualDevelopmentEngine
from .cross_transfer import CrossTransferEngine
from .curiosity import EpistemicCuriosityEngine
from .curriculum import CURRICULUM_SPECS, CurriculumStageId, CurriculumStageSpec
from .environment import BabyWorldEnvironment
from .goal_planning import GoalDirectedPlanningEngine
from .language_grounding import LanguageGroundingEngine
from .metacognition import MetacognitiveEngine
from .metrics import DevelopmentalMetricsTracker
from .perception import (
    DevelopmentalPerceptionAdapter,
    SemanticLeakageViolationError,
)
from .spatial_containment import SpatialContainmentEngine
from .tool_learning import ToolLearningEngine
from .types import (
    AffordanceHypothesis,
    BabyActionType,
    BabyObjectState,
    BabyObjectType,
    BabyRelationType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    ConceptCluster,
    CrossTransferEvaluation,
    DevelopmentalProfile,
    EpistemicUncertaintyReport,
    LexicalCategory,
    LexicalEntry,
    MetacognitiveReport,
    PlanExecutionResult,
    PlanStep,
    PredicateGoal,
    SensoryObservation,
    SpatialRelationFact,
    Vector2D,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "developmental-adapter"
PLUGIN_VERSION = "1.0.0"


def setup(agent: Any = None) -> None:
    """Plugin setup hook."""
    logger.info("Loaded plugin '%s' v%s", PLUGIN_NAME, PLUGIN_VERSION)


def register(bus: Any = None, registry: Any = None) -> list[Any]:
    """Plugin registration hook conforming to HBLLM plugin manager."""
    logger.info("Registered capabilities for '%s'", PLUGIN_NAME)
    return [
        "developmental_observe",
        "developmental_act",
        "developmental_intervene",
        "developmental_discover_causality",
        "developmental_discover_affordances",
        "developmental_spatial_containment",
        "developmental_tool_learning",
        "developmental_goal_planning",
        "developmental_curiosity",
        "developmental_concept_abstraction",
        "developmental_language_grounding",
        "developmental_compositional_language",
        "developmental_continual_learning",
        "developmental_metacognition",
        "developmental_cross_transfer",
        "developmental_evaluate_curriculum",
    ]


__all__ = [
    "ActiveDevelopmentalHCIRCohort",
    "AffordanceDiscoveryEngine",
    "AffordanceHypothesis",
    "BabyActionType",
    "BabyObjectState",
    "BabyObjectType",
    "BabyRelationType",
    "BabyWorldEnvironment",
    "BaseDevelopmentalCohort",
    "BeliefTransitionEvent",
    "BeliefTransitionType",
    "BlankBrainSubstrate",
    "CURRICULUM_SPECS",
    "CausalHypothesis",
    "CohortDiscoveryResult",
    "CompositionalLanguageEngine",
    "ConceptAbstractionEngine",
    "ConceptCluster",
    "ContinualDevelopmentEngine",
    "CrossTransferEngine",
    "CrossTransferEvaluation",
    "CurriculumStageId",
    "CurriculumStageSpec",
    "DevelopmentalActionAdapter",
    "DevelopmentalMetricsTracker",
    "DevelopmentalPerceptionAdapter",
    "DevelopmentalProfile",
    "EpistemicCuriosityEngine",
    "EpistemicUncertaintyReport",
    "GoalDirectedPlanningEngine",
    "InterventionalCausalDiscoveryEngine",
    "LanguageGroundingEngine",
    "LexicalCategory",
    "LexicalEntry",
    "MatureHCIRCohort",
    "MetacognitiveEngine",
    "MetacognitiveReport",
    "NeuralLearnerCohort",
    "PassiveDevelopmentalHCIRCohort",
    "PlanExecutionResult",
    "PlanStep",
    "PredicateGoal",
    "ScriptedCohort",
    "SemanticLeakageViolationError",
    "SensoryObservation",
    "SpatialContainmentEngine",
    "SpatialRelationFact",
    "ToolLearningEngine",
    "Vector2D",
    "create_blank_brain_substrate",
    "register",
    "run_a23_5_benchmark",
    "setup",
]
