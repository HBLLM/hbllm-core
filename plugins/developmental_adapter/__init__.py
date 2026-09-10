"""
Developmental Learning Adapter Plugin for HBLLM Core (Milestone A23).

Bridges the BabyWorld simulator and Piagetian developmental cognitive curriculum
(D0–D16) to HCIR, featuring active interventional causal discovery under confounding.
"""

from __future__ import annotations

import logging
from typing import Any

from .action import DevelopmentalActionAdapter
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
from .curriculum import CURRICULUM_SPECS, CurriculumStageId, CurriculumStageSpec
from .environment import BabyWorldEnvironment
from .metrics import DevelopmentalMetricsTracker
from .perception import (
    DevelopmentalPerceptionAdapter,
    SemanticLeakageViolationError,
)
from .types import (
    BabyActionType,
    BabyObjectState,
    BabyObjectType,
    BabyRelationType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    CausalHypothesis,
    DevelopmentalProfile,
    SensoryObservation,
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
        "developmental_evaluate_curriculum",
    ]


__all__ = [
    "BabyActionType",
    "BabyObjectState",
    "BabyObjectType",
    "BabyRelationType",
    "BabyWorldEnvironment",
    "BeliefTransitionEvent",
    "BeliefTransitionType",
    "BlankBrainSubstrate",
    "CausalHypothesis",
    "CohortDiscoveryResult",
    "CURRICULUM_SPECS",
    "CurriculumStageId",
    "CurriculumStageSpec",
    "DevelopmentalActionAdapter",
    "DevelopmentalMetricsTracker",
    "DevelopmentalPerceptionAdapter",
    "DevelopmentalProfile",
    "InterventionalCausalDiscoveryEngine",
    "ActiveDevelopmentalHCIRCohort",
    "BaseDevelopmentalCohort",
    "MatureHCIRCohort",
    "NeuralLearnerCohort",
    "PassiveDevelopmentalHCIRCohort",
    "ScriptedCohort",
    "SemanticLeakageViolationError",
    "SensoryObservation",
    "Vector2D",
    "create_blank_brain_substrate",
    "run_a23_5_benchmark",
    "setup",
    "register",
]
