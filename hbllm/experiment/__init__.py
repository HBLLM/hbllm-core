"""Scientific Comparison Experiment Package for HBLLM.

Provides experimental harness, cohorts (HBLLM-Core, HBLLM+LLM, LLM-Only, Ablations),
standardized tasks (E1-E7), metrics calculator, independent oracle, leakage auditor,
and master report generators.
"""

from __future__ import annotations

from pathlib import Path

# Extend package search path to include experiments/hbllm_experiment
_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "hbllm_experiment"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))

from hbllm.experiment.ast_audit import ASTLeakageAuditor, ASTViolation
from hbllm.experiment.cohorts import (
    AblatedHBLLMCohort,
    BaseCohort,
    CohortOutput,
    HBLLMCoreCohort,
    HBLLMPlusLLMCohort,
    HeuristicBaselineCohort,
    LLMOnlyCohort,
)
from hbllm.experiment.environments import (
    CanonicalTaskEnvironment,
    EnvironmentObservation,
    IndependentEnvironmentOracle,
    PhysicalEnvironmentState,
)
from hbllm.experiment.leakage_audit import LeakageAuditor, LeakageAuditReport
from hbllm.experiment.manifests import ReproducibilityManifest
from hbllm.experiment.metrics import ExperimentMetricsCalculator
from hbllm.experiment.reports import ScientificExperimentReport
from hbllm.experiment.runner import ExperimentRunner
from hbllm.experiment.statistics import ExperimentStatistics, MetricSummary, ProportionSummary
from hbllm.experiment.tasks import (
    E1_ConceptAcquisitionTask,
    E2_LexicalAcquisitionTask,
    E3_CounterfactualSimulationTask,
    E4_EpistemicCalibrationTask,
    E5_ActiveEpistemicDiscoveryTask,
    E6_RelationalTransferTask,
    E7_LifelongCurriculumTask,
    E8_CausalInterventionTask,
    TaskEvaluationResult,
)

__all__ = [
    "ASTLeakageAuditor",
    "ASTViolation",
    "AblatedHBLLMCohort",
    "BaseCohort",
    "CanonicalTaskEnvironment",
    "CohortOutput",
    "E1_ConceptAcquisitionTask",
    "E2_LexicalAcquisitionTask",
    "E3_CounterfactualSimulationTask",
    "E4_EpistemicCalibrationTask",
    "E5_ActiveEpistemicDiscoveryTask",
    "E6_RelationalTransferTask",
    "E7_LifelongCurriculumTask",
    "E8_CausalInterventionTask",
    "EnvironmentObservation",
    "ExperimentMetricsCalculator",
    "ExperimentRunner",
    "ExperimentStatistics",
    "HBLLMCoreCohort",
    "HBLLMPlusLLMCohort",
    "HeuristicBaselineCohort",
    "IndependentEnvironmentOracle",
    "LLMOnlyCohort",
    "LeakageAuditReport",
    "LeakageAuditor",
    "MetricSummary",
    "PhysicalEnvironmentState",
    "ProportionSummary",
    "ReproducibilityManifest",
    "ScientificExperimentReport",
    "TaskEvaluationResult",
]
