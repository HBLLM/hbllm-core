"""Scientific Comparison Experiment Package for HBLLM.

Provides experimental harness, cohorts (HBLLM-Core, HBLLM+LLM, LLM-Only, Ablations),
standardized tasks (E1-E7), metrics calculator, independent oracle, leakage auditor,
and master report generators.
"""

from hbllm.experiment.ast_audit import ASTLeakageAuditor, ASTViolation
from hbllm.experiment.cohorts import (
    AblatedHBLLMCohort,
    BaseCohort,
    CohortOutput,
    HBLLMCoreCohort,
    HBLLMPlusLLMCohort,
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
from hbllm.experiment.statistics import ExperimentStatistics, MetricSummary
from hbllm.experiment.tasks import (
    E1_ConceptAcquisitionTask,
    E2_LexicalAcquisitionTask,
    E3_CounterfactualSimulationTask,
    E4_EpistemicCalibrationTask,
    E5_ActiveEpistemicDiscoveryTask,
    E6_RelationalTransferTask,
    E7_LifelongCurriculumTask,
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
    "EnvironmentObservation",
    "ExperimentMetricsCalculator",
    "ExperimentRunner",
    "ExperimentStatistics",
    "HBLLMCoreCohort",
    "HBLLMPlusLLMCohort",
    "IndependentEnvironmentOracle",
    "LLMOnlyCohort",
    "LeakageAuditReport",
    "LeakageAuditor",
    "MetricSummary",
    "PhysicalEnvironmentState",
    "ReproducibilityManifest",
    "ScientificExperimentReport",
    "TaskEvaluationResult",
]
