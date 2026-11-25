"""Scientific Comparison Experiment Package for HBLLM.

Provides experimental harness, cohorts (HBLLM-Core, HBLLM+LLM, LLM-Only, Ablations),
standardized tasks (E1-E7), metrics calculator, independent oracle, leakage auditor,
and master report generators.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ASTLeakageAuditor": ("hbllm.experiment.ast_audit", "ASTLeakageAuditor"),
    "ASTViolation": ("hbllm.experiment.ast_audit", "ASTViolation"),
    "AblatedHBLLMCohort": ("hbllm.experiment.cohorts", "AblatedHBLLMCohort"),
    "BaseCohort": ("hbllm.experiment.cohorts", "BaseCohort"),
    "CohortOutput": ("hbllm.experiment.cohorts", "CohortOutput"),
    "HBLLMCoreCohort": ("hbllm.experiment.cohorts", "HBLLMCoreCohort"),
    "HBLLMPlusLLMCohort": ("hbllm.experiment.cohorts", "HBLLMPlusLLMCohort"),
    "HeuristicBaselineCohort": ("hbllm.experiment.cohorts", "HeuristicBaselineCohort"),
    "LLMOnlyCohort": ("hbllm.experiment.cohorts", "LLMOnlyCohort"),
    "CanonicalTaskEnvironment": ("hbllm.experiment.environments", "CanonicalTaskEnvironment"),
    "EnvironmentObservation": ("hbllm.experiment.environments", "EnvironmentObservation"),
    "IndependentEnvironmentOracle": (
        "hbllm.experiment.environments",
        "IndependentEnvironmentOracle",
    ),
    "PhysicalEnvironmentState": ("hbllm.experiment.environments", "PhysicalEnvironmentState"),
    "LeakageAuditReport": ("hbllm.experiment.leakage_audit", "LeakageAuditReport"),
    "LeakageAuditor": ("hbllm.experiment.leakage_audit", "LeakageAuditor"),
    "ReproducibilityManifest": ("hbllm.experiment.manifests", "ReproducibilityManifest"),
    "ExperimentMetricsCalculator": ("hbllm.experiment.metrics", "ExperimentMetricsCalculator"),
    "ScientificExperimentReport": ("hbllm.experiment.reports", "ScientificExperimentReport"),
    "ExperimentRunner": ("hbllm.experiment.runner", "ExperimentRunner"),
    "ExperimentStatistics": ("hbllm.experiment.statistics", "ExperimentStatistics"),
    "MetricSummary": ("hbllm.experiment.statistics", "MetricSummary"),
    "ProportionSummary": ("hbllm.experiment.statistics", "ProportionSummary"),
    "E1_ConceptAcquisitionTask": ("hbllm.experiment.tasks", "E1_ConceptAcquisitionTask"),
    "E2_LexicalAcquisitionTask": ("hbllm.experiment.tasks", "E2_LexicalAcquisitionTask"),
    "E3_CounterfactualSimulationTask": (
        "hbllm.experiment.tasks",
        "E3_CounterfactualSimulationTask",
    ),
    "E4_EpistemicCalibrationTask": ("hbllm.experiment.tasks", "E4_EpistemicCalibrationTask"),
    "E5_ActiveEpistemicDiscoveryTask": (
        "hbllm.experiment.tasks",
        "E5_ActiveEpistemicDiscoveryTask",
    ),
    "E6_RelationalTransferTask": ("hbllm.experiment.tasks", "E6_RelationalTransferTask"),
    "E7_LifelongCurriculumTask": ("hbllm.experiment.tasks", "E7_LifelongCurriculumTask"),
    "E8_CausalInterventionTask": ("hbllm.experiment.tasks", "E8_CausalInterventionTask"),
    "TaskEvaluationResult": ("hbllm.experiment.tasks", "TaskEvaluationResult"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
