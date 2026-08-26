"""A21 Explicit Metacognitive Self-Model Package.

Provides Contextual Competence Profiles, Epistemic Calibration,
Cognitive Resource Budgeting, Metacognitive Monitoring, and Introspective Self-Correction.
"""

from hbllm.brain.self_model.budget import (
    BudgetDecision,
    CognitiveBudget,
    CognitiveBudgetManager,
)
from hbllm.brain.self_model.calibrator import (
    CalibrationBin,
    CalibrationReport,
    EpistemicCalibrator,
)
from hbllm.brain.self_model.metacognitive_self_model import (
    MetacognitiveSelfModel,
)
from hbllm.brain.self_model.monitor import (
    FailureCause,
    FailureDiagnosis,
    MetacognitiveEvent,
    MetacognitiveEventType,
    MetacognitiveMonitor,
    MetacognitiveState,
    StrategyAction,
)
from hbllm.brain.self_model.profile import (
    CompetenceProfile,
    EpistemicMaturity,
    SelfModelEvidence,
    UncertaintyBreakdown,
    UncertaintyType,
)

__all__ = [
    "BudgetDecision",
    "CalibrationBin",
    "CalibrationReport",
    "CognitiveBudget",
    "CognitiveBudgetManager",
    "CompetenceProfile",
    "EpistemicCalibrator",
    "EpistemicMaturity",
    "FailureCause",
    "FailureDiagnosis",
    "MetacognitiveEvent",
    "MetacognitiveEventType",
    "MetacognitiveMonitor",
    "MetacognitiveSelfModel",
    "MetacognitiveState",
    "SelfModelEvidence",
    "StrategyAction",
    "UncertaintyBreakdown",
    "UncertaintyType",
]
