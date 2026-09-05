"""
A12 — Reasoning Operators Package.

Composable cognitive operators that read immutable HCIR views
and propose state transitions.  No operator owns cognitive state.

Public API::

    from hbllm.brain.reasoning.operators import (
        # Core types
        ReasoningOperator,
        ReasoningProblem,
        CognitiveContext,
        CognitiveResult,
        ReasoningBudget,
        FrozenGraphView,
        OperatorTrace,
        OperatorSelectionScore,
        OperatorRegistry,
        ProblemType,
        ResultStatus,

        # New operators (Phase 7)
        DeductionOperator,
        InductionOperator,
        AbductionOperator,
        TemporalOperator,
        SpatialOperator,
        AnalogyOperator,

        # Wrapped existing operators (Phase 8)
        PredictionOperator,
        ContradictionOperator,
        CounterfactualOperator,
        CausalOperator,
        ActiveInferenceOperator,
        SimulationOperator,
        SNNReasoningOperator,
    )
"""

from hbllm.brain.reasoning.operators.abduction import AbductionOperator
from hbllm.brain.reasoning.operators.active_inference import ActiveInferenceOperator
from hbllm.brain.reasoning.operators.analogy import AnalogyOperator
from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    FrozenGraphView,
    OperatorInvocation,
    OperatorSelectionScore,
    OperatorTrace,
    ProblemType,
    ProvenanceChain,
    ReasoningBudget,
    ReasoningOperator,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.brain.reasoning.operators.causal import CausalOperator
from hbllm.brain.reasoning.operators.contradiction import ContradictionOperator
from hbllm.brain.reasoning.operators.counterfactual import CounterfactualOperator

# ── Phase 7: New Operators ───────────────────────────────────────────
from hbllm.brain.reasoning.operators.deduction import DeductionOperator
from hbllm.brain.reasoning.operators.induction import InductionOperator

# ── Phase 8: Wrapped Existing Components ─────────────────────────────
from hbllm.brain.reasoning.operators.prediction import PredictionOperator
from hbllm.brain.reasoning.operators.registry import (
    OperatorRegistry,
    create_default_operator_registry,
)
from hbllm.brain.reasoning.operators.simulation import SimulationOperator
from hbllm.brain.reasoning.operators.snn_reasoning import SNNReasoningOperator
from hbllm.brain.reasoning.operators.spatial import SpatialOperator
from hbllm.brain.reasoning.operators.temporal import TemporalOperator

__all__ = [
    # Core types
    "CognitiveContext",
    "CognitiveResult",
    "FrozenGraphView",
    "OperatorInvocation",
    "OperatorRegistry",
    "create_default_operator_registry",
    "OperatorSelectionScore",
    "OperatorTrace",
    "ProblemType",
    "ProvenanceChain",
    "ReasoningBudget",
    "ReasoningOperator",
    "ReasoningProblem",
    "ResourceCost",
    "ResultStatus",
    # Phase 7: New operators
    "DeductionOperator",
    "InductionOperator",
    "AbductionOperator",
    "TemporalOperator",
    "SpatialOperator",
    "AnalogyOperator",
    # Phase 8: Wrapped existing
    "PredictionOperator",
    "ContradictionOperator",
    "CounterfactualOperator",
    "CausalOperator",
    "ActiveInferenceOperator",
    "SimulationOperator",
    "SNNReasoningOperator",
]
