"""
A12 — Reasoning Operators Package.

Composable cognitive operators that read immutable HCIR views
and propose state transitions.  No operator owns cognitive state.

Public API::

    from hbllm.hcir.reasoning.operators import (
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

from hbllm.hcir.reasoning.operators.abduction import AbductionOperator
from hbllm.hcir.reasoning.operators.active_inference import ActiveInferenceOperator
from hbllm.hcir.reasoning.operators.analogy import AnalogyOperator
from hbllm.hcir.reasoning.operators.base import (
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
from hbllm.hcir.reasoning.operators.causal import CausalOperator
from hbllm.hcir.reasoning.operators.contradiction import ContradictionOperator
from hbllm.hcir.reasoning.operators.counterfactual import CounterfactualOperator

# ── Phase 7: New Operators ───────────────────────────────────────────
from hbllm.hcir.reasoning.operators.deduction import DeductionOperator
from hbllm.hcir.reasoning.operators.embodied_causal import EmbodiedCausalOperator
from hbllm.hcir.reasoning.operators.induction import InductionOperator

# ── Phase 8: Wrapped Existing Components ─────────────────────────────
from hbllm.hcir.reasoning.operators.prediction import PredictionOperator
from hbllm.hcir.reasoning.operators.registry import (
    OperatorRegistry,
    create_default_operator_registry,
)
from hbllm.hcir.reasoning.operators.simulation import SimulationOperator
from hbllm.hcir.reasoning.operators.snn_reasoning import SNNReasoningOperator
from hbllm.hcir.reasoning.operators.spatial import SpatialOperator
from hbllm.hcir.reasoning.operators.temporal import TemporalOperator

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
    # Embodiment
    "EmbodiedCausalOperator",
]
