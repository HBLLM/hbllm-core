"""A18 Embodied Simulation & Counterfactual Mental Sandbox Package.

Provides copy-on-write simulation branches over HCIR, deterministic state-transition
operators, geometric support/stability reasoning, multi-branch counterfactual planning,
and risk/safety evaluation.
"""

from hbllm.brain.simulation.branch import SimulationBranch
from hbllm.brain.simulation.counterfactual_engine import (
    CounterfactualResult,
    ExecutionPlan,
    MentalSandbox,
    PredictedWorldState,
)
from hbllm.brain.simulation.events import SimulationEvent, compute_state_hash
from hbllm.brain.simulation.geometry import (
    BoundingBox,
    SurfaceGeometry,
    derive_surface_geometry,
    evaluate_support_stability,
    is_path_clear,
)
from hbllm.brain.simulation.operators import (
    ActionOperator,
    MoveOperator,
    OperatorExecutionResult,
    PushOperator,
    PutInOperator,
    StackOperator,
)

__all__ = [
    "ActionOperator",
    "BoundingBox",
    "CounterfactualResult",
    "ExecutionPlan",
    "MentalSandbox",
    "MoveOperator",
    "OperatorExecutionResult",
    "PredictedWorldState",
    "PushOperator",
    "PutInOperator",
    "SimulationBranch",
    "SimulationEvent",
    "StackOperator",
    "SurfaceGeometry",
    "compute_state_hash",
    "derive_surface_geometry",
    "evaluate_support_stability",
    "is_path_clear",
]
