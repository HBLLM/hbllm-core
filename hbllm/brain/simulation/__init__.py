"""A18 Embodied Simulation & Counterfactual Mental Sandbox Package.

Provides copy-on-write simulation branches over HCIR, deterministic state-transition
operators, geometric support/stability reasoning, multi-branch counterfactual planning,
and risk/safety evaluation.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "SimulationBranch": ("hbllm.brain.simulation.branch", "SimulationBranch"),
    "CounterfactualResult": (
        "hbllm.brain.simulation.counterfactual_engine",
        "CounterfactualResult",
    ),
    "ExecutionPlan": ("hbllm.brain.simulation.counterfactual_engine", "ExecutionPlan"),
    "MentalSandbox": ("hbllm.brain.simulation.counterfactual_engine", "MentalSandbox"),
    "PredictedWorldState": (
        "hbllm.brain.simulation.counterfactual_engine",
        "PredictedWorldState",
    ),
    "SimulationEvent": ("hbllm.brain.simulation.events", "SimulationEvent"),
    "compute_state_hash": ("hbllm.brain.simulation.events", "compute_state_hash"),
    "BoundingBox": ("hbllm.brain.simulation.geometry", "BoundingBox"),
    "SurfaceGeometry": ("hbllm.brain.simulation.geometry", "SurfaceGeometry"),
    "derive_surface_geometry": ("hbllm.brain.simulation.geometry", "derive_surface_geometry"),
    "evaluate_support_stability": (
        "hbllm.brain.simulation.geometry",
        "evaluate_support_stability",
    ),
    "is_path_clear": ("hbllm.brain.simulation.geometry", "is_path_clear"),
    "ActionOperator": ("hbllm.brain.simulation.operators", "ActionOperator"),
    "MoveOperator": ("hbllm.brain.simulation.operators", "MoveOperator"),
    "OperatorExecutionResult": (
        "hbllm.brain.simulation.operators",
        "OperatorExecutionResult",
    ),
    "PushOperator": ("hbllm.brain.simulation.operators", "PushOperator"),
    "PutInOperator": ("hbllm.brain.simulation.operators", "PutInOperator"),
    "StackOperator": ("hbllm.brain.simulation.operators", "StackOperator"),
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
