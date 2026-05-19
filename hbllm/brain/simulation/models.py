"""Simulation and Anticipatory Reasoning Data Models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.brain.autonomy.task_graph import TaskNode


class PredictionOrigin(StrEnum):
    """The epistemic origin of a simulated prediction."""

    HISTORICAL = "historical"   # Derived from CausalGraph records
    INFERRED = "inferred"       # Derived from deterministic rules or tight correlation
    SPECULATIVE = "speculative" # Hallucinated by LLM due to novel domain


@dataclass
class FutureWorldState:
    """A projected future state of the world, strictly isolated from WorldState."""

    state_id: str = field(default_factory=lambda: f"fws_{uuid.uuid4().hex[:12]}")
    base_clock: int = 0         # The logical clock of the WorldState this branches from
    mutations: dict[str, Any] = field(default_factory=dict) # Entity deltas applied over the base state
    predicted_confidence: float = 1.0 # Overall confidence in this future state
    prediction_origin: PredictionOrigin = PredictionOrigin.INFERRED
    risk_score: float = 0.0     # 0.0 to 1.0 composite risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "base_clock": self.base_clock,
            "mutations": self.mutations,
            "predicted_confidence": self.predicted_confidence,
            "prediction_origin": self.prediction_origin.value,
            "risk_score": self.risk_score,
        }


@dataclass
class CounterfactualScenario:
    """A multi-branch sequence of actions leading to a FutureWorldState."""

    scenario_id: str = field(default_factory=lambda: f"sce_{uuid.uuid4().hex[:12]}")
    goal_id: str = ""
    proposed_tasks: list[TaskNode] = field(default_factory=list)
    predicted_state: FutureWorldState | None = None
    utility_score: float = 0.0
    risk_categories: dict[str, float] = field(default_factory=dict)
