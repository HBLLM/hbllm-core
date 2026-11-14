"""Canonical Task Environments and Independent Ground-Truth Oracle.

Ensures strict parity across all cohorts by providing identical observations,
action vocabulary, and evaluation criteria, while hosting an independent
ground-truth oracle outside all cognitive cohorts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class EnvironmentObservation:
    """The canonical observation package delivered to all cohorts equally."""

    step_index: int
    visible_entities: list[dict[str, Any]]
    spatial_relations: list[dict[str, Any]]
    goal_description: str
    available_actions: list[dict[str, Any]]
    interaction_history: list[dict[str, Any]]
    resource_budget: dict[str, Any]
    feedback_signal: float | None = None
    is_terminal: bool = False


@dataclass
class PhysicalEnvironmentState:
    """True hidden environment state (never exposed directly to cohorts)."""

    entities: dict[str, dict[str, Any]]
    physics_rules: dict[str, Any]
    target_goal: dict[str, Any]


class IndependentEnvironmentOracle:
    """An independent ground-truth evaluator that calculates optimal utility and optimal probes

    without using any cohort's internal reasoning or decision engine.
    """

    def compute_optimal_probe(
        self,
        candidate_probes: list[dict[str, Any]],
        hypotheses_probabilities: dict[str, float],
    ) -> tuple[dict[str, Any] | None, float]:
        """Compute the mathematically optimal probe action based on information entropy reduction per unit cost."""
        if not candidate_probes or not hypotheses_probabilities:
            return None, 0.0

        # Prior entropy
        h_prior = -sum(p * math.log2(p) for p in hypotheses_probabilities.values() if p > 0.0)

        best_probe = None
        best_efficiency = -1.0

        for probe in candidate_probes:
            cost = float(probe.get("cost", 1.0))
            # Expected posterior entropy after probe
            # Discriminative power d in [0, 1]
            d_power = float(probe.get("discriminative_power", 0.5))
            expected_h_post = max(0.0, h_prior * (1.0 - d_power))
            delta_h = max(0.0, h_prior - expected_h_post)
            efficiency = delta_h / max(0.01, cost)

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_probe = probe

        return best_probe, best_efficiency

    def evaluate_true_action_utility(
        self,
        action: dict[str, Any],
        true_state: PhysicalEnvironmentState,
    ) -> float:
        """Compute objective task utility of an action against true physics and goals."""
        act_name = action.get("name", "")
        params = action.get("parameters", {})

        if act_name == "STACK":
            base_id = params.get("base", "")
            base = true_state.entities.get(base_id, {})
            # True physics constraint: support must be flat
            is_flat = base.get("surface_geometry") == "flat"
            is_rigid = base.get("is_rigid", True)
            if is_flat and is_rigid:
                return 1.0
            return -0.80  # Unstable collapse

        if act_name == "PUT_IN":
            container_id = params.get("container", "")
            cont = true_state.entities.get(container_id, {})
            if cont.get("is_open", True):
                return 1.0
            return -0.50

        if act_name == "GENTLE_TAP_PROBE":
            return 0.40  # Low cost information gain

        return 0.0


class CanonicalTaskEnvironment:
    """The canonical deterministic physical environment executing tasks."""

    def __init__(self, domain: str, seed: int = 42) -> None:
        self.domain = domain
        self.seed = seed
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self.oracle = IndependentEnvironmentOracle()
        self.state = self._initialize_state()

    def _initialize_state(self) -> PhysicalEnvironmentState:
        """Build initial true hidden physical state."""
        return PhysicalEnvironmentState(
            entities={
                "obj_support": {
                    "id": "obj_support",
                    "surface_geometry": "flat",
                    "is_rigid": True,
                    "color": "blue",
                },
                "obj_payload": {
                    "id": "obj_payload",
                    "surface_geometry": "flat",
                    "is_rigid": True,
                    "color": "red",
                },
                "obj_curved": {
                    "id": "obj_curved",
                    "surface_geometry": "curved",
                    "is_rigid": True,
                    "color": "green",
                },
            },
            physics_rules={"gravity": 9.81, "friction": 0.5},
            target_goal={"type": "STABLE_SUPPORT", "target": "obj_payload", "base": "obj_support"},
        )

    def reset(self) -> EnvironmentObservation:
        """Reset environment to initial state and return initial canonical observation."""
        self.step_count = 0
        self.history = []
        self.state = self._initialize_state()
        return self._build_canonical_observation()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[EnvironmentObservation, float, bool, dict[str, Any]]:
        """Execute action, advance environment state, and return observation, reward, done, info."""
        self.step_count += 1
        reward = self.oracle.evaluate_true_action_utility(action, self.state)
        is_success = reward > 0.0

        step_record = {
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "is_success": is_success,
        }
        self.history.append(step_record)

        is_terminal = self.step_count >= 10 or is_success
        obs = self._build_canonical_observation(feedback=reward, is_terminal=is_terminal)

        info = {
            "true_utility": reward,
            "is_success": is_success,
            "oracle_optimal_action": "STACK" if is_success else "GENTLE_TAP_PROBE",
        }
        return obs, reward, is_terminal, info

    def _build_canonical_observation(
        self,
        feedback: float | None = None,
        is_terminal: bool = False,
    ) -> EnvironmentObservation:
        """Construct identical public observation for all cohorts."""
        visible = [
            {"id": "obj_support", "color": "blue", "visible_shape": "box"},
            {"id": "obj_payload", "color": "red", "visible_shape": "cube"},
            {"id": "obj_curved", "color": "green", "visible_shape": "cylinder"},
        ]
        relations = [
            {"subject": "obj_support", "relation": "ON", "object": "table"},
            {"subject": "obj_payload", "relation": "ON", "object": "table"},
        ]
        actions = [
            {
                "name": "STACK",
                "parameters": {"item": "obj_payload", "base": "obj_support"},
                "cost": 1.0,
            },
            {
                "name": "STACK",
                "parameters": {"item": "obj_payload", "base": "obj_curved"},
                "cost": 1.0,
            },
            {
                "name": "GENTLE_TAP_PROBE",
                "parameters": {"target": "obj_support"},
                "cost": 0.2,
                "discriminative_power": 0.8,
            },
            {"name": "ABSTAIN", "parameters": {}, "cost": 0.0},
        ]
        return EnvironmentObservation(
            step_index=self.step_count,
            visible_entities=visible,
            spatial_relations=relations,
            goal_description="Construct stable support for obj_payload",
            available_actions=actions,
            interaction_history=list(self.history),
            resource_budget={"max_steps": 10, "max_simulation_depth": 5, "max_tokens": 1024},
            feedback_signal=feedback,
            is_terminal=is_terminal,
        )
