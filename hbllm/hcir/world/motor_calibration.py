"""Motor Calibration — Empirical action dynamics and state mutation models.

Provides domain-agnostic data models for tracking:
- Action → displacement / outcome mappings (ActionDynamicsModel)
- Environmental state change rules (StateMutationModel)

These are learned through empirical probing and trial-and-error interaction,
mimicking human motor and causal discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionDynamicsModel:
    """Empirical causal model mapping an action to an observed displacement vector."""

    action_id: int | str
    delta_r: int = 0
    delta_c: int = 0
    confidence: float = 0.5
    probes_tested: int = 0
    displacement: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.displacement:
            self.displacement = (float(self.delta_r), float(self.delta_c))
        elif self.delta_r == 0 and self.delta_c == 0 and len(self.displacement) >= 2:
            self.delta_r = int(round(self.displacement[0]))
            self.delta_c = int(round(self.displacement[1]))

    def update_from_trial(
        self,
        observed_delta: tuple[int | float, ...],
        success: bool = True,
        learning_rate: float = 0.5,
    ) -> None:
        """Update empirical dynamics through trial-and-error observation."""
        self.probes_tested += 1
        if success:
            if len(observed_delta) >= 2:
                dr, dc = float(observed_delta[0]), float(observed_delta[1])
                new_dr = (1.0 - learning_rate) * self.delta_r + learning_rate * dr
                new_dc = (1.0 - learning_rate) * self.delta_c + learning_rate * dc
                self.delta_r = int(round(new_dr))
                self.delta_c = int(round(new_dc))
                self.displacement = (new_dr, new_dc)
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.0, self.confidence - 0.15)

    def describe(self) -> str:
        return f"Action({self.action_id}) -> (Δr={self.delta_r:+d}, Δc={self.delta_c:+d}) [conf={self.confidence:.2f}]"


@dataclass
class StateMutationModel:
    """Discrete causal rule mapping an environmental trigger to an observable state mutation."""

    trigger_type: str  # e.g., "CONTACT", "ACTION", "PROXIMITY"
    trigger_pos: tuple[int, ...] | None = None
    trigger_feature: Any = None  # Generic feature identifier (visual, tactile, token, color)
    mutation_type: str = "MUTATION"  # "PROPERTY_CHANGE", "BARRIER_OPEN", "ROTATION", "STATE_CHANGE"
    prior_value: Any = None
    posterior_value: Any = None
    confidence: float = 0.5
    occurrences: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        trigger_type: str,
        trigger_pos: tuple[int, ...] | None = None,
        trigger_feature: Any = None,
        mutation_type: str = "MUTATION",
        prior_value: Any = None,
        posterior_value: Any = None,
        confidence: float = 0.5,
        occurrences: int = 1,
        # Backward-compatibility kwargs
        trigger_color: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.trigger_type = trigger_type
        self.trigger_pos = trigger_pos
        self.trigger_feature = trigger_feature if trigger_feature is not None else trigger_color
        self.mutation_type = mutation_type
        self.prior_value = prior_value
        self.posterior_value = posterior_value
        self.confidence = confidence
        self.occurrences = occurrences
        self.metadata = dict(kwargs)

    @property
    def trigger_color(self) -> Any:
        return self.trigger_feature

    @trigger_color.setter
    def trigger_color(self, val: Any) -> None:
        self.trigger_feature = val

    def record_observation(self, observed_posterior: Any, learning_rate: float = 0.2) -> None:
        """Update causal rule confidence and state through trial-and-error observation."""
        self.occurrences += 1
        if observed_posterior == self.posterior_value:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.posterior_value = observed_posterior
            self.confidence = max(0.2, self.confidence - learning_rate)
